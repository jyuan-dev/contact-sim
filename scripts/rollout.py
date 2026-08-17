#!/usr/bin/env python3
"""
Unified Autoregressive Future Rollout Evaluation & Visualization CLI.

Supports:
  - Standard short-horizon clip evaluation (6-16 frames)
  - Long-horizon sequence evaluation (16, 50, or arbitrary frames)
  - Full-episode un-cropped evaluation (100+ frames per episode)
  - Multi-swap episode isolation and diagnostic visualization
  - Multi-panel composite rendering (3-panel: GT / Model Recon / Slot Masks, or Overlay)
  - Standardized ModelOutput evaluation across SAVi, SlotFormer, and LeWM

Usage Examples:
  # Standard clip rollout (6 frames, 2 cond)
  python scripts/rollout.py --ckpt_path scratch/checkpoints/slotformer_pusht/slotformer_best.pt --device cuda

  # Full-episode rollout with 3-panel rendering
  python scripts/rollout.py --ckpt_path scratch/checkpoints/slotformer_pusht/slotformer_best.pt --full_episode --render_mode 3panel

  # Long sequence evaluation (16 frames)
  python scripts/rollout.py --ckpt_path scratch/checkpoints/slotformer_pusht/slotformer_best.pt --n_sample_frames 16

  # Filter and visualize episodes with slot swapping
  python scripts/rollout.py --ckpt_path scratch/checkpoints/slotformer_pusht/slotformer_best.pt --filter_multi_swap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets import DeterministicEpisodeEvalDataset
from src.metrics import greedy_slot_assignments
from src.utils.vis_utils import render_slot_overlay_frame, save_frames_to_gif, SLOT_COLORS_RGB
from src.utils.training_utils import load_checkpoint_state
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
from src.utils.data_utils import find_dataset_path


def render_3panel_composite_frame(
    frame_rgb: np.ndarray,
    recon_rgb: np.ndarray | None,
    pred_masks_t: np.ndarray | None,
    gt_masks_t: np.ndarray | None = None,
    banner_text: str = "",
) -> np.ndarray:
    """
    Renders a 3-panel side-by-side composite frame:
      - Left: Ground Truth RGB frame with GT contours.
      - Middle: Model Autoregressively Reconstructed RGB Image.
      - Right: Color-coded slot mask overlay.
    """
    if frame_rgb.shape[:2] != (64, 64):
        frame_rgb = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)

    # 1. Left Panel: Ground Truth Outlines
    p_gt = frame_rgb.copy()
    if gt_masks_t is not None:
        for m_idx in range(gt_masks_t.shape[0]):
            m_bin = gt_masks_t[m_idx] > 0.5
            if m_bin.any():
                contours, _ = cv2.findContours(m_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color_rgb = {0: (255, 140, 0), 1: (0, 230, 115), 2: (0, 128, 255)}.get(m_idx, (255, 255, 255))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                cv2.drawContours(p_gt, contours, -1, color_bgr, 1)

    # 2. Middle Panel: Model Reconstruction
    if recon_rgb is not None:
        if recon_rgb.shape[:2] != (64, 64):
            p_recon = cv2.resize(recon_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
        else:
            p_recon = recon_rgb.copy()
    else:
        p_recon = np.zeros_like(p_gt)

    # 3. Right Panel: Color-Coded Slot Mask Overlay
    p_slots = frame_rgb.copy().astype(np.float32)
    if pred_masks_t is not None:
        K = pred_masks_t.shape[0]
        slot_map = np.zeros((64, 64, 3), dtype=np.float32)
        weight_sum = np.zeros((64, 64, 1), dtype=np.float32)
        for k in range(K):
            m_k = np.clip(pred_masks_t[k], 0, 1)[..., None]
            color_k = np.array(SLOT_COLORS_RGB[k % len(SLOT_COLORS_RGB)], dtype=np.float32)
            slot_map += m_k * color_k
            weight_sum += m_k

        weight_sum = np.maximum(weight_sum, 1e-6)
        slot_composite = slot_map / weight_sum
        active_mask = (weight_sum > 0.15)
        alpha = 0.60
        p_slots[active_mask[:, :, 0]] = (1.0 - alpha) * p_slots[active_mask[:, :, 0]] + alpha * slot_composite[active_mask[:, :, 0]]
    p_slots_uint8 = np.clip(p_slots, 0, 255).astype(np.uint8)

    combined = np.hstack([p_gt, p_recon, p_slots_uint8])
    combined_large = cv2.resize(combined, (720, 240), interpolation=cv2.INTER_NEAREST)

    if banner_text:
        cv2.rectangle(combined_large, (0, 0), (combined_large.shape[1], 18), (30, 140, 220), -1)
        cv2.putText(combined_large, banner_text, (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return combined_large


def run_full_episode_rollout(
    model: torch.nn.Module,
    h5_path: str,
    n_cond_frames: int = 2,
    num_episodes: int = 2,
    device: str = "cpu",
    render_mode: str = "3panel",
    out_gif_prefix: str = "scratch/rollout_full_episode",
) -> dict:
    """
    Rolls out complete uncropped episodes from start to finish.
    """
    print(f"\n--- Running Full-Episode Rollout on {h5_path} (Episodes: {num_episodes}) ---")
    h5_file = h5py.File(h5_path, "r")
    offs = h5_file["episode_ends_offsets"][:] if "episode_ends_offsets" in h5_file else None
    lens = h5_file["episode_lengths"][:] if "episode_lengths" in h5_file else None

    if offs is None or lens is None:
        ends = h5_file["meta"]["episode_ends"][:]
        offs = np.concatenate([[0], ends[:-1]])
        lens = np.diff(np.concatenate([[0], ends]))

    val_start_ep = int(len(lens) * 0.8)
    ep_results = []

    for i in range(num_episodes):
        ep_idx = val_start_ep + i
        if ep_idx >= len(lens):
            break

        ep_len = int(lens[ep_idx])
        ep_off = int(offs[ep_idx])
        frame_idxs = list(range(ep_off, ep_off + ep_len))

        frames = h5_file["pixels"][frame_idxs]  # [T, H, W, C]
        agent_m = (h5_file["agent_masks"][frame_idxs] > 127).astype(np.float32) if "agent_masks" in h5_file else None
        block_m = (h5_file["block_masks"][frame_idxs] > 127).astype(np.float32) if "block_masks" in h5_file else None
        goal_m = (h5_file["goal_masks"][frame_idxs] > 127).astype(np.float32) if "goal_masks" in h5_file else None

        if agent_m is not None and block_m is not None and goal_m is not None:
            goal_visible = np.clip(goal_m - np.maximum(block_m, agent_m), 0.0, 1.0)
            gt_masks = np.stack([agent_m, block_m, goal_visible], axis=1)  # [T, K, H, W]
            gt_masks_t = torch.from_numpy(gt_masks).unsqueeze(0).to(device)
        else:
            gt_masks = None
            gt_masks_t = None

        video = (frames.astype(np.float32) / 127.5) - 1.0
        img_t = torch.from_numpy(video.transpose(0, 3, 1, 2)).unsqueeze(0).to(device)

        start_t = time.time()
        with torch.no_grad():
            out = model.rollout(img_t, n_cond_frames=n_cond_frames)
        eval_time = time.time() - start_t

        recon_t = out.get("recon_img")
        pred_masks_t = out.get("pred_masks")

        cond_mses, rollout_mses = [], []
        cond_mious, rollout_mious = [], []

        for t in range(ep_len):
            if recon_t is not None:
                mse_t = torch.mean((recon_t[:, t] - img_t[:, t]) ** 2).item()
                if t < n_cond_frames:
                    cond_mses.append(mse_t)
                else:
                    rollout_mses.append(mse_t)

            if pred_masks_t is not None and gt_masks_t is not None:
                p_t = (pred_masks_t[0, t] > 0.5).float()
                g_t = (gt_masks_t[0, t] > 0.5).float()
                min_k = min(p_t.shape[0], g_t.shape[0])
                inter = (p_t[:min_k] * g_t[:min_k]).sum(dim=(-2, -1))
                union = p_t[:min_k].sum(dim=(-2, -1)) + g_t[:min_k].sum(dim=(-2, -1)) - inter
                iou_t = ((inter + 1e-6) / (union + 1e-6)).mean().item()
                if t < n_cond_frames:
                    cond_mious.append(iou_t)
                else:
                    rollout_mious.append(iou_t)

        vis_frames = []
        recon_np = ((recon_t[0].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8) if recon_t is not None else None
        pred_masks_np = pred_masks_t[0].cpu().numpy() if pred_masks_t is not None else None
        gt_masks_np = gt_masks_t[0].cpu().numpy() if gt_masks_t is not None else None

        for t in range(ep_len):
            phase = "COND" if t < n_cond_frames else f"ROLLOUT+{t+1-n_cond_frames}"
            banner = f"Episode {ep_idx} Frame {t+1}/{ep_len} [{phase}]"
            if render_mode == "3panel":
                frame_rendered = render_3panel_composite_frame(
                    frame_rgb=frames[t],
                    recon_rgb=recon_np[t] if recon_np is not None else None,
                    pred_masks_t=pred_masks_np[t] if pred_masks_np is not None else None,
                    gt_masks_t=gt_masks_np[t] if gt_masks_np is not None else None,
                    banner_text=banner,
                )
            else:
                frame_rendered = render_slot_overlay_frame(
                    frame_rgb=frames[t],
                    pred_masks_t=pred_masks_np[t] if pred_masks_np is not None else None,
                    gt_masks_t=gt_masks_np[t] if gt_masks_np is not None else None,
                    banner_text=banner,
                )
            vis_frames.append(frame_rendered)

        out_gif = f"{out_gif_prefix}_ep{ep_idx}.gif"
        os.makedirs(os.path.dirname(os.path.abspath(out_gif)), exist_ok=True)
        save_frames_to_gif(vis_frames, out_gif, fps=6)
        print(f"Saved Episode {ep_idx} ({ep_len} frames) rollout GIF to: {out_gif}")

        ep_results.append({
            "episode_idx": ep_idx,
            "total_frames": ep_len,
            "cond_mse": float(np.mean(cond_mses)) if cond_mses else float("nan"),
            "rollout_mse": float(np.mean(rollout_mses)) if rollout_mses else float("nan"),
            "cond_miou": float(np.mean(cond_mious) * 100.0) if cond_mious else float("nan"),
            "rollout_miou": float(np.mean(rollout_mious) * 100.0) if rollout_mious else float("nan"),
            "eval_time_sec": eval_time,
            "fps": ep_len / max(1e-4, eval_time),
            "out_gif": out_gif,
        })

    h5_file.close()
    return {"full_episode_rollouts": ep_results}


def run_rollout_evaluation(
    ckpt_path: str,
    n_cond_frames: int = 2,
    n_sample_frames: int = 6,
    base_seed: int = 42,
    clips_per_ep: int = 2,
    batch_size: int = 32,
    full_episode: bool = False,
    filter_multi_swap: bool = False,
    render_mode: str = "overlay",
    device: str = "cpu",
    out_gif: str = "scratch/rollout.gif",
    out_json: str = "scratch/rollout_results.json",
) -> dict:
    """
    Main Rollout Evaluation and Benchmarking Suite.
    """
    print("=" * 80)
    print(f"Autoregressive Future Rollout Evaluation: {ckpt_path}")
    print(f"  Cond Frames: {n_cond_frames} | Sample Frames: {n_sample_frames} | Device: {device}")
    print(f"  Mode: {'Full Episode' if full_episode else 'Sampled Clips'} | Render: {render_mode}")
    print("=" * 80)

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Reconstruct the experiment from the checkpoint
    model, cfg = bootstrap_checkpoint(ckpt_path)
    model = model.to(device)
    load_checkpoint_state(model, ckpt_path, device=device)
    model_name = cfg.get("model", {}).get("name", "unknown")
    print(f"[Rollout Eval] Model weights loaded into '{model_name}'.")
    model.eval()

    h5_path = find_dataset_path(None)

    # Branch: Full Episode Mode
    if full_episode:
        res = run_full_episode_rollout(
            model=model,
            h5_path=h5_path,
            n_cond_frames=n_cond_frames,
            num_episodes=2,
            device=device,
            render_mode=render_mode,
            out_gif_prefix="scratch/rollout_full_episode",
        )
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        return res

    # Branch: Clip Evaluation Mode (Standard / Long Sequence / Multi-Swap Filter)
    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=h5_path,
        split="val",
        resolution=(64, 64),
        n_sample_frames=n_sample_frames,
        clips_per_episode=clips_per_ep,
        seed=base_seed,
    )
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    cond_mses, rollout_mses = [], []
    cond_mious, rollout_mious = [], []

    total_rollout_transitions = 0
    swap_rollout_transitions = 0
    swapped_rollout_seqs = 0
    total_rollout_seqs = 0
    multi_swap_seqs = 0

    vis_frames = []
    max_vis_seqs = 4
    found_filtered_ep = False

    start_t = time.time()
    num_batches = len(val_loader)

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            video = batch["img"].to(device)  # [B, T, C, H, W]
            actions = batch.get("action", None)
            if actions is not None:
                actions = actions.to(device)

            out = model.rollout(video, n_cond_frames=n_cond_frames, actions=actions)

            B, T = video.shape[:2]
            recon = out.get("recon_img")
            pred_masks = out.get("pred_masks")
            gt_masks = batch.get("gt_masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)

            # Per-frame MSE and mIoU metrics
            for t in range(T):
                if recon is not None:
                    mse_t = torch.mean((recon[:, t] - video[:, t]) ** 2).item()
                    if t < n_cond_frames:
                        cond_mses.append(mse_t)
                    else:
                        rollout_mses.append(mse_t)

                if pred_masks is not None and gt_masks is not None:
                    min_k = min(pred_masks.shape[2], gt_masks.shape[2])
                    p_t = (pred_masks[:, t, :min_k] > 0.5).float()
                    g_t = (gt_masks[:, t, :min_k] > 0.5).float()
                    inter = (p_t * g_t).sum(dim=(-2, -1))
                    union = p_t.sum(dim=(-2, -1)) + g_t.sum(dim=(-2, -1)) - inter
                    iou_t = ((inter + 1e-6) / (union + 1e-6)).mean().item()
                    if t < n_cond_frames:
                        cond_mious.append(iou_t)
                    else:
                        rollout_mious.append(iou_t)

            # Slot Swapping Analysis over Rollout steps (t >= n_cond_frames)
            if pred_masks is not None and gt_masks is not None:
                assign = greedy_slot_assignments(pred_masks, gt_masks)  # [B, T, K_gt]

                for b in range(B):
                    total_rollout_seqs += 1
                    ep_swap_count = 0

                    for t in range(n_cond_frames, T):
                        total_rollout_transitions += 1
                        if not torch.equal(assign[b, t], assign[b, t - 1]):
                            swap_rollout_transitions += 1
                            ep_swap_count += 1

                    if ep_swap_count >= 1:
                        swapped_rollout_seqs += 1
                    if ep_swap_count >= 2:
                        multi_swap_seqs += 1

                    # Visualization Capture Logic
                    should_capture = False
                    if filter_multi_swap:
                        if ep_swap_count >= 2 and not found_filtered_ep:
                            should_capture = True
                            found_filtered_ep = True
                    else:
                        if len(vis_frames) // T < max_vis_seqs and b == 0:
                            should_capture = True

                    if should_capture:
                        video_np = ((video[b].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                        recon_np = ((recon[b].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8) if recon is not None else None
                        pred_masks_np = pred_masks[b].cpu().numpy() if pred_masks is not None else None
                        gt_masks_np = gt_masks[b].cpu().numpy() if gt_masks is not None else None

                        for t in range(T):
                            phase_label = "COND" if t < n_cond_frames else f"ROLLOUT+{t+1-n_cond_frames}"
                            banner = f"Seq {b_idx*B + b + 1} Frame {t+1}/{T} [{phase_label}] | Swaps: {ep_swap_count}"
                            if render_mode == "3panel":
                                frame_rendered = render_3panel_composite_frame(
                                    frame_rgb=video_np[t],
                                    recon_rgb=recon_np[t] if recon_np is not None else None,
                                    pred_masks_t=pred_masks_np[t] if pred_masks_np is not None else None,
                                    gt_masks_t=gt_masks_np[t] if gt_masks_np is not None else None,
                                    banner_text=banner,
                                )
                            else:
                                frame_rendered = render_slot_overlay_frame(
                                    frame_rgb=video_np[t],
                                    pred_masks_t=pred_masks_np[t] if pred_masks_np is not None else None,
                                    gt_masks_t=gt_masks_np[t] if gt_masks_np is not None else None,
                                    banner_text=banner,
                                )
                            vis_frames.append(frame_rendered)

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated Rollout [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s)")

    rollout_swap_rate = (swap_rollout_transitions / max(1, total_rollout_transitions)) * 100.0
    seq_rollout_swap_rate = (swapped_rollout_seqs / max(1, total_rollout_seqs)) * 100.0
    multi_swap_seq_rate = (multi_swap_seqs / max(1, total_rollout_seqs)) * 100.0

    res = {
        "ckpt_path": ckpt_path,
        "n_cond_frames": n_cond_frames,
        "n_sample_frames": n_sample_frames,
        "n_rollout_frames": n_sample_frames - n_cond_frames,
        "cond_mse": float(np.mean(cond_mses)) if cond_mses else float("nan"),
        "rollout_mse": float(np.mean(rollout_mses)) if rollout_mses else float("nan"),
        "cond_miou": float(np.mean(cond_mious)) * 100.0 if cond_mious else 0.0,
        "rollout_miou": float(np.mean(rollout_mious)) * 100.0 if rollout_mious else 0.0,
        "slot_swap_transition_rate_pct": float(rollout_swap_rate),
        "sequence_swap_rate_pct": float(seq_rollout_swap_rate),
        "multi_swap_sequence_rate_pct": float(multi_swap_seq_rate),
        "total_evaluated_sequences": total_rollout_seqs,
    }

    print("\n" + "=" * 80)
    print(f"Rollout Evaluation Summary for {ckpt_path}:")
    print(f"  Condition Context Frames:               {n_cond_frames}")
    print(f"  Autoregressive Rollout Frames:          {n_sample_frames - n_cond_frames}")
    print(f"  Conditioned Phase MSE:                  {res['cond_mse']:.6f}")
    print(f"  Future Rollout Phase MSE:               {res['rollout_mse']:.6f}")
    print(f"  Conditioned Phase mIoU:                 {res['cond_miou']:.2f}%")
    print(f"  Future Rollout Phase mIoU:              {res['rollout_miou']:.2f}%")
    print(f"  Slot Swap Transition Rate:              {res['slot_swap_transition_rate_pct']:.2f}% ({swap_rollout_transitions}/{total_rollout_transitions})")
    print(f"  Sequences with >= 1 Slot Swap:          {res['sequence_swap_rate_pct']:.2f}%")
    print(f"  Sequences with >= 2 Slot Swaps:         {res['multi_swap_sequence_rate_pct']:.2f}%")
    print("=" * 80 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved rollout evaluation metrics to: {out_json}")

    if vis_frames:
        os.makedirs(os.path.dirname(os.path.abspath(out_gif)), exist_ok=True)
        save_frames_to_gif(vis_frames, out_gif, fps=4 if not full_episode else 6)
        print(f"Saved Rollout visualization GIF to: {out_gif}")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Autoregressive Future Rollout Evaluation & Visualization CLI")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--n_cond_frames", type=int, default=2, help="Number of conditioning context frames")
    parser.add_argument("--n_sample_frames", type=int, default=6, help="Sequence length (e.g. 6, 10, 16, 50)")
    parser.add_argument("--base_seed", type=int, default=42, help="Evaluation random seed")
    parser.add_argument("--clips_per_ep", type=int, default=2, help="Clips per episode to sample")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--full_episode", action="store_true", help="Evaluate full uncropped episodes")
    parser.add_argument("--filter_multi_swap", action="store_true", help="Isolate and visualize episodes with slot swapping")
    parser.add_argument("--render_mode", type=str, default="overlay", choices=["overlay", "3panel"], help="Rendering mode for GIFs")
    parser.add_argument("--out_gif", type=str, default="scratch/rollout.gif", help="Output GIF path")
    parser.add_argument("--out_json", type=str, default="scratch/rollout_results.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_rollout_evaluation(
        ckpt_path=args.ckpt_path,
        n_cond_frames=args.n_cond_frames,
        n_sample_frames=args.n_sample_frames,
        base_seed=args.base_seed,
        clips_per_ep=args.clips_per_ep,
        batch_size=args.batch_size,
        full_episode=args.full_episode,
        filter_multi_swap=args.filter_multi_swap,
        render_mode=args.render_mode,
        out_gif=args.out_gif,
        out_json=args.out_json,
        device=args.device,
    )
