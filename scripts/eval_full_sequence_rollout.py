#!/usr/bin/env python3
"""
Full-Sequence Autoregressive Rollout Evaluation for Stage 2 SlotFormer.

Evaluates Stage 2 SlotFormer on full-length episode trajectories (16 frames and 50 frames),
measuring long-horizon future rollout mIoU, MSE, slot swapping rates, and saving full-sequence rollout GIFs.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.models.rollout import predict_slot_rollout
from src.datasets import DeterministicEpisodeEvalDataset
from src.utils.vis_utils import render_slot_overlay_frame, save_frames_to_gif
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
from src.utils.data_utils import find_dataset_path


def run_full_sequence_rollout(
    ckpt_path: str,
    n_cond_frames: int = 2,
    n_sample_frames: int = 16,
    batch_size: int = 16,
    device: str = "cuda",
    out_gif: str = "scratch/rollout_full_sequence_16f.gif",
):
    print("=" * 80)
    print(f"FULL-SEQUENCE AUTOREGRESSIVE ROLLOUT EVALUATION")
    print(f"Checkpoint:             {ckpt_path}")
    print(f"Context Frames:         {n_cond_frames}")
    print(f"Total Sequence Length:  {n_sample_frames} frames (Rollout horizon: {n_sample_frames - n_cond_frames} steps)")
    print(f"Device:                 {device}")
    print("=" * 80)

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=find_dataset_path(None),
        split="val",
        resolution=(64, 64),
        n_sample_frames=n_sample_frames,
        clips_per_episode=2,
        seed=42,
    )
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Reconstruct the experiment from the checkpoint
    model, cfg = bootstrap_checkpoint(ckpt_path)
    model = model.to(device)
    from src.utils.training_utils import load_checkpoint_state
    load_checkpoint_state(model, ckpt_path, device=device)
    model.eval()

    cond_mses, rollout_mses = [], []
    cond_mious, rollout_mious = [], []

    per_frame_mse = [[] for _ in range(n_sample_frames)]
    per_frame_miou = [[] for _ in range(n_sample_frames)]

    total_rollout_transitions = 0
    swap_rollout_transitions = 0
    total_episodes = 0
    episodes_with_swap = 0
    episodes_multi_swap = 0

    vis_frames_full_ep = []
    max_vis_episodes = 2

    start_t = time.time()
    num_batches = len(val_loader)

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            video = batch["img"].to(device)  # [B, T, C, H, W]
            out = predict_slot_rollout(model, video, n_cond_frames=n_cond_frames)

            B, T = video.shape[:2]
            recon = out["recon_img"]
            pred_masks = out["pred_masks"]
            gt_masks = batch.get("gt_masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)

            # Per-frame breakdown
            for t in range(T):
                mse_t = torch.mean((recon[:, t] - video[:, t]) ** 2).item()
                per_frame_mse[t].append(mse_t)
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
                    per_frame_miou[t].append(iou_t)
                    if t < n_cond_frames:
                        cond_mious.append(iou_t)
                    else:
                        rollout_mious.append(iou_t)

            # Slot Swapping Analysis per sequence
            if pred_masks is not None and gt_masks is not None:
                min_k = min(pred_masks.shape[2], gt_masks.shape[2])
                p_bin = (pred_masks[:, :, :min_k] > 0.5).float()
                g_bin = (gt_masks[:, :, :min_k] > 0.5).float()

                for b in range(B):
                    total_episodes += 1
                    ep_swap_count = 0
                    prev_assign = None

                    for t in range(n_cond_frames - 1, T):
                        p_bt = p_bin[b, t]
                        g_bt = g_bin[b, t]
                        inter_m = (p_bt.unsqueeze(1) * g_bt.unsqueeze(0)).sum(dim=(-2, -1))
                        union_m = p_bt.unsqueeze(1).sum(dim=(-2, -1)) + g_bt.unsqueeze(0).sum(dim=(-2, -1)) - inter_m
                        iou_m = (inter_m + 1e-6) / (union_m + 1e-6)

                        curr_assign = torch.argmax(iou_m, dim=1).tolist()
                        if prev_assign is not None:
                            total_rollout_transitions += 1
                            if curr_assign != prev_assign:
                                swap_rollout_transitions += 1
                                ep_swap_count += 1

                        prev_assign = curr_assign

                    if ep_swap_count >= 1:
                        episodes_with_swap += 1
                    if ep_swap_count >= 2:
                        episodes_multi_swap += 1

            # Render full sequence visualization for first few episodes
            if b_idx < max_vis_episodes:
                for b in range(min(B, 1)):
                    video_np = ((video[b].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                    pred_masks_np = pred_masks[b].cpu().numpy()
                    gt_masks_np = gt_masks[b].cpu().numpy() if gt_masks is not None else None

                    for t in range(T):
                        phase = "COND" if t < n_cond_frames else "ROLLOUT"
                        banner = f"Episode {b_idx+1} Frame {t+1}/{T} [{phase}] | mIoU: {per_frame_miou[t][-1]*100:.1f}%"
                        combined = render_slot_overlay_frame(
                            frame_rgb=video_np[t],
                            pred_masks_t=pred_masks_np[t],
                            gt_masks_t=gt_masks_np[t] if gt_masks_np is not None else None,
                            banner_text=banner,
                        )
                        vis_frames_full_ep.append(combined)

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s)")

    eval_time = time.time() - start_t
    avg_per_frame_miou = [float(np.mean(m) * 100) for m in per_frame_miou]
    avg_per_frame_mse = [float(np.mean(m)) for m in per_frame_mse]

    transition_swap_pct = (swap_rollout_transitions / max(1, total_rollout_transitions)) * 100.0
    ep_swap_pct = (episodes_with_swap / max(1, total_episodes)) * 100.0
    multi_swap_pct = (episodes_multi_swap / max(1, total_episodes)) * 100.0

    print("\n" + "=" * 80)
    print(f"FULL-SEQUENCE ROLLOUT EVALUATION SUMMARY ({n_sample_frames} FRAMES)")
    print("=" * 80)
    print(f"Total Evaluated Episodes:              {total_episodes}")
    print(f"Total Autoregressive Steps per Ep:    {n_sample_frames - n_cond_frames}")
    print(f"Conditioned Context MSE (t < {n_cond_frames}):       {np.mean(cond_mses):.6f}")
    print(f"Conditioned Context mIoU (t < {n_cond_frames}):      {np.mean(cond_mious)*100:.2f}%")
    print(f"Future Rollout MSE (t >= {n_cond_frames}):           {np.mean(rollout_mses):.6f}")
    print(f"Future Rollout mIoU (t >= {n_cond_frames}):          {np.mean(rollout_mious)*100:.2f}%")
    print(f"Transition Slot Swap Rate:             {transition_swap_pct:.2f}% ({swap_rollout_transitions}/{total_rollout_transitions})")
    print(f"Episodes with >= 1 Slot Swap:          {ep_swap_pct:.2f}% ({episodes_with_swap}/{total_episodes})")
    print(f"Episodes with >= 2 Slot Swaps:         {multi_swap_pct:.2f}% ({episodes_multi_swap}/{total_episodes})")
    print(f"Evaluation Wall Time:                  {eval_time:.2f}s")
    print("=" * 80)

    print("\nPer-Frame mIoU Trajectory Breakdown:")
    for t, miou_val in enumerate(avg_per_frame_miou):
        phase = "COND   " if t < n_cond_frames else "ROLLOUT"
        bar = "█" * int(miou_val / 4)
        print(f"  Frame {t+1:02d} [{phase}]: {miou_val:5.2f}% {bar}")

    if vis_frames_full_ep:
        os.makedirs(os.path.dirname(out_gif), exist_ok=True)
        save_frames_to_gif(vis_frames_full_ep, out_gif, fps=4)
        print(f"\nSaved Full-Sequence Rollout Visualization GIF to: {out_gif}")

    res = {
        "ckpt_path": ckpt_path,
        "n_cond_frames": n_cond_frames,
        "n_sample_frames": n_sample_frames,
        "n_rollout_steps": n_sample_frames - n_cond_frames,
        "total_episodes": total_episodes,
        "cond_mse": float(np.mean(cond_mses)),
        "cond_miou_pct": float(np.mean(cond_mious) * 100),
        "rollout_mse": float(np.mean(rollout_mses)),
        "rollout_miou_pct": float(np.mean(rollout_mious) * 100),
        "transition_swap_rate_pct": float(transition_swap_pct),
        "episodes_with_swap_pct": float(ep_swap_pct),
        "episodes_multi_swap_pct": float(multi_swap_pct),
        "per_frame_miou_pct": avg_per_frame_miou,
        "per_frame_mse": avg_per_frame_mse,
        "eval_time_sec": float(eval_time),
    }

    out_json = os.path.splitext(out_gif)[0] + "_metrics.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved full-sequence metrics JSON: {out_json}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Sequence Stage 2 SlotFormer Rollout Evaluation")
    parser.add_argument("--ckpt_path", type=str, default="scratch/checkpoints/slotformer_pusht/slotformer_best.pt")
    parser.add_argument("--n_cond_frames", type=int, default=2)
    parser.add_argument("--n_sample_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_gif", type=str, default="scratch/rollout_full_sequence_16f.gif")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_full_sequence_rollout(
        ckpt_path=args.ckpt_path,
        n_cond_frames=args.n_cond_frames,
        n_sample_frames=args.n_sample_frames,
        batch_size=args.batch_size,
        out_gif=args.out_gif,
        device=args.device,
    )
