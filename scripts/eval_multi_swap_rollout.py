#!/usr/bin/env python3
"""
Full Stage 2 SlotFormer Evaluation & Multi-Swap Episode Visualization Script.

Evaluates Stage 2 SlotFormer on PushT dataset, calculates full metrics,
and specifically isolates and visualizes episodes with multiple slot swaps (> 1 swap).
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


def run_full_evaluation(
    ckpt_path: str,
    n_cond_frames: int = 2,
    n_sample_frames: int = 10,
    base_seed: int = 42,
    batch_size: int = 32,
    device: str = "cuda",
    out_gif: str = "scratch/rollout_multi_swap_episode.gif",
):
    print("=" * 80)
    print(f"FULL STAGE 2 SLOTFORMER EVALUATION (Multi-Swap Isolation)")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Conditioning Frames: {n_cond_frames} | Total Sample Frames: {n_sample_frames} | Device: {device}")
    print("=" * 80)

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    h5_path = "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5"
    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=h5_path,
        split="val",
        resolution=(64, 64),
        n_sample_frames=n_sample_frames,
        clips_per_episode=2,
        seed=base_seed,
    )
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 1. Build Stage 2 SlotFormer model dynamically
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    
    d_model = 128
    ffn_dim = 512
    num_layers = 4
    for k, v in state_dict.items():
        if "rollouter.in_proj.weight" in k:
            d_model = v.shape[0]
        elif "rollouter.transformer_encoder.layers.0.linear1.weight" in k:
            ffn_dim = v.shape[0]
        elif "rollouter.transformer_encoder.layers." in k:
            parts = k.split(".")
            for p in parts:
                if p.isdigit():
                    num_layers = max(num_layers, int(p) + 1)
                    
    cfg = {
        "model": {
            "name": "slotformer",
            "type": "slotformer",
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": 8,
            "ffn_dim": ffn_dim,
            "stage1_ckpt_path": "scratch/checkpoints/savi_pusht/savi_best.pt",
        }
    }

    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device or "cpu").get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    cond_mses, rollout_mses = [], []
    cond_mious, rollout_mious = [], []

    total_rollout_transitions = 0
    swap_rollout_transitions = 0
    total_episodes = 0
    episodes_with_swap = 0
    episodes_multi_swap = 0  # Episode with >= 2 slot swaps

    multi_swap_gif_frames = []
    found_multi_swap_ep = False

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

            # Per-frame MSE and mIoU
            for t in range(T):
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

                    # Save frames for multi-swap episode if found
                    if ep_swap_count >= 1 and not found_multi_swap_ep:
                        found_multi_swap_ep = True
                        video_np = ((video[b].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                        pred_masks_np = pred_masks[b].cpu().numpy()
                        gt_masks_np = gt_masks[b].cpu().numpy() if gt_masks is not None else None

                        for t in range(T):
                            phase_label = "COND" if t < n_cond_frames else "ROLLOUT"
                            banner = f"Multi-Swap Ep (Swaps: {ep_swap_count}) Frame {t+1}/{T} [{phase_label}]"
                            combined = render_slot_overlay_frame(
                                frame_rgb=video_np[t],
                                pred_masks_t=pred_masks_np[t],
                                gt_masks_t=gt_masks_np[t] if gt_masks_np is not None else None,
                                banner_text=banner,
                            )
                            multi_swap_gif_frames.append(combined)

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s)")

    eval_time = time.time() - start_t
    transition_swap_pct = (swap_rollout_transitions / max(1, total_rollout_transitions)) * 100.0
    ep_swap_pct = (episodes_with_swap / max(1, total_episodes)) * 100.0
    multi_swap_pct = (episodes_multi_swap / max(1, total_episodes)) * 100.0

    print("\n" + "=" * 80)
    print("STAGE 2 SLOTFORMER FULL ROLLOUT EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Evaluated Sequences:             {total_episodes}")
    print(f"Total Rollout Transitions:             {total_rollout_transitions}")
    print(f"Conditioned Context MSE (t < 2):       {np.mean(cond_mses):.6f}")
    print(f"Conditioned Context mIoU (t < 2):      {np.mean(cond_mious)*100:.2f}%")
    print(f"Future Rollout MSE (t >= 2):           {np.mean(rollout_mses):.6f}")
    print(f"Future Rollout mIoU (t >= 2):          {np.mean(rollout_mious)*100:.2f}%")
    print(f"Slot Swap Transition Rate:             {transition_swap_pct:.2f}% ({swap_rollout_transitions}/{total_rollout_transitions})")
    print(f"Episodes with >= 1 Slot Swap:          {ep_swap_pct:.2f}% ({episodes_with_swap}/{total_episodes})")
    print(f"Episodes with >= 2 Slot Swaps:         {multi_swap_pct:.2f}% ({episodes_multi_swap}/{total_episodes})")
    print(f"Evaluation Wall Time:                  {eval_time:.2f}s")
    print("=" * 80)

    if multi_swap_gif_frames:
        os.makedirs(os.path.dirname(out_gif), exist_ok=True)
        save_frames_to_gif(multi_swap_gif_frames, out_gif, fps=4)
        print(f"\nSaved Multi-Swap Episode Visualization GIF to: {out_gif}")

    res = {
        "ckpt_path": ckpt_path,
        "n_cond_frames": n_cond_frames,
        "n_sample_frames": n_sample_frames,
        "total_episodes": total_episodes,
        "cond_mse": float(np.mean(cond_mses)),
        "cond_miou_pct": float(np.mean(cond_mious) * 100),
        "rollout_mse": float(np.mean(rollout_mses)),
        "rollout_miou_pct": float(np.mean(rollout_mious) * 100),
        "slot_swap_transition_rate_pct": float(transition_swap_pct),
        "episodes_with_swap_pct": float(ep_swap_pct),
        "episodes_multi_swap_pct": float(multi_swap_pct),
        "eval_time_sec": float(eval_time),
    }

    out_file = os.path.splitext(out_gif)[0] + "_metrics.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved metrics JSON: {out_file}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Stage 2 Evaluation with Multi-Swap Isolation")
    parser.add_argument("--ckpt_path", type=str, default="scratch/checkpoints/slotformer_pusht/slotformer_best.pt")
    parser.add_argument("--n_cond_frames", type=int, default=2)
    parser.add_argument("--n_sample_frames", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_gif", type=str, default="scratch/rollout_multi_swap_episode.gif")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_full_evaluation(
        ckpt_path=args.ckpt_path,
        n_cond_frames=args.n_cond_frames,
        n_sample_frames=args.n_sample_frames,
        batch_size=args.batch_size,
        out_gif=args.out_gif,
        device=args.device,
    )
