#!/usr/bin/env python3
"""
SlotFormer Autoregressive Future Slot Rollout Evaluation & Visualization Script.

Usage:
  python scripts/rollout.py --ckpt_path scratch/checkpoints/slotformer_pusht/slotformer_best.pt --device cuda
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
from src.utils.training_utils import load_checkpoint_state
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
from src.utils.data_utils import find_dataset_path


def run_rollout_evaluation(
    ckpt_path: str,
    n_cond_frames: int = 2,
    base_seed: int = 42,
    clips_per_ep: int = 2,
    batch_size: int = 32,
    device: str = "cpu",
    out_gif: str = "scratch/rollout_best_model.gif",
):
    print("=" * 80)
    print(f"SlotFormer Future Rollout Evaluation: {ckpt_path}")
    print(f"  Condition Frames: {n_cond_frames} | Device: {device} | Base Seed: {base_seed}")
    print("=" * 80)

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=find_dataset_path(None),
        split="val",
        resolution=(64, 64),
        n_sample_frames=6,
        clips_per_episode=clips_per_ep,
        seed=base_seed,
    )
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Reconstruct the experiment from the checkpoint
    model, cfg = bootstrap_checkpoint(ckpt_path)
    model = model.to(device)
    load_checkpoint_state(model, ckpt_path, device=device)
    model_name = cfg.get("model", {}).get("name", "unknown")
    print(f"[Rollout Eval] Model weights loaded into '{model_name}'.")
    model.eval()

    cond_mses, rollout_mses = [], []
    cond_mious, rollout_mious = [], []
    
    total_rollout_transitions = 0
    swap_rollout_transitions = 0
    swapped_rollout_seqs = 0
    total_rollout_seqs = 0

    vis_frames = []
    max_vis_seqs = 4

    start_t = time.time()
    num_batches = len(val_loader)

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            video = batch["img"].to(device)
            out = predict_slot_rollout(model, video, n_cond_frames=n_cond_frames)

            B, T = video.shape[:2]
            recon = out["recon_img"]
            pred_masks = out["pred_masks"]
            gt_masks = batch.get("gt_masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)

            # Split metrics into Conditioned (t < n_cond) vs Rollout (t >= n_cond)
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

            # Slot Swapping Analysis over Rollout steps (t >= n_cond_frames)
            if pred_masks is not None and gt_masks is not None:
                min_k = min(pred_masks.shape[2], gt_masks.shape[2])
                p_bin = (pred_masks[:, :, :min_k] > 0.5).float()
                g_bin = (gt_masks[:, :, :min_k] > 0.5).float()

                for b in range(B):
                    total_rollout_seqs += 1
                    seq_swapped = False
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
                                seq_swapped = True

                        prev_assign = curr_assign

                    if seq_swapped:
                        swapped_rollout_seqs += 1

            # Generate visualization frames for first few sequences
            if b_idx < max_vis_seqs:
                for b in range(min(B, 1)):
                    video_np = ((video[b].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                    pred_masks_np = pred_masks[b].cpu().numpy()
                    gt_masks_np = gt_masks[b].cpu().numpy() if gt_masks is not None else None

                    for t in range(T):
                        phase_label = "COND" if t < n_cond_frames else "ROLLOUT"
                        banner = f"Seq {b_idx+1} Frame {t+1}/{T} [{phase_label}] | Rollout MSE: {rollout_mses[-1]:.6f}"
                        combined = render_slot_overlay_frame(
                            frame_rgb=video_np[t],
                            pred_masks_t=pred_masks_np[t],
                            gt_masks_t=gt_masks_np[t] if gt_masks_np is not None else None,
                            banner_text=banner,
                        )
                        vis_frames.append(combined)

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated Rollout [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s)")

    rollout_swap_rate = (swap_rollout_transitions / total_rollout_transitions * 100.0) if total_rollout_transitions > 0 else 0.0
    seq_rollout_swap_rate = (swapped_rollout_seqs / total_rollout_seqs * 100.0) if total_rollout_seqs > 0 else 0.0

    res = {
        "ckpt_path": ckpt_path,
        "n_cond_frames": n_cond_frames,
        "n_rollout_frames": T - n_cond_frames,
        "cond_mse": float(np.mean(cond_mses)) if cond_mses else float("nan"),
        "rollout_mse": float(np.mean(rollout_mses)) if rollout_mses else float("nan"),
        "cond_miou": float(np.mean(cond_mious)) * 100 if cond_mious else 0.0,
        "rollout_miou": float(np.mean(rollout_mious)) * 100 if rollout_mious else 0.0,
        "future_rollout_slot_swapping_transition_rate_pct": float(rollout_swap_rate),
        "future_rollout_sequence_swapping_rate_pct": float(seq_rollout_swap_rate),
    }

    print("\n" + "=" * 80)
    print(f"SlotFormer Future Rollout Evaluation Summary for {ckpt_path}:")
    print(f"  Condition Context Frames:               {n_cond_frames}")
    print(f"  Autoregressive Rollout Frames:          {T - n_cond_frames}")
    print(f"  Conditioned Phase MSE:                  {res['cond_mse']:.6f}")
    print(f"  Future Rollout Phase MSE:               {res['rollout_mse']:.6f}")
    print(f"  Conditioned Phase mIoU:                 {res['cond_miou']:.2f}%")
    print(f"  Future Rollout Phase mIoU:              {res['rollout_miou']:.2f}%")
    print(f"  Future Rollout Slot Swap Transition Rate: {res['future_rollout_slot_swapping_transition_rate_pct']:.2f}%")
    print(f"  Future Rollout Sequence Swap Rate:      {res['future_rollout_sequence_swapping_rate_pct']:.2f}%")
    print("=" * 80 + "\n")

    # Save rollout metrics in both the checkpoint directory and scratch root
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    exp_out_file = os.path.join(ckpt_dir, "rollout_results.json")
    with open(exp_out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved rollout evaluation metrics to experiment dir: {exp_out_file}")

    out_dir = os.path.join(REPO_ROOT, "scratch")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "rollout_results.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    if vis_frames:
        save_frames_to_gif(vis_frames, out_gif, fps=4)
        print(f"Saved Future Rollout visualization GIF to: {out_gif}")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SlotFormer Autoregressive Future Slot Rollout Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--n_cond_frames", type=int, default=2, help="Number of condition frames")
    parser.add_argument("--base_seed", type=int, default=42, help="Evaluation random seed")
    parser.add_argument("--clips_per_ep", type=int, default=2, help="Clips per episode to sample")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--out_gif", type=str, default="scratch/rollout_slotformer_stage2.gif", help="Output GIF path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_rollout_evaluation(
        ckpt_path=args.ckpt_path,
        n_cond_frames=args.n_cond_frames,
        base_seed=args.base_seed,
        clips_per_ep=args.clips_per_ep,
        batch_size=args.batch_size,
        out_gif=args.out_gif,
        device=args.device,
    )
