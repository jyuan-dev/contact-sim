#!/usr/bin/env python3
"""
Evaluate Stage 2 SlotFormer reconstruction and segmentation performance
across variable rollout step lengths (5 steps, 10 steps, 15 steps, 30 steps).
"""

import os
import sys
import json
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
from src.utils.training_utils import load_checkpoint_state


def evaluate_horizon(ckpt_path: str, rollout_steps: int, n_cond_frames: int = 2, batch_size: int = 64, device: str = "cuda"):
    n_sample_frames = n_cond_frames + rollout_steps
    h5_path = "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5"

    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=h5_path,
        split="val",
        resolution=(64, 64),
        n_sample_frames=n_sample_frames,
        clips_per_episode=2,
        seed=42,
    )
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ckpt_dir = os.path.dirname(ckpt_path)
    from omegaconf import OmegaConf
    saved_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    cfg = OmegaConf.to_container(saved_cfg, resolve=True)

    model = build_model(cfg).to(device)
    load_checkpoint_state(model, ckpt_path, device=device)
    model.eval()

    cond_mses, rollout_mses = [], []
    cond_mious, rollout_mious = [], []
    per_frame_mse = [[] for _ in range(n_sample_frames)]
    per_frame_miou = [[] for _ in range(n_sample_frames)]

    total_rollout_transitions = 0
    swap_rollout_transitions = 0

    with torch.no_grad():
        for batch in val_loader:
            video = batch["img"].to(device)
            gt_masks = batch.get("gt_masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)

            out = predict_slot_rollout(model, video, n_cond_frames=n_cond_frames)
            recon = out["recon_img"]
            pred_masks = out["pred_masks"]
            B, T = video.shape[:2]

            for t in range(T):
                mse_t_per_seq = torch.mean((recon[:, t] - video[:, t]) ** 2, dim=(1, 2, 3))
                per_frame_mse[t].extend(mse_t_per_seq.cpu().tolist())

                if t < n_cond_frames:
                    cond_mses.extend(mse_t_per_seq.cpu().tolist())
                else:
                    rollout_mses.extend(mse_t_per_seq.cpu().tolist())

                if pred_masks is not None and gt_masks is not None:
                    min_k = min(pred_masks.shape[2], gt_masks.shape[2])
                    p_t = (pred_masks[:, t, :min_k] > 0.5).float()
                    g_t = (gt_masks[:, t, :min_k] > 0.5).float()
                    inter = (p_t * g_t).sum(dim=(-2, -1))
                    union = p_t.sum(dim=(-2, -1)) + g_t.sum(dim=(-2, -1)) - inter
                    iou_t_per_seq = ((inter + 1e-6) / (union + 1e-6)).mean(dim=-1)
                    per_frame_miou[t].extend(iou_t_per_seq.cpu().tolist())

                    if t < n_cond_frames:
                        cond_mious.extend(iou_t_per_seq.cpu().tolist())
                    else:
                        rollout_mious.extend(iou_t_per_seq.cpu().tolist())

            # Slot swapping rate over rollout phase
            if pred_masks is not None and gt_masks is not None:
                min_k = min(pred_masks.shape[2], gt_masks.shape[2])
                p_bin = (pred_masks[:, :, :min_k] > 0.5).float()
                g_bin = (gt_masks[:, :, :min_k] > 0.5).float()

                for b in range(B):
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
                        prev_assign = curr_assign

    swap_rate_pct = (swap_rollout_transitions / max(1, total_rollout_transitions)) * 100.0

    return {
        "rollout_steps": rollout_steps,
        "n_sample_frames": n_sample_frames,
        "total_clips": len(eval_dataset),
        "cond_mse": float(np.mean(cond_mses)),
        "rollout_mse": float(np.mean(rollout_mses)),
        "cond_miou_pct": float(np.mean(cond_mious) * 100),
        "rollout_miou_pct": float(np.mean(rollout_mious) * 100),
        "swap_rate_pct": float(swap_rate_pct),
        "per_frame_mse": [float(np.mean(m)) for m in per_frame_mse],
        "per_frame_miou_pct": [float(np.mean(m) * 100) for m in per_frame_miou],
    }


def main():
    ckpts = {
        "1ep_baseline": "scratch/checkpoints/slotformer_pusht/slotformer_best.pt",
        "4ep_model": "scratch/checkpoints/slotformer_pusht_default_4ep/slotformer_best.pt",
        "ocvp_4ep_model": "scratch/checkpoints/ocvp_slotformer_pusht_default_4ep/ocvp_slotformer_best.pt",
    }
    horizons = [5, 10, 15, 30]

    all_results = {}
    for name, path in ckpts.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (path not found: {path})")
            continue
        print(f"\n========================================================")
        print(f" Evaluating {name}: {path}")
        print(f"========================================================")
        all_results[name] = {}
        for h in horizons:
            b_size = 64 if h <= 5 else (32 if h <= 15 else 16)
            print(f"--- Horizon: {h} rollout steps ({h+2} frames total, batch_size={b_size}) ---")
            torch.cuda.empty_cache()
            res = evaluate_horizon(path, rollout_steps=h, batch_size=b_size)
            all_results[name][f"{h}_steps"] = res
            print(f"  Rollout MSE:  {res['rollout_mse']:.6f}")
            print(f"  Rollout mIoU: {res['rollout_miou_pct']:.2f}%")
            print(f"  Swap Rate:    {res['swap_rate_pct']:.2f}%")

    out_file = "scratch/variable_horizon_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved variable horizon evaluation results to {out_file}")


if __name__ == "__main__":
    main()
