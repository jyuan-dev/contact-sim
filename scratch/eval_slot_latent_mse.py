#!/usr/bin/env python3
"""
Evaluate Slot Latent Rollout MSE (the exact training loss metric: slot_mse)
alongside Image Recon MSE, mIoU, and Slot Swapping Rate across variable rollout horizons.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.models.rollout import predict_slot_rollout
from src.datasets import DeterministicEpisodeEvalDataset
from src.utils.training_utils import load_checkpoint_state


def evaluate_horizon_slot_mse(ckpt_path: str, rollout_steps: int, n_cond_frames: int = 2, batch_size: int = 64, device: str = "cuda"):
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

    rollout_slot_mses = []
    rollout_img_mses = []
    rollout_mious = []

    with torch.no_grad():
        for batch in val_loader:
            video = batch["img"].to(device)
            gt_masks = batch.get("gt_masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)

            # 1. Extract ground truth slots for all T frames
            extract_fn = getattr(model, "extract_slots", getattr(model.model, "extract_slots", None))
            gt_all_slots = extract_fn(video)  # [B, T, K, D]

            # 2. Autoregressively roll out predicted slots
            out = predict_slot_rollout(model, video, n_cond_frames=n_cond_frames)
            post_slots = out["post_slots"]  # [B, T, K, D]
            recon_img = out["recon_img"]
            pred_masks = out["pred_masks"]

            # 3. Compute Slot Latent Rollout MSE (the training loss metric)
            gt_rollout_slots = gt_all_slots[:, n_cond_frames:]
            pred_rollout_slots = post_slots[:, n_cond_frames:]

            # Per-clip slot latent MSE over rollout phase
            clip_slot_mse = F.mse_loss(pred_rollout_slots, gt_rollout_slots, reduction="none").mean(dim=(1, 2, 3))
            rollout_slot_mses.extend(clip_slot_mse.cpu().tolist())

            # Pixel Image Rollout MSE
            clip_img_mse = F.mse_loss(recon_img[:, n_cond_frames:], video[:, n_cond_frames:], reduction="none").mean(dim=(1, 2, 3, 4))
            rollout_img_mses.extend(clip_img_mse.cpu().tolist())

            # mIoU over rollout phase
            if pred_masks is not None and gt_masks is not None:
                min_k = min(pred_masks.shape[2], gt_masks.shape[2])
                p_r = (pred_masks[:, n_cond_frames:, :min_k] > 0.5).float()
                g_r = (gt_masks[:, n_cond_frames:, :min_k] > 0.5).float()
                inter = (p_r * g_r).sum(dim=(-2, -1))
                union = p_r.sum(dim=(-2, -1)) + g_r.sum(dim=(-2, -1)) - inter
                iou_clip = ((inter + 1e-6) / (union + 1e-6)).mean(dim=(-2, -1))
                rollout_mious.extend(iou_clip.cpu().tolist())

    return {
        "rollout_steps": rollout_steps,
        "n_sample_frames": n_sample_frames,
        "total_clips": len(eval_dataset),
        "rollout_slot_latent_mse": float(np.mean(rollout_slot_mses)),
        "rollout_img_pixel_mse": float(np.mean(rollout_img_mses)),
        "rollout_miou_pct": float(np.mean(rollout_mious) * 100) if rollout_mious else 0.0,
    }


def main():
    ckpts = {
        "1ep_baseline": "scratch/checkpoints/slotformer_pusht/slotformer_best.pt",
        "4ep_model": "scratch/checkpoints/slotformer_pusht_default_4ep/slotformer_best.pt",
        "ocvp_4ep_model": "scratch/checkpoints/ocvp_slotformer_pusht_default_4ep/ocvp_slotformer_best.pt",
    }
    horizons = [5, 10, 15, 30]

    results = {}
    for name, path in ckpts.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (not found: {path})")
            continue
        print(f"\n========================================================")
        print(f" Measuring Slot Latent Rollout MSE for {name}: {path}")
        print(f"========================================================")
        results[name] = {}
        for h in horizons:
            b_size = 64 if h <= 5 else (32 if h <= 15 else 16)
            print(f"--- Horizon: {h} rollout steps ({h+2} frames, batch_size={b_size}) ---")
            torch.cuda.empty_cache()
            res = evaluate_horizon_slot_mse(path, rollout_steps=h, batch_size=b_size)
            results[name][f"{h}_steps"] = res
            print(f"  Slot Latent Rollout MSE (Training Loss): {res['rollout_slot_latent_mse']:.6f}")
            print(f"  Image Pixel Rollout MSE:                 {res['rollout_img_pixel_mse']:.6f}")
            print(f"  Rollout mIoU:                             {res['rollout_miou_pct']:.2f}%")

    out_path = "scratch/slot_latent_mse_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved slot latent MSE evaluation results to {out_path}")


if __name__ == "__main__":
    main()
