#!/usr/bin/env python3
"""
Full-Episode Autoregressive Rollout Script for Stage 2 SlotFormer.

Rolls out Stage 2 SlotFormer on 2 COMPLETE, un-cropped PushT validation episodes from start to finish
(all 100+ frames per episode) and outputs 2 separate GIF files for the two distinct episodes.
"""

import os
import sys
import json
import argparse
import time
import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.models.rollout import predict_slot_rollout
from src.utils.vis_utils import render_slot_overlay_frame, save_frames_to_gif
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
from src.utils.data_utils import find_dataset_path


def render_3panel_composite_frame(
    frame_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    pred_masks_t: np.ndarray,
    gt_masks_t: np.ndarray | None = None,
    banner_text: str = "",
) -> np.ndarray:
    """
    Renders a 3-panel side-by-side composite frame:
      - Left: Ground Truth RGB frame with GT outlines.
      - Middle: Model Autoregressively Reconstructed RGB Image.
      - Right: Color-coded slot mask overlay.
    """
    if frame_rgb.shape[:2] != (64, 64):
        frame_rgb = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
    if recon_rgb.shape[:2] != (64, 64):
        recon_rgb = cv2.resize(recon_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)

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
    p_recon = recon_rgb.copy()

    # 3. Right Panel: Color-Coded Slot Mask Overlay
    from src.utils.vis_utils import SLOT_COLORS_RGB
    p_slots = frame_rgb.copy().astype(np.float32)
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


def rollout_single_full_episode(
    model: torch.nn.Module,
    h5_file: h5py.File,
    ep_idx: int,
    offs: np.ndarray,
    lens: np.ndarray,
    n_cond_frames: int = 2,
    device: str = "cuda",
    out_gif_path: str = "scratch/rollout_full_episode_1.gif",
) -> dict:
    ep_len = int(lens[ep_idx])
    ep_off = int(offs[ep_idx])

    frame_idxs = list(range(ep_off, ep_off + ep_len))
    frames = h5_file["pixels"][frame_idxs]  # [T, H, W, C]
    
    agent_m = (h5_file["agent_masks"][frame_idxs] > 127).astype(np.float32)
    block_m = (h5_file["block_masks"][frame_idxs] > 127).astype(np.float32)
    goal_m = (h5_file["goal_masks"][frame_idxs] > 127).astype(np.float32)
    goal_visible = np.clip(goal_m - np.maximum(block_m, agent_m), 0.0, 1.0)
    gt_masks = np.stack([agent_m, block_m, goal_visible], axis=1)  # [T, K, H, W]

    video = (frames.astype(np.float32) / 127.5) - 1.0
    img_t = torch.from_numpy(video.transpose(0, 3, 1, 2)).unsqueeze(0).to(device)  # [1, T, C, H, W]
    gt_masks_t = torch.from_numpy(gt_masks).unsqueeze(0).to(device)                # [1, T, K, H, W]

    print(f"\n--- Rolling out Full Episode {ep_idx} (Total Frames: {ep_len}) ---")
    start_t = time.time()

    with torch.no_grad():
        out = predict_slot_rollout(model, img_t, n_cond_frames=n_cond_frames)

    eval_time = time.time() - start_t
    print(f"Completed full episode rollout in {eval_time:.2f}s ({ep_len / eval_time:.1f} frames/s)")

    recon_t = out["recon_img"]
    pred_masks_t = out["pred_masks"]

    T = ep_len
    cond_mses, rollout_mses = [], []
    cond_mious, rollout_mious = [], []
    per_frame_miou = []
    per_frame_mse = []

    for t in range(T):
        mse_t = torch.mean((recon_t[:, t] - img_t[:, t]) ** 2).item()
        per_frame_mse.append(mse_t)
        if t < n_cond_frames:
            cond_mses.append(mse_t)
        else:
            rollout_mses.append(mse_t)

        p_t = (pred_masks_t[0, t] > 0.5).float()
        g_t = (gt_masks_t[0, t] > 0.5).float()
        min_k = min(p_t.shape[0], g_t.shape[0])
        inter = (p_t[:min_k] * g_t[:min_k]).sum(dim=(-2, -1))
        union = p_t[:min_k].sum(dim=(-2, -1)) + g_t[:min_k].sum(dim=(-2, -1)) - inter
        iou_t = ((inter + 1e-6) / (union + 1e-6)).mean().item()
        per_frame_miou.append(iou_t * 100.0)

        if t < n_cond_frames:
            cond_mious.append(iou_t)
        else:
            rollout_mious.append(iou_t)

    # Render full-episode 3-panel GIF frames
    vis_frames = []
    recon_np = ((recon_t[0].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    pred_masks_np = pred_masks_t[0].cpu().numpy()
    gt_masks_np = gt_masks_t[0].cpu().numpy()

    for t in range(T):
        phase = "COND" if t < n_cond_frames else f"ROLLOUT+{t+1-n_cond_frames}"
        banner = f"Episode {ep_idx} Frame {t+1}/{T} [{phase}] | MSE: {per_frame_mse[t]:.5f} | mIoU: {per_frame_miou[t]:.1f}%"
        combined = render_3panel_composite_frame(
            frame_rgb=frames[t],
            recon_rgb=recon_np[t],
            pred_masks_t=pred_masks_np[t],
            gt_masks_t=gt_masks_np[t],
            banner_text=banner,
        )
        vis_frames.append(combined)

    save_frames_to_gif(vis_frames, out_gif_path, fps=6)
    print(f"Saved full episode {ep_idx} 3-panel rollout GIF to: {out_gif_path}")

    res = {
        "episode_idx": ep_idx,
        "total_frames": T,
        "n_cond_frames": n_cond_frames,
        "n_rollout_steps": T - n_cond_frames,
        "cond_mse": float(np.mean(cond_mses)),
        "cond_miou_pct": float(np.mean(cond_mious) * 100.0),
        "rollout_mse": float(np.mean(rollout_mses)),
        "rollout_miou_pct": float(np.mean(rollout_mious) * 100.0),
        "out_gif": out_gif_path,
        "eval_time_sec": float(eval_time),
    }
    return res


def main():
    import random
    parser = argparse.ArgumentParser(description="Full Episode Rollout for Stage 2 SlotFormer")
    parser.add_argument("--ckpt_path", type=str, default="scratch/checkpoints/slotformer_pusht_default_4ep/slotformer_best.pt")
    parser.add_argument("--n_cond_frames", type=int, default=2)
    parser.add_argument("--ep_idx", type=int, default=None, help="Validation episode index (random if None)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    h5_file = h5py.File(find_dataset_path(None), "r")
    offs = h5_file["ep_offset"][:]
    lens = h5_file["ep_len"][:]
    n_episodes = len(lens)

    # Determine validation episodes (last 10% of dataset)
    val_start_ep = int(n_episodes * 0.9)
    val_episodes = list(range(val_start_ep, n_episodes))

    if args.ep_idx is None:
        rng = random.Random(args.seed)
        selected_ep = rng.choice(val_episodes)
    else:
        selected_ep = args.ep_idx

    model, cfg = bootstrap_checkpoint(ckpt_path)
    model = model.to(args.device)
    from src.utils.training_utils import load_checkpoint_state
    load_checkpoint_state(model, ckpt_path, device=args.device)
    model.eval()

    out_gif_path = f"scratch/long_rollout_ep{selected_ep}.gif"
    res = rollout_single_full_episode(
        model=model,
        h5_file=h5_file,
        ep_idx=selected_ep,
        offs=offs,
        lens=lens,
        n_cond_frames=args.n_cond_frames,
        device=args.device,
        out_gif_path=out_gif_path,
    )

    print("\n" + "=" * 80)
    print("LONG-HORIZON AUTOREGRESSIVE ROLLOUT COMPLETED")
    print("=" * 80)
    print(f"Selected Validation Episode Index: {selected_ep}")
    print(f"Total Trajectory Length:           {res['total_frames']} frames")
    print(f"Autoregressive Rollout Steps:     {res['n_rollout_steps']} steps")
    print(f"Conditioned Context MSE (2f):     {res['cond_mse']:.6f} (mIoU: {res['cond_miou_pct']:.2f}%)")
    print(f"Long-Horizon Rollout MSE ({res['n_rollout_steps']}s): {res['rollout_mse']:.6f} (mIoU: {res['rollout_miou_pct']:.2f}%)")
    print(f"Saved 3-Panel Composite GIF:      {res['out_gif']}")
    print("=" * 80)

    h5_file.close()


if __name__ == "__main__":
    main()
