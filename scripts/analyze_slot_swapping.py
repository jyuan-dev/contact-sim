#!/usr/bin/env python3
"""
Single-Script Slot Swapping Analysis & Visualization Tool powered by Hydra.

Usage:
  python scripts/analyze_slot_swapping.py model=savi dataset=pusht ckpt_path=/home/jyuan/.stable-wm/savi_mask_detr/savi_epoch_8.pt
  python scripts/analyze_slot_swapping.py model=detr dataset=pusht ckpt_path=/home/jyuan/.stable-wm/detr_pusht/detr_final.pt
"""

import os
import json
import yaml
import h5py
import hdf5plugin
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.factory import build_model
from src.datasets.pusht import PushTMaskHDF5Dataset
from src.metrics.evaluator import compute_binary_iou_dice, EvaluationSuite
from src.utils.training_utils import get_device

CLASS_NAMES = {0: "Block", 1: "Agent", 2: "Goal"}

# Fixed Color Palette for Slot IDs (0..4)
SLOT_COLORS_RGB = {
    0: (255, 40, 40),     # Slot 0: Bright Red
    1: (40, 220, 40),     # Slot 1: Bright Green
    2: (40, 120, 255),    # Slot 2: Bright Blue
    3: (255, 210, 0),     # Slot 3: Yellow
    4: (230, 40, 230)     # Slot 4: Magenta
}

SLOT_COLORS_PLT = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'gold',
    4: 'magenta'
}

GT_COLORS_RGB = {
    0: (255, 140, 0),    # Orange (Block)
    1: (0, 230, 115),    # Green (Agent)
    2: (0, 128, 255)     # Blue (Goal)
}


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    device = get_device()

    ckpt_path = cfg.get('ckpt_path', None) or cfg.get('ckpt', None) or "/home/jyuan/.stable-wm/savi_mask_detr/savi_epoch_8.pt"
    ep_idx = cfg.get('ep_idx', 6)
    output_dir = cfg.get('output_dir', "scratch")

    model_name = cfg.model.name
    dataset_name = cfg.dataset.name

    print("======================================================================")
    print(f"       Hydra Slot Swapping Analysis Tool ({model_name} / {dataset_name})")
    print("======================================================================")
    print(f"Device: {device}")
    print(f"Loading Checkpoint: {ckpt_path}")

    # 1. Build Model via Factory
    model = build_model(cfg_dict).to(device)

    ckpt_data = torch.load(ckpt_path, map_location=device)
    target_model = model.model if hasattr(model, 'model') else model
    state = ckpt_data.get('model', ckpt_data.get('model_state', ckpt_data))
    state = {k.replace('model.', '').replace('module.', ''): v for k, v in state.items()}
    target_model.load_state_dict(state)

    model.eval()
    print("Model loaded successfully into eval mode!")

    # 2. Extract Episode Sequence Data
    h5_path = cfg.dataset.get('h5_path', '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5')
    if not os.path.exists(h5_path):
        h5_path = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5'

    with h5py.File(h5_path, 'r') as f:
        ep_lens = np.array(f['ep_len'])
        ep_offs = np.array(f['ep_offset'])
        offset = ep_offs[ep_idx]
        length = ep_lens[ep_idx]
        pixels = np.array(f['pixels'][offset : offset + length])
        b_masks = np.array(f['block_masks'][offset : offset + length]) > 0
        a_masks = np.array(f['agent_masks'][offset : offset + length]) > 0
        g_masks = np.array(f['goal_masks'][offset : offset + length]) > 0

    T = length
    print(f"Loaded validation episode index {ep_idx}: {T} frames.")

    imgs_np = pixels.transpose(0, 3, 1, 2) if (pixels.ndim == 4 and pixels.shape[-1] == 3) else pixels
    imgs_torch = torch.tensor(imgs_np, dtype=torch.float32, device=device) / 255.0
    imgs_torch_norm = (imgs_torch - 0.5) / 0.5

    if imgs_torch_norm.shape[-1] != 64 or imgs_torch_norm.shape[-2] != 64:
        imgs_torch_norm = F.interpolate(imgs_torch_norm, size=(64, 64), mode='bilinear', align_corners=False)

    # 3. Model Forward Pass
    with torch.no_grad():
        out = model(imgs_torch_norm.unsqueeze(0))

    pred_masks = out.get('pred_masks', None)
    if pred_masks is not None:
        pred_masks_np = pred_masks[0].cpu().numpy() # [T, K, 64, 64]
    else:
        pred_masks_np = np.zeros((T, 4, 64, 64), dtype=np.float32)

    T, K, H, W = pred_masks_np.shape
    gt_masks_dict = {0: b_masks, 1: a_masks, 2: g_masks}

    # 4. Perform Quantitative Slot Swapping Analysis
    slot_assignments = {c: [] for c in range(3)}
    slot_ious = {c: [] for c in range(3)}
    swap_events = []

    for t in range(T):
        cost_matrix = np.zeros((K, 3), dtype=np.float32)
        for k in range(K):
            for c in range(3):
                iou, _ = compute_binary_iou_dice(pred_masks_np[t, k], gt_masks_dict[c][t])
                cost_matrix[k, c] = -iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for slot_idx, class_idx in zip(row_ind, col_ind):
            best_iou = -cost_matrix[slot_idx, class_idx]
            slot_assignments[class_idx].append(int(slot_idx))
            slot_ious[class_idx].append(best_iou)

            if t > 0 and (gt_masks_dict[class_idx][t].sum() > 0):
                prev_slot = slot_assignments[class_idx][t - 1]
                if prev_slot != slot_idx:
                    swap_events.append({
                        'frame': t,
                        'class_id': class_idx,
                        'class_name': CLASS_NAMES[class_idx],
                        'from_slot': prev_slot,
                        'to_slot': int(slot_idx)
                    })

    metrics = {
        'total_frames': T,
        'total_swap_events': len(swap_events),
        'swap_rate_per_100_frames': float((len(swap_events) / T) * 100.0) if T > 0 else 0.0,
        'class_mIoU': {CLASS_NAMES[c]: float(np.mean(slot_ious[c])) for c in range(3)},
        'overall_mIoU': float(np.mean([np.mean(slot_ious[c]) for c in range(3)]))
    }

    print("\n---------------- Slot Swapping Analysis Report ----------------")
    print(f"Total Frames Analyzed: {T}")
    print(f"Total Slot Swap Events: {metrics['total_swap_events']}")
    print(f"Slot Swap Rate: {metrics['swap_rate_per_100_frames']:.2f} swaps / 100 frames")
    for cls_name, miou in metrics['class_mIoU'].items():
        print(f"Class '{cls_name}': Mean IoU = {miou:.4f}")
    print(f"Overall Mean IoU (mIoU): {metrics['overall_mIoU']:.4f}")
    print("---------------------------------------------------------------\n")

    os.makedirs(output_dir, exist_ok=True)

    # 5. Save JSON Metrics Summary
    json_path = os.path.join(output_dir, "slot_swapping_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved JSON metrics to: {json_path}")

    # 6. Render & Save Fixed Slot Mask Color Swapping GIF
    vis_frames = []
    img_raw_np = (imgs_torch.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

    for t in range(T):
        frame_rgb = img_raw_np[t]
        if frame_rgb.shape[:2] != (64, 64):
            frame_rgb = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Left Panel: GT Outlines
        p_gt = frame_rgb.copy()
        gt_list = [b_masks[t], a_masks[t], g_masks[t]]
        for m_idx in range(3):
            m_bin = gt_list[m_idx]
            if m_bin.any():
                contours, _ = cv2.findContours(m_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color_bgr = (GT_COLORS_RGB[m_idx][2], GT_COLORS_RGB[m_idx][1], GT_COLORS_RGB[m_idx][0])
                cv2.drawContours(p_gt, contours, -1, color_bgr, 1)

        # Right Panel: Pure Slot Mask Overlay
        p_slots = frame_rgb.copy().astype(np.float32)
        masks_t = pred_masks_np[t]

        slot_map = np.zeros((64, 64, 3), dtype=np.float32)
        weight_sum = np.zeros((64, 64, 1), dtype=np.float32)

        for k in range(K):
            m_k = np.clip(masks_t[k], 0, 1)[..., None]
            color_k = np.array(SLOT_COLORS_RGB[k % len(SLOT_COLORS_RGB)], dtype=np.float32)
            slot_map += m_k * color_k
            weight_sum += m_k

        weight_sum = np.maximum(weight_sum, 1e-6)
        slot_composite = slot_map / weight_sum
        active_mask = (weight_sum > 0.15)
        alpha = 0.60
        p_slots[active_mask[:, :, 0]] = (1.0 - alpha) * p_slots[active_mask[:, :, 0]] + alpha * slot_composite[active_mask[:, :, 0]]

        p_slots_uint8 = np.clip(p_slots, 0, 255).astype(np.uint8)

        combined = np.hstack([p_gt, p_slots_uint8])
        combined_large = cv2.resize(combined, (480, 240), interpolation=cv2.INTER_NEAREST)

        banner_text = f"Frame {t:03d}/{T} | Swapping Rate: {metrics['swap_rate_per_100_frames']:.1f}/100f | Right: Fixed Slot Mask Color (S0=Red, S1=Green, S2=Blue)"
        cv2.rectangle(combined_large, (0, 0), (combined_large.shape[1], 18), (30, 140, 220), -1)
        cv2.putText(combined_large, banner_text, (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        vis_frames.append(combined_large)

    gif_path = os.path.join(output_dir, "slot_swapping_demo.gif")
    imageio.mimsave(gif_path, vis_frames, fps=10, loop=0)
    print(f"Saved Slot Swapping GIF to: {gif_path}")

    # Copy to brain dir if available
    brain_dir = "/home/jyuan/.gemini/antigravity-ide/brain/0e62dc39-5378-4e8f-b19c-9d502981fb60"
    if os.path.exists(brain_dir):
        import shutil
        shutil.copy(gif_path, os.path.join(brain_dir, "slot_swapping_demo.gif"))

    # 7. Render & Save 3-Panel Quantitative Analysis Figure (PNG)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, dpi=300)
    plt.subplots_adjust(hspace=0.25)
    frames_axis = np.arange(T)

    # Panel 1: Slot Assignment Timeline
    ax1 = axes[0]
    ax1.set_title(f"1. Frame-by-Frame Slot Assignment per Object (Total Swaps: {metrics['total_swap_events']})", fontsize=11, fontweight='bold')
    ax1.plot(frames_axis, slot_assignments[0], label="T-Block Assigned Slot", color='darkorange', linewidth=2, marker='o', markersize=3)
    ax1.plot(frames_axis, slot_assignments[1], label="Agent Assigned Slot", color='green', linewidth=2, marker='s', markersize=3)
    ax1.plot(frames_axis, slot_assignments[2], label="Goal Area Assigned Slot", color='dodgerblue', linewidth=1.5, linestyle='--')
    ax1.set_ylabel("Slot ID", fontsize=10)
    ax1.set_yticks(range(K))
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)

    # Panel 2: Max Slot IoU Curves Over Time
    ax2 = axes[1]
    ax2.set_title("2. Object Segmentation IoU Scores Over Time", fontsize=11, fontweight='bold')
    ax2.plot(frames_axis, slot_ious[0], label="T-Block IoU", color='darkorange', linewidth=1.8)
    ax2.plot(frames_axis, slot_ious[1], label="Agent IoU", color='green', linewidth=1.8)
    ax2.plot(frames_axis, slot_ious[2], label="Goal IoU", color='dodgerblue', linewidth=1.5, linestyle='--')
    ax2.set_ylabel("IoU Score", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8)

    # Panel 3: Individual Slot Peak Intensities
    ax3 = axes[2]
    ax3.set_title("3. Individual Slot Peak Mask Intensities (Slots 0..3)", fontsize=11, fontweight='bold')
    for k in range(K):
        slot_intensity = pred_masks_np[:, k].max(axis=(1, 2))
        ax3.plot(frames_axis, slot_intensity, label=f"Slot {k} Intensity", color=SLOT_COLORS_PLT.get(k, 'gray'), linewidth=1.5)
    ax3.set_xlabel("Frame Index (Episode 6, 0..108)", fontsize=10)
    ax3.set_ylabel("Peak Intensity", fontsize=10)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right', fontsize=8)

    png_path = os.path.join(output_dir, "slot_swapping_analysis.png")
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Quantitative Analysis Chart to: {png_path}")

    if os.path.exists(brain_dir):
        import shutil
        shutil.copy(png_path, os.path.join(brain_dir, "slot_swapping_analysis.png"))

    print("\nSlot Swapping Analysis Script Execution Completed Successfully!")


if __name__ == "__main__":
    main()
