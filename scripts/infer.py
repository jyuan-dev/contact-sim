#!/usr/bin/env python3
"""
Quick Inspection & Visualization CLI Tool for SAVi / Deformable SAVi Checkpoints.

Usage:
  python scripts/infer.py --ckpt_path scratch/checkpoints/deformable_savi_3class_1ep/deformable_savi_best.pt
  python scripts/infer.py --ckpt_path scratch/checkpoints/deformable_savi_1ep_with_sigreg/deformable_savi_best.pt --num_sequences 5
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.utils.training_utils import get_device

SLOT_COLORS_RGB = {
    0: (255, 40, 40),     # Slot 0: Red (Agent)
    1: (40, 220, 40),     # Slot 1: Green (T-Block)
    2: (40, 120, 255),    # Slot 2: Blue (Goal Target)
    3: (255, 210, 0),     # Slot 3: Yellow
    4: (230, 40, 230)     # Slot 4: Magenta
}

GT_COLORS_RGB = {
    0: (255, 140, 0),    # Orange
    1: (0, 230, 115),    # Green
    2: (0, 128, 255)     # Blue
}


def run_quick_inference(ckpt_path, num_sequences=5, out_gif="scratch/quick_infer_demo.gif", device='cpu'):
    print("=" * 80)
    print(f"Quick Inspection & Visual Inference: {ckpt_path}")
    print(f"  Sequences: {num_sequences} | Device: {device} | Output GIF: {out_gif}")
    print("=" * 80)

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})

    dataset_cfg = {
        'name': 'pusht',
        'type': 'pusht',
        'h5_path': '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5',
        'resolution': [64, 64],
        'n_sample_frames': 6,
        'frame_offset': 1,
        'train_frac': 0.8,
    }
    if 'dataset' not in cfg:
        cfg['dataset'] = dataset_cfg
    if 'model' not in cfg:
        cfg['model'] = {'name': 'savi', 'type': 'savi'}

    val_loader = build_dataloader({'dataset': dataset_cfg}, split='val', batch_size=1, num_workers=2, shuffle=False)

    model = build_model(cfg).to(device)

    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    vis_frames = []
    mses = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= num_sequences:
                break

            video = batch['img'].to(device)  # [1, T, 3, 64, 64]
            out = model(video)

            recon = out.get('recon_img', None)
            if recon is not None:
                mse = torch.mean((recon - video) ** 2).item()
                mses.append(mse)

            pred_masks = out.get('pred_masks', None)  # [1, T, K, 64, 64]
            gt_masks = batch.get('gt_masks', None)     # [1, T, M, 64, 64]

            T = video.shape[1]
            video_np = ((video[0].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            pred_masks_np = pred_masks[0].cpu().numpy() if pred_masks is not None else np.zeros((T, 3, 64, 64))
            gt_masks_np = gt_masks[0].cpu().numpy() if gt_masks is not None else None

            for t in range(T):
                frame_rgb = video_np[t]
                if frame_rgb.shape[:2] != (64, 64):
                    frame_rgb = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)

                # Left Panel: Ground Truth Outlines
                p_gt = frame_rgb.copy()
                if gt_masks_np is not None:
                    for m_idx in range(gt_masks_np.shape[1]):
                        m_bin = gt_masks_np[t, m_idx] > 0.5
                        if m_bin.any():
                            contours, _ = cv2.findContours(m_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            color_bgr = GT_COLORS_RGB.get(m_idx, (255, 255, 255))
                            cv2.drawContours(p_gt, contours, -1, color_bgr, 1)

                # Right Panel: Color-Coded Slot Mask Overlay
                p_slots = frame_rgb.copy().astype(np.float32)
                masks_t = pred_masks_np[t]
                K = masks_t.shape[0]

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

                banner_text = f"Seq {b_idx+1}/{num_sequences} Frame {t+1}/{T} | MSE: {mses[-1]:.6f}"
                cv2.rectangle(combined_large, (0, 0), (combined_large.shape[1], 18), (30, 140, 220), -1)
                cv2.putText(combined_large, banner_text, (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                vis_frames.append(combined_large)

    print("\n---------------- Quick Inspection Report ----------------")
    print(f"Inspected Sequences:         {num_sequences}")
    print(f"Mean Reconstruction MSE:     {np.mean(mses):.6f}")

    os.makedirs("scratch", exist_ok=True)
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in vis_frames]
    if pil_frames:
        pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:], duration=150, loop=0)
    print(f"Saved Quick Inspection GIF to: {out_gif}")
    print("----------------------------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quick inspection and visualization CLI for SAVi models.")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument('--num_sequences', type=int, default=5, help="Number of validation sequences to inspect")
    parser.add_argument('--out_gif', type=str, default="scratch/quick_infer_demo.gif", help="Output GIF path")
    parser.add_argument('--device', type=str, default='cpu', help="Target device (cpu or cuda)")
    args = parser.parse_args()

    run_quick_inference(args.ckpt_path, num_sequences=args.num_sequences, out_gif=args.out_gif, device=args.device)
