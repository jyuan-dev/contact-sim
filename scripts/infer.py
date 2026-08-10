#!/usr/bin/env python3
"""
Quick Inspection & Visualization CLI Tool for SAVi / Deformable SAVi Checkpoints.

Usage:
  python scripts/infer.py --ckpt_path scratch/checkpoints/savi_pusht/savi_best.pt
  python scripts/infer.py --ckpt_path scratch/checkpoints/deformable_savi_pusht/deformable_savi_best.pt --num_sequences 5
"""

import os
import sys
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.utils.vis_utils import render_slot_overlay_frame, save_frames_to_gif


def run_quick_inference(ckpt_path: str, num_sequences: int = 5, out_gif: str = "scratch/quick_infer_demo.gif", device: str = 'cpu'):
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

            pred_masks = out.get('pred_masks', None)
            gt_masks = batch.get('gt_masks', None)

            T = video.shape[1]
            video_np = ((video[0].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            pred_masks_np = pred_masks[0].cpu().numpy() if pred_masks is not None else np.zeros((T, 3, 64, 64))
            gt_masks_np = gt_masks[0].cpu().numpy() if gt_masks is not None else None

            for t in range(T):
                frame_rgb = video_np[t]
                gt_t = gt_masks_np[t] if gt_masks_np is not None else None
                banner_text = f"Seq {b_idx+1}/{num_sequences} Frame {t+1}/{T} | MSE: {mses[-1]:.6f}"

                combined_frame = render_slot_overlay_frame(
                    frame_rgb=frame_rgb,
                    pred_masks_t=pred_masks_np[t],
                    gt_masks_t=gt_t,
                    banner_text=banner_text,
                )
                vis_frames.append(combined_frame)

    print("\n---------------- Quick Inspection Report ----------------")
    print(f"Inspected Sequences:         {num_sequences}")
    print(f"Mean Reconstruction MSE:     {np.mean(mses):.6f}" if mses else "N/A")

    save_frames_to_gif(vis_frames, out_gif, fps=7)
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
