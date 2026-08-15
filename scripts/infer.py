#!/usr/bin/env python3
"""
Quick Inspection & Visualization CLI Tool for SAVi Checkpoints.

Usage:
  python scripts/infer.py --ckpt_path scratch/checkpoints/savi_pusht/savi_best.pt
"""

import os
import sys
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.factory import build_dataloader
from src.metrics import greedy_slot_assignments
from src.utils.vis_utils import render_slot_overlay_frame, save_frames_to_gif
from src.utils.training_utils import load_checkpoint_state
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
from src.utils.data_utils import find_dataset_path


def run_quick_inference(ckpt_path: str, clip_idx: int = None, num_sequences: int = 5, out_gif: str = "scratch/quick_infer_demo.gif", device: str = 'cpu'):
    print("=" * 80)
    print(f"Quick Inspection & Slot Occupation Analysis: {ckpt_path}")
    print(f"  Target Clip: {clip_idx if clip_idx is not None else 'First ' + str(num_sequences)} | Device: {device} | Output GIF: {out_gif}")
    print("=" * 80)

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from src.datasets.pusht import DeterministicEpisodeEvalDataset
    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=find_dataset_path(None),
        split='val',
        resolution=(64, 64),
        n_sample_frames=6,
        clips_per_episode=2,
        base_seed=42
    )

    # ── Reconstruct the experiment from the checkpoint ───────────────────────
    model, cfg = bootstrap_checkpoint(ckpt_path)
    model = model.to(device)
    load_checkpoint_state(model, ckpt_path, device=device)
    model.eval()

    vis_frames = []
    mses = []
    slot_names = {0: 'GT 0 (Robot)', 1: 'GT 1 (T-Block)', 2: 'GT 2 (Goal Target)'}

    target_indices = [clip_idx] if clip_idx is not None else list(range(min(num_sequences, len(eval_dataset))))

    with torch.no_grad():
        for i_idx, c_idx in enumerate(target_indices):
            sample = eval_dataset[c_idx]
            video = sample['img'].unsqueeze(0).to(device)       # [1, T, 3, 64, 64]
            gt_masks = sample['gt_masks'].unsqueeze(0).to(device) if 'gt_masks' in sample else None  # [1, T, 3, 64, 64]

            out = model(video)
            recon = out.get('recon_img', None)
            if recon is not None:
                mse = torch.mean((recon - video) ** 2).item()
                mses.append(mse)

            pred_masks = out.get('pred_masks', None)
            T = video.shape[1]
            video_np = ((video[0].cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            pred_masks_np = pred_masks[0].cpu().numpy() if pred_masks is not None else np.zeros((T, 3, 64, 64))
            gt_masks_np = gt_masks[0].cpu().numpy() if gt_masks is not None else None

            p_bin = (pred_masks[0] > 0.5).float()
            g_bin = (gt_masks[0] > 0.5).float() if gt_masks is not None else None

            # Canonical greedy swap tracking (shared with eval.py + multi-swap rollout)
            if pred_masks is not None and gt_masks is not None:
                swap_info = greedy_slot_assignments(pred_masks[:1], gt_masks[:1], thresh=0.5)
                iou_mats = swap_info['iou_matrices'][0]  # [T, Kp, Kg]
                assigns = swap_info['assignments'][0]    # [T, Kp]

            print(f"\n================ Clip {c_idx} Slot Occupation Breakdown ================")

            for t in range(T):
                frame_rgb = video_np[t]
                gt_t = gt_masks_np[t] if gt_masks_np is not None else None
                banner_text = f"Clip {c_idx} Frame {t+1}/{T} | MSE: {mses[-1]:.6f}"

                combined_frame = render_slot_overlay_frame(
                    frame_rgb=frame_rgb,
                    pred_masks_t=pred_masks_np[t],
                    gt_masks_t=gt_t,
                    banner_text=banner_text,
                )
                vis_frames.append(combined_frame)

                if g_bin is not None:
                    p_t = p_bin[t]
                    iou_mat = iou_mats[t]
                    curr_assignment = assigns[t].tolist()
                    is_swapped = (t > 0 and not torch.equal(assigns[t], assigns[t - 1]))

                    areas_px = p_t.sum(dim=(-2, -1)).cpu().numpy()

                    print(f"--- Frame t = {t+1} {'[SLOT SWAP DETECTED!]' if is_swapped else ''} ---")
                    print(f"  Slot Occupation Areas (Total Image = 4096 px):")
                    for k in range(min(3, p_t.shape[0])):
                        best_gt = curr_assignment[k]
                        iou_val = iou_mat[k, best_gt].item() * 100
                        print(f"    Slot {k}: {int(areas_px[k]):4d} px ({areas_px[k]/4096*100:4.1f}% img) | Max Bound: {slot_names.get(best_gt, f'GT {best_gt}')} (IoU: {iou_val:5.1f}%)")

                    print("  Pairwise IoU Matrix (Slot x GT Object):")
                    print("          GT 0 (Robot)  GT 1 (T-Block)  GT 2 (Goal)")
                    for k in range(min(3, p_t.shape[0])):
                        print(f"  Slot {k}:   {iou_mat[k, 0].item()*100:6.1f}%       {iou_mat[k, 1].item()*100:6.1f}%       {iou_mat[k, 2].item()*100:6.1f}%")

    print("\n---------------- Quick Inspection Report ----------------")
    print(f"Inspected Clips:             {len(target_indices)}")
    print(f"Mean Reconstruction MSE:     {np.mean(mses):.6f}" if mses else "N/A")

    if vis_frames:
        save_frames_to_gif(vis_frames, out_gif, fps=5)
        print(f"Saved Quick Inspection GIF to: {out_gif}")
        save_frames_to_gif(vis_frames, "scratch/rollout_best_model.gif", fps=5)
    print("----------------------------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quick inspection and visualization CLI for SAVi models.")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument('--clip_idx', type=int, default=None, help="Specific clip index to evaluate")
    parser.add_argument('--num_sequences', type=int, default=5, help="Number of validation sequences to inspect")
    parser.add_argument('--out_gif', type=str, default="scratch/quick_infer_demo.gif", help="Output GIF path")
    parser.add_argument('--device', type=str, default='cpu', help="Target device (cpu or cuda)")
    args = parser.parse_args()

    run_quick_inference(args.ckpt_path, clip_idx=args.clip_idx, num_sequences=args.num_sequences, out_gif=args.out_gif, device=args.device)
