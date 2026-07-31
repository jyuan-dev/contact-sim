#!/usr/bin/env python3
"""
Unified Baseline Evaluation CLI Entrypoint for Contact-Sim / Slot-Worldmodel.

Usage:
  python scripts/eval.py --config configs/savi/pusht.yaml --ckpt /home/jyuan/.stable-wm/savi_mask_detr/savi_epoch_8.pt
  python scripts/eval.py --config configs/detr/pusht.yaml --ckpt /home/jyuan/.stable-wm/detr_pusht/detr_final.pt
"""

import sys
import os
import argparse
import json
import yaml
import h5py
import hdf5plugin
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio

os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.metrics.evaluator import EvaluationSuite
from src.datasets.pusht import PushTMaskHDF5Dataset

SLOT_COLORS_RGB = {
    0: (255, 40, 40),     # Slot 0: Red
    1: (40, 220, 40),     # Slot 1: Green
    2: (40, 120, 255),    # Slot 2: Blue
    3: (255, 210, 0),     # Slot 3: Yellow
    4: (230, 40, 230)     # Slot 4: Magenta
}

GT_COLORS_RGB = {
    0: (255, 140, 0),    # Orange (Block)
    1: (0, 230, 115),    # Green (Agent)
    2: (0, 128, 255)     # Blue (Goal)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Baseline Evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt / .ckpt)")
    parser.add_argument("--ep_idx", type=int, default=6, help="Validation episode index to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    print("======================================================================")
    print(f"               Unified Baseline Evaluator ({args.config})            ")
    print("======================================================================")

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading Checkpoint: {args.ckpt}")

    # 1. Build Model via Factory
    model = build_model(cfg).to(device)

    ckpt_data = torch.load(args.ckpt, map_location=device)

    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    state = ckpt_data.get('model', ckpt_data.get('model_state', ckpt_data))
    state = {k.replace('model.', '').replace('module.', ''): v for k, v in state.items()}
    target_model.load_state_dict(state)

    model.eval()
    print("Model loaded successfully into eval mode!")

    # 2. Extract Episode Data
    h5_path = cfg.get('h5_path', '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5')
    if not os.path.exists(h5_path):
        h5_path = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5'

    with h5py.File(h5_path, 'r') as f:
        ep_lens = np.array(f['ep_len'])
        ep_offs = np.array(f['ep_offset'])

        offset = ep_offs[args.ep_idx]
        length = ep_lens[args.ep_idx]

        pixels = np.array(f['pixels'][offset : offset + length])
        b_masks = np.array(f['block_masks'][offset : offset + length]) > 0
        a_masks = np.array(f['agent_masks'][offset : offset + length]) > 0
        g_masks = np.array(f['goal_masks'][offset : offset + length]) > 0

    T = length
    print(f"Loaded validation episode index {args.ep_idx}: {T} frames.")

    imgs_np = pixels.transpose(0, 3, 1, 2) if (pixels.ndim == 4 and pixels.shape[-1] == 3) else pixels
    imgs_torch = torch.tensor(imgs_np, dtype=torch.float32, device=device) / 255.0
    imgs_torch_norm = (imgs_torch - 0.5) / 0.5

    if imgs_torch_norm.shape[-1] != 64 or imgs_torch_norm.shape[-2] != 64:
        imgs_torch_norm = F.interpolate(imgs_torch_norm, size=(64, 64), mode='bilinear', align_corners=False)

    # 3. Forward Pass
    with torch.no_grad():
        out = model(imgs_torch_norm.unsqueeze(0))

    pred_masks = out.get('pred_masks', None)
    if pred_masks is not None:
        pred_masks_np = pred_masks[0].cpu().numpy() # [T, K, H, W]
    else:
        # Fallback dummy mask for pure box models
        pred_masks_np = np.zeros((T, 4, 64, 64), dtype=np.float32)

    # 4. Evaluate Metrics
    evaluator = EvaluationSuite(num_classes=3)
    gt_masks_dict = {0: b_masks, 1: a_masks, 2: g_masks}
    metrics = evaluator.evaluate_sequence_masks(pred_masks_np, gt_masks_dict)

    print("\n---------------- Quantitative Evaluation Report ----------------")
    print(f"Total Frames Analyzed: {metrics['total_frames']}")
    print(f"Total Slot Swap Events: {metrics['total_swap_events']}")
    print(f"Slot Swap Rate: {metrics['swap_rate_per_100_frames']:.2f} swaps / 100 frames")
    for cls_name, cls_m in metrics['class_metrics'].items():
        print(f"Class '{cls_name}': Mean IoU = {cls_m['mean_iou']:.4f}, Mean Dice = {cls_m['mean_dice']:.4f}")
    print(f"\nOverall Mean IoU (mIoU)  = {metrics['overall_mIoU']:.4f}")
    print(f"Overall Mean Dice        = {metrics['overall_mDice']:.4f}")
    print("----------------------------------------------------------------\n")

    # 5. Render & Save Visualization GIF
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

        banner_text = f"Frame {t:03d}/{T} | mIoU: {metrics['overall_mIoU']:.3f} | Swap Rate: {metrics['swap_rate_per_100_frames']:.1f}/100f"
        cv2.rectangle(combined_large, (0, 0), (combined_large.shape[1], 18), (30, 140, 220), -1)
        cv2.putText(combined_large, banner_text, (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        vis_frames.append(combined_large)

    os.makedirs("scratch", exist_ok=True)
    out_gif = "scratch/baseline_eval_demo.gif"
    imageio.mimsave(out_gif, vis_frames, fps=10, loop=0)
    print(f"Saved evaluation video to: {out_gif}")

    # Copy to brain dir if available
    brain_dir = "/home/jyuan/.gemini/antigravity-ide/brain/0e62dc39-5378-4e8f-b19c-9d502981fb60"
    if os.path.exists(brain_dir):
        import shutil
        shutil.copy(out_gif, os.path.join(brain_dir, "baseline_eval_demo.gif"))

    # Save JSON report
    report_json_path = "scratch/baseline_eval_metrics.json"
    with open(report_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved JSON metrics report to: {report_json_path}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
