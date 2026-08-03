#!/usr/bin/env python3
"""
Unified Baseline Evaluation CLI Entrypoint powered by Hydra.

Usage:
  python scripts/eval.py model=savi dataset=pusht ckpt_path=/home/jyuan/.stable-wm/savi_mask_detr/savi_epoch_8.pt
  python scripts/eval.py model=detr dataset=pusht ckpt_path=/home/jyuan/.stable-wm/detr_pusht/detr_final.pt
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.metrics.evaluator import EvaluationSuite

SLOT_COLORS_RGB = {
    0: (255, 40, 40),     # Slot 0: Red
    1: (40, 220, 40),     # Slot 1: Green
    2: (40, 120, 255),    # Slot 2: Blue
    3: (255, 210, 0),     # Slot 3: Yellow
    4: (230, 40, 230)     # Slot 4: Magenta
}

GT_COLORS_RGB = {
    0: (255, 140, 0),    # Orange
    1: (0, 230, 115),    # Green
    2: (0, 128, 255)     # Blue
}


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ckpt_path = cfg.get('ckpt_path', None) or cfg.get('ckpt', None)
    model_name = cfg.model.name
    dataset_name = cfg.dataset.name

    print("======================================================================")
    print(f"            Hydra Baseline Evaluator ({model_name} / {dataset_name})  ")
    print("======================================================================")
    print(f"Device: {device}")

    # 1. Build Model via Factory
    model = build_model(cfg_dict).to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading Checkpoint: {ckpt_path}")
        ckpt_data = torch.load(ckpt_path, map_location=device)
        target_model = model.model if hasattr(model, 'model') else model
        state = ckpt_data.get('model', ckpt_data.get('model_state', ckpt_data))
        state = {k.replace('model.', '').replace('module.', ''): v for k, v in state.items()}
        target_model.load_state_dict(state, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print("No valid checkpoint specified or file not found. Running with initial weights.")

    model.eval()

    # 2. Build Dataloader via Factory
    val_loader = build_dataloader(cfg_dict, split='val', batch_size=1, num_workers=2, shuffle=False)

    evaluator = EvaluationSuite(num_classes=3)
    batch = next(iter(val_loader))

    imgs_torch = batch['img'] if 'img' in batch else batch['video']
    imgs_torch = imgs_torch.to(device)
    gt_masks = batch.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = gt_masks.to(device)

    # 3. Forward Pass
    with torch.no_grad():
        out = model(imgs_torch)

    pred_masks = out.get('pred_masks', None)
    if pred_masks is not None:
        pred_masks_np = pred_masks[0].cpu().numpy()
    else:
        T = imgs_torch.shape[1] if imgs_torch.ndim == 5 else 1
        pred_masks_np = np.zeros((T, 4, 64, 64), dtype=np.float32)

    # 4. Evaluate Metrics
    if gt_masks is not None:
        gt_masks_np = gt_masks[0].cpu().numpy() # [T, M, H, W]
        gt_masks_dict = {m_idx: (gt_masks_np[:, m_idx] > 0.5) for m_idx in range(gt_masks_np.shape[1])}
    else:
        gt_masks_dict = {}

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
    video_tensor = imgs_torch[0].cpu()
    if video_tensor.ndim == 3:
        video_tensor = video_tensor.unsqueeze(0)

    T = video_tensor.shape[0]
    img_raw_np = ((video_tensor.permute(0, 2, 3, 1).numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    for t in range(T):
        frame_rgb = img_raw_np[t]
        if frame_rgb.shape[:2] != (64, 64):
            frame_rgb = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Left Panel: GT Outlines
        p_gt = frame_rgb.copy()
        if gt_masks is not None:
            for m_idx in range(gt_masks_np.shape[1]):
                m_bin = gt_masks_np[t, m_idx] > 0.5
                if m_bin.any():
                    contours, _ = cv2.findContours(m_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color_bgr = GT_COLORS_RGB.get(m_idx, (255, 255, 255))
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

    # Save JSON report
    report_json_path = "scratch/baseline_eval_metrics.json"
    with open(report_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved JSON metrics report to: {report_json_path}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
