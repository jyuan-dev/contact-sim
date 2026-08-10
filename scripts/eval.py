#!/usr/bin/env python3
"""
Unified Baseline Evaluation CLI Entrypoint powered by Hydra.

Usage:
  python scripts/eval.py model=savi dataset=pusht ckpt_path=/home/jyuan/.stable-wm/savi_mask_detr/savi_epoch_8.pt
  python scripts/eval.py model=detr dataset=pusht ckpt_path=/home/jyuan/.stable-wm/detr_pusht/detr_final.pt
"""

import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.metrics.evaluator import EvaluationSuite
from src.utils.training_utils import get_device

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

    device = get_device()

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
    num_eval_batches = cfg.get('eval_batches', 20)
    all_metrics = []

    print(f"Running Evaluation on {num_eval_batches} validation sequences...")
    for step_idx, batch in enumerate(val_loader):
        if step_idx >= num_eval_batches:
            break
        imgs_torch = batch['img'].to(device)
        gt_masks = batch.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = gt_masks.to(device)

        with torch.no_grad():
            out = model(imgs_torch)

        pred_masks = out.get('pred_masks', None)
        pred_boxes = out.get('pred_boxes', None)
        pred_logits = out.get('pred_logits', None)
        T = imgs_torch.shape[1] if imgs_torch.ndim == 5 else 1

        if pred_masks is not None:
            pred_masks_np = pred_masks[0].cpu().numpy()
        elif pred_boxes is not None and pred_logits is not None:
            Q = pred_boxes.shape[1]
            pred_masks_np = np.zeros((T, Q, 64, 64), dtype=np.float32)
            pred_classes = pred_logits.argmax(dim=-1).cpu().numpy()
            pred_boxes_np = pred_boxes.cpu().numpy()
            num_classes_detr = pred_logits.shape[-1] - 1
            
            for t_idx in range(min(T, pred_boxes_np.shape[0])):
                for q in range(Q):
                    cls_id = pred_classes[t_idx, q]
                    if cls_id < num_classes_detr:
                        cx, cy, w, h = pred_boxes_np[t_idx, q]
                        x0 = int(np.clip((cx - 0.5 * w) * 64, 0, 64))
                        y0 = int(np.clip((cy - 0.5 * h) * 64, 0, 64))
                        x1 = int(np.clip((cx + 0.5 * w) * 64, 0, 64))
                        y1 = int(np.clip((cy + 0.5 * h) * 64, 0, 64))
                        if x1 > x0 and y1 > y0:
                            pred_masks_np[t_idx, q, y0:y1, x0:x1] = 1.0
        else:
            pred_masks_np = np.zeros((T, 4, 64, 64), dtype=np.float32)

        if gt_masks is not None:
            gt_masks_np = gt_masks[0].cpu().numpy() # [T, M, H, W]
            gt_masks_dict = {m_idx: (gt_masks_np[:, m_idx] > 0.5) for m_idx in range(gt_masks_np.shape[1])}
        else:
            gt_masks_dict = {}

        seq_metrics = evaluator.evaluate_sequence_masks(pred_masks_np, gt_masks_dict)
        all_metrics.append(seq_metrics)

    # Aggregate metrics across evaluated validation sequences
    total_frames = sum(m['total_frames'] for m in all_metrics)
    total_swaps = sum(m['total_swap_events'] for m in all_metrics)
    overall_mIoU = float(np.mean([m['overall_mIoU'] for m in all_metrics]))
    overall_mDice = float(np.mean([m['overall_mDice'] for m in all_metrics]))
    swap_rate = (total_swaps / max(total_frames, 1)) * 100.0

    class_names_map = {0: "Agent", 1: "Block", 2: "Goal"}
    class_metrics = {}
    for cls_idx, cls_name in class_names_map.items():
        cls_ious = [m['class_metrics'].get(cls_name, {}).get('mean_iou', 0.0) for m in all_metrics if cls_name in m['class_metrics']]
        cls_dices = [m['class_metrics'].get(cls_name, {}).get('mean_dice', 0.0) for m in all_metrics if cls_name in m['class_metrics']]
        class_metrics[cls_name] = {
            'mean_iou': float(np.mean(cls_ious)) if cls_ious else 0.0,
            'mean_dice': float(np.mean(cls_dices)) if cls_dices else 0.0,
        }

    metrics = {
        'total_frames': total_frames,
        'total_swap_events': total_swaps,
        'swap_rate_per_100_frames': float(swap_rate),
        'class_metrics': class_metrics,
        'overall_mIoU': overall_mIoU,
        'overall_mDice': overall_mDice,
    }

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

        # Right Panel: Pure Slot Mask Overlay or Bounding Box Overlay for DETR
        p_slots = frame_rgb.copy().astype(np.float32)
        masks_t = pred_masks_np[t]
        K = masks_t.shape[0]

        if pred_boxes is not None and pred_logits is not None:
            pred_classes = pred_logits.argmax(dim=-1).cpu().numpy()
            pred_boxes_np = pred_boxes.cpu().numpy()
            num_classes_detr = pred_logits.shape[-1] - 1
            
            p_slots_uint8 = frame_rgb.copy()
            t_idx = min(t, pred_boxes_np.shape[0] - 1)
            for q in range(K):
                cls_id = pred_classes[t_idx, q]
                if cls_id < num_classes_detr:
                    cx, cy, w, h = pred_boxes_np[t_idx, q]
                    x0 = int(np.clip((cx - 0.5 * w) * 64, 0, 64))
                    y0 = int(np.clip((cy - 0.5 * h) * 64, 0, 64))
                    x1 = int(np.clip((cx + 0.5 * w) * 64, 0, 64))
                    y1 = int(np.clip((cy + 0.5 * h) * 64, 0, 64))
                    color_k = SLOT_COLORS_RGB[q % len(SLOT_COLORS_RGB)]
                    cv2.rectangle(p_slots_uint8, (x0, y0), (x1, y1), color_k, 1)
                    cls_name = {0: "B", 1: "A", 2: "G"}.get(cls_id, str(cls_id))
                    cv2.putText(p_slots_uint8, cls_name, (x0 + 2, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color_k, 1, cv2.LINE_AA)
        else:
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
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in vis_frames]
    if pil_frames:
        pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
    print(f"Saved evaluation video to: {out_gif}")

    # Save JSON report
    report_json_path = "scratch/baseline_eval_metrics.json"
    with open(report_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved JSON metrics report to: {report_json_path}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
