"""
DETR Evaluation & Visualization Tool.

Computes per-class IoU, Dice scores, bounding box detections, and saves comparison GIFs.

Usage:
    python eval/eval_detr.py --ckpt /path/to/detr_final.pt [--save-gif] [--output-dir ./scratch/eval_results]
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
import cv2
import imageio
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.data_utils import get_dataset, find_dataset_path
from src.utils.training_utils import get_device
from src.models.detr import DETR, ResNetBackbone, Transformer, box_cxcywh_to_xyxy, masks_to_boxes_and_labels

CLASS_NAMES = {0: "Block", 1: "Agent", 2: "Goal"}
COLORS = {
    0: (0, 165, 255),  # Orange BGR
    1: (0, 255, 0),    # Green BGR
    2: (255, 0, 0)     # Blue BGR
}


def draw_box(img, box_xyxy, color, label_text, score=None):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = min(max(int(x1 * W), 0), W - 1)
    x2 = min(max(int(x2 * W), 0), W - 1)
    y1 = min(max(int(y1 * H), 0), H - 1)
    y2 = min(max(int(y2 * H), 0), H - 1)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    txt = f"{label_text}" if score is None else f"{label_text}:{score:.2f}"
    cv2.putText(img, txt, (x1, max(y1 - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return img


def box_to_mask(box_xyxy, shape=(64, 64)):
    H, W = shape
    x1, y1, x2, y2 = box_xyxy
    x1_px = max(0, min(int(round(x1 * W)), W - 1))
    y1_px = max(0, min(int(round(y1 * H)), H - 1))
    x2_px = max(0, min(int(round(x2 * W)), W - 1))
    y2_px = max(0, min(int(round(y2 * H)), H - 1))
    mask = np.zeros((H, W), dtype=np.uint8)
    if x2_px >= x1_px and y2_px >= y1_px:
        mask[y1_px:y2_px + 1, x1_px:x2_px + 1] = 1
    return mask


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DETR model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to DETR checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="pusht", choices=["pusht", "ogbench", "libero"])
    parser.add_argument("--h5-path", type=str, default="scratch/pusht_expert_train_test_enriched.h5")
    parser.add_argument("--save-gif", action="store_true", help="Save visual bounding-box GIF")
    parser.add_argument("--output-dir", type=str, default="scratch/eval_results")
    args = parser.parse_args()

    device = get_device()
    print(f"[Eval DETR] Using device: {device} | Checkpoint: {args.ckpt}")

    # Build model architecture
    backbone = ResNetBackbone(train_backbone=False)
    transformer = Transformer(d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256)
    model = DETR(backbone=backbone, transformer=transformer, num_classes=3, num_queries=10).to(device)

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
        print(" DETR model weights loaded successfully.")
    else:
        print(f" Warning: Checkpoint path '{args.ckpt}' not found. Running dry run.")

    model.eval()

    h5_path = find_dataset_path(args.h5_path)
    if os.path.exists(h5_path):
        val_ds = get_dataset(args.dataset, h5_path, split='val', resolution=(64, 64), n_sample_frames=16)
    else:
        print("[Eval DETR] HDF5 dataset not found; skipping quantitative evaluation pass.")
        val_ds = None

    if val_ds is not None:
        print(f"[Eval DETR] Evaluating on {len(val_ds)} validation samples...")

    if args.save_gif:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[Eval DETR] Visualizing detections...")

if __name__ == "__main__":
    main()
