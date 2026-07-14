"""
Evaluate the trained DETR model on the PushT validation set.

Computes per-class IoU and Dice scores and optionally saves a comparison GIF.

Usage:
    python eval/infer_detr.py --ckpt-path /path/to/detr_final.pt [--save-gif] [--output-dir ./eval_results]
"""
import sys
import os
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.pusht import PushTMaskHDF5Dataset
from src.models.detr import DETR, ResNetBackbone, Transformer, box_cxcywh_to_xyxy


def box_to_mask(box_xyxy, shape=(64, 64)):
    """Convert a normalized bounding box [x1, y1, x2, y2] to a binary mask."""
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
    parser = argparse.ArgumentParser(description='DETR segmentation evaluation on PushT val set')
    parser.add_argument('--ckpt-path', required=True, help='Path to DETR checkpoint (.pt file)')
    parser.add_argument('--output-dir', default='.', help='Directory to save outputs (default: current dir)')
    parser.add_argument('--save-gif', action='store_true', help='Save visualization GIF to output-dir')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    CFG = ckpt['config']
    print("Checkpoint config loaded.")

    backbone = ResNetBackbone(train_backbone=False)
    transformer = Transformer(
        d_model=CFG['d_model'],
        nhead=CFG['nhead'],
        num_encoder_layers=CFG['num_encoder_layers'],
        num_decoder_layers=CFG['num_decoder_layers'],
        dim_feedforward=CFG['dim_feedforward']
    )
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=CFG['num_classes'],
        num_queries=CFG['num_queries']
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("DETR model loaded and set to eval mode.")

    print("Loading validation dataset...")
    val_ds = PushTMaskHDF5Dataset(
        CFG['h5_path'], split='val',
        resolution=CFG['resolution'],
        n_sample_frames=CFG['n_sample_frames'],
        frame_offset=CFG['frame_offset'],
        train_frac=CFG['train_frac'],
    )

    val_ep_idx = val_ds._episode_indices[0]
    print(f"Running inference on validation episode index: {val_ep_idx}")
    data = val_ds.get_video(val_ep_idx)
    video = data['video'].float().to(device)          # [T, 3, 64, 64]
    gt_masks = data['gt_masks'].float().cpu().numpy() # [T, 3, 64, 64]

    T = video.shape[0]
    print(f"Episode length: {T} frames")

    with torch.no_grad():
        outputs = model(video)
        pred_logits = outputs['pred_logits']  # [T, Q, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [T, Q, 4]

    print("Running evaluation...")
    iou_scores  = {0: [], 1: [], 2: []}
    dice_scores = {0: [], 1: [], 2: []}

    # Colors for block (Coral), agent (Emerald), goal (Blue) in BGR
    COLORS = {
        0: (107, 107, 255),
        1: (196, 205, 78),
        2: (254, 182, 69),
    }
    class_names = {0: 'Block', 1: 'Agent', 2: 'Goal'}

    gif_frames = []

    for t in range(T):
        img_tensor = video[t]
        img_rgb = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        gt_overlay   = img_bgr.copy()
        pred_overlay = img_bgr.copy()

        for c in range(3):
            # GT overlay
            mask_c = gt_masks[t, c] > 0.5
            color_mask = np.zeros_like(img_bgr)
            color_mask[mask_c] = COLORS[c]
            gt_overlay = cv2.addWeighted(gt_overlay, 1.0, color_mask, 0.5, 0)

            # DETR prediction
            probs_t    = pred_logits[t].softmax(-1)
            best_q     = torch.argmax(probs_t[:, c]).item()
            best_prob  = probs_t[best_q, c].item()

            box_cxcywh = pred_boxes[t, best_q]
            box_xyxy   = box_cxcywh_to_xyxy(box_cxcywh).cpu().numpy()

            if best_prob > 0.3:
                mask = box_to_mask(box_xyxy, shape=(64, 64))
            else:
                mask = np.zeros((64, 64), dtype=np.uint8)

            # Metrics
            gt_mask_c    = (gt_masks[t, c] > 0.5).astype(np.uint8)
            intersection = np.logical_and(mask, gt_mask_c).sum()
            union        = np.logical_or(mask, gt_mask_c).sum()
            iou          = intersection / union if union > 0 else 1.0
            denom        = mask.sum() + gt_mask_c.sum()
            dice         = 2.0 * intersection / denom if denom > 0 else 1.0
            iou_scores[c].append(iou)
            dice_scores[c].append(dice)

            if best_prob > 0.3:
                cm = np.zeros_like(img_bgr)
                cm[mask > 0] = COLORS[c]
                pred_overlay = cv2.addWeighted(pred_overlay, 1.0, cm, 0.5, 0)
                x1, y1, x2, y2 = box_xyxy
                cv2.rectangle(pred_overlay,
                              (int(round(x1 * 64)), int(round(y1 * 64))),
                              (int(round(x2 * 64)), int(round(y2 * 64))),
                              COLORS[c], 1)

        if args.save_gif:
            sz = 256
            orig_l = cv2.resize(img_bgr,    (sz, sz), interpolation=cv2.INTER_NEAREST)
            gt_l   = cv2.resize(gt_overlay, (sz, sz), interpolation=cv2.INTER_NEAREST)
            pred_l = cv2.resize(pred_overlay,(sz, sz), interpolation=cv2.INTER_NEAREST)
            cv2.putText(orig_l,  'Original Frame',  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(gt_l,    'GT Segmentation', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(pred_l,  'DETR Predicted',  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            combined = np.hstack([orig_l, gt_l, pred_l])
            gif_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

    print('\n--- Segmentation Metrics Summary ---')
    all_ious, all_dices = [], []
    for c in range(3):
        avg_iou  = np.mean(iou_scores[c])
        avg_dice = np.mean(dice_scores[c])
        all_ious.append(avg_iou)
        all_dices.append(avg_dice)
        print(f'{class_names[c]}: Mean IoU = {avg_iou:.4f}, Mean Dice = {avg_dice:.4f}')
    print(f'Overall Mean IoU  (mIoU) = {np.mean(all_ious):.4f}')
    print(f'Overall Mean Dice        = {np.mean(all_dices):.4f}')

    if args.save_gif:
        os.makedirs(args.output_dir, exist_ok=True)
        out_gif = os.path.join(args.output_dir, 'detr_segmentation_comparison.gif')
        pil_frames = [Image.fromarray(f) for f in gif_frames]
        pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:], duration=150, loop=0)
        print(f'\nSaved GIF to: {out_gif}')


if __name__ == '__main__':
    main()
