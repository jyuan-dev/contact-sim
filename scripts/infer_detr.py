import os
import sys
import argparse
import yaml
import numpy as np
import torch
import cv2
import imageio
from PIL import Image

# Must import hdf5plugin before h5py to support Zstd decompression
import hdf5plugin
import h5py

# Add workspace root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.detr import (
    DETR,
    ResNetBackbone,
    Transformer,
    box_cxcywh_to_xyxy,
    masks_to_boxes_and_labels
)

# Colors: Class 0 (Block): Orange/Red, Class 1 (Agent): Green, Class 2 (Goal): Blue
CLASS_NAMES = {0: "Block", 1: "Agent", 2: "Goal"}
COLORS = {
    0: (0, 165, 255),  # Orange BGR
    1: (0, 255, 0),    # Green BGR
    2: (255, 0, 0)     # Blue BGR
}

def draw_box(img, box_xyxy, color, label_text, is_gt=False, score=None):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = min(max(int(x1 * W), 0), W - 1)
    x2 = min(max(int(x2 * W), 0), W - 1)
    y1 = min(max(int(y1 * H), 0), H - 1)
    y2 = min(max(int(y2 * H), 0), H - 1)
    
    # Draw rectangle
    thickness = 1 if is_gt else 2
    line_type = cv2.LINE_4 if is_gt else cv2.LINE_AA
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, line_type)
    
    # Label text
    suffix = " (GT)" if is_gt else f" ({score:.2f})"
    text = f"{label_text}{suffix}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    text_thickness = 1
    
    # Draw text background
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    tx = x1
    ty = y1 - 4 if y1 - 4 > th else y1 + th + 2
    
    cv2.rectangle(img, (tx, ty - th - 2), (tx + tw, ty + baseline), (30, 30, 30), -1)
    cv2.putText(img, text, (tx, ty), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='/home/jyuan/.stable-wm/detr_pusht/detr_epoch_2.pt',
                        help='Path to DETR checkpoint to run')
    parser.add_argument('--dataset', type=str, default='pusht',
                        help='Dataset config name')
    parser.add_argument('--episode_idx', type=int, default=0,
                        help='Index of the validation episode to run inference on')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Score threshold for displaying predictions')
    parser.add_argument('--max_frames', type=int, default=60,
                        help='Maximum frames to process in the output GIF')
    args = parser.parse_args()

    # Load Config from YAML
    config_path = os.path.join(REPO_ROOT, 'configs', 'detr', f'{args.dataset}.yaml')
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build validation dataset
    print(f"Building dataset for {args.dataset}...")
    from src.datasets.pusht import PushTMaskHDF5Dataset
    val_ds = PushTMaskHDF5Dataset(
        h5_path=cfg['h5_path'],
        split='val',
        resolution=tuple(cfg['resolution']),
        n_sample_frames=cfg['n_sample_frames'],
        frame_offset=cfg['frame_offset'],
        train_frac=cfg['train_frac']
    )

    # Load Model
    print("Building DETR model...")
    backbone = ResNetBackbone(train_backbone=False)
    transformer = Transformer(
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_encoder_layers=cfg['num_encoder_layers'],
        num_decoder_layers=cfg['num_decoder_layers'],
        dim_feedforward=cfg['dim_feedforward']
    )
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries']
    ).to(device)

    print(f"Loading checkpoint from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Get episode data
    ep_idx = val_ds._episode_indices[args.episode_idx]
    print(f"Running inference on validation episode index: {ep_idx}")
    data = val_ds.get_video(ep_idx)
    video = data['video'].float().to(device)
    gt_m = data['gt_masks'].float().to(device)

    T = min(video.shape[0], args.max_frames)
    frames_out = []

    for t in range(T):
        img_tensor = video[t] # [3, H, W]
        gt_mask_t = gt_m[t]   # [3, H, W]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0))
        
        pred_logits = outputs['pred_logits'][0] # [num_queries, num_classes + 1]
        pred_boxes = outputs['pred_boxes'][0]   # [num_queries, 4]

        # Convert back to HWC uint8 for drawing
        img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8).copy()
        
        # Convert BGR (OpenCV) to RGB for final rendering
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw Ground Truth bounding boxes derived from segment masks
        gt_targets = masks_to_boxes_and_labels(gt_mask_t.unsqueeze(0))[0]
        for box, label in zip(gt_targets['boxes'], gt_targets['labels']):
            box_xyxy = box_cxcywh_to_xyxy(box)
            label_idx = label.item()
            img_bgr = draw_box(img_bgr, box_xyxy, COLORS[label_idx], CLASS_NAMES[label_idx], is_gt=True)

        # Draw Predicted bounding boxes from DETR
        probs = pred_logits.softmax(-1)
        scores, labels = probs[:, :-1].max(-1)
        
        for score, label, box in zip(scores, labels, pred_boxes):
            if score > args.threshold:
                box_xyxy = box_cxcywh_to_xyxy(box)
                label_idx = label.item()
                img_bgr = draw_box(img_bgr, box_xyxy, COLORS[label_idx], CLASS_NAMES[label_idx], is_gt=False, score=score.item())

        # Convert BGR back to RGB for imageio
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames_out.append(img_rgb)

    # Save outputs
    output_dir = os.path.join(REPO_ROOT, 'scratch')
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, 'detr_inference.gif')
    imageio.mimsave(gif_path, frames_out, fps=4)
    print(f"Successfully saved inference video to: {gif_path}")
    
    # Save a comparison grid image of some frames
    step = max(1, T // 4)
    sel_frames = [frames_out[i] for i in range(0, T, step)[:4]]
    grid_img = np.hstack(sel_frames)
    png_path = os.path.join(output_dir, 'detr_inference.png')
    Image.fromarray(grid_img).save(png_path)
    print(f"Successfully saved grid preview to: {png_path}")

if __name__ == '__main__':
    main()
