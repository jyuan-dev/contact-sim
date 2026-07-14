import sys
import os
import h5py
import numpy as np
import torch
import cv2
from PIL import Image

# Setup paths to import from datasets and models
REPO_ROOT = '/home/jyuan/jyuan-ws/contact-sim'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.pusht import PushTMaskHDF5Dataset
from src.models.detr import DETR, ResNetBackbone, Transformer, box_cxcywh_to_xyxy

def box_to_mask(box_xyxy, shape=(64, 64)):
    """
    Converts a normalized bounding box [x1, y1, x2, y2] to a binary mask.
    """
    H, W = shape
    x1, y1, x2, y2 = box_xyxy
    
    # Scale to pixel coordinates
    x1_px = int(round(x1 * W))
    y1_px = int(round(y1 * H))
    x2_px = int(round(x2 * W))
    y2_px = int(round(y2 * H))
    
    # Clamp to grid size
    x1_px = max(0, min(x1_px, W - 1))
    x2_px = max(0, min(x2_px, W - 1))
    y1_px = max(0, min(y1_px, H - 1))
    y2_px = max(0, min(y2_px, H - 1))
    
    mask = np.zeros((H, W), dtype=np.uint8)
    if x2_px >= x1_px and y2_px >= y1_px:
        mask[y1_px:y2_px + 1, x1_px:x2_px + 1] = 1
    return mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt_path = '/home/jyuan/.stable-wm/detr_pusht/detr_final.pt'
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    CFG = ckpt['config']
    print("Checkpoint config loaded.")
    
    # Build model using saved config
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
    print("DETR Model loaded and set to eval mode.")
    
    # Load dataset
    print("Loading validation dataset...")
    val_ds = PushTMaskHDF5Dataset(
        CFG['h5_path'], split='val',
        resolution=CFG['resolution'],
        n_sample_frames=CFG['n_sample_frames'],
        frame_offset=CFG['frame_offset'],
        train_frac=CFG['train_frac'],
    )
    
    # We will run inference on the first validation episode
    val_ep_idx = val_ds._episode_indices[0]
    print(f"Running inference on validation episode index: {val_ep_idx}")
    data = val_ds.get_video(val_ep_idx)
    video = data['video'].float().to(device)       # [T, 3, 64, 64]
    gt_masks = data['gt_masks'].float().cpu().numpy()  # [T, 3, 64, 64] - block, agent, goal
    
    T = video.shape[0]
    print(f"Episode length: {T} frames")
    
    # Predict
    with torch.no_grad():
        outputs = model(video)
        pred_logits = outputs['pred_logits']  # [T, Q, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [T, Q, 4] (cx, cy, w, h)
        
    print("Running evaluation and creating masks...")
    iou_scores = {0: [], 1: [], 2: []}  # class: score list
    dice_scores = {0: [], 1: [], 2: []}
    
    # Colors for block (Coral), agent (Emerald), goal (Blue)
    COLORS = {
        0: (107, 107, 255),  # block (Red-ish in BGR is (107,107,255))
        1: (196, 205, 78),   # agent (Green-ish in BGR is (196,205,78))
        2: (254, 182, 69)    # goal (Blue-ish in BGR is (254,182,69))
    }
    
    gif_frames = []
    
    for t in range(T):
        img_tensor = video[t]
        # Convert [-1, 1] tensor to [0, 255] RGB numpy image
        img_rgb = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 1. Ground Truth Overlay
        gt_overlay = img_bgr.copy()
        for c in range(3):
            mask_c = gt_masks[t, c] > 0.5
            color_mask = np.zeros_like(img_bgr)
            color_mask[mask_c] = COLORS[c]
            gt_overlay = cv2.addWeighted(gt_overlay, 1.0, color_mask, 0.5, 0)
            
        # 2. DETR Predicted Overlay
        pred_overlay = img_bgr.copy()
        pred_boxes_t = pred_boxes[t]  # [Q, 4] (cx, cy, w, h)
        logits_t = pred_logits[t]      # [Q, num_classes+1]
        probs_t = logits_t.softmax(-1)  # [Q, 4]
        
        # For each class c, find the query with highest confidence
        # Classes: 0 (block), 1 (agent), 2 (goal)
        pred_masks_t = {}
        for c in range(3):
            c_probs = probs_t[:, c]  # [Q]
            best_q = torch.argmax(c_probs).item()
            best_prob = c_probs[best_q].item()
            
            box_cxcywh = pred_boxes_t[best_q]
            box_xyxy = box_cxcywh_to_xyxy(box_cxcywh).cpu().numpy()
            
            # If the probability is above 0.3, treat it as a valid prediction
            if best_prob > 0.3:
                mask = box_to_mask(box_xyxy, shape=(64, 64))
            else:
                mask = np.zeros((64, 64), dtype=np.uint8)
                
            pred_masks_t[c] = mask
            
            # Compute metrics
            gt_mask_c = (gt_masks[t, c] > 0.5).astype(np.uint8)
            
            intersection = np.logical_and(mask, gt_mask_c).sum()
            union = np.logical_or(mask, gt_mask_c).sum()
            iou = intersection / union if union > 0 else 1.0
            
            dice = 2.0 * intersection / (mask.sum() + gt_mask_c.sum()) if (mask.sum() + gt_mask_c.sum()) > 0 else 1.0
            
            iou_scores[c].append(iou)
            dice_scores[c].append(dice)
            
            # Draw on predicted overlay
            if best_prob > 0.3:
                color_mask = np.zeros_like(img_bgr)
                color_mask[mask > 0] = COLORS[c]
                pred_overlay = cv2.addWeighted(pred_overlay, 1.0, color_mask, 0.5, 0)
                # Draw predicted bounding box boundary
                x1, y1, x2, y2 = box_xyxy
                x1_px = int(round(x1 * 64))
                y1_px = int(round(y1 * 64))
                x2_px = int(round(x2 * 64))
                y2_px = int(round(y2 * 64))
                cv2.rectangle(pred_overlay, (x1_px, y1_px), (x2_px, y2_px), COLORS[c], 1)
                
        # Resize to make it readable
        sz = 256
        orig_large = cv2.resize(img_bgr, (sz, sz), interpolation=cv2.INTER_NEAREST)
        gt_large = cv2.resize(gt_overlay, (sz, sz), interpolation=cv2.INTER_NEAREST)
        pred_large = cv2.resize(pred_overlay, (sz, sz), interpolation=cv2.INTER_NEAREST)
        
        # Put labels on the images
        cv2.putText(orig_large, "Original Frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(gt_large, "GT Segmentation", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(pred_large, "DETR Predicted", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Concatenate side by side
        combined = np.hstack([orig_large, gt_large, pred_large])
        
        # Convert BGR back to RGB for PIL GIF
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        gif_frames.append(combined_rgb)
        
    # Calculate average metrics
    print("\n--- Segmentation Metrics Summary ---")
    class_names = {0: "Block", 1: "Agent", 2: "Goal"}
    all_ious = []
    all_dices = []
    for c in range(3):
        avg_iou = np.mean(iou_scores[c])
        avg_dice = np.mean(dice_scores[c])
        all_ious.append(avg_iou)
        all_dices.append(avg_dice)
        print(f"{class_names[c]}: Mean IoU = {avg_iou:.4f}, Mean Dice = {avg_dice:.4f}")
    print(f"Overall Mean IoU (mIoU) = {np.mean(all_ious):.4f}")
    print(f"Overall Mean Dice = {np.mean(all_dices):.4f}")
    
    # Save GIF
    artifact_dir = '/home/jyuan/.gemini/antigravity-ide/brain/9593b9bf-bf14-456b-baaf-a168c84b378d'
    os.makedirs(artifact_dir, exist_ok=True)
    out_gif = os.path.join(artifact_dir, 'detr_segmentation_comparison.gif')
    
    pil_frames = [Image.fromarray(f) for f in gif_frames]
    pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:], duration=150, loop=0)
    print(f"\nSaved visualization GIF to: {out_gif}")

if __name__ == '__main__':
    main()
