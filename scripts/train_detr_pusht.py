"""
Train an authentic DETR model on PushT data for object detection.

Predicts coordinates [cx, cy, w, h] and labels for:
  - Class 0: block
  - Class 1: agent
  - Class 2: goal
  - Class 3: background (no-object)

Optimized via Hungarian bipartite matching using L1 box loss, GIoU loss, and Cross Entropy classification.
"""

import sys
import os
import argparse
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import h5py
import hdf5plugin
import cv2
from scipy.optimize import linear_sum_assignment

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOTFORMER = os.path.join(REPO_ROOT, 'third_party', 'cjepa', 'src',
                          'third_party', 'slotformer')
HDF5_DS    = os.path.join(SLOTFORMER, 'base_slots')

for p in [REPO_ROOT, SLOTFORMER, HDF5_DS]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    h5_path        = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5',
    resolution     = (64, 64),
    n_sample_frames= 6,
    frame_offset   = 1,
    train_frac     = 0.8,

    # Model parameters
    num_queries    = 5,      # 3 target objects + 2 background slots
    num_classes    = 3,      # 0: block, 1: agent, 2: goal
    d_model        = 128,    # Keep it lightweight for 64x64 inputs
    nhead          = 4,
    num_encoder_layers = 3,
    num_decoder_layers = 3,
    dim_feedforward = 512,

    # Training
    max_epochs      = 15,
    batch_size      = 256,
    num_workers     = 8,
    lr              = 2e-4,
    clip_grad       = 0.1,
    warmup_pct      = 0.05,

    # Loss weights
    weight_class   = 1.0,
    weight_bbox    = 5.0,
    weight_giou    = 2.0,
    eos_coef       = 0.1,    # Scale class loss for background queries

    # I/O
    ckpt_dir        = '/home/jyuan/.stable-wm/detr_pusht',
    tb_dir          = '/home/jyuan/.stable-wm/detr_pusht/tb_logs',
    save_every_n_epochs = 2,
    vis_every_n_epochs  = 1,
    n_vis_samples       = 4,
)


from datasets.pusht import PushTMaskHDF5Dataset
from models.detr import (
    DETR,
    ResNetBackbone,
    Transformer,
    HungarianMatcher,
    SetCriterion,
    box_cxcywh_to_xyxy,
    masks_to_boxes_and_labels
)



def draw_box(img, box_xyxy, color, thickness=1):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = min(max(int(x1 * W), 0), W - 1)
    x2 = min(max(int(x2 * W), 0), W - 1)
    y1 = min(max(int(y1 * H), 0), H - 1)
    y2 = min(max(int(y2 * H), 0), H - 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


@torch.no_grad()
def visualize_detr(model, val_ds, n_samples, device, writer, epoch):
    model.eval()
    grids = []
    
    for i in range(min(n_samples, len(val_ds._episode_indices))):
        ep_idx = val_ds._episode_indices[i]
        data   = val_ds.get_video(ep_idx)
        video  = data['video'].float().to(device)
        gt_m   = data['gt_masks'].float().to(device)
        T      = min(video.shape[0], CFG['n_sample_frames'])
        
        clip   = video[:T]
        gt_clip= gt_m[:T]
        
        outputs = model(clip)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        gt_targets = masks_to_boxes_and_labels(gt_clip)
        
        rows = []
        for t in range(T):
            img_tensor = clip[t]
            img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            
            # 1. Ground Truth image
            gt_img = img_np.copy()
            gt_t = gt_targets[t]
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_t['boxes'])
            for box, label in zip(gt_boxes_xyxy, gt_t['labels']):
                color = (255, 0, 0) if label == 0 else ((0, 255, 0) if label == 1 else (0, 0, 255))
                gt_img = draw_box(gt_img, box.cpu().numpy(), color, thickness=1)
                
            # 2. Predicted image (confidence threshold score > 0.3)
            pred_img = img_np.copy()
            probs = pred_logits[t].softmax(-1)
            scores, labels = probs[:, :-1].max(-1)
            
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[t])
            for q_idx in range(len(scores)):
                score = scores[q_idx].item()
                if score > 0.3:
                    label = labels[q_idx].item()
                    color = (255, 0, 0) if label == 0 else ((0, 255, 0) if label == 1 else (0, 0, 255))
                    pred_img = draw_box(pred_img, pred_boxes_xyxy[q_idx].cpu().numpy(), color, thickness=1)
                    
            gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0
            pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).float() / 255.0
            orig_tensor = (img_tensor * 0.5 + 0.5).clamp(0, 1).cpu()
            
            # Row contents: Original Frame | GT Bounding Boxes | Pred Bounding Boxes
            rows.append(torch.stack([orig_tensor, gt_tensor, pred_tensor], dim=0))
            
        grid = torch.stack(rows, dim=0).flatten(0, 1)
        grids.append(vutils.make_grid(grid, nrow=3, pad_value=1.0))
        
    writer.add_image('val/detections',
                     torch.stack(grids, dim=0).mean(0),
                     global_step=epoch)
    model.train()


# ── Training Helpers ──────────────────────────────────────────────────────────

def cosine_anneal_with_warmup(step, total_steps, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--limit_train_batches', type=int, default=None,
                        help='Limit train batches per epoch (for verification)')
    parser.add_argument('--limit_val_batches', type=int, default=None,
                        help='Limit validation batches per epoch (for verification)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override maximum training epochs')
    args = parser.parse_args()

    if args.max_epochs is not None:
        CFG['max_epochs'] = args.max_epochs

    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    os.makedirs(CFG['tb_dir'],   exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building datasets …")
    train_ds = PushTMaskHDF5Dataset(
        CFG['h5_path'], split='train',
        resolution=CFG['resolution'],
        n_sample_frames=CFG['n_sample_frames'],
        frame_offset=CFG['frame_offset'],
        train_frac=CFG['train_frac'],
    )
    val_ds = PushTMaskHDF5Dataset(
        CFG['h5_path'], split='val',
        resolution=CFG['resolution'],
        n_sample_frames=CFG['n_sample_frames'],
        frame_offset=CFG['frame_offset'],
        train_frac=CFG['train_frac'],
    )

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=CFG['num_workers'],
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=CFG['num_workers'],
                              pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building DETR model …")
    backbone = ResNetBackbone(train_backbone=True)
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ── Matcher & Loss ────────────────────────────────────────────────────────
    matcher = HungarianMatcher(
        cost_class=CFG['weight_class'],
        cost_bbox=CFG['weight_bbox'],
        cost_giou=CFG['weight_giou']
    )
    
    weight_dict = {
        'loss_ce': CFG['weight_class'],
        'loss_bbox': CFG['weight_bbox'],
        'loss_giou': CFG['weight_giou']
    }
    
    criterion = SetCriterion(
        num_classes=CFG['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=CFG['eos_coef'],
        losses=['labels', 'boxes']
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    steps_per_epoch = len(train_loader) if args.limit_train_batches is None else args.limit_train_batches
    total_steps  = CFG['max_epochs'] * steps_per_epoch
    warmup_steps = int(CFG['warmup_pct'] * total_steps)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=CFG['tb_dir'])
    writer.add_text('config', str(CFG), global_step=0)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Starting training: {CFG['max_epochs']} epochs, {steps_per_epoch} steps/epoch")

    for epoch in range(start_epoch, CFG['max_epochs']):
        model.train()
        epoch_losses = {'total': 0., 'ce': 0., 'bbox': 0., 'giou': 0.}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_idx >= args.limit_train_batches:
                break

            img      = batch['img'].to(device, non_blocking=True)       # [B, T, 3, H, W]
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)  # [B, T, 3, H, W]

            # Flatten temporal dim: treats each frame as a standalone object detection image
            # B, T, C, H, W -> B * T, C, H, W
            B, T = img.shape[:2]
            img_flat = img.flatten(0, 1)
            gt_flat = gt_masks.flatten(0, 1)

            # Convert GT masks to bounding box targets
            targets = masks_to_boxes_and_labels(gt_flat)

            # LR schedule
            lr = cosine_anneal_with_warmup(global_step, total_steps,
                                           warmup_steps, CFG['lr'],
                                           CFG['lr'] / 100.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward
            outputs = model(img_flat)
            
            # Loss calculation
            loss_dict = criterion(outputs, targets)
            
            loss_ce = loss_dict['loss_ce']
            loss_bbox = loss_dict['loss_bbox']
            loss_giou = loss_dict['loss_giou']

            loss = (CFG['weight_class'] * loss_ce +
                    CFG['weight_bbox']  * loss_bbox +
                    CFG['weight_giou']  * loss_giou)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if CFG['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
            optimizer.step()

            # Accumulate
            epoch_losses['total'] += loss.item()
            epoch_losses['ce']    += loss_ce.item()
            epoch_losses['bbox']  += loss_bbox.item()
            epoch_losses['giou']  += loss_giou.item()

            writer.add_scalar('train/lr',        lr,               global_step)
            writer.add_scalar('train/loss',      loss.item(),      global_step)
            writer.add_scalar('train/loss_ce',   loss_ce.item(),   global_step)
            writer.add_scalar('train/loss_bbox', loss_bbox.item(), global_step)
            writer.add_scalar('train/loss_giou', loss_giou.item(), global_step)

            if global_step % 100 == 0:
                print(f"  Step {global_step:6d}/{total_steps:6d} | "
                      f"loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                      f"bbox={loss_bbox.item():.4f} giou={loss_giou.item():.4f} "
                      f"lr={lr:.2e}", flush=True)

            global_step += 1

        # ── End of epoch ──────────────────────────────────────────────────────
        n = steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{CFG['max_epochs']} | "
              f"loss={epoch_losses['total']/n:.4f}  "
              f"ce={epoch_losses['ce']/n:.4f}  "
              f"bbox={epoch_losses['bbox']/n:.4f}  "
              f"giou={epoch_losses['giou']/n:.4f}  "
              f"lr={lr:.2e}  [{elapsed:.0f}s]", flush=True)

        writer.add_scalar('epoch/loss',      epoch_losses['total'] / n, epoch)
        writer.add_scalar('epoch/loss_ce',   epoch_losses['ce'] / n, epoch)
        writer.add_scalar('epoch/loss_bbox', epoch_losses['bbox'] / n, epoch)
        writer.add_scalar('epoch/loss_giou', epoch_losses['giou'] / n, epoch)

        # Validation
        model.eval()
        val_losses = {'total': 0., 'ce': 0., 'bbox': 0., 'giou': 0.}
        val_steps = len(val_loader) if args.limit_val_batches is None else args.limit_val_batches

        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_loader):
                if args.limit_val_batches is not None and val_idx >= args.limit_val_batches:
                    break

                img      = val_batch['img'].to(device, non_blocking=True)
                gt_masks = val_batch['gt_masks'].to(device, non_blocking=True)
                
                B_v, T_v = img.shape[:2]
                img_v_flat = img.flatten(0, 1)
                gt_v_flat = gt_masks.flatten(0, 1)
                
                targets_v = masks_to_boxes_and_labels(gt_v_flat)
                
                outputs_v = model(img_v_flat)
                loss_dict_v = criterion(outputs_v, targets_v)
                
                l_ce = loss_dict_v['loss_ce']
                l_bbox = loss_dict_v['loss_bbox']
                l_giou = loss_dict_v['loss_giou']
                
                val_losses['ce']    += l_ce.item()
                val_losses['bbox']  += l_bbox.item()
                val_losses['giou']  += l_giou.item()
                val_losses['total'] += (CFG['weight_class'] * l_ce +
                                        CFG['weight_bbox']  * l_bbox +
                                        CFG['weight_giou']  * l_giou).item()

        vn = val_steps
        writer.add_scalar('val/loss',      val_losses['total'] / vn, epoch)
        writer.add_scalar('val/loss_ce',   val_losses['ce'] / vn, epoch)
        writer.add_scalar('val/loss_bbox', val_losses['bbox'] / vn, epoch)
        writer.add_scalar('val/loss_giou', val_losses['giou'] / vn, epoch)
        print(f"          Val  | loss={val_losses['total']/vn:.4f}  "
              f"ce={val_losses['ce']/vn:.4f}  "
              f"bbox={val_losses['bbox']/vn:.4f}  "
              f"giou={val_losses['giou']/vn:.4f}", flush=True)

        if (epoch + 1) % CFG['vis_every_n_epochs'] == 0:
            visualize_detr(model, val_ds, CFG['n_vis_samples'], device, writer, epoch)

        if (epoch + 1) % CFG['save_every_n_epochs'] == 0:
            ckpt_path = os.path.join(CFG['ckpt_dir'], f'detr_epoch_{epoch+1}.pt')
            torch.save({
                'epoch':       epoch,
                'global_step': global_step,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'config':      CFG,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    final_path = os.path.join(CFG['ckpt_dir'], 'detr_final.pt')
    torch.save({'epoch': CFG['max_epochs'] - 1, 'global_step': global_step,
                'model': model.state_dict(), 'config': CFG}, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()


if __name__ == '__main__':
    main()
