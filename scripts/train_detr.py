"""
Train an authentic DETR model on PushT or other datasets for object detection.

Optimized via Hungarian bipartite matching using L1 box loss, GIoU loss, and Cross Entropy classification.
"""

import sys
import os
import argparse
import time
import math
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import cv2

# Add workspace root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.detr import (
    DETR,
    ResNetBackbone,
    Transformer,
    HungarianMatcher,
    SetCriterion,
    box_cxcywh_to_xyxy,
    masks_to_boxes_and_labels
)

# ── Dynamic Dataset Loader ───────────────────────────────────────────────────
def get_dataset(dataset_name, h5_path, split, resolution, n_sample_frames, frame_offset, train_frac):
    if dataset_name == 'pusht':
        from src.datasets.pusht import PushTMaskHDF5Dataset
        return PushTMaskHDF5Dataset(
            h5_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            train_frac=train_frac
        )
    elif dataset_name == 'ogbench':
        from src.datasets.ogbench import OGBenchCubeDataset
        return OGBenchCubeDataset(
            data_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames
        )
    elif dataset_name == 'libero':
        from src.datasets.libero import LiberoDataset
        return LiberoDataset(
            data_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Box Processing & Visualization Helpers ────────────────────────────────────
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
def visualize_detr(model, val_ds, n_samples, device, writer, epoch, cfg):
    model.eval()
    grids = []
    
    # We check if dataset is a placeholder stub
    if not hasattr(val_ds, '_episode_indices'):
        print("[visualize_detr] Skipping visualization: dataset does not support episode indexing (placeholder stub).")
        return

    for i in range(min(n_samples, len(val_ds._episode_indices))):
        ep_idx = val_ds._episode_indices[i]
        data   = val_ds.get_video(ep_idx)
        video  = data['video'].float().to(device)
        gt_m   = data['gt_masks'].float().to(device)
        T      = min(video.shape[0], cfg['n_sample_frames'])
        
        clip   = video[:T]
        gt_clip= gt_m[:T]
        
        outputs = model(clip)
        pred_logits = outputs['pred_logits']
        pred_boxes  = outputs['pred_boxes']
        
        # Colors: Class 0: Blue, Class 1: Orange, Class 2: Green
        colors = [(255, 0, 0), (0, 165, 255), (0, 255, 0)]
        rows = []
        
        for t in range(T):
            img_tensor = clip[t]
            img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8).copy()
            
            # Ground truth
            gt_img = img_np.copy()
            gt_targets = masks_to_boxes_and_labels(gt_clip[t].unsqueeze(0))[0]
            for box, label in zip(gt_targets['boxes'], gt_targets['labels']):
                box_xyxy = box_cxcywh_to_xyxy(box)
                gt_img = draw_box(gt_img, box_xyxy, colors[label.item()], thickness=1)
                
            # Predictions
            pred_img = img_np.copy()
            probs = pred_logits[t].softmax(-1)
            scores, labels = probs[:, :-1].max(-1)
            
            for score, label, box in zip(scores, labels, pred_boxes[t]):
                if score > 0.4:
                    box_xyxy = box_cxcywh_to_xyxy(box)
                    pred_img = draw_box(pred_img, box_xyxy, colors[label.item()], thickness=1)
                    
            gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0
            pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).float() / 255.0
            orig_tensor = (img_tensor * 0.5 + 0.5).clamp(0, 1).cpu()
            
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
    parser.add_argument('--dataset', type=str, default='pusht', choices=['pusht', 'ogbench', 'libero'],
                        help='Dataset name to load config and run training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--limit_train_batches', type=int, default=None,
                        help='Limit train batches per epoch (for verification)')
    parser.add_argument('--limit_val_batches', type=int, default=None,
                        help='Limit validation batches per epoch (for verification)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override maximum training epochs')
    args = parser.parse_args()

    # Load Config from YAML file
    config_path = os.path.join(REPO_ROOT, 'configs', 'detr', f'{args.dataset}.yaml')
    print(f"Loading DETR configuration from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.max_epochs is not None:
        cfg['max_epochs'] = args.max_epochs

    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    os.makedirs(cfg['tb_dir'],   exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Building datasets for {args.dataset} …")
    train_ds = get_dataset(
        dataset_name=args.dataset,
        h5_path=cfg['h5_path'],
        split='train',
        resolution=tuple(cfg['resolution']),
        n_sample_frames=cfg['n_sample_frames'],
        frame_offset=cfg['frame_offset'],
        train_frac=cfg['train_frac']
    )
    val_ds = get_dataset(
        dataset_name=args.dataset,
        h5_path=cfg['h5_path'],
        split='val',
        resolution=tuple(cfg['resolution']),
        n_sample_frames=cfg['n_sample_frames'],
        frame_offset=cfg['frame_offset'],
        train_frac=cfg['train_frac']
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'],
                              pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building DETR model …")
    backbone = ResNetBackbone(train_backbone=True)
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ── Matcher & Loss ────────────────────────────────────────────────────────
    matcher = HungarianMatcher(
        cost_class=cfg['weight_class'],
        cost_bbox=cfg['weight_bbox'],
        cost_giou=cfg['weight_giou']
    )
    
    weight_dict = {
        'loss_ce': cfg['weight_class'],
        'loss_bbox': cfg['weight_bbox'],
        'loss_giou': cfg['weight_giou']
    }
    
    criterion = SetCriterion(
        num_classes=cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg['eos_coef'],
        losses=['labels', 'boxes']
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=1e-4)
    steps_per_epoch = len(train_loader) if args.limit_train_batches is None else args.limit_train_batches
    total_steps  = cfg['max_epochs'] * steps_per_epoch
    warmup_steps = int(cfg['warmup_pct'] * total_steps)

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
    writer = SummaryWriter(log_dir=cfg['tb_dir'])

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"Starting training: {cfg['max_epochs']} epochs, {steps_per_epoch} steps/epoch")
    model.train()
    
    for epoch in range(start_epoch, cfg['max_epochs']):
        t0 = time.time()
        epoch_losses = {'total': 0., 'ce': 0., 'bbox': 0., 'giou': 0.}
        
        for step, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and step >= args.limit_train_batches:
                break
                
            # Cosine LR warmup/decay
            lr = cosine_anneal_with_warmup(global_step, total_steps, warmup_steps, float(cfg['lr']), 1e-6)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            img      = batch['img'].to(device, non_blocking=True)
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)

            # Flatten time and batch dimensions for DETR box prediction
            B, T, C, H, W = img.shape
            img_flat = img.flatten(0, 1)        # [B * T, C, H, W]
            gt_flat  = gt_masks.flatten(0, 1)   # [B * T, 3, H, W]

            # Parse bounding boxes from target masks
            targets = masks_to_boxes_and_labels(gt_flat)

            # Forward
            outputs = model(img_flat)
            loss_dict = criterion(outputs, targets)

            loss_ce   = loss_dict['loss_ce']
            loss_bbox = loss_dict['loss_bbox']
            loss_giou = loss_dict['loss_giou']

            loss = cfg['weight_class'] * loss_ce + cfg['weight_bbox'] * loss_bbox + cfg['weight_giou'] * loss_giou

            # Backward & Optimize
            optimizer.zero_grad()
            loss.backward()
            if cfg['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            optimizer.step()

            # Record
            epoch_losses['total'] += loss.item()
            epoch_losses['ce']    += loss_ce.item()
            epoch_losses['bbox']  += loss_bbox.item()
            epoch_losses['giou']  += loss_giou.item()

            writer.add_scalar('train/loss',      loss.item(),      global_step)
            writer.add_scalar('train/loss_ce',   loss_ce.item(),   global_step)
            writer.add_scalar('train/loss_bbox', loss_bbox.item(), global_step)
            writer.add_scalar('train/loss_giou', loss_giou.item(), global_step)
            writer.add_scalar('train/lr',        lr,               global_step)

            if global_step % 100 == 0:
                print(f"Epoch {epoch+1:2d} | Step {step:4d}/{steps_per_epoch} | "
                      f"loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                      f"bbox={loss_bbox.item():.4f} giou={loss_giou.item():.4f} "
                      f"lr={lr:.2e}", flush=True)

            global_step += 1

        # ── End of epoch ──────────────────────────────────────────────────────
        n = steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{cfg['max_epochs']} | "
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
                val_losses['total'] += (cfg['weight_class'] * l_ce +
                                        cfg['weight_bbox']  * l_bbox +
                                        cfg['weight_giou']  * l_giou).item()

        vn = val_steps
        writer.add_scalar('val/loss',      val_losses['total'] / vn, epoch)
        writer.add_scalar('val/loss_ce',   val_losses['ce'] / vn, epoch)
        writer.add_scalar('val/loss_bbox', val_losses['bbox'] / vn, epoch)
        writer.add_scalar('val/loss_giou', val_losses['giou'] / vn, epoch)
        print(f"          Val  | loss={val_losses['total']/vn:.4f}  "
              f"ce={val_losses['ce']/vn:.4f}  "
              f"bbox={val_losses['bbox']/vn:.4f}  "
              f"giou={val_losses['giou']/vn:.4f}", flush=True)

        if (epoch + 1) % cfg['vis_every_n_epochs'] == 0:
            visualize_detr(model, val_ds, cfg['n_vis_samples'], device, writer, epoch, cfg)

        if (epoch + 1) % cfg['save_every_n_epochs'] == 0:
            ckpt_path = os.path.join(cfg['ckpt_dir'], f'detr_epoch_{epoch+1}.pt')
            torch.save({
                'epoch':       epoch,
                'global_step': global_step,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'config':      cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    final_path = os.path.join(cfg['ckpt_dir'], 'detr_final.pt')
    torch.save({'epoch': cfg['max_epochs'] - 1, 'global_step': global_step,
                'model': model.state_dict(), 'config': cfg}, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()


if __name__ == '__main__':
    main()
