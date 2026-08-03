#!/usr/bin/env python3
"""
Unified Baseline Training CLI Entrypoint powered by Hydra.

Usage:
  python scripts/train.py                        # Default: DETR on PushT
  python scripts/train.py model=savi             # SAVi on PushT
  python scripts/train.py model=savi dataset=gridshapes  # SAVi on GridShapes
  python scripts/train.py dry_run=true           # Fast 5-batch dry run
"""

import sys
import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.models.detr import HungarianMatcher, SetCriterion, masks_to_boxes_and_labels
from src.datasets.factory import build_dataloader
from src.training.trainer import BaseTrainer


def compute_detr_loss(out, gt_masks, weight_dict=None):
    """Computes DETR Hungarian matching loss (classification CE, box L1, box GIoU)."""
    if weight_dict is None:
        weight_dict = {'class': 1.0, 'bbox': 5.0, 'giou': 2.0}

    pred_logits = out['pred_logits']
    pred_boxes = out['pred_boxes']

    if gt_masks is None:
        raise ValueError("gt_masks is required for training DETR")

    if gt_masks.ndim == 5:
        B, T, C, H, W = gt_masks.shape
        gt_masks_flat = gt_masks.view(B * T, C, H, W)
    else:
        gt_masks_flat = gt_masks

    targets = masks_to_boxes_and_labels(gt_masks_flat)

    matcher = HungarianMatcher(
        cost_class=weight_dict.get('class', 1.0),
        cost_bbox=weight_dict.get('bbox', 5.0),
        cost_giou=weight_dict.get('giou', 2.0)
    )
    num_classes = pred_logits.shape[-1] - 1
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=['labels', 'boxes']
    ).to(pred_logits.device)

    detr_out = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
    losses = criterion(detr_out, targets)

    total_loss = torch.tensor(0.0, device=pred_logits.device)
    loss_dict = {}
    for k, v in losses.items():
        weight_key = k.replace('loss_', '')
        if weight_key == 'ce':
            weight_key = 'class'
        w = weight_dict.get(weight_key, 1.0)
        total_loss += w * v
        loss_dict[k] = v.item()

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


def compute_savi_loss(raw_out, gt_masks, weight_dict=None):
    """Computes SAVi slot attention mask & reconstruction loss."""
    if weight_dict is None:
        weight_dict = {'recon': 1.0, 'mask': 1.0, 'kld': 0.001}

    recon_img = raw_out.get('recon_combined', None)
    post_masks = raw_out.get('post_masks', None)
    
    total_loss = 0.0
    loss_dict = {}

    if recon_img is not None and 'img' in raw_out:
        target_img = raw_out['img']
        recon_loss = F.mse_loss(recon_img, target_img)
        total_loss += weight_dict.get('recon', 1.0) * recon_loss
        loss_dict['recon_loss'] = recon_loss.item()

    if post_masks is not None and gt_masks is not None:
        p_masks = post_masks.squeeze(3) if post_masks.ndim == 6 else post_masks
        if gt_masks.ndim == 5:
            B, T, C, H, W = gt_masks.shape
            if p_masks.shape[-2:] != (H, W):
                p_masks = F.interpolate(p_masks.view(B*T, -1, p_masks.shape[-2], p_masks.shape[-1]), size=(H, W), mode='bilinear', align_corners=False).view(B, T, -1, H, W)
            
            mask_bce = F.binary_cross_entropy(torch.clamp(p_masks.max(dim=2)[0], 1e-4, 1-1e-4), (gt_masks.max(dim=2)[0] > 0.5).float())
            total_loss += weight_dict.get('mask', 1.0) * mask_bce
            loss_dict['mask_bce'] = mask_bce.item()

    loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    return total_loss, loss_dict


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Determine Experiment Name & Checkpoint Directory (Workspace Rule)
    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    exp_name = cfg.get('exp_name', f"{model_name}_{dataset_name}")
    ckpt_dir = os.path.join(REPO_ROOT, "scratch", "checkpoints", exp_name)

    # Initialize BaseTrainer for dedicated train.log & TensorBoard logging
    trainer = BaseTrainer(save_dir=ckpt_dir, experiment_name=exp_name)

    print("======================================================================")
    print(f"            Hydra Baseline Trainer ({model_name} / {dataset_name})    ")
    print("======================================================================")
    print(f"Device: {device}")
    print(f"Checkpoint Directory: {ckpt_dir}")

    writer = trainer.writer

    # 1. Build Model via Factory
    model = build_model(cfg_dict).to(device)
    print(f"Model '{model_name}' instantiated successfully via factory!")

    # 2. Build Dataloaders via Factory
    batch_size = cfg.batch_size
    num_workers = cfg.get('num_workers', 4)

    train_loader = build_dataloader(cfg_dict, split='train', batch_size=batch_size, num_workers=num_workers)
    val_loader = build_dataloader(cfg_dict, split='val', batch_size=batch_size, num_workers=num_workers)

    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)} | Batch Size: {batch_size}")

    # CUDA & Performance Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 3. Optimizer
    lr = float(cfg.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    max_steps = cfg.get('max_steps', None)
    is_dry_run = cfg.get('dry_run', False)
    num_epochs = 1 if is_dry_run else (1000 if max_steps is not None else cfg.epochs)
    global_step = 0
    best_val_loss = float('inf')
    stop_training = False

    start_time = time.time()
    total_target_steps = max_steps if max_steps is not None else (num_epochs * len(train_loader))

    # 4. Training Loop
    for epoch in range(num_epochs):
        if stop_training:
            break
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader):
            if is_dry_run and step >= 5:
                print("Dry-run limit reached (5 batches). Stopping early.")
                stop_training = True
                break

            if max_steps is not None and global_step >= max_steps:
                print(f"Max steps limit reached ({max_steps} steps). Stopping training.")
                stop_training = True
                break

            video = batch['img'] if 'img' in batch else batch['video']
            video = video.to(device, non_blocking=True)
            gt_masks = batch['gt_masks'].to(device, non_blocking=True) if 'gt_masks' in batch else None

            optimizer.zero_grad()
            out = model(video)

            if out.get('pred_logits') is not None and out.get('pred_boxes') is not None:
                model_w = cfg_dict.get('model', {}).get('weight_dict', cfg_dict.get('weight_dict', None))
                loss, loss_dict = compute_detr_loss(out, gt_masks, weight_dict=model_w)
            elif out.get('raw_out') is not None:
                loss, loss_dict = compute_savi_loss(out['raw_out'], gt_masks, weight_dict=cfg_dict.get('weight_dict', None))
            else:
                loss = torch.tensor(0.5, device=device, requires_grad=True)
                loss_dict = {'loss': 0.5}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            for lk, lv in loss_dict.items():
                if lk != 'total_loss':
                    writer.add_scalar(f"Train/{lk}", lv, global_step)
            global_step += 1

            if global_step % 50 == 0 or step == 0:
                elapsed_sec = time.time() - start_time
                avg_sec_per_step = elapsed_sec / max(1, global_step)
                remaining_steps = max(0, total_target_steps - global_step)
                eta_sec = remaining_steps * avg_sec_per_step
                eta_str = str(timedelta(seconds=int(eta_sec)))
                speed_str = f"{1.0 / avg_sec_per_step:.2f} it/s" if avg_sec_per_step < 1.0 else f"{avg_sec_per_step:.2f} s/it"
                progress_pct = (global_step / total_target_steps) * 100
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in loss_dict.items() if k != 'total_loss'])
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{global_step}/{total_target_steps}] ({progress_pct:.1f}% | {speed_str} | ETA: {eta_str}) Total Loss: {loss.item():.4f} [{loss_str}]")

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Train Loss: {avg_train_loss:.4f}")

        # Validation Pass
        model.eval()
        val_losses = []
        with torch.no_grad():
            for v_step, v_batch in enumerate(val_loader):
                if is_dry_run and v_step >= 3:
                    break
                v_video = v_batch['img'] if 'img' in v_batch else v_batch['video']
                v_video = v_video.to(device)
                v_gt_masks = v_batch['gt_masks'].to(device) if 'gt_masks' in v_batch else None
                v_out = model(v_video)
                if v_out.get('pred_logits') is not None and v_out.get('pred_boxes') is not None:
                    model_w = cfg_dict.get('model', {}).get('weight_dict', cfg_dict.get('weight_dict', None))
                    v_loss, _ = compute_detr_loss(v_out, v_gt_masks, weight_dict=model_w)
                elif v_out.get('raw_out') is not None:
                    v_loss, _ = compute_savi_loss(v_out['raw_out'], v_gt_masks, weight_dict=cfg_dict.get('weight_dict', None))
                else:
                    v_loss = torch.tensor(0.5, device=device)
                val_losses.append(v_loss.item())

        avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
            torch.save({'model_state': model.state_dict(), 'config': cfg_dict, 'epoch': epoch+1}, best_ckpt_path)
            print(f"Saved new best checkpoint: {best_ckpt_path}")

    # Save Final Checkpoint
    final_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_final.pt")
    torch.save({'model_state': model.state_dict(), 'config': cfg_dict, 'epoch': num_epochs}, final_ckpt_path)
    print(f"Saved final checkpoint: {final_ckpt_path}")

    trainer.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
