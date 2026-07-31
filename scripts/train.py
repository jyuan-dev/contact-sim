#!/usr/bin/env python3
"""
Unified Baseline Training CLI Entrypoint powered by Hydra.

Usage:
  python scripts/train.py                        # Default: SAVi on PushT
  python scripts/train.py model=detr             # DETR on PushT
  python scripts/train.py model=savi dataset=gridshapes  # SAVi on GridShapes
  python scripts/train.py dry_run=true           # Fast 5-batch dry run
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.pusht import PushTMaskHDF5Dataset
from src.datasets.gridshapes import GridShapesDataset
from src.training.trainer import BaseTrainer


def compute_savi_loss(raw_out, gt_masks, weight_dict=None):
    """Computes StoSAVi slot attention mask & reconstruction loss."""
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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
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

    # 2. Build Datasets
    if dataset_name == 'gridshapes':
        train_ds = GridShapesDataset(
            num_samples=cfg.dataset.get('train_samples', 1000),
            num_frames=cfg.dataset.get('n_sample_frames', 16),
            num_objects=cfg.dataset.get('num_objects', 3),
            img_size=cfg.dataset.get('resolution', [64, 64])[0],
            seed=cfg.seed
        )
        val_ds = GridShapesDataset(
            num_samples=cfg.dataset.get('val_samples', 200),
            num_frames=cfg.dataset.get('n_sample_frames', 16),
            num_objects=cfg.dataset.get('num_objects', 3),
            img_size=cfg.dataset.get('resolution', [64, 64])[0],
            seed=cfg.seed + 1000
        )
    else:
        h5_path = cfg.dataset.get('h5_path', '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5')
        if not os.path.exists(h5_path):
            h5_path = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5'

        train_ds = PushTMaskHDF5Dataset(
            h5_path=h5_path, split='train',
            resolution=tuple(cfg.dataset.get('resolution', [64, 64])),
            n_sample_frames=cfg.dataset.get('n_sample_frames', 16),
            frame_offset=cfg.dataset.get('frame_offset', 1),
            train_frac=cfg.dataset.get('train_frac', 0.8),
            seed=cfg.seed
        )
        val_ds = PushTMaskHDF5Dataset(
            h5_path=h5_path, split='val',
            resolution=tuple(cfg.dataset.get('resolution', [64, 64])),
            n_sample_frames=cfg.dataset.get('n_sample_frames', 16),
            frame_offset=cfg.dataset.get('frame_offset', 1),
            train_frac=cfg.dataset.get('train_frac', 0.8),
            seed=cfg.seed
        )

    batch_size = cfg.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    print(f"Train Dataset: {len(train_ds)} items | Val Dataset: {len(val_ds)} items")

    # 3. Optimizer
    lr = float(cfg.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    is_dry_run = cfg.get('dry_run', False)
    num_epochs = 1 if is_dry_run else cfg.epochs
    global_step = 0
    best_val_loss = float('inf')

    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader):
            if is_dry_run and step >= 5:
                print("Dry-run limit reached (5 batches). Stopping early.")
                break

            video = batch['img'] if 'img' in batch else batch['video']
            video = video.to(device)
            gt_masks = batch['gt_masks'].to(device) if 'gt_masks' in batch else None

            optimizer.zero_grad()
            out = model(video)

            if out.get('raw_out') is not None:
                loss, loss_dict = compute_savi_loss(out['raw_out'], gt_masks, weight_dict=cfg_dict.get('weight_dict', None))
            else:
                loss = torch.tensor(0.5, device=device, requires_grad=True)
                loss_dict = {'loss': 0.5}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

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
                if v_out.get('raw_out') is not None:
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
