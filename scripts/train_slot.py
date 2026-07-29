"""
Unified Training Script for Slot Attention models (StoSAVi & Slot-PIDM).

Usage:
    # Train StoSAVi (Stage 1):
    python scripts/train_slot.py --config configs/savi/pusht.yaml

    # Train Slot-PIDM (Stage 2):
    python scripts/train_slot.py --config configs/savi/slot_pidm_pusht.yaml --mode slot_pidm
"""

import sys
import os
import argparse
import time
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils

# Add workspace root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.data_utils import get_dataset, find_dataset_path
from src.utils.training_utils import cosine_anneal_with_warmup, set_seed, get_device
from src.training.trainer import BaseTrainer
from src.models.slot_attention import StoSAVi, DETRHungarianMatcher, DETRMaskLoss
from src.models.slot_pidm import SlotPIDMAgent


# ── Slot-PIDM Trajectory Wrapper ──────────────────────────────────────────────
class SlotPIDMTrajectoryDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        img = data['img']            # [T, 3, H, W]
        masks = data['gt_masks']      # [T, M, H, W]

        T = img.shape[0]
        t = np.random.randint(0, T - 1)

        img_t = img[t]
        img_next = img[t + 1]
        mask_t = masks[t]
        mask_next = masks[t + 1]

        # Infer proxy 2D action from agent mask centroid displacement (agent = mask 0)
        agent_t = mask_t[0] > 0.5
        agent_next = mask_next[0] > 0.5

        if agent_t.any() and agent_next.any():
            grid_y, grid_x = torch.nonzero(agent_t, as_tuple=True)
            center_t = torch.stack([grid_x.float().mean(), grid_y.float().mean()])
            grid_y_next, grid_x_next = torch.nonzero(agent_next, as_tuple=True)
            center_next = torch.stack([grid_x_next.float().mean(), grid_y_next.float().mean()])
            action = (center_next - center_t) / img.shape[-1]
        else:
            action = torch.zeros(2)

        return {
            'img_t': img_t,
            'img_next': img_next,
            'mask_t': mask_t,
            'mask_next': mask_next,
            'action': action
        }


# ── Synthetic Dataset Fallback for Dry Runs ───────────────────────────────────
class SyntheticSlotDataset(Dataset):
    def __init__(self, mode='savi', num_samples=32, num_frames=6, res=64, num_masks=3):
        self.mode = mode
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.res = res
        self.num_masks = num_masks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'savi':
            return {
                'img': torch.rand(self.num_frames, 3, self.res, self.res),
                'gt_masks': (torch.rand(self.num_frames, self.num_masks, self.res, self.res) > 0.5).float()
            }
        else:
            return {
                'img_t': torch.rand(3, self.res, self.res),
                'img_next': torch.rand(3, self.res, self.res),
                'mask_t': (torch.rand(self.num_masks, self.res, self.res) > 0.5).float(),
                'mask_next': (torch.rand(self.num_masks, self.res, self.res) > 0.5).float(),
                'action': torch.randn(2)
            }


# ── Train StoSAVi (Stage 1) ───────────────────────────────────────────────────
def train_savi(config, args, device):
    save_dir = config.get('ckpt_dir', 'scratch/checkpoints/savi')
    trainer = BaseTrainer(save_dir=save_dir, experiment_name="StoSAVi-Train")

    dataset_name = config.get('dataset_name', 'pusht')
    h5_path = find_dataset_path(config.get('h5_path', ''))
    resolution = tuple(config.get('resolution', [64, 64]))
    n_sample_frames = config.get('n_sample_frames', 6)
    frame_offset = config.get('frame_offset', 1)
    train_frac = config.get('train_frac', 0.8)

    if os.path.exists(h5_path):
        train_ds = get_dataset(dataset_name, h5_path, 'train', resolution, n_sample_frames, frame_offset, train_frac)
    else:
        print(f"[Train StoSAVi] Dataset path '{h5_path}' not found. Using synthetic dataset for dry run.")
        train_ds = SyntheticSlotDataset(mode='savi', res=resolution[0], num_frames=n_sample_frames)

    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=='cuda'))

    model = StoSAVi(
        resolution=resolution,
        clip_len=n_sample_frames,
        slot_dict=config['slot_dict'],
        enc_dict=config['enc_dict'],
        dec_dict=config['dec_dict'],
        pred_dict=config['pred_dict'],
        loss_dict=config.get('loss_dict', None)
    ).to(device)


    matcher = DETRHungarianMatcher(cost_mask=1.0, cost_dice=1.0)
    criterion = DETRMaskLoss(matcher=matcher, weight_mask=config.get('mask_loss_w', 1.0), weight_dice=1.0).to(device)

    lr = float(config.get('lr', 2e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    max_epochs = args.max_epochs or config.get('max_epochs', 10)
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(total_steps * config.get('warmup_pct', 0.05))

    global_step = 0
    print(f"[Train StoSAVi] Starting training for {max_epochs} epochs ({total_steps} steps)...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            global_step += 1
            curr_lr = cosine_anneal_with_warmup(global_step, total_steps, warmup_steps, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

            imgs = batch['img'].to(device)         # [B, T, 3, H, W]
            gt_masks = batch['gt_masks'].to(device) # [B, T, M, H, W]

            optimizer.zero_grad()
            masks_pred, slots, loss_dict = model(imgs, gt_masks)
            
            # Hungarian matching mask loss across frames
            B, T, K, H, W = masks_pred.shape
            pred_masks_flat = masks_pred.view(B * T, K, H, W)
            gt_masks_flat = gt_masks.view(B * T, -1, H, W)

            loss_detr, mask_iou = criterion(pred_masks_flat, gt_masks_flat)
            recon_loss = loss_dict.get('recon_loss', torch.tensor(0.0, device=device))
            kld_loss = loss_dict.get('kld_loss', torch.tensor(0.0, device=device))

            total_loss = loss_detr + config.get('recon_loss_w', 1.0) * recon_loss + config.get('kld_loss_w', 1e-4) * kld_loss
            total_loss.backward()

            if config.get('clip_grad', 0.0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])

            optimizer.step()
            epoch_loss += total_loss.item()

            if global_step % 10 == 0:
                trainer.log_scalar('train/loss', total_loss.item(), global_step)
                trainer.log_scalar('train/mask_iou', mask_iou.item(), global_step)
                trainer.log_scalar('train/lr', curr_lr, global_step)

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch}/{max_epochs}] Average Loss: {avg_loss:.4f}")

        if epoch % config.get('save_every_n_epochs', 5) == 0 or epoch == max_epochs:
            trainer.save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config
            }, filename=f"savi_epoch_{epoch}.pt")

    trainer.save_checkpoint({'model_state': model.state_dict(), 'config': config}, filename="savi_final.pt")
    trainer.close()


# ── Train Slot-PIDM (Stage 2) ─────────────────────────────────────────────────
def train_slot_pidm(config, args, device):
    tr_cfg = config.get('training', {})
    save_dir = tr_cfg.get('save_dir', 'scratch/checkpoints/slot_pidm_pusht')
    trainer = BaseTrainer(save_dir=save_dir, experiment_name="Slot-PIDM-Train")

    ds_cfg = config.get('dataset', {})
    dataset_name = ds_cfg.get('name', 'pusht')
    h5_path = find_dataset_path(ds_cfg.get('h5_path', ''))
    resolution = tuple(ds_cfg.get('resolution', [64, 64]))
    n_sample_frames = ds_cfg.get('n_sample_frames', 6)
    frame_offset = ds_cfg.get('frame_offset', 1)
    train_frac = ds_cfg.get('train_frac', 0.9)

    if os.path.exists(h5_path):
        base_ds = get_dataset(dataset_name, h5_path, 'train', resolution, n_sample_frames, frame_offset, train_frac)
        train_ds = SlotPIDMTrajectoryDataset(base_ds)
    else:
        print(f"[Train Slot-PIDM] Dataset path '{h5_path}' not found. Using synthetic dataset for dry run.")
        train_ds = SyntheticSlotDataset(mode='slot_pidm', res=resolution[0])

    batch_size = tr_cfg.get('batch_size', 64)
    num_workers = tr_cfg.get('num_workers', 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=='cuda'))

    m_cfg = config.get('model', {})
    agent = SlotPIDMAgent(
        d_model=m_cfg.get('d_model', 128),
        action_dim=m_cfg.get('action_dim', 2),
        k_slots=m_cfg.get('k_slots', 4),
        num_heads=m_cfg.get('num_heads', 4),
        num_iterations=m_cfg.get('num_iterations', 2),
        weight_action_loss=m_cfg.get('weight_action_loss', 1.0),
        weight_slot_loss=m_cfg.get('weight_slot_loss', 1.0),
        weight_sigreg_loss=m_cfg.get('weight_sigreg_loss', 0.01)
    ).to(device)

    if getattr(args, 'savi_ckpt', None) and os.path.exists(args.savi_ckpt):
        print(f"[Train Slot-PIDM] Loading pretrained StoSAVi checkpoint from {args.savi_ckpt}")
        ckpt = torch.load(args.savi_ckpt, map_location=device)
        state_dict = ckpt.get('model_state', ckpt)
        agent.savi.load_state_dict(state_dict, strict=False)

    lr = float(tr_cfg.get('lr', 1e-3))
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=tr_cfg.get('weight_decay', 1e-4))

    max_epochs = args.max_epochs or tr_cfg.get('max_epochs', 20)
    total_steps = max_epochs * len(train_loader)
    warmup_steps = tr_cfg.get('warmup_epochs', 2) * len(train_loader)

    global_step = 0
    print(f"[Train Slot-PIDM] Starting training for {max_epochs} epochs ({total_steps} steps)...")

    for epoch in range(1, max_epochs + 1):
        agent.train()
        epoch_loss = 0.0
        for batch in train_loader:
            global_step += 1
            curr_lr = cosine_anneal_with_warmup(global_step, total_steps, warmup_steps, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

            img_t = batch['img_t'].to(device)
            img_next = batch['img_next'].to(device)
            mask_t = batch['mask_t'].to(device)
            action = batch['action'].to(device)

            optimizer.zero_grad()
            out = agent(img_t=img_t, img_next=img_next, gt_masks=mask_t, action_gt=action)
            slot_loss = out['loss_slot']
            pidm_loss = out['loss_action']
            total_loss = out['loss_total']

            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()


            epoch_loss += total_loss.item()

            if global_step % tr_cfg.get('log_interval', 10) == 0:
                trainer.log_scalar('train/total_loss', total_loss.item(), global_step)
                trainer.log_scalar('train/slot_loss', slot_loss.item(), global_step)
                trainer.log_scalar('train/pidm_loss', pidm_loss.item(), global_step)
                trainer.log_scalar('train/lr', curr_lr, global_step)

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch}/{max_epochs}] Average Loss: {avg_loss:.4f}")

        if epoch % tr_cfg.get('checkpoint_interval', 5) == 0 or epoch == max_epochs:
            trainer.save_checkpoint({
                'epoch': epoch,
                'model_state': agent.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config
            }, filename=f"slot_pidm_epoch_{epoch}.pt")

    trainer.save_checkpoint({'model_state': agent.state_dict(), 'config': config}, filename="slot_pidm_final.pt")
    trainer.close()


# ── Main Entrypoint ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for Slot Attention Models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config file")
    parser.add_argument("--mode", type=str, choices=['auto', 'savi', 'slot_pidm'], default='auto',
                        help="Training mode: 'savi' (Stage 1 StoSAVi) or 'slot_pidm' (Stage 2 Slot-PIDM)")
    parser.add_argument("--savi_ckpt", type=str, default=None, help="Path to Stage 1 StoSAVi checkpoint (for slot_pidm)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max training epochs")
    parser.add_argument("--device", type=str, default=None, help="Specify device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mode = args.mode
    if mode == 'auto':
        if 'slot_pidm' in args.config or 'model' in config and 'd_model' in config['model']:
            mode = 'slot_pidm'
        else:
            mode = 'savi'

    print(f"[Train Slot] Mode: '{mode.upper()}' | Device: {device} | Config: {args.config}")

    if mode == 'savi':
        train_savi(config, args, device)
    elif mode == 'slot_pidm':
        train_slot_pidm(config, args, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
