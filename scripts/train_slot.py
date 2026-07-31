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
from src.models.slot_attention import StoSAVi, DETRHungarianMatcher, DETRMaskLoss, build_savi_model

from src.models.slot_pidm import SlotPIDMAgent
from src.losses.sigreg import SIGRegLoss
from src.losses.contrastive import TemporalSlotContrastiveLoss
from src.metrics.eval_metrics import (
    compute_psnr, compute_ssim, compute_fg_ari, compute_latent_std, compute_sigreg_stat
)


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


# ── Early Stopping Helper ──────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            best_val = -self.best_score if self.mode == 'min' else self.best_score
            print(f"[EarlyStopping] Patience counter: {self.counter}/{self.patience} (Best val loss: {best_val:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


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
        val_ds = get_dataset(dataset_name, h5_path, 'val', resolution, n_sample_frames, frame_offset, train_frac)
    else:
        print(f"[Train StoSAVi] Dataset path '{h5_path}' not found. Using synthetic dataset for dry run.")
        train_ds = SyntheticSlotDataset(mode='savi', res=resolution[0], num_frames=n_sample_frames)
        val_ds = SyntheticSlotDataset(mode='savi', res=resolution[0], num_frames=n_sample_frames)

    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

    model = build_savi_model(
        resolution=resolution,
        clip_len=n_sample_frames,
        slot_dict=config['slot_dict'],
        enc_dict=config['enc_dict'],
        dec_dict=config['dec_dict'],
        pred_dict=config['pred_dict'],
        loss_dict=config.get('loss_dict', None)
    ).to(device)


    print(f"[Hardware Verification] StoSAVi model parameters allocated on: {next(model.parameters()).device} ({torch.cuda.get_device_name(device)})", flush=True)

    ckpt_path = getattr(args, 'ckpt_path', None)
    if not ckpt_path and getattr(args, 'resume', False):
        for fname in ['savi_best.pt', 'savi_latest.pt', 'savi_final.pt']:
            candidate = os.path.join(save_dir, fname)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"[Resume Weights] Successfully loaded StoSAVi model weights from: '{ckpt_path}'", flush=True)



    matcher = DETRHungarianMatcher(cost_bce=1.0, cost_dice=1.0)
    criterion = DETRMaskLoss(weight_bce=config.get('mask_loss_w', 1.0), weight_dice=1.0).to(device)
    sigreg_criterion = SIGRegLoss().to(device)
    contrast_criterion = TemporalSlotContrastiveLoss().to(device)

    ablation_mode = getattr(args, 'ablation', 'none')
    use_mask_loss = config.get('loss_dict', {}).get('use_mask_loss', True)
    if ablation_mode != 'none':
        use_mask_loss = False  # Ablation runs are fully self-supervised

    sigreg_w = config.get('sigreg_loss_w', 0.1) if ablation_mode in ['sigreg', 'full'] or (ablation_mode == 'none' and not use_mask_loss) else 0.0
    contrast_w = config.get('contrast_loss_w', 0.05) if ablation_mode == 'full' else 0.0

    print(f"[Self-Supervised Setup] Batch Size: {batch_size} | Mask GT Loss: {use_mask_loss} | SIGReg Weight: {sigreg_w} | Contrast Weight: {contrast_w}", flush=True)

    lr = float(config.get('lr', 2e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    max_epochs = args.max_epochs or config.get('max_epochs', 10)
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(total_steps * config.get('warmup_pct', 0.05))

    es_cfg = config.get('early_stopping', {})
    es_enabled = es_cfg.get('enabled', True)
    es_patience = es_cfg.get('patience', 3)
    es_min_delta = es_cfg.get('min_delta', 1e-4)
    val_every_n_steps = es_cfg.get('val_every_n_steps', 50)
    max_val_batches = es_cfg.get('max_val_batches', 50)
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta, mode='min') if es_enabled else None


    @torch.no_grad()
    def evaluate_val():
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_latent_std_sum = 0.0
        val_sigreg_sum = 0.0
        val_fg_ari_sum = 0.0
        ari_batches = 0
        n_batches = 0

        for i, val_batch in enumerate(val_loader):
            if i >= max_val_batches:
                break
            imgs = val_batch['img'].to(device)
            out_dict = model({'img': imgs})
            masks_pred = out_dict['post_masks']
            loss_dict = model.calc_train_loss({'img': imgs}, out_dict)
            
            recon_loss = loss_dict.get('post_recon_loss', torch.tensor(0.0, device=device))
            kld_loss = loss_dict.get('kld_loss', torch.tensor(0.0, device=device))
            sigreg_loss = sigreg_criterion(out_dict['post_slots']) if sigreg_w > 0 else torch.tensor(0.0, device=device)
            contrast_loss = contrast_criterion(out_dict['post_slots']) if contrast_w > 0 else torch.tensor(0.0, device=device)

            if use_mask_loss and 'gt_masks' in val_batch:
                gt_masks = val_batch['gt_masks'].to(device)
                B, T = imgs.shape[0], imgs.shape[1]
                if masks_pred.ndim == 6:
                    K, C_m, H, W = masks_pred.shape[2:]
                    pred_masks_5d = masks_pred.view(B * T, K, C_m, H, W)
                else:
                    K, H, W = masks_pred.shape[2:]
                    pred_masks_5d = masks_pred.view(B * T, K, 1, H, W)
                pred_masks_4d = pred_masks_5d[:, :, 0, :, :]
                gt_masks_flat = gt_masks.view(B * T, -1, H, W)
                indices = matcher(pred_masks_4d, gt_masks_flat)
                loss_detr, bce_l, dice_l = criterion(pred_masks_5d, gt_masks_flat, indices)
                mask_iou = 1.0 - dice_l
            else:
                loss_detr = torch.tensor(0.0, device=device)
                mask_iou = torch.tensor(0.0, device=device)

            total_loss = loss_detr + config.get('recon_loss_w', 1.0) * recon_loss + config.get('kld_loss_w', 1e-4) * kld_loss + sigreg_w * sigreg_loss + contrast_w * contrast_loss

            # Quantitative evaluation metrics (see src.metrics.eval_metrics for docstrings)

            recon_img = out_dict['post_recon_combined']
            psnr = compute_psnr(recon_img, imgs)
            ssim = compute_ssim(recon_img, imgs)
            latent_std = compute_latent_std(out_dict['post_slots'])
            sigreg_stat = compute_sigreg_stat(out_dict['post_slots'])


            if i == 0:
                # Random sample selection from first validation batch
                sample_idx = torch.randint(0, imgs.shape[0], (1,)).item()
                gt_sample = imgs[sample_idx, 0].clamp(0.0, 1.0)
                recon_sample = recon_img[sample_idx, 0].clamp(0.0, 1.0)
                m_sample = masks_pred[sample_idx, 0]
                if m_sample.ndim == 4:
                    m_sample = m_sample.squeeze(1) # [K, H, W]
                m_sample_rgb = m_sample.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0.0, 1.0) # [K, 3, H, W]
                vis_list = [gt_sample, recon_sample] + [m_sample_rgb[k] for k in range(m_sample_rgb.shape[0])]
                vis_grid = torch.cat(vis_list, dim=2) # Concat along width (W)

            if i < 10 and 'gt_masks' in val_batch:
                fg_ari = compute_fg_ari(masks_pred, val_batch['gt_masks'])
                val_fg_ari_sum += fg_ari
                ari_batches += 1

            val_loss_sum += total_loss.item()
            val_iou_sum += mask_iou.item()
            val_psnr_sum += psnr
            val_ssim_sum += ssim
            val_latent_std_sum += latent_std
            val_sigreg_sum += sigreg_stat
            n_batches += 1

        model.train()
        return {
            'loss': val_loss_sum / max(1, n_batches),
            'iou': val_iou_sum / max(1, n_batches),
            'psnr': val_psnr_sum / max(1, n_batches),
            'ssim': val_ssim_sum / max(1, n_batches),
            'latent_std': val_latent_std_sum / max(1, n_batches),
            'sigreg_stat': val_sigreg_sum / max(1, n_batches),
            'fg_ari': val_fg_ari_sum / max(1, ari_batches),
            'vis_grid': vis_grid if 'vis_grid' in locals() else None
        }


    global_step = 0
    should_stop = False
    print(f"[Train StoSAVi] Starting training for {max_epochs} epochs ({total_steps} steps, batch_size={batch_size}, val every {val_every_n_steps} steps)...", flush=True)


    for epoch in range(1, max_epochs + 1):
        if should_stop:
            break
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            global_step += 1
            curr_lr = cosine_anneal_with_warmup(global_step, total_steps, warmup_steps, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

            imgs = batch['img'].to(device)

            optimizer.zero_grad()
            out_dict = model({'img': imgs})
            masks_pred = out_dict['post_masks']
            loss_dict = model.calc_train_loss({'img': imgs}, out_dict)

            recon_loss = loss_dict.get('post_recon_loss', torch.tensor(0.0, device=device))
            kld_loss = loss_dict.get('kld_loss', torch.tensor(0.0, device=device))
            sigreg_loss = sigreg_criterion(out_dict['post_slots']) if sigreg_w > 0 else torch.tensor(0.0, device=device)
            contrast_loss = contrast_criterion(out_dict['post_slots']) if contrast_w > 0 else torch.tensor(0.0, device=device)

            if use_mask_loss and 'gt_masks' in batch:
                gt_masks = batch['gt_masks'].to(device)
                B, T = imgs.shape[0], imgs.shape[1]
                if masks_pred.ndim == 6:
                    K, C_m, H, W = masks_pred.shape[2:]
                    pred_masks_5d = masks_pred.view(B * T, K, C_m, H, W)
                else:
                    K, H, W = masks_pred.shape[2:]
                    pred_masks_5d = masks_pred.view(B * T, K, 1, H, W)
                pred_masks_4d = pred_masks_5d[:, :, 0, :, :]
                gt_masks_flat = gt_masks.view(B * T, -1, H, W)
                indices = matcher(pred_masks_4d, gt_masks_flat)
                loss_detr, bce_l, dice_l = criterion(pred_masks_5d, gt_masks_flat, indices)
                mask_iou = 1.0 - dice_l
            else:
                loss_detr = torch.tensor(0.0, device=device)
                mask_iou = torch.tensor(0.0, device=device)

            total_loss = loss_detr + config.get('recon_loss_w', 1.0) * recon_loss + config.get('kld_loss_w', 1e-4) * kld_loss + sigreg_w * sigreg_loss + contrast_w * contrast_loss
            total_loss.backward()

            if config.get('clip_grad', 0.0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])

            optimizer.step()
            epoch_loss += total_loss.item()




            if global_step % 10 == 0:
                train_recon = recon_loss.item()
                train_sigreg = sigreg_loss.item()
                train_std = compute_latent_std(out_dict['post_slots'])
                trainer.log_scalar('train/loss', total_loss.item(), global_step)
                trainer.log_scalar('train/recon_loss', train_recon, global_step)
                trainer.log_scalar('train/sigreg_loss', train_sigreg, global_step)
                trainer.log_scalar('train/latent_std', train_std, global_step)
                trainer.log_scalar('train/lr', curr_lr, global_step)
                print(f"[Step {global_step}/{total_steps}] Train Loss: {total_loss.item():.4f} | Recon: {train_recon:.4f} | SIGReg: {train_sigreg:.4f} | Latent Std: {train_std:.4f} | LR: {curr_lr:.6f}", flush=True)


            if global_step % val_every_n_steps == 0 or global_step == total_steps:
                val_res = evaluate_val()
                val_loss = val_res['loss']
                trainer.log_scalar('val/loss', val_loss, global_step)
                trainer.log_scalar('val/psnr', val_res['psnr'], global_step)
                trainer.log_scalar('val/ssim', val_res['ssim'], global_step)
                trainer.log_scalar('val/fg_ari', val_res['fg_ari'], global_step)
                trainer.log_scalar('val/latent_std', val_res['latent_std'], global_step)
                trainer.log_scalar('val/sigreg_stat', val_res['sigreg_stat'], global_step)

                if val_res.get('vis_grid') is not None:
                    trainer.log_image('val/reconstruction_visualization', val_res['vis_grid'], global_step)

                print(f"[Step {global_step}/{total_steps}] Val Loss: {val_loss:.4f} | PSNR: {val_res['psnr']:.2f}dB | SSIM: {val_res['ssim']:.4f} | FG-ARI: {val_res['fg_ari']*100:.1f}% | Latent Std: {val_res['latent_std']:.4f}", flush=True)



                if early_stopper:
                    improved = early_stopper(val_loss)
                    if improved:
                        trainer.save_checkpoint({
                            'epoch': epoch,
                            'step': global_step,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'config': config
                        }, filename="savi_best.pt")
                        print(f" -> Saved new best checkpoint 'savi_best.pt' (val_loss: {val_loss:.4f})", flush=True)
                    if early_stopper.early_stop:
                        print(f"[EarlyStopping] Early stopping triggered at step {global_step}! Saving final checkpoint.", flush=True)
                        trainer.save_checkpoint({
                            'epoch': epoch,
                            'step': global_step,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'config': config
                        }, filename="savi_early_stopped.pt")
                        should_stop = True
                        break

            limit_batches = getattr(args, 'limit_train_batches', None)
            if limit_batches and global_step >= limit_batches:
                print(f"[LimitBatches] Reached limit of {limit_batches} train batches.", flush=True)
                should_stop = True
                break


        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch}/{max_epochs}] Average Loss: {avg_loss:.4f}", flush=True)

        # Overwrite savi_latest.pt at the end of each epoch (no individual per-epoch files)
        trainer.save_checkpoint({
            'epoch': epoch,
            'step': global_step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'config': config
        }, filename="savi_latest.pt")

    trainer.save_checkpoint({
        'epoch': epoch if 'epoch' in locals() else max_epochs,
        'step': global_step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': config
    }, filename="savi_final.pt")
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
    parser.add_argument("--limit_train_batches", type=int, default=None, help="Limit number of train batches")
    parser.add_argument("--ablation", type=str, choices=['none', 'baseline', 'sigreg', 'full'], default='none', help="Ablation study variant")
    parser.add_argument("--encoder_type", type=str, choices=['cnn', 'tinyvit'], default=None,
                        help="Encoder architecture: 'cnn' (standard 4-layer Conv2D) or 'tinyvit' (ImageNet-pretrained TinyViT-5M)")

    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint in ckpt_dir if available")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Explicit path to checkpoint file to load weights from")
    parser.add_argument("--device", type=str, default=None, help="Specify device (cuda/cpu)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.encoder_type is not None:
        config.setdefault('enc_dict', {})['encoder_type'] = args.encoder_type


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
