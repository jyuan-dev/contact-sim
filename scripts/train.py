#!/usr/bin/env python3
"""
Unified Baseline Training CLI Entrypoint powered by Hydra.

Usage:
  python scripts/train.py                        # Default: DETR on PushT
  python scripts/train.py model=savi             # SAVi on PushT
  python scripts/train.py model=savi dataset=gridshapes  # SAVi on GridShapes
  python scripts/train.py dry_run=true           # Fast 5-batch dry run
"""

import os
import time
from datetime import timedelta
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.training.trainer import BaseTrainer
from src.utils.training_utils import get_device

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _save_checkpoint(model, cfg_dict, ckpt_path, epoch):
    torch.save({'model_state': model.state_dict(), 'config': cfg_dict, 'epoch': epoch}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    device = get_device()

    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    exp_name = cfg.get('exp_name', f"{model_name}_{dataset_name}")
    ckpt_dir = os.path.join(REPO_ROOT, "scratch", "checkpoints", exp_name)

    trainer = BaseTrainer(save_dir=ckpt_dir, experiment_name=exp_name)

    print("=" * 70)
    print(f"            Hydra Baseline Trainer ({model_name} / {dataset_name})    ")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint Directory: {ckpt_dir}")

    # 1. Build Model via Factory (wraps criterion/loss internally)
    model = build_model(cfg_dict).to(device)
    match_mode = getattr(model, '_weight_dict', {}).get('match_mode', 'hungarian') if hasattr(model, '_weight_dict') and model._weight_dict else 'hungarian'
    print(f"Slot Matching Supervision Mode: '{match_mode.upper()}' (Option B fixed 1-to-1 matching: {match_mode == 'fixed'})")
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
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = cfg.get('max_steps')
    is_dry_run = cfg.get('dry_run', False)
    num_epochs = 1 if is_dry_run else (1000 if max_steps is not None else cfg.epochs)
    global_step = 0
    best_val_loss = float('inf')
    stop_training = False

    start_time = time.time()
    total_target_steps = max_steps if max_steps is not None else (num_epochs * len(train_loader))

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def _step(batch):
        """Single forward + loss step. Returns (loss, loss_dict) on device."""
        video = batch['img']
        video = video.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(video)
            loss, loss_dict = model.compute_loss(out, batch)
        return loss, loss_dict

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

            optimizer.zero_grad()
            loss, loss_dict = _step(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            train_losses.append(loss_val)
            trainer.log_scalar("Train/Loss", loss_val, global_step)
            for lk, lv in loss_dict.items():
                if lk != 'total_loss':
                    trainer.log_scalar(f"Train/{lk}", lv, global_step)
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
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{global_step}/{total_target_steps}] ({progress_pct:.1f}% | {speed_str} | ETA: {eta_str}) Total Loss: {loss_val:.4f} [{loss_str}]")

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Train Loss: {avg_train_loss:.4f}")

        # Validation Pass
        model.eval()
        val_losses = []
        with torch.no_grad():
            for v_step, v_batch in enumerate(val_loader):
                if is_dry_run and v_step >= 3:
                    break
                v_loss, _ = _step(v_batch)
                val_losses.append(v_loss.item())

        avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
        trainer.log_scalar("Val/Loss", avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            _save_checkpoint(model, cfg_dict,
                             os.path.join(ckpt_dir, f"{model_name}_best.pt"), epoch + 1)

    # Save Final Checkpoint
    _save_checkpoint(model, cfg_dict,
                     os.path.join(ckpt_dir, f"{model_name}_final.pt"), num_epochs)

    trainer.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
