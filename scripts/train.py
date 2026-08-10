#!/usr/bin/env python3
"""
Unified Baseline Training CLI Entrypoint powered by Hydra.

This script is a thin entrypoint: it resolves the Hydra config, builds the
model / dataloaders / optimizer, then delegates the full training loop to
``src.training.train_loop.run_training``.

Usage
-----
  python scripts/train.py                              # DETR on PushT (default)
  python scripts/train.py model=savi                  # SAVi on PushT
  python scripts/train.py model=savi dataset=gridshapes
  python scripts/train.py dry_run=true                # Fast 5-batch smoke test
  python scripts/train.py model=deformable_savi ckpt_path=scratch/checkpoints/.../model_best.pt
"""

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.training.trainer import BaseTrainer
from src.training.train_loop import TrainConfig, run_training
from src.utils.training_utils import get_device, set_seed, load_checkpoint_state

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _save_checkpoint(model, path: str, epoch: int) -> None:
    """Save model state dict + epoch to ``path``."""
    torch.save({"model_state": model.state_dict(), "epoch": epoch}, path)
    print(f"Saved checkpoint: {path}")


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    device = get_device(cfg.get("device"))
    set_seed(int(cfg.get("seed", 42)))

    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    exp_name = cfg.get("exp_name", f"{model_name}_{dataset_name}")
    ckpt_dir = os.path.join(REPO_ROOT, "scratch", "checkpoints", exp_name)

    # ── Set up trainer (TensorBoard + file logging) ───────────────────────
    trainer = BaseTrainer(save_dir=ckpt_dir, experiment_name=exp_name)

    # Save human-readable config snapshot for experiment tracking
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg_save_path = os.path.join(ckpt_dir, "config.yaml")
    with open(cfg_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved experiment configuration to: {cfg_save_path}")

    print("=" * 70)
    print(f"            Hydra Baseline Trainer ({model_name} / {dataset_name})")
    print("=" * 70)
    print(f"Device:               {device}")
    print(f"Checkpoint Directory: {ckpt_dir}")

    # ── Build model ───────────────────────────────────────────────────────
    if cfg.get("ckpt_path"):
        ckpt_path = cfg.ckpt_path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
        model = build_model(cfg_dict).to(device)
        load_checkpoint_state(model, ckpt_path, device=device)
        print(f"Loaded weights from: {ckpt_path}")
    else:
        model = build_model(cfg_dict).to(device)

    match_mode = (
        getattr(model, "_weight_dict", {}) or {}
    ).get("match_mode", "hungarian")
    print(f"Slot Matching Mode: '{match_mode.upper()}'")
    print(f"Model '{model_name}' ready (via registry).")

    # ── Build dataloaders ─────────────────────────────────────────────────
    batch_size = cfg.batch_size
    num_workers = cfg.get("num_workers", 4)
    train_loader = build_dataloader(cfg_dict, split="train", batch_size=batch_size, num_workers=num_workers)
    val_loader = build_dataloader(cfg_dict, split="val", batch_size=batch_size, num_workers=num_workers)
    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)} | Batch Size: {batch_size}")

    # ── Build optimizer + scaler ──────────────────────────────────────────
    train_cfg = TrainConfig.from_cfg(cfg_dict, ckpt_dir=ckpt_dir, model_name=model_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=train_cfg.use_amp)
    print(f"AMP: {'enabled (FP16)' if train_cfg.use_amp else 'disabled (FP32)'}")

    # ── Delegate to run_training ──────────────────────────────────────────
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        cfg=train_cfg,
        trainer=trainer,
        save_checkpoint_fn=_save_checkpoint,
    )

    trainer.close()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
