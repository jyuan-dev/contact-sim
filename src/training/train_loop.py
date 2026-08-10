"""
Training Loop — decoupled from Hydra and CLI concerns.

Provides:
  - ``TrainConfig``     — dataclass holding all training hyperparameters
  - ``run_epoch()``     — single epoch forward + backward pass
  - ``run_training()``  — full training loop with validation, checkpointing, NaN guards

The ``scripts/train.py`` Hydra entrypoint is responsible for:
  - Parsing / resolving the Hydra config
  - Building the model, dataloaders, optimizer, and scaler
  - Constructing a ``TrainConfig`` from the resolved config
  - Calling ``run_training()``

This separation makes the training logic independently testable (with mock
models / loaders) and reusable from non-Hydra scripts.
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.models.base import BaseModelWrapper
from src.training.trainer import BaseTrainer
from src.utils.training_utils import cosine_anneal_with_warmup


# ── Training Configuration ────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    All hyperparameters for a single training run.

    Build from a resolved Hydra config dict via ``TrainConfig.from_cfg(cfg_dict)``.
    All fields have sensible defaults so the dataclass can be constructed
    directly in unit tests without a full config file.
    """

    # Optimisation
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Schedule
    epochs: int = 8
    max_steps: Optional[int] = None

    # Data
    batch_size: int = 128
    num_workers: int = 8

    # Runtime
    seed: int = 42
    dry_run: bool = False
    use_amp: bool = True

    # Checkpoint paths (resolved absolute paths)
    ckpt_dir: str = "scratch/checkpoints/default"
    model_name: str = "model"

    # LR scheduler name (e.g. 'cosine'; null = constant LR)
    scheduler: Optional[str] = None

    # Scheduler hyperparameters
    warmup_steps: int = 1000
    min_lr: float = 1e-5

    @classmethod
    def from_cfg(cls, cfg: dict, ckpt_dir: str, model_name: str) -> "TrainConfig":
        """Construct a ``TrainConfig`` from a fully-resolved config dict."""
        device_is_cuda = torch.cuda.is_available()
        return cls(
            lr=float(cfg.get("lr", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            grad_clip_norm=float(cfg.get("grad_clip_norm", 1.0)),
            epochs=int(cfg.get("epochs", 8)),
            max_steps=cfg.get("max_steps"),
            batch_size=int(cfg.get("batch_size", 128)),
            num_workers=int(cfg.get("num_workers", 8)),
            seed=int(cfg.get("seed", 42)),
            dry_run=bool(cfg.get("dry_run", False)),
            use_amp=bool(cfg.get("use_amp", device_is_cuda)),
            ckpt_dir=ckpt_dir,
            model_name=model_name,
            scheduler=cfg.get("scheduler"),
            warmup_steps=int(cfg.get("warmup_steps", 1000)),
            min_lr=float(cfg.get("min_lr", 1e-5)),
        )


# ── Epoch Runner ──────────────────────────────────────────────────────────────

def run_epoch(
    *,
    model: BaseModelWrapper,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    trainer: BaseTrainer,
    global_step: int,
    cfg: TrainConfig,
    is_train: bool,
    total_target_steps: int,
    start_time: float,
    epoch: int,
    num_epochs: int,
    stop_training_flag: list,   # mutable flag: [False]; set to [True] to stop
) -> tuple[float, int]:
    """
    Run a single training or validation epoch.

    Args:
        model:              The model wrapper.
        loader:             DataLoader for this split.
        optimizer:          Optimizer (``None`` during validation).
        scaler:             AMP GradScaler (``None`` during validation).
        device:             Target device.
        trainer:            ``BaseTrainer`` for TensorBoard logging.
        global_step:        Current global step counter (training only).
        cfg:                ``TrainConfig`` with all hyperparameters.
        is_train:           ``True`` for training, ``False`` for validation.
        total_target_steps: Total steps in this run (for ETA computation).
        start_time:         ``time.time()`` at start of training (for ETA).
        epoch:              0-indexed current epoch number.
        num_epochs:         Total number of epochs.
        stop_training_flag: Single-element list used as a mutable bool flag.

    Returns:
        (avg_loss, new_global_step)
    """
    if is_train:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    sub_losses: dict[str, list[float]] = {}

    ctx = torch.no_grad() if not is_train else _null_context()

    with ctx:
        for step, batch in enumerate(loader):
            # ── Dry-run / max_steps early exit ───────────────────────────
            if is_train:
                if cfg.dry_run and step >= 5:
                    print("Dry-run limit reached (5 batches). Stopping early.")
                    stop_training_flag[0] = True
                    break
                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    print(f"Max steps limit reached ({cfg.max_steps}). Stopping training.")
                    stop_training_flag[0] = True
                    break
            else:
                if cfg.dry_run and step >= 3:
                    break

            # ── Forward pass ─────────────────────────────────────────────
            video = batch["img"].to(device, non_blocking=True)

            try:
                with torch.amp.autocast(device.type, enabled=cfg.use_amp):
                    out = model(video)
                    loss, loss_dict = model.compute_loss(out, batch)
            except ValueError as err:
                if is_train:
                    print("\n" + "!" * 80)
                    print(f"⚠️  [CRITICAL WARNING / NAN AT SOURCE] Step [{global_step + 1}]: {err}")
                    print("   Full traceback follows for diagnosis:")
                    traceback.print_exc()
                    print("   Skipping optimizer update for this batch.")
                    print("!" * 80 + "\n")
                    continue
                raise

            # ── NaN / Inf guard (training only) ──────────────────────────
            if is_train:
                has_nan = torch.isnan(loss) or torch.isinf(loss)
                nan_keys = [k for k, v in loss_dict.items() if math.isnan(v) or math.isinf(v)]
                if has_nan or nan_keys:
                    print("\n" + "!" * 80)
                    print(f"⚠️  [WARNING / NAN DETECTED] Epoch [{epoch + 1}] Step [{global_step + 1}]")
                    print(f"   Total Loss: {loss.item()} | NaN sub-losses: {nan_keys}")
                    print("   Skipping optimizer update for this batch.")
                    print("!" * 80 + "\n")
                    continue

            # ── Backward / optimize ───────────────────────────────────────
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()

                # ── Per-step LR scheduling ────────────────────────────────
                if cfg.scheduler == 'cosine':
                    new_lr = cosine_anneal_with_warmup(
                        global_step, total_target_steps, cfg.warmup_steps,
                        cfg.lr, cfg.min_lr,
                    )
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

            # ── Logging ───────────────────────────────────────────────────
            loss_val = loss.item()
            losses.append(loss_val)

            if is_train:
                split = "Train"
                trainer.log_scalar(f"{split}/Loss", loss_val, global_step)
                for lk, lv in loss_dict.items():
                    if lk != "total_loss":
                        trainer.log_scalar(f"{split}/{lk}", lv, global_step)
                global_step += 1

                if global_step % 50 == 0 or step == 0:
                    elapsed = time.time() - start_time
                    avg_sec = elapsed / max(1, global_step)
                    eta_sec = max(0, total_target_steps - global_step) * avg_sec
                    eta_str = str(timedelta(seconds=int(eta_sec)))
                    speed = (
                        f"{1.0 / avg_sec:.2f} it/s"
                        if avg_sec < 1.0
                        else f"{avg_sec:.2f} s/it"
                    )
                    pct = (global_step / total_target_steps) * 100
                    loss_str = " ".join(
                        f"{k}={v:.4f}"
                        for k, v in loss_dict.items()
                        if k != "total_loss"
                    )
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}] "
                        f"Step [{global_step}/{total_target_steps}] "
                        f"({pct:.1f}% | {speed} | ETA: {eta_str}) "
                        f"Total Loss: {loss_val:.4f} [{loss_str}]"
                    )
            else:
                for vk, vv in loss_dict.items():
                    if vk != "total_loss":
                        sub_losses.setdefault(vk, []).append(vv)

    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, global_step


# ── Full Training Loop ────────────────────────────────────────────────────────

def run_training(
    *,
    model: BaseModelWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    trainer: BaseTrainer,
    save_checkpoint_fn,
) -> None:
    """
    Full training loop: trains for ``cfg.epochs`` epochs (or ``cfg.max_steps``
    steps), validates after each epoch, and checkpoints best + final models.

    Args:
        model:              The model wrapper (already on ``device``).
        train_loader:       DataLoader for training split.
        val_loader:         DataLoader for validation split.
        optimizer:          AdamW or compatible optimizer.
        scaler:             AMP GradScaler (disabled automatically when
                            ``cfg.use_amp`` is ``False``).
        device:             Target device.
        cfg:                ``TrainConfig`` with all hyperparameters.
        trainer:            ``BaseTrainer`` instance for TensorBoard logging.
        save_checkpoint_fn: Callable ``(model, path, epoch)`` that saves the
                            model state dict.  Signature:

                                save_checkpoint_fn(model, cfg_dict, path, epoch)
    """
    # CUDA optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    num_epochs = 1 if cfg.dry_run else (
        1000 if cfg.max_steps is not None else cfg.epochs
    )
    total_target_steps = (
        cfg.max_steps
        if cfg.max_steps is not None
        else num_epochs * len(train_loader)
    )

    global_step = 0
    best_val_loss = float("inf")
    stop_training_flag = [False]
    start_time = time.time()

    for epoch in range(num_epochs):
        if stop_training_flag[0]:
            break

        # ── Training epoch ────────────────────────────────────────────────
        avg_train_loss, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            trainer=trainer,
            global_step=global_step,
            cfg=cfg,
            is_train=True,
            total_target_steps=total_target_steps,
            start_time=start_time,
            epoch=epoch,
            num_epochs=num_epochs,
            stop_training_flag=stop_training_flag,
        )
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Average Train Loss: {avg_train_loss:.4f}"
        )

        # ── Validation epoch ──────────────────────────────────────────────
        avg_val_loss, _ = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            trainer=trainer,
            global_step=global_step,
            cfg=cfg,
            is_train=False,
            total_target_steps=total_target_steps,
            start_time=start_time,
            epoch=epoch,
            num_epochs=num_epochs,
            stop_training_flag=stop_training_flag,
        )
        trainer.log_scalar("Val/Loss", avg_val_loss, epoch)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        import os
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_best.pt")
            save_checkpoint_fn(model, best_path, epoch + 1)

    # Save final checkpoint
    import os
    final_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_final.pt")
    save_checkpoint_fn(model, final_path, num_epochs)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _null_context:
    """No-op context manager (replaces ``contextlib.nullcontext`` for Py 3.6 compat)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
