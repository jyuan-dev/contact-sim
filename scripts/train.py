#!/usr/bin/env python3
"""
Hydra Modular Training Entrypoint.

Usage:
    python scripts/train.py                           # Default: Deformable SAVi on PushT
    python scripts/train.py model=savi loss=savi_sigreg  # Standard SAVi with SIGReg loss
    python scripts/train.py experiment=slotformer_pusht  # Stage 2 SlotFormer training
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.training import BaseTrainer, TrainConfig, run_training
from src.utils.training_utils import set_seed, load_checkpoint_state, save_checkpoint


def _auto_dvc_commit(best_path: str) -> None:
    """Track best checkpoint with DVC and execute dvc commit at the end of training."""
    if not os.path.isfile(best_path):
        return

    import subprocess
    dvc_bin = os.path.join(os.path.dirname(sys.executable), "dvc")
    add_cmd = [dvc_bin, "add", best_path] if os.path.exists(dvc_bin) else [sys.executable, "-m", "dvc", "add", best_path]
    dvc_file = f"{best_path}.dvc"
    commit_cmd = [dvc_bin, "commit", "-f", dvc_file] if os.path.exists(dvc_bin) else [sys.executable, "-m", "dvc", "commit", "-f", dvc_file]

    try:
        res_add = subprocess.run(add_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if res_add.returncode == 0:
            print(f"DVC: Added {best_path}")
            res_commit = subprocess.run(commit_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            if res_commit.returncode == 0:
                print(f"DVC: Successfully committed {dvc_file} with DVC!")
            else:
                print(f"DVC: Notice — `dvc commit` result: {res_commit.stdout.strip() or res_commit.stderr.strip()}")
        else:
            print(f"DVC: Warning — `dvc add` failed for {best_path}: {res_add.stderr.strip()}")
    except Exception as e:
        print(f"DVC: Notice — DVC tracking skipped ({e}).")


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    ckpt_dir = os.path.join(REPO_ROOT, "scratch", "checkpoints", cfg.exp_name)
    # Under --multirun / sweeps, each trial must get its own checkpoint dir —
    # otherwise trials overwrite each other's config snapshots and weights.
    job_id = os.environ.get("HYDRA_JOB_ID")
    if job_id:
        ckpt_dir = f"{ckpt_dir}_trial_{job_id}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save resolved config snapshot to checkpoint directory. Skipped on dry
    # runs — they must not clobber the saved-config contract of an existing
    # checkpoint dir (checkpoint loading depends on this file matching the
    # saved weights).
    if not cfg.get("dry_run", False):
        config_save_path = os.path.join(ckpt_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    print("=" * 70)
    print(f"            Hydra Baseline Trainer ({cfg.model.name} / {cfg.dataset.name})")
    print("=" * 70)
    print(f"Device:               {cfg.device}")
    print(f"Checkpoint Directory: {ckpt_dir}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 1. Build Model ────────────────────────────────────────────────────
    model = build_model(cfg_dict).to(device)
    model_name = cfg.model.name

    # ── Option: Resume from checkpoint if specified ───────────────────────
    if cfg.ckpt_path is not None:
        ckpt_path = cfg.ckpt_path
        if not os.path.isabs(ckpt_path):
            # Hydra chdirs into outputs/ at runtime — resolve against the repo root.
            ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
        load_checkpoint_state(model, ckpt_path, device=device)

    # ── 2. Build DataLoaders ──────────────────────────────────────────────
    train_loader = build_dataloader(cfg_dict, split="train")
    val_loader = build_dataloader(cfg_dict, split="val")

    # ── 3. Build Trainer (loss is injected by build_model via the factory) ──
    trainer = BaseTrainer(save_dir=ckpt_dir, experiment_name=cfg.exp_name, use_wandb=cfg.get("use_wandb", False), cfg_dict=cfg_dict)

    # ── 4. Build Optimizer & Scaler ───────────────────────────────────────
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
        save_checkpoint_fn=save_checkpoint,
    )

    trainer.close()
    print("Training finished successfully!")

    # ── Auto DVC tracking & commit ────────────────────────────────────────
    if cfg.get("auto_dvc", True):
        best_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
        _auto_dvc_commit(best_path)


if __name__ == "__main__":
    main()
