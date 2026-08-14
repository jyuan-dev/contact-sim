#!/usr/bin/env python3
"""
Verify the SIGReg statistic using the newly trained Stage 1 SAVi model.

Loads a trained SAVi checkpoint, extracts slot latents (post_slots) on a few
train/val clips, and computes the SIGReg ECF statistic with training-identical
settings (num_proj=1024). Also runs synthetic baselines to sanity-check the
statistic itself (Gaussian floor, variance sensitivity, batch-size invariance)
and compares against the compute_sigreg_stat default path (num_proj=64).

Usage:
    python scripts/verify_sigreg_trained.py
    python scripts/verify_sigreg_trained.py --ckpt scratch/checkpoints/savi_pusht_sigreg_0001/savi_best.pt
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataset
from src.utils.training_utils import get_device, load_checkpoint_state
from src.losses.sigreg import SIGRegLoss
from src.metrics.eval_metrics import compute_sigreg_stat, compute_latent_std


def compute_sigreg(slots, num_proj, knots=17, t_max=3.0, n_calls=3):
    """Mean raw SIGReg statistic over `n_calls` (projections are resampled per call)."""
    loss_fn = SIGRegLoss(weight=1.0, num_proj=num_proj, knots=knots, t_max=t_max)
    vals = []
    for _ in range(n_calls):
        _, info = loss_fn(slots)
        vals.append(info["sigreg_loss"])
    return sum(vals) / len(vals)


def extract_slots(model, loader, num_batches, device):
    """Run the wrapper forward over the first `num_batches` batches, collect post_slots."""
    model.eval()
    collected = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            video = batch["img"].to(device, non_blocking=True)
            out = model(video)
            slots = out["post_slots"]
            if slots is None:
                print(f"  [WARN] batch {i}: no post_slots in output, skipping")
                continue
            collected.append(slots.float().cpu())
    if not collected:
        raise RuntimeError("No slot latents extracted — check model output keys.")
    return torch.cat(collected, dim=0)  # [N, T, K, D]


def main():
    parser = argparse.ArgumentParser(description="Verify SIGReg with a trained SAVi model")
    parser.add_argument("--ckpt", type=str,
                        default="scratch/checkpoints/savi_pusht_sigreg_0001/savi_best.pt")
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device("cuda")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    # ── Load training config + checkpoint ────────────────────────────────
    from omegaconf import OmegaConf
    saved_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    cfg_dict = OmegaConf.to_container(saved_cfg, resolve=True)
    print(f"Loaded config from: {os.path.join(ckpt_dir, 'config.yaml')}")

    model = build_model(cfg_dict).to(device)
    load_checkpoint_state(model, args.ckpt, device=device)
    print(f"Loaded checkpoint: {args.ckpt}")

    # ── Extract slots (train + val) ──────────────────────────────────────
    ds_cfg = dict(cfg_dict["dataset"])
    print(f"Dataset: {ds_cfg.get('name')} ({ds_cfg.get('h5_path')})")

    splits = {}
    for split in ("train", "val"):
        try:
            ds = build_dataset(ds_cfg, split=split)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
            splits[split] = loader
            print(f"  {split}: {len(ds)} clips")
        except Exception as e:
            print(f"  [WARN] split '{split}' unavailable: {e}")

    report = {}
    for split, loader in splits.items():
        slots = extract_slots(model, loader, args.num_batches, device)
        report[split] = slots
        print(f"{split}: extracted {slots.shape[0]} clips, slots {tuple(slots.shape)}")

    # ── SIGReg on trained latents ────────────────────────────────────────
    print("\n=== SIGReg on newly trained SAVi latents ===")
    for split, slots in report.items():
        stat_1024 = compute_sigreg(slots, num_proj=1024)
        stat_64 = compute_sigreg(slots, num_proj=64)
        helper_64 = compute_sigreg_stat(slots)  # eval_metrics default path (num_proj=64)
        print(f"  {split:5s}: latent_std={compute_latent_std(slots):.4f} | "
              f"sigreg(num_proj=1024)={stat_1024:.3f} | "
              f"sigreg(num_proj=64)={stat_64:.3f} | "
              f"compute_sigreg_stat={helper_64:.3f}")

    # ── Synthetic baselines (same shape [B, T=6, K=4, D=64]) ─────────────
    print("\n=== Synthetic baselines [B, 6, 4, 64] ===")
    def baseline(dist, B=128, std=None):
        if std is not None:
            lat = torch.randn(B, 6, 4, 64) * std
        else:
            lat = dist((B, 6, 4, 64))
        return compute_sigreg(lat, num_proj=1024)

    std_normal = baseline(torch.randn)
    print(f"  N(0,1)            : {std_normal:.3f}   (expected: near-statistic floor)")
    print(f"  N(0,0.1)          : {baseline(torch.randn, std=0.1):.3f}   (collapsed variance -> should be LARGER)")
    print(f"  N(0,2)            : {baseline(torch.randn, std=2.0):.3f}   (inflated variance -> should be LARGER)")
    print(f"  Uniform[-1,1]     : {baseline(lambda s: torch.rand(s) * 2 - 1):.3f}   (non-Gaussian -> should be LARGER)")

    # Batch-size invariance: statistic should not depend on B (iid samples)
    print("\n=== Batch-size invariance (N(0,1), num_proj=1024) ===")
    for B in (16, 32, 128):
        print(f"  B={B:4d}: {baseline(torch.randn, B=B):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
