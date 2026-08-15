#!/usr/bin/env python3
"""
Pre-extract slot representations for PushT dataset using trained Stage 1 SAVi model.
Saves slot latents to scratch/pusht_slots_savi.pt to eliminate online SAVi extraction overhead
and achieve 100% sustained GPU utilization for Stage 2 SlotFormer training.
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataset
from src.utils.training_utils import get_device
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
from src.utils.data_utils import find_dataset_path


import argparse

def extract_all_slots(
    savi_ckpt_path: str = "scratch/checkpoints/savi_pusht_default_4ep/savi_best.pt",
    output_path: str = "scratch/pusht_slots_savi_default_4ep.pt",
    batch_size: int = 512,
    num_workers: int = 4,
):
    from src.utils.training_utils import load_checkpoint_state
    device = get_device("cuda")
    print(f"Loading Stage 1 SAVi model from: {savi_ckpt_path}")

    savi_model, cfg_dict = bootstrap_checkpoint(savi_ckpt_path)
    savi_model = savi_model.to(device)
    load_checkpoint_state(savi_model, savi_ckpt_path, device=device)
    savi_model.eval()

    # Get inner savi
    inner_savi = savi_model.inner_savi()

    # Build dataset
    ds_cfg = {
        "name": "pusht",
        "h5_path": find_dataset_path(None),
        "n_sample_frames": 6,
        "load_masks": False,
        "preload_ram": True,
    }
    dataset = build_dataset(ds_cfg, split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Extracting slots for {len(dataset)} clips with batch_size={batch_size} on {device}...")
    start_t = time.time()

    all_slots_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Slots"):
            video = batch["img"].to(device, non_blocking=True)  # [B, T, C, H, W]

            if hasattr(inner_savi, "_reset_rnn"):
                inner_savi._reset_rnn()

            post_slots, _ = inner_savi.encode(video)
            batch_slots = post_slots.half().cpu()  # [B, T, K, D] in FP16
            all_slots_list.append(batch_slots)

    all_slots_tensor = torch.cat(all_slots_list, dim=0)  # [N, T, K, D]
    elapsed = time.time() - start_t
    print(f"Extraction completed in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} clips/s). Tensor shape: {all_slots_tensor.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"slots": all_slots_tensor}, output_path)
    print(f"Saved slot dataset to: {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-extract slot representations for PushT dataset")
    parser.add_argument("--savi_ckpt_path", type=str, default="scratch/checkpoints/savi_pusht_default_4ep/savi_best.pt")
    parser.add_argument("--output_path", type=str, default="scratch/pusht_slots_savi_default_4ep.pt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    extract_all_slots(
        savi_ckpt_path=args.savi_ckpt_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
