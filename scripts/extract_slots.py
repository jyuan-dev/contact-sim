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

    # Discover training config or build model
    ckpt_dir = os.path.dirname(os.path.abspath(savi_ckpt_path))
    config_candidates = [
        os.path.join(ckpt_dir, "config.yaml"),
        os.path.join(ckpt_dir, ".hydra", "config.yaml"),
    ]
    saved_cfg = None
    for cand in config_candidates:
        if os.path.exists(cand):
            try:
                from omegaconf import OmegaConf
                saved_cfg = OmegaConf.load(cand)
                print(f"[Auto-Config] Loaded Stage 1 training config from: {cand}")
                break
            except Exception:
                pass

    if saved_cfg is not None:
        cfg_dict = OmegaConf.to_container(saved_cfg, resolve=True)
    else:
        cfg_dict = {
            "model": {
                "name": "savi",
                "type": "savi",
                "num_slots": 4,
                "slot_dim": 64,
                "in_channels": 3,
                "resolution": [64, 64],
            }
        }

    savi_model = build_model(cfg_dict).to(device)
    load_checkpoint_state(savi_model, savi_ckpt_path, device=device)
    savi_model.eval()

    # Get inner savi
    if hasattr(savi_model, "model"):
        inner_savi = savi_model.model
    else:
        inner_savi = savi_model

    # Build dataset
    ds_cfg = {
        "name": "pusht",
        "h5_path": "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5",
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
            B, T, C, H, W = video.shape

            if hasattr(inner_savi, "_reset_rnn"):
                inner_savi._reset_rnn()

            video_flat = video.flatten(0, 1)  # [B*T, C, H, W]
            enc_out_all = inner_savi._get_encoder_out(video_flat).unflatten(0, (B, T))

            init_latents = inner_savi.init_latents.repeat(B, 1, 1)
            prev_slots = None
            clip_slots = []

            for t in range(T):
                enc_out_t = enc_out_all[:, t]
                if prev_slots is None:
                    latents = init_latents
                else:
                    latents = inner_savi.predictor(prev_slots)

                kernel_dist = inner_savi.kernel_dist_layer(latents)
                kernels = inner_savi._sample_dist(kernel_dist)
                post_slots = inner_savi.slot_attention(enc_out_t, kernels)
                clip_slots.append(post_slots.half().cpu())  # Save as FP16 on CPU
                prev_slots = post_slots

            batch_slots = torch.stack(clip_slots, dim=1)  # [B, T, K, D] in FP16
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
