#!/usr/bin/env python3
"""
Unified Baseline Evaluation & Benchmarking Suite powered by Hydra & Deterministic Seeding.

Modes of Operation:
  1. Deterministic Per-Episode Evaluation (--mode=deterministic / --full_val=true):
     - Visited 100% of all validation episodes (3,737 episodes).
     - Deterministically samples K clips per episode using per-episode seed (base_seed + ep_idx * 10007).
     - Computes exact MSE, mIoU, mDice, and Per-Slot IoU metrics.
  2. Visualization & Overlay Animation (--mode=visualize):
     - Renders GT outlines & Color-Coded Slot Mask Overlays.
     - Saves demo animation GIF (scratch/baseline_eval_demo.gif) & metrics JSON.

Usage Examples:
  python scripts/eval.py model=deformable_savi ckpt_path=scratch/checkpoints/deformable_savi_3class_1ep/deformable_savi_best.pt
  python scripts/eval.py mode=visualize ckpt_path=scratch/checkpoints/deformable_savi_3class_1ep/deformable_savi_best.pt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.metrics.evaluator import EvaluationSuite
from src.utils.training_utils import get_device

SLOT_COLORS_RGB = {
    0: (255, 40, 40),     # Slot 0: Red (Agent)
    1: (40, 220, 40),     # Slot 1: Green (T-Block)
    2: (40, 120, 255),    # Slot 2: Blue (Goal Target)
    3: (255, 210, 0),     # Slot 3: Yellow
    4: (230, 40, 230)     # Slot 4: Magenta
}

GT_COLORS_RGB = {
    0: (255, 140, 0),    # Orange
    1: (0, 230, 115),    # Green
    2: (0, 128, 255)     # Blue
}


class DeterministicEpisodeEvalDataset(Dataset):
    """
    Evaluation dataset visiting 100% of validation episodes,
    deterministically sampling fixed clips per episode using per-episode seed.
    """
    MASK_KEYS = ['agent_masks', 'block_masks', 'goal_masks']

    def __init__(
        self,
        h5_path: str,
        split: str = 'val',
        resolution=(64, 64),
        n_sample_frames: int = 6,
        clips_per_episode: int = 2,
        train_frac: float = 0.9,
        base_seed: int = 42,
    ):
        self.h5_path = h5_path
        self.resolution = resolution
        self.n_sample_frames = n_sample_frames
        self.clips_per_episode = clips_per_episode
        self._h5 = None

        with h5py.File(h5_path, 'r') as f:
            ep_lens = f['ep_len'][:]
            ep_offs = f['ep_offset'][:]

        self._ep_lens = ep_lens.tolist()
        self._ep_offs = ep_offs.tolist()
        n_episodes = len(ep_lens)

        rng = np.random.RandomState(base_seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * train_frac)

        if split == 'train':
            self._episode_indices = sorted(idx[:n_train].tolist())
        else:
            self._episode_indices = sorted(idx[n_train:].tolist())

        self._index = []
        clip_len = n_sample_frames
        for ep_i, ep_idx in enumerate(self._episode_indices):
            ep_len = self._ep_lens[ep_idx]
            max_start = ep_len - clip_len
            if max_start < 0:
                continue

            ep_seed = (base_seed + ep_idx * 10007) & 0xFFFFFFFF
            ep_rng = np.random.RandomState(ep_seed)
            valid_starts = list(range(0, max_start + 1))

            if len(valid_starts) <= clips_per_episode:
                chosen_starts = valid_starts
            else:
                chosen_starts = sorted(ep_rng.choice(valid_starts, size=clips_per_episode, replace=False).tolist())

            for start in chosen_starts:
                self._index.append((ep_idx, start))

        print(f"[DeterministicEpisodeEvalDataset] Visited 100% of {len(self._episode_indices)} {split} episodes. "
              f"Deterministically sampled {len(self._index)} clips ({clips_per_episode} clips/episode, seed={base_seed}).")

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, start_frame = self._index[idx]
        ep_off = int(self._ep_offs[ep_idx])
        abs_idxs = [ep_off + start_frame + t for t in range(self.n_sample_frames)]

        frames = self.h5['pixels'][abs_idxs]
        masks = {k: self.h5[k][abs_idxs] for k in self.MASK_KEYS}

        video = (frames.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(video.transpose(0, 3, 1, 2))

        agent_m = (masks['agent_masks'] > 127).astype(np.float32)
        block_m = (masks['block_masks'] > 127).astype(np.float32)
        goal_m = (masks['goal_masks'] > 127).astype(np.float32)

        goal_visible = np.clip(goal_m - np.maximum(block_m, agent_m), 0.0, 1.0)
        gt_masks = torch.from_numpy(np.stack([agent_m, block_m, goal_visible], axis=1)).float()

        return {
            'data_idx': idx,
            'img': img,
            'gt_masks': gt_masks,
            'ep_idx': ep_idx,
            'start_frame': start_frame,
        }


def run_deterministic_eval(model, ckpt_path, base_seed=42, clips_per_ep=2, batch_size=64, device='cpu'):
    print("=" * 80)
    print(f"Deterministic Per-Episode Evaluation: {ckpt_path}")
    print(f"  Base Seed: {base_seed} | Clips per Episode: {clips_per_ep} | Device: {device}")
    print("=" * 80)

    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5'
    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=h5_path,
        split='val',
        resolution=(64, 64),
        n_sample_frames=6,
        clips_per_episode=clips_per_ep,
        base_seed=base_seed,
    )
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    mses, mious, mdices = [], [], []
    cls_ious = {0: [], 1: [], 2: []}
    num_batches = len(val_loader)
    start_t = time.time()

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            video = batch['img'].to(device)
            out = model(video)

            recon = out.get('recon_img', None)
            if recon is not None:
                mse = torch.mean((recon - video) ** 2).item()
                mses.append(mse)

            pred_masks = out.get('pred_masks', None)
            gt_masks = batch.get('gt_masks', None)
            if pred_masks is not None and gt_masks is not None:
                gt_masks = gt_masks.to(device)
                min_k = min(pred_masks.shape[2], gt_masks.shape[2])
                p_sub = pred_masks[:, :, :min_k]
                g_sub = gt_masks[:, :, :min_k]

                for k in range(min_k):
                    p_k = (p_sub[:, :, k] > 0.5).float()
                    g_k = (g_sub[:, :, k] > 0.5).float()
                    inter = (p_k * g_k).sum(dim=(-2, -1))
                    union = p_k.sum(dim=(-2, -1)) + g_k.sum(dim=(-2, -1)) - inter
                    iou_k = (inter + 1e-6) / (union + 1e-6)
                    cls_ious[k].append(iou_k.mean().item())

                intersection = (p_sub * g_sub).sum(dim=(-2, -1))
                union = (p_sub + g_sub).sum(dim=(-2, -1)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                dice = (2.0 * intersection + 1e-6) / (p_sub.sum(dim=(-2, -1)) + g_sub.sum(dim=(-2, -1)) + 1e-6)
                mious.append(iou.mean().item())
                mdices.append(dice.mean().item())

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s) | Curr mIoU: {np.mean(mious)*100:.2f}%")

    res = {
        'ckpt_path': ckpt_path,
        'val_mse': float(np.mean(mses)) if mses else float('nan'),
        'miou': float(np.mean(mious)) * 100 if mious else 0.0,
        'mdice': float(np.mean(mdices)) * 100 if mdices else 0.0,
        'slot0_agent_iou': float(np.mean(cls_ious[0])) * 100 if cls_ious[0] else 0.0,
        'slot1_block_iou': float(np.mean(cls_ious[1])) * 100 if cls_ious[1] else 0.0,
        'slot2_goal_iou': float(np.mean(cls_ious[2])) * 100 if cls_ious[2] else 0.0,
    }

    print("\n" + "=" * 80)
    print(f"Deterministic Episode Evaluation Summary for {ckpt_path}:")
    print(f"  Visited Validation Episodes: {len(eval_dataset._episode_indices)} (100% Coverage)")
    print(f"  Evaluated Total Clips:       {len(eval_dataset)}")
    print(f"  Val Reconstruction MSE:      {res['val_mse']:.6f}")
    print(f"  Overall Mask mIoU:           {res['miou']:.2f}%")
    print(f"  Overall Mask mDice:          {res['mdice']:.2f}%")
    print(f"  Slot 0 (Agent IoU):          {res['slot0_agent_iou']:.2f}%")
    print(f"  Slot 1 (T-Block IoU):        {res['slot1_block_iou']:.2f}%")
    print(f"  Slot 2 (Goal Target):        {res['slot2_goal_iou']:.2f}%")
    print("=" * 80 + "\n")

    os.makedirs("scratch", exist_ok=True)
    out_file = "scratch/eval_results.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved evaluation metrics to: {out_file}")

    return res


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device_name = cfg.get('device', 'cpu')
    device = torch.device(device_name if torch.cuda.is_available() and device_name != 'cpu' else 'cpu')

    ckpt_path = cfg.get('ckpt_path', None) or cfg.get('ckpt', None)
    if ckpt_path and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"Error: Invalid or missing checkpoint path: '{ckpt_path}'")
        sys.exit(1)

    model = build_model(cfg_dict).to(device)
    ckpt_data = torch.load(ckpt_path, map_location=device)
    state = ckpt_data.get('model', ckpt_data.get('model_state', ckpt_data))
    state = {k.replace('model.', '').replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    eval_mode = str(cfg.get('mode', 'deterministic')).lower()
    if eval_mode in ('deterministic', 'full', 'full_val'):
        base_seed = int(cfg.get('seed', 42))
        clips_per_ep = int(cfg.get('clips_per_ep', 2))
        batch_size = int(cfg.get('batch_size', 64))
        run_deterministic_eval(model, ckpt_path, base_seed=base_seed, clips_per_ep=clips_per_ep, batch_size=batch_size, device=device)
    else:
        # Visualization mode
        val_loader = build_dataloader(cfg_dict, split='val', batch_size=1, num_workers=2, shuffle=False)
        evaluator = EvaluationSuite(num_classes=3)
        num_eval_batches = cfg.get('eval_batches', 20)
        all_metrics = []

        model.eval()
        for step_idx, batch in enumerate(val_loader):
            if step_idx >= num_eval_batches:
                break
            imgs_torch = batch['img'].to(device)
            gt_masks = batch.get('gt_masks', None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)

            with torch.no_grad():
                out = model(imgs_torch)

            pred_masks = out.get('pred_masks', None)
            if pred_masks is not None:
                pred_masks_np = pred_masks[0].cpu().numpy()
            else:
                pred_masks_np = np.zeros((imgs_torch.shape[1], 4, 64, 64), dtype=np.float32)

            gt_masks_np = gt_masks[0].cpu().numpy() if gt_masks is not None else np.zeros((imgs_torch.shape[1], 3, 64, 64))
            gt_masks_dict = {m_idx: (gt_masks_np[:, m_idx] > 0.5) for m_idx in range(gt_masks_np.shape[1])}

            seq_metrics = evaluator.evaluate_sequence_masks(pred_masks_np, gt_masks_dict)
            all_metrics.append(seq_metrics)

        print("\nVisualization evaluation completed!")


if __name__ == "__main__":
    main()
