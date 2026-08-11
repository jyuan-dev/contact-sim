#!/usr/bin/env python3
"""
Unified Baseline Evaluation & Benchmarking Suite powered by Hydra & Deterministic Seeding.

Usage Examples:
  python scripts/eval.py model=savi ckpt_path=scratch/checkpoints/savi_pusht/savi_best.pt
  python scripts/eval.py model=deformable_savi ckpt_path=scratch/checkpoints/deformable_savi_pusht/deformable_savi_best.pt
"""

import os
import sys
import json
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets import build_dataloader, DeterministicEpisodeEvalDataset
from src.metrics import EvaluationSuite
from src.utils.training_utils import load_checkpoint_state


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
    total_transitions = 0
    swap_transitions = 0
    swapped_sequences = 0
    total_sequences = 0
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
                B, T, K_pred = pred_masks.shape[:3]
                min_k = min(K_pred, gt_masks.shape[2])
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

                # ── Slot Swapping Analysis ──────────────────────────────────
                # For each sequence b in batch: find best GT mask j for each slot k across frames
                p_bin = (p_sub > 0.5).float()
                g_bin = (g_sub > 0.5).float()

                for b in range(B):
                    total_sequences += 1
                    seq_swapped = False
                    prev_assignments = None

                    for t in range(T):
                        # Calculate pairwise IoU matrix between slots (min_k) and GT masks (min_k)
                        p_bt = p_bin[b, t]  # [min_k, H, W]
                        g_bt = g_bin[b, t]  # [min_k, H, W]
                        inter_mat = (p_bt.unsqueeze(1) * g_bt.unsqueeze(0)).sum(dim=(-2, -1)) # [min_k, min_k]
                        union_mat = p_bt.unsqueeze(1).sum(dim=(-2, -1)) + g_bt.unsqueeze(0).sum(dim=(-2, -1)) - inter_mat
                        iou_mat = (inter_mat + 1e-6) / (union_mat + 1e-6)

                        # Best GT assignment per slot
                        curr_assignments = torch.argmax(iou_mat, dim=1).tolist()

                        if prev_assignments is not None:
                            total_transitions += 1
                            if curr_assignments != prev_assignments:
                                swap_transitions += 1
                                seq_swapped = True

                        prev_assignments = curr_assignments

                    if seq_swapped:
                        swapped_sequences += 1

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s) | Curr mIoU: {np.mean(mious)*100:.2f}%")

    slot_swap_rate = (swap_transitions / total_transitions * 100.0) if total_transitions > 0 else 0.0
    seq_swap_rate = (swapped_sequences / total_sequences * 100.0) if total_sequences > 0 else 0.0

    res = {
        'ckpt_path': ckpt_path,
        'val_mse': float(np.mean(mses)) if mses else float('nan'),
        'miou': float(np.mean(mious)) * 100 if mious else 0.0,
        'mdice': float(np.mean(mdices)) * 100 if mdices else 0.0,
        'slot0_agent_iou': float(np.mean(cls_ious[0])) * 100 if cls_ious[0] else 0.0,
        'slot1_block_iou': float(np.mean(cls_ious[1])) * 100 if cls_ious[1] else 0.0,
        'slot2_goal_iou': float(np.mean(cls_ious[2])) * 100 if cls_ious[2] else 0.0,
        'slot_swapping_rate_pct': float(slot_swap_rate),
        'sequence_swapping_rate_pct': float(seq_swap_rate),
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
    print(f"  Slot Swapping Transition Rate: {res['slot_swapping_rate_pct']:.2f}%")
    print(f"  Sequence Slot Swapping Rate:  {res['sequence_swapping_rate_pct']:.2f}%")
    print("=" * 80 + "\n")

    # Save evaluation metrics in both the checkpoint directory and scratch root
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    exp_out_file = os.path.join(ckpt_dir, "eval_results.json")
    with open(exp_out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved evaluation metrics to experiment dir: {exp_out_file}")

    out_dir = os.path.join(REPO_ROOT, "scratch")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_results.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

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
    load_checkpoint_state(model, ckpt_path, device=device)

    eval_mode = str(cfg.get('mode', 'deterministic')).lower()
    if eval_mode in ('deterministic', 'full', 'full_val'):
        base_seed = int(cfg.get('seed', 42))
        clips_per_ep = int(cfg.get('clips_per_ep', 2))
        batch_size = int(cfg.get('batch_size', 64))
        run_deterministic_eval(model, ckpt_path, base_seed=base_seed, clips_per_ep=clips_per_ep, batch_size=batch_size, device=device)
    else:
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
