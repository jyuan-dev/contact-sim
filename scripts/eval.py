#!/usr/bin/env python3
"""
Unified Baseline Evaluation & Benchmarking Suite powered by Hydra & Deterministic Seeding.

Usage Examples:
  python scripts/eval.py model=savi ckpt_path=scratch/checkpoints/savi_pusht/savi_best.pt
"""

import os
import sys
import json
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets import DeterministicEpisodeEvalDataset
from src.metrics import DeterministicEvaluator
from src.utils.training_utils import load_checkpoint_state
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
from src.utils.data_utils import find_dataset_path


def run_deterministic_eval(model, ckpt_path, base_seed=42, clips_per_ep=2, batch_size=64, device='cpu', h5_path=None):
    print("=" * 80)
    print(f"Deterministic Per-Episode Evaluation: {ckpt_path}")
    print(f"  Base Seed: {base_seed} | Clips per Episode: {clips_per_ep} | Device: {device}")
    print("=" * 80)

    eval_dataset = DeterministicEpisodeEvalDataset(
        h5_path=find_dataset_path(h5_path),
        split='val',
        resolution=(64, 64),
        n_sample_frames=6,
        clips_per_episode=clips_per_ep,
        base_seed=base_seed,
    )
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    slot_names = {0: "Agent / Robot", 1: "T-Block Object", 2: "Goal Target Area"}
    # Built lazily on the first batch so the slot count matches the model
    # instead of hardcoding 3-slot PushT semantics.
    evaluator = None

    num_batches = len(val_loader)
    start_t = time.time()

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            video = batch['img'].to(device)
            out = model(video)

            gt_masks = batch.get('gt_masks', None)
            if evaluator is None and gt_masks is not None:
                num_classes = gt_masks.shape[2]
                names = {k: slot_names.get(k, f"Slot {k}") for k in range(num_classes)}
                evaluator = DeterministicEvaluator(num_classes=num_classes, slot_names=names, thresh=0.5)
            evaluator.update(
                pred_masks=out.get('pred_masks'),
                gt_masks=gt_masks.to(device) if gt_masks is not None else None,
                recon=out.get('recon_img'),
                video=video,
                episode_idx=batch.get('episode_idx'),
                start_frame=batch.get('start_frame'),
            )

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s) | Curr mIoU: {evaluator.running_miou_mean()*100:.2f}%")

    if evaluator is None:
        raise RuntimeError("No batches carried gt_masks — evaluation produced no results")

    raw = evaluator.finalize()

    # ── Presentation: percentage conversion + JSON layout (script-owned) ─────
    per_slot_stats = {}
    for key, s in raw['per_slot'].items():
        per_slot_stats[key] = {
            "name": s['name'],
            "iou_mean_pct": s['iou'].get("mean", 0) * 100,
            "iou_std_pct": s['iou'].get("std", 0) * 100,
            "iou_median_pct": s['iou'].get("median", 0) * 100,
            "iou_q25_pct": s['iou'].get("q25", 0) * 100,
            "iou_q75_pct": s['iou'].get("q75", 0) * 100,
            "dice_mean_pct": s['dice'].get("mean", 0) * 100,
            "dice_std_pct": s['dice'].get("std", 0) * 100,
        }

    per_frame_stats = []
    for f in raw['per_frame']:
        per_frame_stats.append({
            "frame_idx": f['frame_idx'],
            "mse_mean": f['mse_mean'],
            "miou_mean_pct": f['miou_mean'] * 100,
            "mdice_mean_pct": f['mdice_mean'] * 100,
            "frame_swap_rate_pct": f['swap_rate'] * 100.0,
        })

    per_sequence_records = []
    for r in raw['per_sequence']:
        per_sequence_records.append({
            "clip_idx": r['clip_idx'],
            "episode_idx": r['episode_idx'],
            "start_frame": r['start_frame'],
            "mse": r['mse'],
            "miou": r['miou'] * 100,
            "mdice": r['mdice'] * 100,
            "slot_ious": {k: v * 100 for k, v in r['slot_ious'].items()},
            "swapped": r['swapped'],
            "swap_count": r['swap_count'],
        })

    summary = raw['summary']
    slot_swap_rate = summary['slot_swapping_rate'] * 100.0
    seq_swap_rate = summary['sequence_swapping_rate'] * 100.0

    res = {
        'ckpt_path': ckpt_path,
        'summary': {
            'val_mse': summary['val_mse'],
            'miou': {k: v * 100 for k, v in summary['miou'].items()},
            'mdice': {k: v * 100 for k, v in summary['mdice'].items()},
            'slot_swapping_rate_pct': float(slot_swap_rate),
            'sequence_swapping_rate_pct': float(seq_swap_rate),
            'total_episodes_evaluated': len(per_sequence_records),
        },
        'val_mse': summary['val_mse'].get('mean', float('nan')),
        'miou': summary['miou'].get('mean', 0.0) * 100,
        'mdice': summary['mdice'].get('mean', 0.0) * 100,
        'slot0_agent_iou': per_slot_stats.get("slot_0", {}).get("iou_mean_pct", 0.0),
        'slot1_block_iou': per_slot_stats.get("slot_1", {}).get("iou_mean_pct", 0.0),
        'slot2_goal_iou': per_slot_stats.get("slot_2", {}).get("iou_mean_pct", 0.0),
        'slot_swapping_rate_pct': float(slot_swap_rate),
        'sequence_swapping_rate_pct': float(seq_swap_rate),
        'per_slot': per_slot_stats,
        'per_frame': per_frame_stats,
        'per_sequence': per_sequence_records,
    }

    print("\n" + "=" * 80)
    print(f"Deterministic Episode Evaluation Summary for {ckpt_path}:")
    print(f"  Visited Validation Episodes: {len(eval_dataset._episode_indices)} (100% Coverage)")
    print(f"  Evaluated Total Clips:       {len(eval_dataset)}")
    print(f"  Val Reconstruction MSE:      {res['val_mse']:.6f} (std: {res['summary']['val_mse']['std']:.6f})")
    print(f"  Overall Mask mIoU:           {res['miou']:.2f}% (median: {res['summary']['miou']['median']:.2f}%)")
    print(f"  Overall Mask mDice:          {res['mdice']:.2f}%")
    print(f"  Slot 0 (Agent IoU):          {res['slot0_agent_iou']:.2f}% (std: {res['per_slot']['slot_0']['iou_std_pct']:.2f}%)")
    print(f"  Slot 1 (T-Block IoU):        {res['slot1_block_iou']:.2f}% (std: {res['per_slot']['slot_1']['iou_std_pct']:.2f}%)")
    print(f"  Slot 2 (Goal Target):        {res['slot2_goal_iou']:.2f}% (std: {res['per_slot']['slot_2']['iou_std_pct']:.2f}%)")
    print(f"  Slot Swapping Transition Rate: {res['slot_swapping_rate_pct']:.2f}%")
    print(f"  Sequence Slot Swapping Rate:  {res['sequence_swapping_rate_pct']:.2f}%")
    print("=" * 80 + "\n")

    # Save detailed evaluation metrics in both the checkpoint directory and scratch root
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    exp_out_file = os.path.join(ckpt_dir, "eval_results_detailed.json")
    with open(exp_out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved detailed evaluation metrics to experiment dir: {exp_out_file}")

    # Also save standard eval_results.json for backwards compatibility
    exp_std_file = os.path.join(ckpt_dir, "eval_results.json")
    summary_std = {k: v for k, v in res.items() if k not in ["per_sequence", "per_frame"]}
    with open(exp_std_file, "w") as f:
        json.dump(summary_std, f, indent=2)

    out_dir = os.path.join(REPO_ROOT, "scratch")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_results_detailed.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(summary_std, f, indent=2)

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

    # ── Reconstruct the experiment from the checkpoint ───────────────────────
    # Preserve user CLI execution overrides (device, batch_size, etc.)
    cli_keys = ['device', 'batch_size', 'ckpt_path', 'ckpt', 'seed', 'clips_per_ep', 'mode']
    model, cfg_dict = bootstrap_checkpoint(
        ckpt_path, cli_overrides={k: cfg_dict[k] for k in cli_keys if k in cfg_dict})
    model = model.to(device)
    load_checkpoint_state(model, ckpt_path, device=device)

    eval_mode = str(cfg.get('mode', 'deterministic')).lower()
    if eval_mode in ('deterministic', 'full', 'full_val'):
        base_seed = int(cfg.get('seed', 42))
        clips_per_ep = int(cfg.get('clips_per_ep', 2))
        batch_size = int(cfg.get('batch_size', 64))
        run_deterministic_eval(model, ckpt_path, base_seed=base_seed, clips_per_ep=clips_per_ep, batch_size=batch_size, device=device)
    else:
        # The former visualize branch evaluated zero-filled masks and rendered
        # nothing — fail loudly instead of reporting fake numbers.
        raise NotImplementedError(
            f"Unknown eval mode: '{eval_mode}'. Supported modes: "
            "'deterministic' (full val coverage).")


if __name__ == "__main__":
    main()
