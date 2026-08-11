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
    cls_ious = {k: [] for k in range(3)}
    cls_dices = {k: [] for k in range(3)}
    
    # Temporal & sequence tracking
    frame_mses = {}        # t -> list of MSEs
    frame_mious = {}       # t -> list of mIoUs
    frame_mdices = {}      # t -> list of mDices
    frame_swaps = {}       # t -> (swaps, total)
    
    per_sequence_records = []
    total_transitions = 0
    swap_transitions = 0
    swapped_sequences = 0
    total_sequences = 0
    num_batches = len(val_loader)
    start_t = time.time()

    slot_names = {0: "Agent / Robot", 1: "T-Block Object", 2: "Goal Target Area"}

    with torch.no_grad():
        clip_count = 0
        for b_idx, batch in enumerate(val_loader):
            video = batch['img'].to(device)
            out = model(video)

            recon = out.get('recon_img', None)
            B, T, C, H, W = video.shape
            
            # Per-frame MSE computation
            if recon is not None:
                mse_per_seq = torch.mean((recon - video) ** 2, dim=(2, 3, 4))  # [B, T]
                batch_mse = mse_per_seq.mean().item()
                mses.append(batch_mse)

                for t in range(T):
                    frame_mses.setdefault(t, []).extend(mse_per_seq[:, t].cpu().tolist())

            pred_masks = out.get('pred_masks', None)
            gt_masks = batch.get('gt_masks', None)
            if pred_masks is not None and gt_masks is not None:
                gt_masks = gt_masks.to(device)
                K_pred = pred_masks.shape[2]
                min_k = min(K_pred, gt_masks.shape[2])
                p_sub = pred_masks[:, :, :min_k]
                g_sub = gt_masks[:, :, :min_k]

                # Per-slot IoU and Dice
                for k in range(min_k):
                    p_k = (p_sub[:, :, k] > 0.5).float()
                    g_k = (g_sub[:, :, k] > 0.5).float()
                    inter_k = (p_k * g_k).sum(dim=(-2, -1))
                    union_k = p_k.sum(dim=(-2, -1)) + g_k.sum(dim=(-2, -1)) - inter_k
                    iou_k = (inter_k + 1e-6) / (union_k + 1e-6)
                    dice_k = (2.0 * inter_k + 1e-6) / (p_k.sum(dim=(-2, -1)) + g_k.sum(dim=(-2, -1)) + 1e-6)

                    cls_ious[k].extend(iou_k.reshape(-1).cpu().tolist())
                    cls_dices[k].extend(dice_k.reshape(-1).cpu().tolist())

                intersection = (p_sub * g_sub).sum(dim=(-2, -1))  # [B, T, min_k]
                union = (p_sub + g_sub).sum(dim=(-2, -1)) - intersection
                iou_seq_frame = (intersection + 1e-6) / (union + 1e-6)  # [B, T, min_k]
                miou_per_seq_frame = iou_seq_frame.mean(dim=-1)         # [B, T]

                dice_seq_frame = (2.0 * intersection + 1e-6) / (p_sub.sum(dim=(-2, -1)) + g_sub.sum(dim=(-2, -1)) + 1e-6)
                mdice_per_seq_frame = dice_seq_frame.mean(dim=-1)      # [B, T]

                mious.extend(miou_per_seq_frame.mean(dim=-1).cpu().tolist())
                mdices.extend(mdice_per_seq_frame.mean(dim=-1).cpu().tolist())

                for t in range(T):
                    frame_mious.setdefault(t, []).extend(miou_per_seq_frame[:, t].cpu().tolist())
                    frame_mdices.setdefault(t, []).extend(mdice_per_seq_frame[:, t].cpu().tolist())

                # ── Slot Swapping & Per-Sequence Analysis ────────────────────
                p_bin = (p_sub > 0.5).float()
                g_bin = (g_sub > 0.5).float()

                for b in range(B):
                    global_clip_idx = clip_count + b
                    clip_info = eval_dataset.clips_info[global_clip_idx] if hasattr(eval_dataset, 'clips_info') else {}
                    ep_idx = clip_info.get('ep_idx', global_clip_idx)
                    start_frame = clip_info.get('start_frame', 0)

                    total_sequences += 1
                    seq_swapped = False
                    seq_swap_count = 0
                    prev_assignments = None

                    for t in range(T):
                        p_bt = p_bin[b, t]
                        g_bt = g_bin[b, t]
                        inter_mat = (p_bt.unsqueeze(1) * g_bt.unsqueeze(0)).sum(dim=(-2, -1))
                        union_mat = p_bt.unsqueeze(1).sum(dim=(-2, -1)) + g_bt.unsqueeze(0).sum(dim=(-2, -1)) - inter_mat
                        iou_mat = (inter_mat + 1e-6) / (union_mat + 1e-6)

                        curr_assignments = torch.argmax(iou_mat, dim=1).tolist()

                        if prev_assignments is not None:
                            total_transitions += 1
                            frame_swaps.setdefault(t, [0, 0])
                            frame_swaps[t][1] += 1

                            if curr_assignments != prev_assignments:
                                swap_transitions += 1
                                seq_swapped = True
                                seq_swap_count += 1
                                frame_swaps[t][0] += 1

                        prev_assignments = curr_assignments

                    if seq_swapped:
                        swapped_sequences += 1

                    per_sequence_records.append({
                        "clip_idx": global_clip_idx,
                        "episode_idx": ep_idx,
                        "start_frame": start_frame,
                        "mse": float(mse_per_seq[b].mean().item()) if recon is not None else float("nan"),
                        "miou": float(miou_per_seq_frame[b].mean().item() * 100),
                        "mdice": float(mdice_per_seq_frame[b].mean().item() * 100),
                        "slot_ious": {k: float(iou_seq_frame[b, :, k].mean().item() * 100) for k in range(min_k)},
                        "swapped": seq_swapped,
                        "swap_count": seq_swap_count,
                    })

            clip_count += B

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed = time.time() - start_t
                speed = (b_idx + 1) / elapsed
                print(f"Evaluated [{b_idx+1}/{num_batches}] batches ({speed:.1f} batch/s) | Curr mIoU: {np.mean(mious)*100:.2f}%")

    slot_swap_rate = (swap_transitions / total_transitions * 100.0) if total_transitions > 0 else 0.0
    seq_swap_rate = (swapped_sequences / total_sequences * 100.0) if total_sequences > 0 else 0.0

    # Calculate percentiles & distribution statistics
    def get_stats(arr):
        if not arr:
            return {}
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # Per-slot detailed statistics
    per_slot_stats = {}
    for k in cls_ious:
        iou_st = get_stats(cls_ious[k])
        dice_st = get_stats(cls_dices[k])
        name = slot_names.get(k, f"Slot {k}")
        per_slot_stats[f"slot_{k}"] = {
            "name": name,
            "iou_mean_pct": iou_st.get("mean", 0) * 100,
            "iou_std_pct": iou_st.get("std", 0) * 100,
            "iou_median_pct": iou_st.get("median", 0) * 100,
            "iou_q25_pct": iou_st.get("q25", 0) * 100,
            "iou_q75_pct": iou_st.get("q75", 0) * 100,
            "dice_mean_pct": dice_st.get("mean", 0) * 100,
            "dice_std_pct": dice_st.get("std", 0) * 100,
        }

    # Per-frame detailed statistics
    per_frame_stats = []
    for t in sorted(frame_mious.keys()):
        swaps, total_t = frame_swaps.get(t, [0, 0])
        per_frame_stats.append({
            "frame_idx": t + 1,
            "mse_mean": float(np.mean(frame_mses[t])) if t in frame_mses else float("nan"),
            "miou_mean_pct": float(np.mean(frame_mious[t])) * 100 if t in frame_mious else 0.0,
            "mdice_mean_pct": float(np.mean(frame_mdices[t])) * 100 if t in frame_mdices else 0.0,
            "frame_swap_rate_pct": float(swaps / total_t * 100.0) if total_t > 0 else 0.0,
        })

    res = {
        'ckpt_path': ckpt_path,
        'summary': {
            'val_mse': get_stats(mses),
            'miou': {k: v * 100 for k, v in get_stats(mious).items()},
            'mdice': {k: v * 100 for k, v in get_stats(mdices).items()},
            'slot_swapping_rate_pct': float(slot_swap_rate),
            'sequence_swapping_rate_pct': float(seq_swap_rate),
            'total_episodes_evaluated': len(per_sequence_records),
        },
        'val_mse': float(np.mean(mses)) if mses else float('nan'),
        'miou': float(np.mean(mious)) * 100 if mious else 0.0,
        'mdice': float(np.mean(mdices)) * 100 if mdices else 0.0,
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

    # ── Auto-discover training config.yaml ────────────────────────────────────
    ckpt_dir = os.path.dirname(ckpt_path)
    config_candidates = [
        os.path.join(ckpt_dir, "config.yaml"),
        os.path.join(ckpt_dir, ".hydra", "config.yaml"),
    ]
    saved_cfg = None
    for cand in config_candidates:
        if os.path.exists(cand):
            try:
                saved_cfg = OmegaConf.load(cand)
                print(f"[Auto-Config] Loaded training configuration from: {cand}")
                break
            except Exception as e:
                print(f"[Auto-Config] Warning: failed to load {cand}: {e}")

    if saved_cfg is not None:
        saved_dict = OmegaConf.to_container(saved_cfg, resolve=True)
        # Preserve user CLI execution overrides (device, batch_size, etc.)
        cli_keys = ['device', 'batch_size', 'ckpt_path', 'ckpt', 'seed', 'clips_per_ep', 'mode']
        for k in cli_keys:
            if k in cfg_dict:
                saved_dict[k] = cfg_dict[k]
        cfg_dict = saved_dict
    else:
        # Fallback state_dict key inspection
        ckpt_state = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt_state.get('model_state', ckpt_state)
        is_deformable = any('deform' in k for k in state_dict.keys()) or ('deformable' in ckpt_path.lower())
        detected_model = 'deformable_savi' if is_deformable else 'savi'
        if cfg_dict.get('model', {}).get('name') != detected_model:
            print(f"[Auto-Config] Auto-detected model type '{detected_model}' from checkpoint weights.")
            cfg_dict.setdefault('model', {})['name'] = detected_model
            cfg_dict['model']['type'] = detected_model

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
