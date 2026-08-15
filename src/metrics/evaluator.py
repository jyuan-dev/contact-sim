"""
Comprehensive Evaluation Metrics Suite for Contact-Sim / Slot-Worldmodel.

Computes:
  1. Segmentation Metrics: Mean IoU (mIoU) and Dice coefficient per object class.
  2. Bounding Box Metrics: Hungarian Matching Bounding Box IoU / AP.
  3. Temporal Slot-Swapping Metrics: Frame-by-frame Hungarian optimal assignment tracking,
     counting total swap events and swap rate (swaps / 100 frames).
  4. Reconstruction Metrics: MSE and PSNR.
"""

import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard

patch_typeguard()


def compute_binary_iou_dice(pred_mask, gt_mask, thresh=0.3):
    """Computes binary IoU and Dice score for a single frame mask."""
    pred_bin = (pred_mask > thresh).astype(np.float32)
    gt_bin = (gt_mask > 0.5).astype(np.float32)

    if gt_bin.shape != pred_bin.shape:
        gt_bin = cv2.resize(gt_bin, (pred_bin.shape[1], pred_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    iou = float(intersection / union) if union > 0 else (1.0 if gt_bin.sum() == 0 else 0.0)
    dice = float(2.0 * intersection / (pred_bin.sum() + gt_bin.sum())) if (pred_bin.sum() + gt_bin.sum()) > 0 else 1.0

    return iou, dice


class EvaluationSuite:
    """Evaluates segmentation, bounding box, reconstruction, and temporal slot swapping metrics."""

    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or {0: "Block", 1: "Agent", 2: "Goal"}

    def evaluate_sequence_masks(self, pred_masks, gt_masks_dict, thresh=0.3):
        """
        Evaluates sequence of predicted slot masks [T, K, H, W] against GT masks dict {class_id: [T, H, W]}.

        Returns:
            dict: Metrics summary containing per-class IoU, Dice, total swap events, and swap rate.
        """
        T, K, H, W = pred_masks.shape

        slot_assignments = {c: [] for c in range(self.num_classes)}
        slot_ious = {c: [] for c in range(self.num_classes)}
        slot_dices = {c: [] for c in range(self.num_classes)}

        swap_events = []

        for t in range(T):
            # Cost matrix for Hungarian matching between K slots and num_classes
            cost_matrix = np.zeros((K, self.num_classes), dtype=np.float32)
            for k in range(K):
                for c in range(self.num_classes):
                    iou, _ = compute_binary_iou_dice(pred_masks[t, k], gt_masks_dict[c][t], thresh=thresh)
                    cost_matrix[k, c] = -iou

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for slot_idx, class_idx in zip(row_ind, col_ind):
                best_iou = -cost_matrix[slot_idx, class_idx]
                _, best_dice = compute_binary_iou_dice(pred_masks[t, slot_idx], gt_masks_dict[class_idx][t], thresh=thresh)

                slot_assignments[class_idx].append(slot_idx)
                slot_ious[class_idx].append(best_iou)
                slot_dices[class_idx].append(best_dice)

                # Check for Slot Swap Event — only meaningful when the class
                # was visible (non-empty GT mask) in both the current and the
                # previous frame, and had a valid assignment at t-1.
                if (t > 0
                        and len(slot_assignments[class_idx]) >= 2
                        and (gt_masks_dict[class_idx][t].sum() > 0)
                        and (gt_masks_dict[class_idx][t - 1].sum() > 0)):
                    prev_slot = slot_assignments[class_idx][-2]
                    if prev_slot != slot_idx:
                        swap_events.append({
                            'frame': t,
                            'class_id': class_idx,
                            'class_name': self.class_names.get(class_idx, f"Class {class_idx}"),
                            'from_slot': prev_slot,
                            'to_slot': slot_idx
                        })

        metrics = {
            'total_frames': T,
            'total_swap_events': len(swap_events),
            'swap_rate_per_100_frames': (len(swap_events) / T) * 100.0 if T > 0 else 0.0,
            'class_metrics': {}
        }

        all_ious = []
        all_dices = []

        for c in range(self.num_classes):
            mean_iou = float(np.mean(slot_ious[c])) if len(slot_ious[c]) > 0 else 0.0
            mean_dice = float(np.mean(slot_dices[c])) if len(slot_dices[c]) > 0 else 0.0
            all_ious.append(mean_iou)
            all_dices.append(mean_dice)

            metrics['class_metrics'][self.class_names.get(c, str(c))] = {
                'mean_iou': mean_iou,
                'mean_dice': mean_dice
            }

        metrics['overall_mIoU'] = float(np.mean(all_ious))
        metrics['overall_mDice'] = float(np.mean(all_dices))

        return metrics


@typechecked
def greedy_slot_assignments(
    pred_masks: TensorType["B", "T", "K", "H", "W"],
    gt_masks: TensorType["B", "T", "M", "H", "W"],
    thresh=0.5,
):
    """
    Greedy per-slot argmax assignment + swap tracking — the canonical swap metric.

    Each predicted slot is assigned the GT slot with max IoU per frame; a swap
    is an assignment change between consecutive frames. All computation on
    torch tensors.

    Args:
        pred_masks: [B, T, Kp, H, W] float
        gt_masks: [B, T, Kg, H, W] float
        thresh: binarization threshold (applied to both)

    Returns dict:
        'swap_transitions': int — transitions with a changed assignment
        'total_transitions': int — transitions evaluated
        'seq_records': list per batch item of {'swapped': bool, 'swap_count': int}
        'frame_swaps': {t: [swaps, total]} — transitions counted entering frame t
        'assignments': [B, T, Kp] long tensor of assigned GT slot indices
        'iou_matrices': [B, T, Kp, Kg] float tensor

    The annotations own the shape contract: same-named dims (B, T, H, W)
    are cross-checked between the two arguments by the typechecker.
    """
    B, T = pred_masks.shape[:2]
    p_bin = (pred_masks > thresh).float()
    g_bin = (gt_masks > thresh).float()
    Kp, Kg = p_bin.shape[2], g_bin.shape[2]

    inter_mat = torch.einsum('btkhw,btmhw->btkm', p_bin, g_bin)  # [B, T, Kp, Kg]
    p_area = p_bin.sum(dim=(-2, -1))  # [B, T, Kp]
    g_area = g_bin.sum(dim=(-2, -1))  # [B, T, Kg]
    iou_matrices = (inter_mat + 1e-6) / (p_area.unsqueeze(-1) + g_area.unsqueeze(-2) - inter_mat + 1e-6)

    assignments = torch.argmax(iou_matrices, dim=-1)  # [B, T, Kp]

    changed = (assignments[:, 1:] != assignments[:, :-1]).any(dim=-1)  # [B, T-1]
    total_transitions = int(changed.numel())
    swap_transitions = int(changed.sum().item())

    frame_swaps = {}
    for t in range(1, T):
        frame_swaps[t] = [int(changed[:, t - 1].sum().item()), B]

    seq_records = []
    for b in range(B):
        seq_changed = changed[b]
        seq_records.append({
            'swapped': bool(seq_changed.any().item()),
            'swap_count': int(seq_changed.sum().item()),
        })

    return {
        'swap_transitions': swap_transitions,
        'total_transitions': total_transitions,
        'seq_records': seq_records,
        'frame_swaps': frame_swaps,
        'assignments': assignments,
        'iou_matrices': iou_matrices,
    }


def _distribution_stats(values):
    """Distribution stats of a list of floats, computed on torch."""
    if not values:
        return {}
    t = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.25, 0.75], dtype=torch.float64))
    return {
        'mean': float(t.mean()),
        'std': float(t.std()),
        'median': float(t.median()),
        'q25': float(q[0]),
        'q75': float(q[1]),
        'min': float(t.min()),
        'max': float(t.max()),
    }


def _batch_meta(x, b_idx):
    """Normalize dataset batch metadata (tensor or list) to a scalar."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return int(x[b_idx].item())
    return x[b_idx]


class DeterministicEvaluator:
    """
    Batch-level deterministic evaluation — the canonical eval path.

    Accumulates per-slot IoU/Dice, per-frame mIoU/mDice/MSE, greedy swap
    tracking, and per-sequence records; ``finalize()`` returns raw-fraction
    aggregates. All computation on torch tensors — scripts own presentation
    and serialization.
    """

    def __init__(self, num_classes=3, slot_names=None, thresh=0.5):
        self.num_classes = num_classes
        self.slot_names = slot_names or {k: f"Slot {k}" for k in range(num_classes)}
        self.thresh = thresh
        self._reset()

    def _reset(self):
        self.mses = []   # per-batch mean MSE
        self.mious = []  # per-sequence mean IoU
        self.mdices = []  # per-sequence mean Dice
        self.cls_ious = {k: [] for k in range(self.num_classes)}
        self.cls_dices = {k: [] for k in range(self.num_classes)}
        self.frame_mses = {}
        self.frame_mious = {}
        self.frame_mdices = {}
        self.frame_swaps = {}
        self.per_sequence_records = []
        self.total_transitions = 0
        self.swap_transitions = 0
        self.swapped_sequences = 0
        self.total_sequences = 0
        self.clip_count = 0

    def update(self, pred_masks, gt_masks, recon=None, video=None,
               episode_idx=None, start_frame=None, clip_idx=None):
        """
        Accumulate one batch.

        pred_masks/gt_masks: [B, T, K, H, W]; recon/video: [B, T, C, H, W]
        (MSE); episode_idx/start_frame/clip_idx: batch metadata (tensor or
        list) — episode_idx/start_frame come from the dataset batch and are
        the source of truth for per-sequence records.
        """
        # Validate inputs before any computation (annotations handle the
        # per-argument tensor checks; cross-argument relations stay inline).
        for name, t in (("pred_masks", pred_masks), ("gt_masks", gt_masks),
                        ("recon", recon), ("video", video)):
            if t is not None and (not torch.is_tensor(t) or t.ndim != 5):
                raise ValueError(
                    f"{name} must be a 5-dimensional tensor, "
                    f"got {type(t).__name__ if not torch.is_tensor(t) else tuple(t.shape)}")
        if recon is not None and video is not None and recon.shape[:2] != video.shape[:2]:
            raise ValueError(
                f"recon and video must share (B, T), got "
                f"{tuple(recon.shape[:2])} vs {tuple(video.shape[:2])}")
        if pred_masks is not None and gt_masks is not None \
                and pred_masks.shape[:2] != gt_masks.shape[:2]:
            raise ValueError(
                f"pred_masks and gt_masks must share (B, T), got "
                f"{tuple(pred_masks.shape[:2])} vs {tuple(gt_masks.shape[:2])}")

        mse_per_seq = None
        if recon is not None and video is not None:
            mse_per_seq = torch.mean((recon - video) ** 2, dim=(2, 3, 4))  # [B, T]
            self.mses.append(mse_per_seq.mean().item())
            for t in range(video.shape[1]):
                self.frame_mses.setdefault(t, []).extend(mse_per_seq[:, t].tolist())

        if pred_masks is None or gt_masks is None:
            self.clip_count += video.shape[0] if video is not None else 0
            return

        B, T = pred_masks.shape[:2]

        min_k = min(pred_masks.shape[2], gt_masks.shape[2])
        p_sub = pred_masks[:, :, :min_k]
        g_sub = gt_masks[:, :, :min_k]

        p_bin = (p_sub > self.thresh).float()
        g_bin = (g_sub > self.thresh).float()

        # Per-slot IoU / Dice.
        for k in range(min_k):
            inter_k = (p_bin[:, :, k] * g_bin[:, :, k]).sum(dim=(-2, -1))
            p_sum = p_bin[:, :, k].sum(dim=(-2, -1))
            g_sum = g_bin[:, :, k].sum(dim=(-2, -1))
            iou_k = (inter_k + 1e-6) / (p_sum + g_sum - inter_k + 1e-6)
            dice_k = (2.0 * inter_k + 1e-6) / (p_sum + g_sum + 1e-6)
            self.cls_ious[k].extend(iou_k.reshape(-1).tolist())
            self.cls_dices[k].extend(dice_k.reshape(-1).tolist())

        intersection = (p_bin * g_bin).sum(dim=(-2, -1))  # [B, T, min_k]
        union = (p_bin + g_bin).sum(dim=(-2, -1)) - intersection
        iou_seq_frame = (intersection + 1e-6) / (union + 1e-6)
        dice_seq_frame = (2.0 * intersection + 1e-6) / \
            (p_bin.sum(dim=(-2, -1)) + g_bin.sum(dim=(-2, -1)) + 1e-6)
        miou_per_seq_frame = iou_seq_frame.mean(dim=-1)  # [B, T]
        mdice_per_seq_frame = dice_seq_frame.mean(dim=-1)

        self.mious.extend(miou_per_seq_frame.mean(dim=-1).tolist())
        self.mdices.extend(mdice_per_seq_frame.mean(dim=-1).tolist())
        for t in range(T):
            self.frame_mious.setdefault(t, []).extend(miou_per_seq_frame[:, t].tolist())
            self.frame_mdices.setdefault(t, []).extend(mdice_per_seq_frame[:, t].tolist())

        # Greedy swap tracking.
        swaps = greedy_slot_assignments(p_bin, g_bin, thresh=0.0)  # already binarized
        self.total_transitions += swaps['total_transitions']
        self.swap_transitions += swaps['swap_transitions']
        self.swapped_sequences += sum(1 for r in swaps['seq_records'] if r['swapped'])
        for t, (sw, tot) in swaps['frame_swaps'].items():
            entry = self.frame_swaps.setdefault(t, [0, 0])
            entry[0] += sw
            entry[1] += tot

        # Per-sequence records.
        for b in range(B):
            self.total_sequences += 1
            record = swaps['seq_records'][b]
            self.per_sequence_records.append({
                'clip_idx': _batch_meta(clip_idx, b) if clip_idx is not None else self.clip_count + b,
                'episode_idx': _batch_meta(episode_idx, b) if episode_idx is not None else self.clip_count + b,
                'start_frame': _batch_meta(start_frame, b) if start_frame is not None else 0,
                'mse': float(mse_per_seq[b].mean().item()) if mse_per_seq is not None else float('nan'),
                'miou': float(miou_per_seq_frame[b].mean().item()),
                'mdice': float(mdice_per_seq_frame[b].mean().item()),
                'slot_ious': {k: float(iou_seq_frame[b, :, k].mean().item()) for k in range(min_k)},
                'swapped': record['swapped'],
                'swap_count': record['swap_count'],
            })
        self.clip_count += B

    def running_miou_mean(self):
        """Current mean mIoU over accumulated sequences (for progress prints)."""
        return float(torch.tensor(self.mious).mean()) if self.mious else 0.0

    def finalize(self):
        """Return raw-fraction aggregates (no percentage conversion)."""
        slot_swap_rate = (self.swap_transitions / self.total_transitions) if self.total_transitions > 0 else 0.0
        seq_swap_rate = (self.swapped_sequences / self.total_sequences) if self.total_sequences > 0 else 0.0

        per_slot = {}
        for k in range(self.num_classes):
            per_slot[f"slot_{k}"] = {
                'name': self.slot_names.get(k, f"Slot {k}"),
                'iou': _distribution_stats(self.cls_ious.get(k, [])),
                'dice': _distribution_stats(self.cls_dices.get(k, [])),
            }

        per_frame = []
        for t in sorted(self.frame_mious):
            swaps, total_t = self.frame_swaps.get(t, [0, 0])
            per_frame.append({
                'frame_idx': t + 1,
                'mse_mean': float(torch.tensor(self.frame_mses[t]).mean()) if t in self.frame_mses else float('nan'),
                'miou_mean': float(torch.tensor(self.frame_mious[t]).mean()) if t in self.frame_mious else 0.0,
                'mdice_mean': float(torch.tensor(self.frame_mdices[t]).mean()) if t in self.frame_mdices else 0.0,
                'swap_rate': (swaps / total_t) if total_t > 0 else 0.0,
            })

        return {
            'summary': {
                'val_mse': _distribution_stats(self.mses),
                'miou': _distribution_stats(self.mious),
                'mdice': _distribution_stats(self.mdices),
                'slot_swapping_rate': slot_swap_rate,
                'sequence_swapping_rate': seq_swap_rate,
                'total_sequences': self.total_sequences,
            },
            'per_slot': per_slot,
            'per_frame': per_frame,
            'per_sequence': self.per_sequence_records,
        }
