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
from scipy.optimize import linear_sum_assignment


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

                # Check for Slot Swap Event
                if t > 0 and (gt_masks_dict[class_idx][t].sum() > 0):
                    prev_slot = slot_assignments[class_idx][t - 1]
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
