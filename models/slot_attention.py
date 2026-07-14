import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ── 1. StoSAVi Path Setup & Import ────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOTFORMER = os.path.join(REPO_ROOT, 'third_party', 'cjepa', 'src', 'third_party', 'slotformer')
HDF5_DS    = os.path.join(SLOTFORMER, 'base_slots')

for p in [REPO_ROOT, SLOTFORMER, HDF5_DS, os.path.join(SLOTFORMER, 'base_slots', 'models')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from base_slots.models.savi import StoSAVi


# ── 2. DETR Hungarian Matcher for Slot Masks ──────────────────────────────────
class DETRHungarianMatcher(nn.Module):
    """
    Computes bipartite matching between slot predicted masks and ground-truth masks.
    Uses scale-invariant Dice loss and Binary Cross Entropy (BCE) cost.
    """
    def __init__(self, cost_bce: float = 1.0, cost_dice: float = 1.0):
        super().__init__()
        self.cost_bce = cost_bce
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, pred_masks, gt_masks):
        """
        Args:
            pred_masks: Tensor of shape [N, S, H, W] in [0, 1] (predictions)
            gt_masks: Tensor of shape [N, M, H, W] in [0, 1] (ground truth targets)
        Returns:
            A list of size N, containing tuples of (index_i, index_j) where:
                - index_i is the indices of slots chosen for matching (size M)
                - index_j is the indices of matched ground-truth masks (size M, e.g. [0, 1, 2])
        """
        N, S, H, W = pred_masks.shape
        M = gt_masks.shape[1]
        eps = 1e-8

        # Flatten spatial dims to [N, S, HW] and [N, M, HW]
        pred_flat = pred_masks.flatten(2)
        gt_flat = gt_masks.flatten(2)

        # 1. Compute BCE cost: BCE = gt * log((1-p)/p) - log(1-p)
        p_clamped = pred_flat.clamp(eps, 1.0 - eps)
        B = torch.log((1.0 - p_clamped) / p_clamped)  # [N, S, HW]
        A = -torch.log(1.0 - p_clamped)              # [N, S, HW]
        A_sum = A.sum(dim=-1, keepdim=True)          # [N, S, 1]

        # gt_B is [N, S, M]
        gt_B = torch.bmm(B, gt_flat.transpose(1, 2))
        cost_bce = (A_sum + gt_B) / (H * W)          # [N, S, M]

        # 2. Compute Dice cost: 1 - 2 * (p * g).sum() / (p.sum() + g.sum())
        intersection = torch.bmm(pred_flat, gt_flat.transpose(1, 2))  # [N, S, M]
        sum_pred = pred_flat.sum(dim=-1, keepdim=True)                # [N, S, 1]
        sum_gt = gt_flat.sum(dim=-1, keepdim=True).transpose(1, 2)    # [N, 1, M]
        denominator = sum_pred + sum_gt
        cost_dice = 1.0 - (2.0 * intersection) / (denominator + eps)  # [N, S, M]

        # Combined cost matrix
        cost = self.cost_bce * cost_bce + self.cost_dice * cost_dice
        cost = cost.cpu()

        indices = []
        for i in range(N):
            c = cost[i]
            if not torch.isfinite(c).all():
                c = torch.nan_to_num(c, nan=1e5, posinf=1e5, neginf=-1e5)
            row_ind, col_ind = linear_sum_assignment(c.numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices


# ── 3. DETR Mask Loss ─────────────────────────────────────────────────────────
class DETRMaskLoss(nn.Module):
    """
    Computes BCE and Dice loss for matched pairs.
    """
    def __init__(self, weight_bce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred_masks, gt_masks, indices):
        """
        Args:
            pred_masks: [N, S, 1, H, W] in [0, 1]
            gt_masks: [N, M, H, W] in [0, 1]
            indices: List of matching index tuples of length N
        """
        N, S, _, H, W = pred_masks.shape
        M = gt_masks.shape[1]
        eps = 1e-8

        gt_masks = gt_masks.clamp(0.0, 1.0)
        pred_masks = pred_masks[:, :, 0, :, :]  # [N, S, H, W]

        # Collate matched indices across the batch
        batch_idx = []
        slot_idx = []
        gt_idx = []
        for i, (src, tgt) in enumerate(indices):
            batch_idx.append(torch.full_like(src, i))
            slot_idx.append(src)
            gt_idx.append(tgt)

        batch_idx = torch.cat(batch_idx).to(pred_masks.device)
        slot_idx = torch.cat(slot_idx).to(pred_masks.device)
        gt_idx = torch.cat(gt_idx).to(pred_masks.device)

        # Extract matched pairs
        src_masks = pred_masks[batch_idx, slot_idx]  # [N * M, H, W]
        target_masks = gt_masks[batch_idx, gt_idx]   # [N * M, H, W]

        # BCE Loss
        bce_loss = F.binary_cross_entropy(src_masks, target_masks, reduction='mean')

        # Dice Loss
        src_flat = src_masks.flatten(1)
        target_flat = target_masks.flatten(1)
        intersection = (src_flat * target_flat).sum(dim=-1)
        denom = src_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        dice_loss = (1.0 - (2.0 * intersection) / (denom + eps)).mean()

        total_loss = self.weight_bce * bce_loss + self.weight_dice * dice_loss
        return total_loss, bce_loss, dice_loss
