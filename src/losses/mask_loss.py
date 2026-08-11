"""
Mask Segmentation Loss Module (BCE + Dice).
Supports 1-to-1 Fixed Channel Matching and Hungarian Bipartite Matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskSegmentationLoss(nn.Module):
    """
    Mask Segmentation Loss combining Binary Cross-Entropy (BCE) and Dice Loss.
    """

    def __init__(self, weight: float = 1.0, match_mode: str = "fixed"):
        super().__init__()
        self.weight = weight
        self.match_mode = match_mode

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        post_masks = out.get("pred_masks")
        gt_masks = batch.get("gt_masks") if isinstance(batch, dict) else batch

        if post_masks is None or gt_masks is None:
            device = post_masks.device if post_masks is not None else "cpu"
            return torch.tensor(0.0, device=device), {"mask_bce": 0.0, "mask_dice": 0.0}

        if post_masks.ndim == 6 and post_masks.shape[3] == 1:
            post_masks = post_masks.squeeze(3)  # [B, T, S, H, W]

        if gt_masks.ndim == 5:
            B, T, C, H, W = gt_masks.shape
            if post_masks.shape[-2:] != (H, W):
                post_masks = F.interpolate(
                    post_masks.view(B * T, -1, post_masks.shape[-2], post_masks.shape[-1]),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).view(B, T, -1, H, W)

            pred_m = post_masks.view(B * T, post_masks.shape[2], H, W)
            gt_m = gt_masks.to(pred_m.device).view(B * T, C, H, W)
            N_tot, S_slots = pred_m.shape[:2]
            eps = 1e-6

            if self.match_mode == "fixed":
                num_matched = min(S_slots, C)
                p_matched = pred_m[:, :num_matched].reshape(-1, H, W).float()
                p_matched = torch.nan_to_num(p_matched, nan=eps).clamp(eps, 1.0 - eps)
                g_matched = (gt_m[:, :num_matched].to(p_matched.device).reshape(-1, H, W) > 0.5).float()

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    mask_bce = F.binary_cross_entropy(p_matched, g_matched)

                p_flat_m = p_matched.flatten(1)
                g_flat_m = g_matched.flatten(1)
                num = 2.0 * (p_flat_m * g_flat_m).sum(dim=-1) + eps
                den = p_flat_m.sum(dim=-1) + g_flat_m.sum(dim=-1) + eps
                mask_dice = (1.0 - num / den).mean()
            else:
                with torch.no_grad():
                    pred_flat = pred_m.flatten(2).float()
                    gt_flat = (gt_m.flatten(2) > 0.5).float()

                    p_clamped = pred_flat.clamp(eps, 1.0 - eps)
                    bce_cost = -(
                        gt_flat.unsqueeze(1) * torch.log(p_clamped.unsqueeze(2))
                        + (1.0 - gt_flat.unsqueeze(1)) * torch.log(1.0 - p_clamped.unsqueeze(2))
                    ).mean(dim=-1)

                    inter = torch.einsum("nsh,nch->nsc", pred_flat, gt_flat)
                    card = pred_flat.sum(dim=-1, keepdim=True) + gt_flat.sum(dim=-1).unsqueeze(1)
                    dice_cost = 1.0 - (2.0 * inter + eps) / (card + eps)

                    cost_matrix = (bce_cost + dice_cost).cpu().numpy()
                    from scipy.optimize import linear_sum_assignment

                    matched_src, matched_tgt = [], []
                    for i in range(N_tot):
                        r_idx, c_idx = linear_sum_assignment(cost_matrix[i])
                        matched_src.append(torch.tensor(r_idx, device=pred_m.device) + i * S_slots)
                        matched_tgt.append(torch.tensor(c_idx, device=pred_m.device) + i * C)

                    matched_src = torch.cat(matched_src)
                    matched_tgt = torch.cat(matched_tgt)

                p_matched = pred_m.reshape(-1, H, W)[matched_src].float()
                p_matched = p_matched.clamp(eps, 1.0 - eps)
                g_matched = (gt_m.reshape(-1, H, W)[matched_tgt] > 0.5).float()

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    mask_bce = F.binary_cross_entropy(p_matched, g_matched)

                p_flat_m = p_matched.flatten(1)
                g_flat_m = g_matched.flatten(1)
                num = 2.0 * (p_flat_m * g_flat_m).sum(dim=-1) + eps
                den = p_flat_m.sum(dim=-1) + g_flat_m.sum(dim=-1) + eps
                mask_dice = (1.0 - num / den).mean()

            mask_loss = mask_bce + mask_dice
            weighted_loss = self.weight * mask_loss
            return weighted_loss, {"mask_bce": mask_bce.item(), "mask_dice": mask_dice.item()}

        return torch.tensor(0.0, device=post_masks.device), {"mask_bce": 0.0, "mask_dice": 0.0}
