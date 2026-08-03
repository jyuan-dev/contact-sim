"""
Model-specific loss functions for DETR and SAVi.

Provides:
  - compute_detr_loss: Hungarian-matching classification + box regression loss.
  - compute_savi_loss: Slot attention mask BCE + reconstruction MSE loss.
"""

import torch
import torch.nn.functional as F

from src.models.detr import masks_to_boxes_and_labels


def compute_detr_loss(out, gt_masks, criterion, weight_dict):
    """
    Computes DETR Hungarian matching loss (classification CE, box L1, box GIoU)
    using a pre-initialized SetCriterion.

    Args:
        out (dict): Model output with keys 'pred_logits' and 'pred_boxes'.
        gt_masks (Tensor): Ground-truth masks [B, C, H, W] or [B, T, C, H, W].
        criterion (SetCriterion): Pre-initialized DETR loss criterion.
        weight_dict (dict): Per-loss-term weights, e.g. {'class': 1.0, 'bbox': 5.0, 'giou': 2.0}.

    Returns:
        total_loss (Tensor): Weighted scalar loss.
        loss_dict (dict): Per-term loss values and 'total_loss'.
    """
    pred_logits = out['pred_logits']
    pred_boxes = out['pred_boxes']

    if gt_masks is None:
        raise ValueError("gt_masks is required for training DETR")

    if gt_masks.ndim == 5:
        B, T, C, H, W = gt_masks.shape
        gt_masks_flat = gt_masks.view(B * T, C, H, W)
    else:
        gt_masks_flat = gt_masks

    targets = masks_to_boxes_and_labels(gt_masks_flat)

    detr_out = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
    losses = criterion(detr_out, targets)

    total_loss = torch.tensor(0.0, device=pred_logits.device)
    loss_dict = {}
    for k, v in losses.items():
        weight_key = k.replace('loss_', '')
        if weight_key == 'ce':
            weight_key = 'class'
        w = weight_dict.get(weight_key, 1.0)
        total_loss += w * v
        loss_dict[k] = v.item()

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


def compute_savi_loss(raw_out, gt_masks, weight_dict=None):
    """
    Computes SAVi slot attention mask BCE + reconstruction MSE loss.

    Args:
        raw_out (dict): Model output with optional keys 'recon_combined', 'post_masks', 'img'.
        gt_masks (Tensor or None): Ground-truth masks [B, T, C, H, W].
        weight_dict (dict): Per-term weights, e.g. {'recon': 1.0, 'mask': 1.0, 'kld': 0.001}.

    Returns:
        total_loss (Tensor or float): Weighted scalar loss.
        loss_dict (dict): Per-term loss values and 'total_loss'.
    """
    if weight_dict is None:
        weight_dict = {'recon': 1.0, 'mask': 1.0, 'kld': 0.001}

    recon_img = raw_out.get('recon_combined', None)
    post_masks = raw_out.get('post_masks', None)

    total_loss = 0.0
    loss_dict = {}

    if recon_img is not None and 'img' in raw_out:
        target_img = raw_out['img']
        recon_loss = F.mse_loss(recon_img, target_img)
        total_loss += weight_dict.get('recon', 1.0) * recon_loss
        loss_dict['recon_loss'] = recon_loss.item()

    if post_masks is not None and gt_masks is not None:
        p_masks = post_masks.squeeze(3) if post_masks.ndim == 6 else post_masks
        if gt_masks.ndim == 5:
            B, T, C, H, W = gt_masks.shape
            if p_masks.shape[-2:] != (H, W):
                p_masks = F.interpolate(
                    p_masks.view(B * T, -1, p_masks.shape[-2], p_masks.shape[-1]),
                    size=(H, W), mode='bilinear', align_corners=False
                ).view(B, T, -1, H, W)

            mask_bce = F.binary_cross_entropy(
                torch.clamp(p_masks.max(dim=2)[0], 1e-4, 1 - 1e-4),
                (gt_masks.max(dim=2)[0] > 0.5).float()
            )
            total_loss += weight_dict.get('mask', 1.0) * mask_bce
            loss_dict['mask_bce'] = mask_bce.item()

    loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    return total_loss, loss_dict
