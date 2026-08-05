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
    Computes DETR Hungarian matching loss with auxiliary decoding losses.

    Supports 3D logits [B, Q, C] (single layer, backward-compat) and 4D logits
    [B, L, Q, C] (multi-layer with aux losses per DETR paper).

    Args:
        out (dict): Model output with keys 'pred_logits' and 'pred_boxes'.
        gt_masks (Tensor): Ground-truth masks [B, C, H, W] or [B, T, C, H, W].
        criterion (SetCriterion): Pre-initialized DETR loss criterion.
        weight_dict (dict): Per-loss-term weights. For aux layers, keys like
            'class_aux', 'bbox_aux', 'giou_aux' override the main weights.

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
        # Map criterion keys to weight-dict keys: 'loss_ce_aux_0' → 'class_aux'
        #                                       'loss_ce'       → 'class'
        base = k.replace('loss_', '')  # e.g. 'ce_aux_0', 'ce', 'bbox', 'giou'
        if base.startswith('ce'):
            base = 'class' + base[2:]  # 'class_aux_0' or 'class'

        if '_aux_' in base:
            # Try specific aux weight first, then generic aux weight, then main weight
            w = weight_dict.get(base, weight_dict.get('class_aux' if 'class' in base else
                                                       'bbox_aux' if 'bbox' in base else
                                                       'giou_aux' if 'giou' in base else
                                                       base.split('_aux_')[0], 1.0))
        else:
            w = weight_dict.get(base, 1.0)

        total_loss += w * v
        loss_dict[k] = v.item()

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


def compute_savi_loss(out, gt_masks, weight_dict=None):
    """
    Computes SAVi slot attention mask BCE + reconstruction MSE loss.

    Consumes standardized contract keys (recon_img, pred_masks, input_img).

    Args:
        out (dict): Standardized model output with keys 'recon_img', 'pred_masks', 'input_img'.
        gt_masks (Tensor or None): Ground-truth masks [B, T, C, H, W].
        weight_dict (dict): Per-term weights, e.g. {'recon': 1.0, 'mask': 1.0, 'kld': 0.001}.

    Returns:
        total_loss (Tensor): Weighted scalar loss.
        loss_dict (dict): Per-term loss values and 'total_loss'.
    """
    if weight_dict is None:
        weight_dict = {'recon': 1.0, 'mask': 1.0}

    recon_img = out.get('recon_img')
    post_masks = out.get('pred_masks')
    input_img = out.get('input_img')

    terms = []
    loss_dict = {}

    if recon_img is not None and input_img is not None:
        recon_loss = F.mse_loss(recon_img, input_img)
        w = weight_dict.get('recon', 1.0)
        terms.append(w * recon_loss)
        loss_dict['recon_loss'] = recon_loss.item()

    if post_masks is not None and gt_masks is not None:
        if post_masks.ndim == 6 and post_masks.shape[3] == 1:
            post_masks = post_masks.squeeze(3)  # [B, T, S, H, W]

        if gt_masks.ndim == 5:
            B, T, C, H, W = gt_masks.shape
            if post_masks.shape[-2:] != (H, W):
                post_masks = F.interpolate(
                    post_masks.view(B * T, -1, post_masks.shape[-2], post_masks.shape[-1]),
                    size=(H, W), mode='bilinear', align_corners=False,
                ).view(B, T, -1, H, W)

            # Reshape for Hungarian bipartite matching across timesteps
            # pred_m: [B*T, S, H, W], gt_m: [B*T, C, H, W]
            pred_m = post_masks.view(B * T, post_masks.shape[2], H, W)
            gt_m = gt_masks.to(pred_m.device).view(B * T, C, H, W)
            N_tot, S_slots = pred_m.shape[:2]
            eps = 1e-6

            # Check matching mode: 'fixed' (1-to-1 class index assignment) vs 'hungarian' (bipartite dynamic assignment)
            match_mode = weight_dict.get('match_mode', 'hungarian') if weight_dict else 'hungarian'

            if match_mode == 'fixed':
                # Option B: Direct 1-to-1 correspondence (Slot k strictly matched to GT Channel k)
                num_matched = min(S_slots, C)
                with torch.amp.autocast('cuda', enabled=False):
                    p_matched = pred_m[:, :num_matched].reshape(-1, H, W).float()
                    p_matched = torch.nan_to_num(p_matched, nan=eps, posinf=1.0 - eps, neginf=eps)
                    p_matched = p_matched.clamp(eps, 1.0 - eps)

                    g_matched = (gt_m[:, :num_matched].to(p_matched.device).reshape(-1, H, W) > 0.5).float()

                    mask_bce = F.binary_cross_entropy(p_matched, g_matched)

                    p_flat_m = p_matched.flatten(1)
                    g_flat_m = g_matched.flatten(1)
                    num = 2.0 * (p_flat_m * g_flat_m).sum(dim=-1) + eps
                    den = p_flat_m.sum(dim=-1) + g_flat_m.sum(dim=-1) + eps
                    mask_dice = (1.0 - num / den).mean()
            else:
                # 1. Hungarian Matching per batch-timestep
                with torch.no_grad():
                    pred_flat = pred_m.flatten(2).float() # [N, S, HW]
                    gt_flat = (gt_m.flatten(2) > 0.5).float() # [N, C, HW]
                    
                    # BCE Cost
                    p_clamped = pred_flat.clamp(eps, 1.0 - eps)
                    bce_cost = - (gt_flat.unsqueeze(1) * torch.log(p_clamped.unsqueeze(2)) +
                                  (1.0 - gt_flat.unsqueeze(1)) * torch.log(1.0 - p_clamped.unsqueeze(2))).mean(dim=-1) # [N, S, C]
                    
                    # Dice Cost
                    inter = torch.einsum('nsh,nch->nsc', pred_flat, gt_flat)
                    card = pred_flat.sum(dim=-1, keepdim=True) + gt_flat.sum(dim=-1).unsqueeze(1)
                    dice_cost = 1.0 - (2.0 * inter + eps) / (card + eps) # [N, S, C]

                    cost_matrix = (bce_cost + dice_cost).cpu().numpy()
                    import numpy as np
                    cost_matrix = np.nan_to_num(cost_matrix, nan=1e5, posinf=1e5, neginf=-1e5)

                    from scipy.optimize import linear_sum_assignment

                    matched_src, matched_tgt = [], []
                    for i in range(N_tot):
                        r_idx, c_idx = linear_sum_assignment(cost_matrix[i])
                        matched_src.append(torch.tensor(r_idx, device=pred_m.device) + i * S_slots)
                        matched_tgt.append(torch.tensor(c_idx, device=pred_m.device) + i * C)

                    matched_src = torch.cat(matched_src)
                    matched_tgt = torch.cat(matched_tgt)

                # 2. Compute BCE and Dice loss on matched slot-target pairs
                with torch.amp.autocast('cuda', enabled=False):
                    p_matched = pred_m.reshape(-1, H, W)[matched_src].float()
                    p_matched = torch.nan_to_num(p_matched, nan=eps, posinf=1.0 - eps, neginf=eps)
                    p_matched = p_matched.clamp(eps, 1.0 - eps)

                    g_matched = (gt_m.reshape(-1, H, W)[matched_tgt] > 0.5).float()

                    mask_bce = F.binary_cross_entropy(p_matched, g_matched)

                    p_flat_m = p_matched.flatten(1)
                    g_flat_m = g_matched.flatten(1)
                    num = 2.0 * (p_flat_m * g_flat_m).sum(dim=-1) + eps
                    den = p_flat_m.sum(dim=-1) + g_flat_m.sum(dim=-1) + eps
                    mask_dice = (1.0 - num / den).mean()

            mask_loss = mask_bce + mask_dice
            w = weight_dict.get('mask', 1.0)
            terms.append(w * mask_loss)
            loss_dict['mask_bce'] = mask_bce.item()
            loss_dict['mask_dice'] = mask_dice.item()


    total_loss = sum(terms) if terms else torch.tensor(0.0, device=post_masks.device if post_masks is not None else
                                                       recon_img.device if recon_img is not None else
                                                       torch.device('cpu'))
    loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    return total_loss, loss_dict

