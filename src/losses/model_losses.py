"""
DETR Loss Computation Module.

Provides ``compute_detr_loss`` — the loss function for DETR and Deformable DETR
model families, using Hungarian bipartite matching (``SetCriterion``).

Supports:
  - Single-layer logits [B, Q, C] (backward-compatible, no aux losses)
  - Multi-layer logits [B, L, Q, C] (aux decoding losses, per the DETR paper)
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_detr_loss(
    out: dict,
    gt_masks,
    criterion,
    weight_dict: dict,
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute DETR Hungarian matching loss with auxiliary decoding losses.

    Supports 3-D logits ``[B, Q, C]`` (single layer, backward-compatible) and
    4-D logits ``[B, L, Q, C]`` (multi-layer with aux losses per DETR paper).

    Args:
        out (dict): Model output with keys ``"pred_logits"`` and
            ``"pred_boxes"``.  Multi-layer variants have shape
            ``[B, L, Q, C]`` / ``[B, L, Q, 4]``; single-layer are ``[B, Q, C]``
            / ``[B, Q, 4]``.
        gt_masks (Tensor or None): Ground-truth masks
            ``[B, C, H, W]`` or ``[B, T, C, H, W]``.
            If ``None``, training labels are all background (useful for
            unsupervised / mask-free pretraining).
        criterion (SetCriterion): Pre-initialized DETR loss criterion.
        weight_dict (dict): Per-loss-term weights.  For aux layers, keys like
            ``"class_aux"``, ``"bbox_aux"``, ``"giou_aux"`` override the main
            weights.

    Returns:
        total_loss (Tensor): Weighted scalar loss.
        loss_dict (dict): Per-term loss values and ``"total_loss"``.
    """
    from src.models.detr import masks_to_boxes_and_labels

    pred_logits = out["pred_logits"]
    pred_boxes = out["pred_boxes"]

    if gt_masks is None:
        raise ValueError("gt_masks is required for training DETR")

    # The DETR wrapper flattens temporal inputs [B, T, C, H, W] → [B*T, C, H, W].
    # The pred_logits dimension reflects this B*T batching.
    # To match, we replicate targets for each timestep.
    pred_logits = out["pred_logits"]  # [B*T, Q, C] or [B*T, L, Q, C]
    effective_batch = pred_logits.shape[0]  # B*T

    if gt_masks.ndim == 5:
        # [B, T, C, H, W] → flatten to [B*T, C, H, W]
        B, T, C, H, W = gt_masks.shape
        gt_masks_2d = gt_masks.reshape(B * T, C, H, W)
    elif gt_masks.ndim == 4:
        # [B, C, H, W] — check if we need replication
        B_gt = gt_masks.shape[0]
        if effective_batch != B_gt and effective_batch % B_gt == 0:
            T = effective_batch // B_gt
            gt_masks_2d = gt_masks.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(effective_batch, *gt_masks.shape[1:])
        else:
            gt_masks_2d = gt_masks
    else:
        gt_masks_2d = gt_masks

    targets = masks_to_boxes_and_labels(gt_masks_2d)

    is_multilayer = pred_logits.ndim == 4  # [B, L, Q, C]

    loss_dict: dict[str, float] = {}
    terms: list[Tensor] = []

    if is_multilayer:
        # ── Multi-layer (aux losses per DETR paper) ───────────────────────
        num_layers = pred_logits.shape[1]

        for layer_idx in range(num_layers):
            layer_out = {
                "pred_logits": pred_logits[:, layer_idx],
                "pred_boxes": pred_boxes[:, layer_idx],
            }
            layer_losses = criterion(layer_out, targets)

            is_last = layer_idx == num_layers - 1
            suffix = "" if is_last else f"_aux_{layer_idx}"
            # Determine weight key prefix
            w_prefix = "" if is_last else "_aux"

            for k, v in layer_losses.items():
                # Map criterion key → weight_dict key
                # criterion emits: 'loss_ce', 'loss_bbox', 'loss_giou'
                base = k.replace("loss_", "")  # 'ce' → class, 'bbox', 'giou'
                base = base.replace("ce", "class")
                w_key = f"{base}{w_prefix}" if not is_last else base
                w = weight_dict.get(w_key, weight_dict.get(base, 1.0))

                log_key = k if is_last else k + suffix
                terms.append(w * v)
                loss_dict[log_key] = v.item()
    else:
        # ── Single-layer (no aux losses) ──────────────────────────────────
        layer_losses = criterion(
            {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
            targets,
        )
        for k, v in layer_losses.items():
            base = k.replace("loss_", "").replace("ce", "class")
            w = weight_dict.get(base, 1.0)
            terms.append(w * v)
            loss_dict[k] = v.item()

    total_loss: Tensor = sum(terms)  # type: ignore[assignment]
    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict
