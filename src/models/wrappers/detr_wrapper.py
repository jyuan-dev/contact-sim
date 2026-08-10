"""
Standardized DETR wrapper.

Wraps the base DETR object-detection model to conform to the
``BaseModelWrapper`` interface (forward → ModelOutput, compute_loss,
build classmethod).

Registered under: ``"detr"``
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseModelWrapper
from src.models.factory import register_model
from src.models.model_output import ModelOutput


def _flatten_video(x: Tensor) -> Tensor:
    """Flatten [B, T, C, H, W] → [B*T, C, H, W] if 5-D."""
    if x.ndim == 5:
        B, T, C, H, W = x.shape
        return x.reshape(B * T, C, H, W)
    return x


@register_model("detr")
class StandardizedDETRWrapper(BaseModelWrapper):
    """
    DETR wrapper conforming to ``BaseModelWrapper``.

    The matcher, criterion, and weight dict are all constructed inside
    ``build()``, keeping ``build_model()`` clean.

    Auxiliary losses (per-decoder-layer) are transparently handled via the
    ``pred_logits_all`` / ``pred_boxes_all`` passthrough keys in the output.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        weight_dict: dict,
    ) -> None:
        super().__init__()
        self.model = model
        self._criterion = criterion
        self._weight_dict = weight_dict

    # ------------------------------------------------------------------
    # build() — self-contained constructor from config
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedDETRWrapper":
        """Construct from model config sub-dict."""
        from src.models.detr import (
            DETR,
            ResNetBackbone,
            Transformer,
            HungarianMatcher,
            SetCriterion,
        )

        backbone = ResNetBackbone(
            train_backbone=model_cfg.get("train_backbone", False)
        )
        transformer = Transformer(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 6),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 6),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
        )
        base_model = DETR(
            backbone=backbone,
            transformer=transformer,
            num_classes=model_cfg.get("num_classes", 3),
            num_queries=model_cfg.get("num_queries", 5),
        )

        w_dict = dict(
            model_cfg.get(
                "weight_dict", {"class": 1.0, "bbox": 5.0, "giou": 2.0}
            )
        )
        # Ensure auxiliary loss weight keys are present (defaults = same as main)
        for prefix in ("class", "bbox", "giou"):
            w_dict.setdefault(f"{prefix}_aux", w_dict.get(prefix, 1.0))

        matcher = HungarianMatcher(
            cost_class=w_dict.get("class", 1.0),
            cost_bbox=w_dict.get("bbox", 5.0),
            cost_giou=w_dict.get("giou", 2.0),
        )
        criterion = SetCriterion(
            num_classes=model_cfg.get("num_classes", 3),
            matcher=matcher,
            weight_dict=w_dict,
            eos_coef=model_cfg.get("eos_coef", 0.1),
            losses=["labels", "boxes"],
        )

        return cls(base_model, criterion=criterion, weight_dict=w_dict)

    # ------------------------------------------------------------------
    # forward / compute_loss
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> ModelOutput:
        """Forward pass. Input: [B, C, H, W] or [B, T, C, H, W]."""
        x_in = _flatten_video(x)
        raw_out = self.model(x_in)

        # Extract last decoder layer for standardized contract (eval/metrics).
        # raw_out['pred_logits'] is [B, L, Q, C] (multi-layer) or [B, Q, C].
        pred_logits = raw_out["pred_logits"]
        pred_boxes = raw_out["pred_boxes"]
        if pred_logits.ndim == 4:
            final_logits = pred_logits[:, -1]
            final_boxes = pred_boxes[:, -1]
        else:
            final_logits = pred_logits
            final_boxes = pred_boxes

        return {
            "input_img": x,
            "pred_boxes": final_boxes,
            "pred_logits": final_logits,
            "pred_masks": None,
            "recon_img": None,
            "post_slots": None,
            # Pass full layer-stacked outputs to compute_loss for aux losses.
            "pred_logits_all": pred_logits,
            "pred_boxes_all": pred_boxes,
        }

    def compute_loss(
        self, out: ModelOutput, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute DETR loss using the pre-built criterion (with aux losses)."""
        from src.losses.model_losses import compute_detr_loss

        gt_masks = batch.get("gt_masks") if isinstance(batch, dict) else None
        # Use full layer-stacked outputs for aux losses.
        loss_out = {
            "pred_logits": out.get("pred_logits_all", out["pred_logits"]),
            "pred_boxes": out.get("pred_boxes_all", out["pred_boxes"]),
        }
        return compute_detr_loss(
            loss_out, gt_masks, self._criterion, self._weight_dict
        )
