"""
Standardized Deformable DETR wrapper.

Wraps ``DeformableDETR`` to conform to the ``BaseModelWrapper`` interface.
Reuses the same DETR criterion / matcher stack (``HungarianMatcher`` +
``SetCriterion``) as the standard DETR wrapper.

Registered under: ``"deformable_detr"``
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.models.base import BaseModelWrapper
from src.models.factory import register_model
from src.models.model_output import ModelOutput
from src.models.wrappers.detr_wrapper import _flatten_video, StandardizedDETRWrapper


@register_model(["deformable_detr", "deformable-detr"])
class StandardizedDeformableDETRWrapper(StandardizedDETRWrapper):
    """
    Deformable DETR wrapper conforming to ``BaseModelWrapper``.

    Inherits ``forward()`` and ``compute_loss()`` from
    ``StandardizedDETRWrapper`` — the only difference is how the base model
    is constructed in ``build()``.
    """

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedDeformableDETRWrapper":
        """Construct from model config sub-dict."""
        from src.models.deformable_detr import DeformableDETR
        from src.models.detr import HungarianMatcher, SetCriterion

        base_model = DeformableDETR(
            num_classes=model_cfg.get("num_classes", 3),
            num_queries=model_cfg.get("num_queries", 10),
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 3),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
            backbone_name=model_cfg.get("backbone_name", "resnet18"),
            train_backbone=model_cfg.get("train_backbone", True),
        )

        w_dict = dict(
            model_cfg.get(
                "weight_dict", {"class": 1.0, "bbox": 5.0, "giou": 2.0}
            )
        )
        # Ensure auxiliary loss weight keys are present
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
