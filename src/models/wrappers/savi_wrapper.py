"""
Standardized SAVi / StoSAVi wrapper.

Wraps ``SAVi`` (which in turn wraps the third-party ``StoSAVi``) to conform
to the ``BaseModelWrapper`` interface.

Registered under: ``"savi"``, ``"stosavi"``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseModelWrapper, ModelOutput
from src.models.factory import register_model


@register_model(["savi", "stosavi"])
class StandardizedSAViWrapper(BaseModelWrapper):
    """
    SAVi wrapper conforming to ``BaseModelWrapper``.

    Handles:
      - Tensor / dict input normalization
      - Resolution up/down-sampling to the model's configured resolution
      - Output key normalization (post_masks / prior_masks / masks variants)
      - Delegate loss to ``src.losses.savi_loss.compute_savi_loss``
    """

    def __init__(
        self,
        model: nn.Module,
        weight_dict: dict | None = None,
        loss_fn: nn.Module | None = None,
        resolution: tuple = (64, 64),
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self._weight_dict = weight_dict or {"recon": 1.0, "mask": 1.0}
        self.resolution = resolution

    # ------------------------------------------------------------------
    # build() — self-contained constructor from config
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedSAViWrapper":
        """Construct from model config sub-dict."""
        from src.models.savi import SAVi

        res = tuple(model_cfg.get("resolution", [64, 64]))
        clip_len = model_cfg.get("n_sample_frames", model_cfg.get("clip_len", 6))

        base_model = SAVi(
            resolution=res,
            clip_len=clip_len,
            num_slots=model_cfg.get("num_slots", 4),
            slot_dim=model_cfg.get("slot_dim", 64),
            num_iterations=model_cfg.get("num_iterations", 3),
            in_channels=model_cfg.get("in_channels", 3),
            slot_dict=model_cfg.get("slot_dict"),
            enc_dict=model_cfg.get("enc_dict"),
            dec_dict=model_cfg.get("dec_dict"),
            pred_dict=model_cfg.get("pred_dict"),
            loss_dict=model_cfg.get("loss_dict"),
        )

        weight_dict = dict(model_cfg.get("weight_dict") or {})
        weight_dict.setdefault("recon", 1.0)
        weight_dict.setdefault("mask", 1.0)

        loss_fn = model_cfg.get("loss_fn")

        return cls(base_model, weight_dict=weight_dict, loss_fn=loss_fn, resolution=res)

    # ------------------------------------------------------------------
    # forward / compute_loss
    # ------------------------------------------------------------------

    def forward(self, x) -> ModelOutput:
        """
        Forward pass.

        Args:
            x: ``Tensor [B, T, C, H, W]`` or ``dict {"img": Tensor, ...}``.
        """
        if isinstance(x, torch.Tensor):
            x = {"img": x}

        img_tensor: Tensor = x["img"]
        target_size = self.resolution
        if img_tensor.shape[-2:] != target_size:
            B, T, C, H, W = img_tensor.shape
            img_tensor = F.interpolate(
                img_tensor.view(B * T, C, H, W),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).view(B, T, C, target_size[0], target_size[1])
            x = dict(x, img=img_tensor)

        raw_out = self.model(x)

        post_masks = raw_out.get(
            "post_masks", raw_out.get("masks", raw_out.get("prior_masks"))
        )
        if post_masks is not None and post_masks.ndim == 6 and post_masks.shape[3] == 1:
            post_masks = post_masks.squeeze(3)

        recon_img = raw_out.get(
            "post_recon_combined", raw_out.get("recon_combined")
        )

        return {
            "input_img": img_tensor,
            "pred_boxes": None,
            "pred_logits": None,
            "pred_masks": post_masks,
            "recon_img": recon_img,
            "post_slots": raw_out.get("post_slots", raw_out.get("slots")),
        }

    def compute_loss(
        self, out: ModelOutput, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute model loss via configured loss module."""
        if self.loss_fn is None:
            from src.losses import build_loss

            self.loss_fn = build_loss(None)
        return self.loss_fn(out, batch)


# ── Deformable SAVi variant ────────────────────────────────────────────────────

@register_model(["deformable_savi", "deformable-savi"])
class StandardizedDeformableSAViWrapper(StandardizedSAViWrapper):
    """Deformable SAVi wrapper.  Inherits ``forward()`` and ``compute_loss()``
    from ``StandardizedSAViWrapper`` — only ``build()`` differs."""

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedDeformableSAViWrapper":
        from src.models.deformable_savi import DeformableSAVi

        res = tuple(model_cfg.get("resolution", [64, 64]))
        clip_len = model_cfg.get("n_sample_frames", model_cfg.get("clip_len", 6))

        base_model = DeformableSAVi(
            resolution=res,
            clip_len=clip_len,
            num_slots=model_cfg.get("num_slots", 4),
            slot_dim=model_cfg.get("slot_dim", 64),
            num_iterations=model_cfg.get("num_iterations", 3),
            n_heads=model_cfg.get("n_heads", 4),
            n_points=model_cfg.get("n_points", 4),
            in_channels=model_cfg.get("in_channels", 3),
            slot_dict=model_cfg.get("slot_dict"),
            enc_dict=model_cfg.get("enc_dict"),
            dec_dict=model_cfg.get("dec_dict"),
            pred_dict=model_cfg.get("pred_dict"),
            loss_dict=model_cfg.get("loss_dict"),
        )

        weight_dict = dict(model_cfg.get("weight_dict") or {})
        weight_dict.setdefault("recon", 1.0)
        weight_dict.setdefault("mask", 1.0)
        weight_dict.setdefault("sigreg", 0.1)

        loss_fn = model_cfg.get("loss_fn")

        return cls(base_model, weight_dict=weight_dict, loss_fn=loss_fn, resolution=res)

