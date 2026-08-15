"""
Standardized SAVi / StoSAVi wrapper.

Wraps ``SAVi`` (which in turn wraps the native ``StoSAVi`` core) to conform
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
      - Output key normalization (one shape per key, see ModelOutput)
      - Delegate loss to the injected ``loss_fn`` (built by the factory)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        resolution: tuple = (64, 64),
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
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

        use_encoder_bn = model_cfg.get("use_encoder_bn", model_cfg.get("use_bn", False))
        use_residual_bn = model_cfg.get("use_residual_bn", model_cfg.get("use_bn", False))

        base_model = SAVi(
            resolution=res,
            clip_len=clip_len,
            num_slots=model_cfg.get("num_slots", 4),
            slot_dim=model_cfg.get("slot_dim", 64),
            num_iterations=model_cfg.get("num_iterations", 3),
            in_channels=model_cfg.get("in_channels", 3),
            use_encoder_bn=use_encoder_bn,
            use_residual_bn=use_residual_bn,
            slot_dict=model_cfg.get("slot_dict"),
            enc_dict=model_cfg.get("enc_dict"),
            dec_dict=model_cfg.get("dec_dict"),
            pred_dict=model_cfg.get("pred_dict"),
            loss_dict=model_cfg.get("loss_dict"),
        )

        loss_fn = model_cfg.get("loss_fn")

        return cls(base_model, loss_fn=loss_fn, resolution=res)

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

        # One shape per key: the wrapper owns normalization, consumers take
        # the keys strictly (see ModelOutput in src/models/base.py).
        raw_out = self.model(x)
        post_masks = raw_out["post_masks"]
        if post_masks.ndim == 6 and post_masks.shape[3] == 1:
            post_masks = post_masks.squeeze(3)  # [B, T, K, H, W]

        return {
            "input_img": img_tensor,
            "pred_boxes": None,
            "pred_logits": None,
            "pred_masks": post_masks,
            "recon_img": raw_out["post_recon_combined"],
            "post_slots": raw_out["post_slots"],
        }

    def inner_savi(self):
        """Return the core StoSAVi model (typed accessor)."""
        return self.model.model

    def compute_loss(
        self, out: ModelOutput, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute model loss via the injected loss module."""
        if self.loss_fn is None:
            raise RuntimeError(
                f"{type(self).__name__} has no loss module injected — "
                "build the wrapper through build_model() so the factory "
                "injects the configured loss."
            )
        return self.loss_fn(out, batch)

