"""
Standardized SAVi / StoSAVi wrapper.

Wraps ``SAVi`` (which in turn wraps the native ``StoSAVi`` core) to conform
to the ``BaseModelWrapper`` interface.

Registered under: ``"savi"``, ``"stosavi"``
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

from src.models.base import BaseModelWrapper, ModelOutput
from src.models.factory import register_model
from src.utils.tensor_checks import typechecked


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

    @typechecked
    def encode_slots(self, video: Float[Tensor, "B T C H W"]) -> Float[Tensor, "B T K D"]:
        """Extract per-frame slots [B, T, K, D] for video [B, T, C, H, W]."""
        return self.model.encode_slots(video)

    @typechecked
    def decode_slots(self, slots: Float[Tensor, "..."]) -> tuple[Tensor, Tensor]:
        """Decode slots to (recon_img, pred_masks)."""
        return self.model.decode_slots(slots)

    def inner_savi(self) -> Any:
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

    @typechecked
    def rollout(
        self,
        video: Float[Tensor, "B T C H W"],
        n_cond_frames: int = 2,
        actions: Float[Tensor, "B Tact ActDim"] | None = None,
        goal_slots: Float[Tensor, "B K D"] | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """
        Perform autoregressive future slot rollout using Stage 1 predictor.
        """
        if isinstance(n_cond_frames, bool) or not 1 <= n_cond_frames <= video.shape[1]:
            raise ValueError(f"n_cond_frames must be in [1, T={video.shape[1]}], got {n_cond_frames!r}")

        B, T, C, H, W = video.shape
        device = video.device

        # Resample if needed
        if (H, W) != self.resolution:
            video_resized = F.interpolate(
                video.view(B * T, C, H, W),
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
            ).view(B, T, C, self.resolution[0], self.resolution[1])
        else:
            video_resized = video

        inner = self.inner_savi()
        if hasattr(inner, "_reset_rnn"):
            inner._reset_rnn()

        cond_slots, _ = inner.encode(video_resized[:, :n_cond_frames])  # [B, n_cond_frames, K, D]
        prev_slots = cond_slots[:, -1]

        rollout_len = T - n_cond_frames
        if rollout_len > 0:
            rollout_slots = []
            for _ in range(n_cond_frames, T):
                rollout_latents = inner.predictor(prev_slots)
                rollout_slots.append(rollout_latents)
                prev_slots = rollout_latents
            rollout_slots_tensor = torch.stack(rollout_slots, dim=1)
            slots_stacked = torch.cat([cond_slots, rollout_slots_tensor], dim=1)
        else:
            slots_stacked = cond_slots

        recon_img, pred_masks = self.decode_slots(slots_stacked)
        is_rollout_mask = torch.tensor(
            [t >= n_cond_frames for t in range(T)], device=device, dtype=torch.bool
        )

        return {
            "input_img": video,
            "pred_masks": pred_masks,
            "recon_img": recon_img,
            "post_slots": slots_stacked,
            "is_rollout_mask": is_rollout_mask,
        }

