"""
Standardized SlotFormer (Stage 2) Wrapper.

Wraps ``SlotFormerModel`` to conform to the ``BaseModelWrapper`` interface.
Registered under: ``"slotformer"``, ``"slot_former"``
"""

from __future__ import annotations

import os
from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

from src.models.base import BaseModelWrapper, ModelOutput
from src.models.factory import register_model, build_model
from src.utils.tensor_checks import typechecked

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@register_model([
    "slotformer", "slot_former", "ocvp_slotformer", "ocvp-slotformer",
    "factorized_slotformer", "cocvp_slotformer", "cocvp-slotformer",
    "cocvp_slotformer_film", "cocvp_slotformer_sum", "cocvp_slotformer_concat",
    "intact_slotformer", "intact_ocvp_slotformer", "ocvp_intact_slotformer",
    "pidm", "pidm_slotformer", "pidm-slotformer", "pidm_pusht",
])
class StandardizedSlotFormerWrapper(BaseModelWrapper):
    """
    SlotFormer Stage 2 wrapper conforming to ``BaseModelWrapper``.
    Supports standard SlotFormer, OCVP Factorized, Action-Conditioned (cOCVP),
    INTACT RobotSlotIntentActionActor, and PIDM (Predictive Inverse Dynamics Model) variants.
    """

    def __init__(
        self,
        model: nn.Module,
        resolution: tuple = (64, 64),
    ) -> None:
        super().__init__()
        self.model = model
        self.resolution = resolution

    @typechecked
    def encode_slots(self, video: Float[Tensor, "B T C H W"]) -> Float[Tensor, "B T K D"]:
        """Extract per-frame slots [B, T, K, D] for video [B, T, C, H, W]."""
        return self.model.extract_slots(video)

    @typechecked
    def decode_slots(self, slots: Float[Tensor, "..."]) -> tuple[Tensor, Tensor]:
        """Decode slots to (recon_img, pred_masks)."""
        stage1 = getattr(self.model, "stage1_model", None)
        if stage1 is not None and hasattr(stage1, "decode_slots"):
            return stage1.decode_slots(slots)
        inner_savi = self.inner_savi()
        is_5d = (slots.ndim == 4)
        if is_5d:
            B, T = slots.shape[:2]
            slots_flat = slots.flatten(0, 1)
        else:
            slots_flat = slots
        recon_flat, _, masks_flat, _ = inner_savi.decode(slots_flat)
        if is_5d:
            return recon_flat.unflatten(0, (B, T)), masks_flat.squeeze(2).unflatten(0, (B, T))
        return recon_flat, masks_flat.squeeze(2)

    def inner_savi(self) -> Any:
        """Return the Stage-1 StoSAVi core (typed accessor)."""
        stage1 = getattr(self.model, "stage1_model", None)
        if stage1 is None:
            raise AttributeError(f"{type(self.model).__name__} has no stage1_model")
        return stage1.model.model

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedSlotFormerWrapper":
        """Construct from model config sub-dict."""
        from src.models.slotformer import SlotFormerModel
        from src.models.pidm import PIDMModel

        stage1_ckpt_path = model_cfg.get("stage1_ckpt_path", "scratch/checkpoints/savi_pusht_default_4ep/savi_best.pt")
        if stage1_ckpt_path and not os.path.isabs(stage1_ckpt_path):
            stage1_ckpt_path = os.path.join(REPO_ROOT, stage1_ckpt_path)

        stage1_model_type = model_cfg.get("stage1_model_type", "") or "savi"

        res = tuple(model_cfg.get("resolution", [64, 64]))
        stage1_cfg = {
            "model": {
                "name": stage1_model_type,
                "type": stage1_model_type,
                "resolution": res,
            }
        }
        if stage1_ckpt_path and os.path.exists(stage1_ckpt_path):
            # Reconstruct the stage-1 experiment from ITS OWN saved config —
            # num_slots / slot_dim / BN flags live there, not in this config.
            from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
            from src.utils.training_utils import load_checkpoint_state

            stage1_wrapper, _ = bootstrap_checkpoint(stage1_ckpt_path)
            load_checkpoint_state(stage1_wrapper, stage1_ckpt_path)
            print(f"[SlotFormer] Loaded pretrained Stage 1 experiment from: {stage1_ckpt_path}")
        else:
            if stage1_ckpt_path:
                # A configured-but-missing stage-1 checkpoint means stage-2
                # would silently train against random frozen slots — fail
                # loudly instead.
                raise FileNotFoundError(
                    f"[SlotFormer] Stage 1 checkpoint not found: {stage1_ckpt_path}")
            stage1_wrapper = build_model(stage1_cfg)

        model_type_str = str(model_cfg.get("type", model_cfg.get("name", ""))).lower()

        # Each model family owns its own kwarg defaults/aliases via `from_config`;
        # this wrapper only decides *which* family to build.
        if "pidm" in model_type_str:
            inner_model = PIDMModel.from_config(model_cfg, stage1_wrapper)
        else:
            inner_model = SlotFormerModel.from_config(model_cfg, stage1_wrapper)

        return cls(inner_model, resolution=res)

    def forward(self, x: dict | Tensor) -> ModelOutput:
        out_dict = self.model(x)
        res = {
            "input_img": out_dict.get("input_img"),
            "pred_masks": out_dict.get("pred_masks"),
            "recon_img": out_dict.get("recon_img"),
            "post_slots": out_dict.get("post_slots"),
            "gt_slots": out_dict.get("gt_slots"),
            "pred_slots": out_dict.get("pred_slots"),
        }
        if "action_nll_dict" in out_dict:
            res["action_nll_dict"] = out_dict["action_nll_dict"]
        return cast(ModelOutput, res)

    def compute_loss(
        self, out: ModelOutput, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        return self.model.calc_train_loss(out, batch)

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
        Perform autoregressive future slot rollout using Stage 2 Transformer rollouter
        (or falling back to Stage 1 SAVi predictor if unconfigured).
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

        # 1. Conditioned encoding
        cond_slots = self.encode_slots(video_resized[:, :n_cond_frames])  # [B, n_cond_frames, K, D]
        rollout_len = T - n_cond_frames

        # 2. Rollout
        if rollout_len > 0:
            rollouter = getattr(self.model, "rollouter", None)
            if rollouter is not None:
                roll_kwargs: dict[str, Any] = {"pred_len": rollout_len}
                if actions is not None:
                    roll_kwargs["actions"] = actions
                if goal_slots is not None:
                    roll_kwargs["goal_slots"] = goal_slots
                rollout_slots_tensor = rollouter(cond_slots, **roll_kwargs)
            else:
                inner = self.inner_savi()
                prev_slots = cond_slots[:, -1]
                rollout_slots = []
                for _ in range(n_cond_frames, T):
                    rollout_latents = inner.predictor(prev_slots)
                    rollout_slots.append(rollout_latents)
                    prev_slots = rollout_latents
                rollout_slots_tensor = torch.stack(rollout_slots, dim=1)

            slots_stacked = torch.cat([cond_slots, rollout_slots_tensor], dim=1)
        else:
            slots_stacked = cond_slots

        # 3. Decode
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

