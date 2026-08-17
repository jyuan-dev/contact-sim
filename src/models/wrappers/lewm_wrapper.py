"""
Standardized LeWorldModel (LeWM) Wrapper.

Wraps ``LeWM`` to conform to the ``BaseModelWrapper`` interface.
Registered under: ``"lewm"``, ``"leworldmodel"``, ``"le_wm"``.
"""

from __future__ import annotations

from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseModelWrapper, ModelOutput
from src.models.factory import register_model
from src.models.lewm import LeWM
from src.losses.lewm_loss import LeWMLoss


@register_model(["lewm", "leworldmodel", "le_wm"])
class StandardizedLeWMWrapper(BaseModelWrapper):
    """
    LeWorldModel wrapper conforming to ``BaseModelWrapper``.
    """

    def __init__(
        self,
        model: LeWM,
        loss_fn: nn.Module | None = None,
        resolution: tuple[int, int] = (64, 64),
        n_preds: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn or LeWMLoss()
        self.resolution = resolution
        self.n_preds = n_preds

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedLeWMWrapper":
        """Construct from model configuration dictionary."""
        res = tuple(model_cfg.get("resolution", [64, 64]))
        in_channels = model_cfg.get("in_channels", 3)
        action_dim = model_cfg.get("action_dim", 2)
        embed_dim = model_cfg.get("embed_dim", 192)
        hidden_dim = model_cfg.get("hidden_dim", 2048)
        num_frames = model_cfg.get("num_frames", model_cfg.get("clip_len", model_cfg.get("n_sample_frames", 16)))

        predictor_cfg = model_cfg.get("predictor", {})
        predictor_depth = predictor_cfg.get("depth", model_cfg.get("depth", 6))
        predictor_heads = predictor_cfg.get("heads", model_cfg.get("heads", 8))
        predictor_dim_head = predictor_cfg.get("dim_head", model_cfg.get("dim_head", 64))
        predictor_mlp_dim = predictor_cfg.get("mlp_dim", model_cfg.get("mlp_dim", 1024))
        dropout = predictor_cfg.get("dropout", model_cfg.get("dropout", 0.1))

        core_model = LeWM(
            resolution=res,
            in_channels=in_channels,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            predictor_depth=predictor_depth,
            predictor_heads=predictor_heads,
            predictor_dim_head=predictor_dim_head,
            predictor_mlp_dim=predictor_mlp_dim,
            dropout=dropout,
        )

        loss_fn = model_cfg.get("loss_fn")
        if loss_fn is None:
            sigreg_cfg = model_cfg.get("loss", {}).get("sigreg", {})
            sigreg_weight = sigreg_cfg.get("weight", model_cfg.get("sigreg_weight", 0.09))
            loss_fn = LeWMLoss(sigreg_weight=sigreg_weight)

        n_preds = model_cfg.get("num_preds", model_cfg.get("n_preds", 1))

        return cls(
            model=core_model,
            loss_fn=loss_fn,
            resolution=res,
            n_preds=n_preds,
        )

    def forward(self, x: Union[Tensor, dict[str, Any]]) -> ModelOutput:
        """
        Run the LeWM forward pass.
        """
        if isinstance(x, dict):
            video = x.get("img", x.get("video", x.get("pixels")))
            actions = x.get("actions", x.get("action"))
        else:
            video = x
            actions = None

        if video is None:
            raise ValueError("LeWM forward: input dictionary must contain 'img', 'video', or 'pixels'.")

        # Normalize 4D [B, C, H, W] -> 5D [B, 1, C, H, W]
        if video.ndim == 4:
            video = video.unsqueeze(1)

        # Spatial resolution interpolation if needed
        B, T, C, H, W = video.shape
        target_H, target_W = self.resolution
        if (H, W) != (target_H, target_W):
            video = F.interpolate(
                video.view(B * T, C, H, W),
                size=(target_H, target_W),
                mode="bilinear",
                align_corners=False,
            ).view(B, T, C, target_H, target_W)

        out = self.model(video, actions=actions, n_preds=self.n_preds)

        model_output: ModelOutput = {
            "input_img": video,
            "recon_img": None,
            "pred_masks": None,
            "post_slots": out["emb"].unsqueeze(2),  # [B, T, 1, D] for slot compatibility
            "init_slots": None,
            "pred_boxes": None,
            "pred_logits": None,
            "rollout_slots": None,
            "attn_masks": None,
            "extra": {
                "emb": out["emb"],
                "pred_emb": out["pred_emb"],
                "target_emb": out["target_emb"],
                "act_emb": out["act_emb"],
                "pred_loss": out["pred_loss"],
            },
        }
        return model_output

    def compute_loss(
        self,
        out: ModelOutput,
        batch: dict,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute training loss (prediction MSE + SIGReg regularization).
        """
        loss_input = dict(out["extra"]) if "extra" in out and out["extra"] is not None else {}
        loss_res = self.loss_fn(loss_input, batch)

        if isinstance(loss_res, dict):
            total_loss = loss_res["loss"]
            loss_dict = {k: v.item() if isinstance(v, Tensor) else float(v) for k, v in loss_res.items()}
        else:
            total_loss = loss_res
            loss_dict = {"total_loss": total_loss.item()}

        return total_loss, loss_dict
