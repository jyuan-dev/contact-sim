"""
Slot Loss Modules for Stage 2 World Models.

Provides:
  - SlotMSELoss: Temporal Slot MSE prediction loss with optional exponential decay.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotMSELoss(nn.Module):
    """
    Slot MSE Prediction Loss for Stage 2 World Models.

    Evaluates Mean Squared Error between predicted future slots and ground-truth slots,
    with optional exponential discount factor across the rollout horizon.
    """

    def __init__(
        self,
        decay_factor: float = 1.0,
        action_loss_weight: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.decay_factor = float(decay_factor)
        self.action_loss_weight = float(action_loss_weight)
        self.weight = float(weight)

    def forward(
        self,
        model_output: dict[str, Any],
        batch: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute slot MSE loss and optional action NLL loss.
        """
        gt_slots = model_output["gt_slots"]      # [B, rollout_len, K, D]
        pred_slots = model_output["pred_slots"]  # [B, rollout_len, K, D]

        slots_loss = F.mse_loss(pred_slots, gt_slots, reduction="none")

        if self.decay_factor < 1.0:
            w = self.decay_factor ** torch.arange(gt_slots.shape[1], device=gt_slots.device)
            w = w / w.sum() * gt_slots.shape[1]
            slots_loss = slots_loss * w[None, :, None, None]

        slot_mse = slots_loss.mean()
        total_loss = self.weight * slot_mse

        loss_dict: dict[str, torch.Tensor] = {
            "loss": total_loss,
            "slot_mse": slot_mse,
        }

        # Optional Action NLL loss (e.g. from INTACT or PIDM)
        act_dict = model_output.get("action_nll_dict") or model_output.get("act_loss_dict")
        if act_dict is not None and "loss" in act_dict:
            act_loss = act_dict["loss"]
            total_loss = total_loss + self.action_loss_weight * act_loss
            loss_dict["loss"] = total_loss
            loss_dict["action_nll"] = act_loss
            for k in ("action_mae", "action_rmse"):
                if k in act_dict:
                    loss_dict[k] = act_dict[k]

        return loss_dict
