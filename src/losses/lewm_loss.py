"""
LeWM Loss Module.

Combines next-embedding MSE prediction loss in latent space with
Sketched Isotropic Gaussian Regularization (SIGReg).

Ref:
    Maes, Le Lidec, Scieur, LeCun, Balestriero (2026/2024).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.sigreg import compute_sigreg_statistic


class LeWMLoss(nn.Module):
    """
    LeWorldModel (LeWM) Joint Loss:
      L_total = L_pred + sigreg_weight * L_SIGReg
    """

    def __init__(
        self,
        sigreg_weight: float = 0.09,
        num_proj: int = 1024,
        knots: int = 17,
        t_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.sigreg_weight = float(sigreg_weight)
        self.num_proj = int(num_proj)
        self.knots = int(knots)
        self.t_max = float(t_max)

    def forward(
        self,
        model_output: dict[str, Any],
        batch: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute LeWM prediction loss and SIGReg latent regularization.
        """
        emb = model_output["emb"]            # [B, T, D]
        pred_emb = model_output["pred_emb"]  # [B, T_pred, D]
        target_emb = model_output["target_emb"]  # [B, T_pred, D]

        # 1. Prediction MSE in latent space
        pred_loss = F.mse_loss(pred_emb, target_emb)

        # 2. SIGReg Regularization on all video embeddings [B, T, 1, D]
        sig_stat = compute_sigreg_statistic(
            emb.unsqueeze(2),
            num_proj=self.num_proj,
            knots=self.knots,
            t_max=self.t_max,
        )  # (1,)
        sigreg_loss = sig_stat.mean()

        total_loss = pred_loss + self.sigreg_weight * sigreg_loss

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
        }
