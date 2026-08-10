"""
src.losses — Loss functions for Contact-Sim / Slot-Worldmodel baselines.

Modules:
  - contrastive:   TemporalSlotContrastiveLoss (InfoNCE across video frames)
  - sigreg:        SIGRegLoss (Sketched Isotropic Gaussian Regularization)
  - savi_loss:     compute_savi_loss (Reconstruction MSE, Mask BCE/Dice, SIGReg)
  - model_losses:  compute_detr_loss (DETR Hungarian matching loss + aux layers)
"""

from src.losses.contrastive import TemporalSlotContrastiveLoss
from src.losses.sigreg import SIGRegLoss
from src.losses.savi_loss import compute_savi_loss
from src.losses.model_losses import compute_detr_loss

__all__ = [
    "TemporalSlotContrastiveLoss",
    "SIGRegLoss",
    "compute_savi_loss",
    "compute_detr_loss",
]
