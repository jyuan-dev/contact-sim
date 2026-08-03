"""
src.losses — Loss functions for Contact-Sim / Slot-Worldmodel baselines.

Modules:
  - contrastive:   TemporalSlotContrastiveLoss (InfoNCE across video frames)
  - sigreg:        SIGRegLoss (Sketched Isotropic Gaussian Regularization)
  - model_losses:  compute_detr_loss, compute_savi_loss
"""

from src.losses.contrastive import TemporalSlotContrastiveLoss
from src.losses.sigreg import SIGRegLoss
from src.losses.model_losses import compute_detr_loss, compute_savi_loss

__all__ = [
    "TemporalSlotContrastiveLoss",
    "SIGRegLoss",
    "compute_detr_loss",
    "compute_savi_loss",
]
