"""
src.losses — Modular loss functions and composite loss aggregator.

Modules:
  - recon_loss:    ReconstructionMSELoss (MSE between recon and input images)
  - mask_loss:     MaskSegmentationLoss (BCE + Dice mask loss)
  - sigreg:        SIGRegLoss (Sketched Isotropic Gaussian Regularization)
  - contrastive:   TemporalSlotContrastiveLoss (InfoNCE across video frames)
  - composite:     CompositeLoss (aggregates arbitrary sub-losses configured via Hydra)
"""

from src.losses.recon_loss import ReconstructionMSELoss
from src.losses.mask_loss import MaskSegmentationLoss
from src.losses.sigreg import SIGRegLoss
from src.losses.contrastive import TemporalSlotContrastiveLoss
from src.losses.composite import CompositeLoss
from src.losses.lewm_loss import LeWMLoss


def build_loss(cfg_loss):
    """
    Instantiate a loss module from a Hydra loss config dict using hydra.utils.instantiate.
    """
    if cfg_loss is None:
        return CompositeLoss(
            recon=ReconstructionMSELoss(weight=1.0),
            mask=MaskSegmentationLoss(weight=1.0),
        )
    if isinstance(cfg_loss, dict) and "_target_" in cfg_loss:
        import hydra

        return hydra.utils.instantiate(cfg_loss)
    if isinstance(cfg_loss, dict):
        return CompositeLoss(losses=cfg_loss)
    return cfg_loss


__all__ = [
    "ReconstructionMSELoss",
    "MaskSegmentationLoss",
    "SIGRegLoss",
    "TemporalSlotContrastiveLoss",
    "CompositeLoss",
    "LeWMLoss",
    "build_loss",
]
