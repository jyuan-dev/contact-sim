"""
src.metrics — Quantitative evaluation metrics for object-centric slot models.
"""

from src.metrics.evaluator import (
    EvaluationSuite,
    DeterministicEvaluator,
    compute_binary_iou_dice,
    greedy_slot_assignments,
)
from src.metrics.eval_metrics import (
    compute_psnr,
    compute_ssim,
    compute_fg_ari,
    compute_latent_std,
    compute_sigreg_stat,
    compute_collapse_diagnostics,
)

__all__ = [
    "EvaluationSuite",
    "DeterministicEvaluator",
    "compute_binary_iou_dice",
    "greedy_slot_assignments",
    "compute_psnr",
    "compute_ssim",
    "compute_fg_ari",
    "compute_latent_std",
    "compute_sigreg_stat",
    "compute_collapse_diagnostics",
]
