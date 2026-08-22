"""
src.metrics — Quantitative evaluation metrics for object-centric slot models.
"""

from src.metrics.evaluator import (
    DeterministicEvaluator,
    greedy_slot_assignments,
)
from src.metrics.eval_metrics import (
    compute_fg_ari,
    compute_latent_std,
    compute_sigreg_stat,
    compute_collapse_diagnostics,
)

__all__ = [
    "DeterministicEvaluator",
    "greedy_slot_assignments",
    "compute_fg_ari",
    "compute_latent_std",
    "compute_sigreg_stat",
    "compute_collapse_diagnostics",
]
