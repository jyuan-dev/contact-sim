"""
Tests for the single SIGReg statistic core — loss and metric share it.
"""

import unittest

import torch

from src.losses.sigreg import SIGRegLoss, compute_sigreg_statistic
from src.metrics.eval_metrics import compute_sigreg_stat


class TestSigRegStatisticCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.latents = torch.randn(4, 6, 3, 32)  # [B, T, K, D]

    def test_loss_matches_metric_with_same_seed(self):
        """SIGRegLoss(weight=1.0) == compute_sigreg_stat under the same seed."""
        loss_fn = SIGRegLoss(weight=1.0, num_proj=64, seed=7)
        weighted, info = loss_fn(self.latents)

        metric = compute_sigreg_stat(self.latents, sketch_dim=64, seed=7)

        self.assertAlmostEqual(weighted.item(), metric, places=6)
        self.assertAlmostEqual(info["sigreg_loss"], metric, places=6)

    def test_seeded_calls_are_deterministic(self):
        first = compute_sigreg_statistic(self.latents, 64, seed=7)
        second = compute_sigreg_statistic(self.latents, 64, seed=7)
        self.assertTrue(torch.equal(first, second))

    def test_unseeded_calls_resample(self):
        """Without a seed, projections resample per call (values differ)."""
        a = compute_sigreg_statistic(self.latents, 64)
        b = compute_sigreg_statistic(self.latents, 64)
        self.assertFalse(torch.equal(a, b))

    def test_per_slot_output_shape(self):
        per_slot = compute_sigreg_statistic(self.latents, 64)
        self.assertEqual(per_slot.shape, (3,))

    def test_loss_per_slot_breakdown(self):
        _, info = SIGRegLoss(weight=1.0, num_proj=64, seed=3)(self.latents)
        self.assertIn("sigreg_loss", info)
        for k in range(3):
            self.assertIn(f"sigreg_slot{k}", info)

    def test_small_batch_returns_zero(self):
        weighted, info = SIGRegLoss(weight=1.0)(torch.randn(1, 4, 3, 32))
        self.assertEqual(weighted.item(), 0.0)
        self.assertEqual(info["sigreg_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
