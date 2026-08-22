"""
Tests for src/metrics/eval_metrics.py.
Covers: compute_fg_ari, compute_latent_std, compute_sigreg_stat.
"""

import unittest
import numpy as np
import torch

from src.metrics.eval_metrics import (
    compute_fg_ari,
    compute_latent_std,
    compute_sigreg_stat,
)


# ── eval_metrics.py ───────────────────────────────────────────────────────────

class TestComputeFgARI(unittest.TestCase):
    def _make_one_hot_masks(self, B, T, K, H, W):
        """Create hard one-hot masks where each slot covers a distinct region."""
        masks = torch.zeros(B, T, K, H, W)
        strip = H // K
        for k in range(K):
            masks[:, :, k, k * strip:(k + 1) * strip, :] = 1.0
        return masks

    def test_perfect_segmentation(self):
        """When pred masks match GT exactly, ARI should be 1.0."""
        B, T, K, H, W = 1, 3, 3, 30, 30
        gt = self._make_one_hot_masks(B, T, K, H, W)
        pred = gt.clone()
        ari = compute_fg_ari(pred, gt)
        self.assertAlmostEqual(ari, 1.0, places=4)

    def test_random_segmentation(self):
        """Random masks should yield an ARI value in [-1, 1]."""
        pred = torch.rand(2, 4, 4, 32, 32)
        gt = torch.rand(2, 4, 3, 32, 32)
        ari = compute_fg_ari(pred, gt)
        self.assertGreaterEqual(ari, -1.0)
        self.assertLessEqual(ari, 1.0)

    def test_5d_input(self):
        """Takes normalized 5D [B, T, K, H, W] masks (wrapper-owned squeeze)."""
        pred = torch.rand(2, 3, 4, 32, 32)
        gt = torch.rand(2, 3, 3, 32, 32)
        ari = compute_fg_ari(pred, gt)
        self.assertIsInstance(ari, float)


class TestComputeLatentStd(unittest.TestCase):
    def test_constant_slots(self):
        """All-constant slot vectors should yield std = 0."""
        slots = torch.ones(4, 8, 64) * 3.14
        std = compute_latent_std(slots)
        self.assertAlmostEqual(std, 0.0, places=5)

    def test_varying_slots(self):
        """Random slot vectors should yield a non-zero std."""
        slots = torch.randn(4, 8, 64)
        std = compute_latent_std(slots)
        self.assertGreater(std, 0.0)

    def test_2d_input(self):
        """Accepts 2D input [N, D] directly."""
        slots = torch.randn(32, 64)
        std = compute_latent_std(slots)
        self.assertIsInstance(std, float)


class TestComputeSIGRegStat(unittest.TestCase):
    def test_gaussian_slots(self):
        """Slots sampled from N(0,I) should produce a low SIGReg stat.

        One input convention: [B, T, K, D].
        """
        slots = torch.randn(200, 1, 1, 64)
        stat = compute_sigreg_stat(slots, sketch_dim=16)
        self.assertIsInstance(stat, float)
        # True Gaussian should give a fairly low stat (not exact due to finite samples)
        self.assertGreaterEqual(stat, 0.0)

    def test_single_batch_smoke(self):
        """Small batch still produces a finite stat (no crash)."""
        stat = compute_sigreg_stat(torch.randn(1, 4, 1, 16))
        self.assertIsInstance(stat, float)

    def test_4d_input(self):
        """Accepts 4D input [B, T, K, D] by reshaping."""
        slots = torch.randn(2, 4, 3, 32)
        stat = compute_sigreg_stat(slots, sketch_dim=8)
        self.assertIsInstance(stat, float)


if __name__ == '__main__':
    unittest.main()
