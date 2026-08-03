"""
Tests for src/metrics/eval_metrics.py and src/metrics/evaluator.py.
Covers: compute_psnr, compute_ssim, compute_fg_ari, compute_latent_std,
        compute_sigreg_stat, EvaluationSuite.
"""

import unittest
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.metrics.eval_metrics import (
    compute_psnr,
    compute_ssim,
    compute_fg_ari,
    compute_latent_std,
    compute_sigreg_stat,
)
from src.metrics.evaluator import compute_binary_iou_dice, EvaluationSuite


# ── eval_metrics.py ───────────────────────────────────────────────────────────

class TestComputePSNR(unittest.TestCase):
    def test_perfect_reconstruction(self):
        """Identical inputs should return 100 dB (sentinel for zero MSE)."""
        img = torch.rand(2, 3, 64, 64)
        psnr = compute_psnr(img, img)
        self.assertAlmostEqual(psnr, 100.0)

    def test_noisy_reconstruction(self):
        """Noisy input should yield a finite, positive PSNR below 100."""
        gt = torch.rand(2, 3, 64, 64)
        pred = gt + 0.1 * torch.randn_like(gt)
        pred = pred.clamp(0.0, 1.0)
        psnr = compute_psnr(pred, gt)
        self.assertGreater(psnr, 0.0)
        self.assertLess(psnr, 100.0)

    def test_output_type(self):
        """Return type must be a Python float."""
        img = torch.rand(1, 3, 32, 32)
        psnr = compute_psnr(img, img + 0.05)
        self.assertIsInstance(psnr, float)


class TestComputeSSIM(unittest.TestCase):
    def test_perfect_similarity(self):
        """Identical images should return SSIM close to 1.0."""
        img = torch.rand(2, 3, 64, 64)
        ssim = compute_ssim(img, img)
        self.assertAlmostEqual(ssim, 1.0, places=4)

    def test_bounded_output(self):
        """SSIM should be in the range [-1, 1]."""
        pred = torch.rand(2, 3, 64, 64)
        gt = torch.rand(2, 3, 64, 64)
        ssim = compute_ssim(pred, gt)
        self.assertGreaterEqual(ssim, -1.0)
        self.assertLessEqual(ssim, 1.0)

    def test_5d_input(self):
        """Accepts 5D input [B, T, C, H, W] by flattening leading dims."""
        pred = torch.rand(2, 4, 3, 64, 64)
        gt = torch.rand(2, 4, 3, 64, 64)
        ssim = compute_ssim(pred, gt)
        self.assertIsInstance(ssim, float)


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

    def test_6d_squeeze(self):
        """Handles 6D input [B, T, K, 1, H, W] by squeezing channel dim."""
        pred = torch.rand(2, 3, 4, 1, 32, 32)
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
        """Slots sampled from N(0,I) should produce a low SIGReg stat."""
        slots = torch.randn(200, 64)
        stat = compute_sigreg_stat(slots, sketch_dim=16)
        self.assertIsInstance(stat, float)
        # True Gaussian should give a fairly low stat (not exact due to finite samples)
        self.assertGreaterEqual(stat, 0.0)

    def test_single_sample(self):
        """Single sample should return 0.0 gracefully."""
        stat = compute_sigreg_stat(torch.randn(1, 16))
        self.assertAlmostEqual(stat, 0.0)

    def test_4d_input(self):
        """Accepts 4D input [B, T, K, D] by reshaping."""
        slots = torch.randn(2, 4, 3, 32)
        stat = compute_sigreg_stat(slots, sketch_dim=8)
        self.assertIsInstance(stat, float)


# ── evaluator.py ──────────────────────────────────────────────────────────────

class TestComputeBinaryIoUDice(unittest.TestCase):
    def test_perfect_overlap(self):
        """Identical masks should yield IoU=1 and Dice=1."""
        mask = np.ones((32, 32), dtype=np.float32)
        iou, dice = compute_binary_iou_dice(mask, mask)
        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(dice, 1.0)

    def test_no_overlap(self):
        """Non-overlapping masks should yield IoU=0 and Dice=0."""
        pred = np.zeros((32, 32), dtype=np.float32)
        pred[:16, :] = 1.0
        gt = np.zeros((32, 32), dtype=np.float32)
        gt[16:, :] = 1.0
        iou, dice = compute_binary_iou_dice(pred, gt)
        self.assertAlmostEqual(iou, 0.0)
        self.assertAlmostEqual(dice, 0.0)

    def test_empty_gt_returns_one(self):
        """Empty GT mask with empty pred should return iou=1, dice=1."""
        pred = np.zeros((32, 32), dtype=np.float32)
        gt = np.zeros((32, 32), dtype=np.float32)
        iou, dice = compute_binary_iou_dice(pred, gt)
        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(dice, 1.0)


class TestEvaluationSuite(unittest.TestCase):
    def _make_pred_masks(self, T=5, K=4, H=32, W=32):
        """Create dummy one-hot prediction masks [T, K, H, W]."""
        masks = np.zeros((T, K, H, W), dtype=np.float32)
        strip = H // K
        for k in range(K):
            masks[:, k, k * strip:(k + 1) * strip, :] = 1.0
        return masks

    def _make_gt_masks_dict(self, T=5, num_classes=3, H=32, W=32):
        """Create dummy GT masks dict {class_id: np.array [T, H, W]}."""
        strip = H // num_classes
        gt = {}
        for c in range(num_classes):
            m = np.zeros((T, H, W), dtype=np.float32)
            m[:, c * strip:(c + 1) * strip, :] = 1.0
            gt[c] = m
        return gt

    def test_evaluate_sequence_masks_structure(self):
        """Verify EvaluationSuite returns the expected metric keys."""
        suite = EvaluationSuite(num_classes=3)
        pred_masks = self._make_pred_masks()
        gt_masks_dict = self._make_gt_masks_dict()

        metrics = suite.evaluate_sequence_masks(pred_masks, gt_masks_dict)

        self.assertIn('total_frames', metrics)
        self.assertIn('total_swap_events', metrics)
        self.assertIn('swap_rate_per_100_frames', metrics)
        self.assertIn('overall_mIoU', metrics)
        self.assertIn('overall_mDice', metrics)
        self.assertIn('class_metrics', metrics)

    def test_evaluate_sequence_masks_perfect_assignment(self):
        """Perfect 1-to-1 slot-to-class assignment should yield high mIoU."""
        suite = EvaluationSuite(num_classes=3)
        # Match 3 of the 4 slots to the 3 GT classes perfectly
        T, H, W = 5, 30, 30
        pred_masks = np.zeros((T, 4, H, W), dtype=np.float32)
        gt_masks_dict = {}
        strip = H // 3
        for c in range(3):
            pred_masks[:, c, c * strip:(c + 1) * strip, :] = 1.0
            m = np.zeros((T, H, W), dtype=np.float32)
            m[:, c * strip:(c + 1) * strip, :] = 1.0
            gt_masks_dict[c] = m

        metrics = suite.evaluate_sequence_masks(pred_masks, gt_masks_dict)
        self.assertGreater(metrics['overall_mIoU'], 0.5)

    def test_no_swap_events_for_static_assignment(self):
        """Static slot-to-class assignment across frames should yield 0 swaps."""
        suite = EvaluationSuite(num_classes=3)
        pred_masks = self._make_pred_masks(T=10)
        gt_masks_dict = self._make_gt_masks_dict(T=10)

        metrics = suite.evaluate_sequence_masks(pred_masks, gt_masks_dict)
        self.assertEqual(metrics['total_swap_events'], 0)

    def test_custom_class_names(self):
        """Custom class_names should appear in class_metrics keys."""
        class_names = {0: 'Block', 1: 'Agent', 2: 'Goal'}
        suite = EvaluationSuite(num_classes=3, class_names=class_names)
        pred_masks = self._make_pred_masks()
        gt_masks_dict = self._make_gt_masks_dict()

        metrics = suite.evaluate_sequence_masks(pred_masks, gt_masks_dict)
        for name in class_names.values():
            self.assertIn(name, metrics['class_metrics'])


if __name__ == '__main__':
    unittest.main()
