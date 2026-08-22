"""
Tests for src/utils/data_utils.py and src/utils/training_utils.py.
Covers: find_dataset_path, set_seed, cosine_anneal_with_warmup.
"""

import unittest
import os
import math
import torch

from src.utils.data_utils import find_dataset_path
from src.utils.training_utils import set_seed, cosine_anneal_with_warmup


# ── data_utils.py ─────────────────────────────────────────────────────────────

class TestFindDatasetPath(unittest.TestCase):
    def test_existing_path_returned_as_is(self):
        """When the given path exists, it should be returned unchanged."""
        result = find_dataset_path(__file__)   # __file__ always exists
        self.assertEqual(result, __file__)

    def test_missing_path_raises(self):
        """Missing paths raise FileNotFoundError listing probed locations."""
        non_existent = "/tmp/does_not_exist_12345.h5"
        with self.assertRaises(FileNotFoundError) as ctx:
            find_dataset_path(non_existent, default_filename="does_not_exist_12345.h5")
        self.assertIn("does_not_exist_12345.h5", str(ctx.exception))

    def test_none_path_raises_when_nothing_found(self):
        """None probes fallbacks; raises when no candidate exists."""
        with self.assertRaises(FileNotFoundError):
            find_dataset_path(None, default_filename="non_existent_dummy_123.h5")

    def test_none_path_finds_default_dataset(self):
        """None probes fallbacks and finds the real PushT dataset."""
        result = find_dataset_path(None)
        self.assertTrue(os.path.exists(result))

    def test_empty_string_raises(self):
        """Empty-string path is treated as missing."""
        with self.assertRaises(FileNotFoundError):
            find_dataset_path("", default_filename="non_existent_dummy_123.h5")


# ── training_utils.py ─────────────────────────────────────────────────────────

class TestSetSeed(unittest.TestCase):
    def test_reproducibility(self):
        """Two identical seed calls should produce the same random tensors."""
        set_seed(0)
        t1 = torch.randn(10)
        set_seed(0)
        t2 = torch.randn(10)
        self.assertTrue(torch.allclose(t1, t2))

    def test_different_seeds_differ(self):
        """Different seeds should (almost always) produce different tensors."""
        set_seed(1)
        t1 = torch.randn(10)
        set_seed(2)
        t2 = torch.randn(10)
        self.assertFalse(torch.allclose(t1, t2))


class TestCosineAnnealWithWarmup(unittest.TestCase):
    def test_warmup_phase(self):
        """During warmup, LR should grow linearly from ~0 to lr."""
        lr = cosine_anneal_with_warmup(step=5, total_steps=100, warmup_steps=10, lr=1.0)
        self.assertAlmostEqual(lr, 0.5, places=4)

    def test_at_warmup_end(self):
        """Exactly at the end of warmup, LR should equal the peak lr."""
        lr = cosine_anneal_with_warmup(step=10, total_steps=100, warmup_steps=10, lr=1.0)
        self.assertAlmostEqual(lr, 1.0, places=4)

    def test_cosine_decay(self):
        """After warmup, LR should decrease monotonically toward min_lr."""
        lrs = [
            cosine_anneal_with_warmup(step=s, total_steps=100, warmup_steps=10, lr=1.0, min_lr=0.0)
            for s in range(10, 101)
        ]
        # Verify strictly non-increasing
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i] + 1e-9, lrs[i + 1])

    def test_final_lr_approaches_min(self):
        """At the final step, LR should be at min_lr."""
        lr = cosine_anneal_with_warmup(step=100, total_steps=100, warmup_steps=10, lr=1.0, min_lr=1e-5)
        self.assertAlmostEqual(lr, 1e-5, places=6)

    def test_zero_warmup_steps(self):
        """When warmup_steps=0, step 0 should still be handled without division by zero."""
        lr = cosine_anneal_with_warmup(step=0, total_steps=100, warmup_steps=0, lr=0.5)
        self.assertIsInstance(lr, float)

    def test_output_bounds(self):
        """LR should always stay in [min_lr, lr]."""
        for step in range(0, 110, 5):
            lr = cosine_anneal_with_warmup(step=step, total_steps=100, warmup_steps=10, lr=1.0, min_lr=1e-4)
            self.assertGreaterEqual(lr, 1e-4 - 1e-8)
            self.assertLessEqual(lr, 1.0 + 1e-8)


if __name__ == '__main__':
    unittest.main()
