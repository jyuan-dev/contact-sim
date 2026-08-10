"""
Tests for src/utils/data_utils.py and src/utils/training_utils.py.
Covers: find_dataset_path, get_dataset (stub), set_seed, cosine_anneal_with_warmup, get_device.
"""

import unittest
import os
import math
import torch

from src.utils.data_utils import find_dataset_path, get_dataset
from src.utils.training_utils import set_seed, cosine_anneal_with_warmup, get_device


# ── data_utils.py ─────────────────────────────────────────────────────────────

class TestFindDatasetPath(unittest.TestCase):
    def test_existing_path_returned_as_is(self):
        """When the given path exists, it should be returned unchanged."""
        result = find_dataset_path(__file__)   # __file__ always exists
        self.assertEqual(result, __file__)

    def test_missing_path_falls_back(self):
        """When path is missing, the function searches fallback locations
        and ultimately returns the original path if nothing is found."""
        non_existent = "/tmp/does_not_exist_12345.h5"
        result = find_dataset_path(non_existent, default_filename="does_not_exist_12345.h5")
        # Should not raise; returns something (either a fallback or the original)
        self.assertIsInstance(result, str)

    def test_none_path_falls_back(self):
        """Passing None should return fallback if found, or None if dummy default_filename is non-existent."""
        result = find_dataset_path(None, default_filename="non_existent_dummy_123.h5")
        self.assertIsNone(result)

    def test_empty_string_falls_back(self):
        """Empty-string path is treated as missing."""
        result = find_dataset_path("", default_filename="non_existent_dummy_123.h5")
        self.assertEqual(result, "")


class TestGetDataset(unittest.TestCase):
    def test_gridshapes_via_get_dataset(self):
        """get_dataset should successfully build a GridShapesDataset."""
        ds = get_dataset(
            dataset_name="gridshapes",
            n_sample_frames=4,
            split="train",
        )
        self.assertTrue(len(ds) > 0)
        sample = ds[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)

    def test_pusht_skipped_if_missing(self):
        """get_dataset for pusht should raise if no HDF5 file is found."""
        h5_path = "/tmp/nonexistent_pusht_9999.h5"
        with self.assertRaises(Exception):
            # Pass a dummy filename that does not exist anywhere on system
            from src.datasets.pusht import PushTMaskHDF5Dataset
            _ = PushTMaskHDF5Dataset(h5_path=h5_path, split="train")


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


class TestGetDevice(unittest.TestCase):
    def test_returns_device(self):
        """get_device() should return a torch.device instance."""
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_explicit_cpu(self):
        """Requesting 'cpu' should return a cpu device."""
        device = get_device('cpu')
        self.assertEqual(device.type, 'cpu')

    def test_auto_device_is_cuda_or_cpu(self):
        """Auto device should be either cuda or cpu."""
        device = get_device()
        self.assertIn(device.type, ('cuda', 'cpu'))


if __name__ == '__main__':
    unittest.main()
