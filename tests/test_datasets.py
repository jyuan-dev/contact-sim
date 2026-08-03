import unittest
import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.factory import build_dataset, build_dataloader
from src.datasets.pusht import PushTMaskHDF5Dataset, normalize_img, denormalize_img, augment_background
from src.datasets.gridshapes import GridShapesDataset
import numpy as np


# ── pusht.py utilities ────────────────────────────────────────────────────────

class TestPushTUtilities(unittest.TestCase):
    def test_normalize_denormalize_roundtrip(self):
        """normalize followed by denormalize should approximately recover the original."""
        img = torch.rand(3, 64, 64)
        recovered = denormalize_img(normalize_img(img))
        self.assertTrue(torch.allclose(img, recovered, atol=1e-5))

    def test_normalize_shifts_range(self):
        """After normalisation, mean should be close to 0 for ImageNet-like inputs."""
        # An all-0.5 image normalises to approx 0 (mean subtracted)
        img = torch.ones(3, 64, 64) * 0.5
        normalized = normalize_img(img)
        self.assertTrue(normalized.abs().max() < 2.0)

    def test_augment_background_replaces_white_pixels(self):
        """augment_background should change purely white pixel values."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 255  # all white
        augmented = augment_background(img)
        # Some pixels should have changed
        self.assertFalse(np.all(augmented == 255))


# ── GridShapesDataset ───────────────────────────────────────────────────────────

class TestGridShapesDataset(unittest.TestCase):
    def setUp(self):
        self.ds = GridShapesDataset(
            num_samples=20, num_frames=6, num_objects=3, img_size=64, seed=0
        )

    def test_length(self):
        self.assertEqual(len(self.ds), 20)

    def test_output_keys(self):
        """Each sample should have img, gt_masks, video, and data_idx."""
        sample = self.ds[0]
        for key in ('img', 'gt_masks', 'video', 'data_idx'):
            self.assertIn(key, sample)

    def test_image_shape(self):
        """img tensor should be [T, C, H, W]."""
        sample = self.ds[0]
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))

    def test_mask_shape(self):
        """gt_masks tensor should be [T, K, H, W]."""
        sample = self.ds[0]
        self.assertEqual(sample['gt_masks'].shape, (6, 3, 64, 64))

    def test_image_value_range(self):
        """img values should be in [-1, 1] (normalised)."""
        sample = self.ds[0]
        self.assertGreaterEqual(sample['img'].min().item(), -1.0 - 1e-5)
        self.assertLessEqual(sample['img'].max().item(), 1.0 + 1e-5)

    def test_mask_value_range(self):
        """gt_masks values should be in [0, 1]."""
        sample = self.ds[0]
        self.assertGreaterEqual(sample['gt_masks'].min().item(), 0.0)
        self.assertLessEqual(sample['gt_masks'].max().item(), 1.0)

    def test_deterministic_with_seed(self):
        """Same seed should produce identical samples."""
        ds2 = GridShapesDataset(num_samples=20, num_frames=6, num_objects=3, img_size=64, seed=0)
        s1 = self.ds[5]['img']
        s2 = ds2[5]['img']
        self.assertTrue(torch.allclose(s1, s2))


# ── Dataset Factory ─────────────────────────────────────────────────────────────

class TestDatasetFactory(unittest.TestCase):
    def test_gridshapes_via_factory(self):
        """build_dataset should return a GridShapesDataset."""
        ds = build_dataset(
            {'dataset': {'name': 'gridshapes', 'num_samples': 10, 'num_frames': 4}},
            split='train'
        )
        self.assertEqual(len(ds), 10)
        sample = ds[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        self.assertEqual(sample['img'].shape, (4, 3, 64, 64))
        self.assertEqual(sample['gt_masks'].shape, (4, 3, 64, 64))

    def test_unknown_dataset_raises(self):
        """build_dataset with an unknown name should raise ValueError."""
        with self.assertRaises(ValueError):
            build_dataset({'dataset': {'name': 'nonexistent_dataset_xyz'}}, split='train')


class TestDataloaderFactory(unittest.TestCase):
    def test_gridshapes_dataloader_batch(self):
        """build_dataloader should yield correctly shaped batches."""
        dl = build_dataloader(
            {'dataset': {'name': 'gridshapes', 'num_samples': 16, 'num_frames': 4}},
            split='train',
            batch_size=4,
            num_workers=0,
        )
        batch = next(iter(dl))
        self.assertIn('img', batch)
        self.assertEqual(batch['img'].shape, (4, 4, 3, 64, 64))  # [B, T, C, H, W]


# ── PushT Real-File Integration (skipped if missing) ────────────────────────────

class TestPushTMaskDatasetIntegration(unittest.TestCase):
    H5_PATH = "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5"

    def setUp(self):
        if not os.path.exists(self.H5_PATH):
            self.skipTest(f"PushT dataset not found at {self.H5_PATH}. Skipping.")

    def test_dataset_length_nonzero(self):
        ds = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="train",
            resolution=(64, 64), n_sample_frames=6, frame_offset=1, train_frac=0.8
        )
        self.assertTrue(len(ds) > 0)

    def test_sample_shapes(self):
        ds = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="train",
            resolution=(64, 64), n_sample_frames=6, frame_offset=1, train_frac=0.8
        )
        sample = ds[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))
        self.assertEqual(sample['gt_masks'].shape, (6, 3, 64, 64))

    def test_mask_value_range(self):
        ds = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="train",
            resolution=(64, 64), n_sample_frames=6, frame_offset=1, train_frac=0.8
        )
        sample = ds[0]
        self.assertGreaterEqual(sample['gt_masks'].min().item(), 0.0)
        self.assertLessEqual(sample['gt_masks'].max().item(), 1.0)


if __name__ == '__main__':
    unittest.main()

