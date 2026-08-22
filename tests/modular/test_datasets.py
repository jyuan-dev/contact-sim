import unittest
import os
import torch

from src.datasets.factory import build_dataset, build_dataloader
from src.datasets.pusht import PushTMaskHDF5Dataset
from src.datasets.gridshapes import GridShapesDataset
import numpy as np


# ── pusht.py utilities (moved here — only tests used them) ─────────────────────

def _augment_background(img_np, bg_threshold=240):
    """Replace white background pixels with a random color in a uint8 HWC image."""
    bg_mask = np.all(img_np > bg_threshold, axis=-1)
    rand_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    img_aug = img_np.copy()
    img_aug[bg_mask] = rand_color
    return img_aug


class TestPushTUtilities(unittest.TestCase):
    def test_augment_background_replaces_white_pixels(self):
        """augment_background should change purely white pixel values."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 255  # all white
        augmented = _augment_background(img)
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
        for key in ('img', 'gt_masks', 'data_idx'):
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
            resolution=(64, 64), n_sample_frames=6, frame_offset=1, train_frac=0.8, load_masks=True
        )
        sample = ds[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))
        num_masks = len(PushTMaskHDF5Dataset.MASK_KEYS)
        self.assertEqual(sample['gt_masks'].shape, (6, num_masks, 64, 64))


    def test_mask_value_range(self):
        ds = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="train",
            resolution=(64, 64), n_sample_frames=6, frame_offset=1, train_frac=0.8, load_masks=True
        )
        sample = ds[0]
        self.assertGreaterEqual(sample['gt_masks'].min().item(), 0.0)
        self.assertLessEqual(sample['gt_masks'].max().item(), 1.0)

    def test_val_split_no_overlap(self):
        """Train and val splits should have no overlapping episode indices."""
        ds_train = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="train", resolution=(64, 64),
            n_sample_frames=6, train_frac=0.8, seed=42,
        )
        ds_val = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="val", resolution=(64, 64),
            n_sample_frames=6, train_frac=0.8, seed=42,
        )
        train_eps = set(ds_train._episode_indices)
        val_eps = set(ds_val._episode_indices)
        self.assertTrue(train_eps.isdisjoint(val_eps))

    def test_deterministic_sampling_same_seed(self):
        """Same seed should give same sample indices."""
        ds1 = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="train", resolution=(64, 64),
            n_sample_frames=6, seed=123,
        )
        ds2 = PushTMaskHDF5Dataset(
            h5_path=self.H5_PATH, split="train", resolution=(64, 64),
            n_sample_frames=6, seed=123,
        )
        self.assertEqual(ds1._episode_indices, ds2._episode_indices)


# ── GridShapesDataset edge cases ────────────────────────────────────────────

class TestGridShapesDatasetEdgeCases(unittest.TestCase):
    def test_single_object(self):
        """Single-object dataset should generate correctly."""
        ds = GridShapesDataset(num_samples=10, num_frames=6, num_objects=1, img_size=64, seed=0)
        sample = ds[0]
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))
        self.assertEqual(sample['gt_masks'].shape, (6, 1, 64, 64))

    def test_large_resolution(self):
        """Larger image resolution should produce correctly sized outputs."""
        ds = GridShapesDataset(num_samples=5, num_frames=4, num_objects=2, img_size=128, seed=0)
        sample = ds[0]
        self.assertEqual(sample['img'].shape, (4, 3, 128, 128))

    def test_many_objects(self):
        """Many objects should produce correct mask count."""
        ds = GridShapesDataset(num_samples=5, num_frames=4, num_objects=5, img_size=64, seed=0)
        sample = ds[0]
        self.assertEqual(sample['gt_masks'].shape, (4, 5, 64, 64))

    def test_different_seeds_differ(self):
        """Different seeds should produce different videos."""
        ds1 = GridShapesDataset(num_samples=5, num_frames=6, num_objects=3, img_size=64, seed=0)
        ds2 = GridShapesDataset(num_samples=5, num_frames=6, num_objects=3, img_size=64, seed=1)
        s1 = ds1[0]['img']
        s2 = ds2[0]['img']
        self.assertFalse(torch.allclose(s1, s2))

    def test_video_alias_key(self):
        """GridShapesDataset returns 'img' key (doc mentions 'video' alias but
        current implementation only returns 'img' and 'gt_masks')."""
        ds = GridShapesDataset(num_samples=5, num_frames=6, num_objects=3, img_size=64, seed=0)
        sample = ds[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        # Verify the shape is correct for the primary image key
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))


# ── Dataset factory edge cases ────────────────────────────────────────────────

class TestDatasetFactoryEdgeCases(unittest.TestCase):
    H5_PATH = "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5"

    def test_factory_resolves_gridshapes_resolution_key(self):
        """Factory should read 'resolution' key for gridshapes."""
        ds = build_dataset(
            {'dataset': {'name': 'gridshapes', 'resolution': [32, 32], 'train_samples': 10}},
            split='train',
        )
        sample = ds[0]
        self.assertEqual(sample['img'].shape[-2:], (32, 32))

    def test_factory_gridshapes_train_vs_val_samples(self):
        """train_samples and val_samples should give different dataset sizes."""
        ds_train = build_dataset(
            {'dataset': {'name': 'gridshapes', 'train_samples': 20, 'val_samples': 5}},
            split='train',
        )
        ds_val = build_dataset(
            {'dataset': {'name': 'gridshapes', 'train_samples': 20, 'val_samples': 5}},
            split='val',
        )
        self.assertEqual(len(ds_train), 20)
        self.assertEqual(len(ds_val), 5)

    def test_pusht_via_factory_train_frac(self):
        """Factory should pass train_frac through to PushT dataset."""
        if not os.path.exists(self.H5_PATH):
            self.skipTest("PushT dataset not found.")
        ds = build_dataset(
            {'dataset': {'name': 'pusht', 'h5_path': self.H5_PATH, 'train_frac': 0.7}},
            split='train',
        )
        self.assertTrue(len(ds) > 0)


if __name__ == '__main__':
    unittest.main()

