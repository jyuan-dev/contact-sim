import unittest
import os
import sys
import torch

# Ensure workspace root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.pusht import PushTMaskHDF5Dataset, PushTDataset
from src.datasets.ogbench import OGBenchCubeDataset
from src.datasets.libero import LiberoDataset

class TestDatasets(unittest.TestCase):
    def test_imports(self):
        """Verify that all dataset modules can be imported correctly."""
        self.assertIsNotNone(PushTMaskHDF5Dataset)
        self.assertIsNotNone(PushTDataset)
        self.assertIsNotNone(OGBenchCubeDataset)
        self.assertIsNotNone(LiberoDataset)

    def test_ogbench_stub(self):
        """Test instantiation and output structure of the OGBench CUBE stub loader."""
        dataset = OGBenchCubeDataset(
            data_path="dummy_path.h5",
            split="train",
            resolution=(64, 64),
            n_sample_frames=6
        )
        self.assertEqual(len(dataset), 950)
        sample = dataset[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))
        self.assertEqual(sample['gt_masks'].shape, (6, 3, 64, 64))

    def test_libero_stub(self):
        """Test instantiation and output structure of the LIBERO stub loader."""
        dataset = LiberoDataset(
            data_path="dummy_path.h5",
            split="train",
            resolution=(64, 64),
            n_sample_frames=6
        )
        self.assertEqual(len(dataset), 100)
        sample = dataset[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))
        self.assertEqual(sample['gt_masks'].shape, (6, 3, 64, 64))

    def test_pusht_mask_dataset_real_or_stub(self):
        """Verify PushTMaskHDF5Dataset loads successfully if the dataset exists."""
        h5_path = "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5"
        if not os.path.exists(h5_path):
            self.skipTest(f"PushT dataset file missing at {h5_path}. Skipping real loading test.")
            
        dataset = PushTMaskHDF5Dataset(
            h5_path=h5_path,
            split="train",
            resolution=(64, 64),
            n_sample_frames=6,
            frame_offset=1,
            train_frac=0.8
        )
        self.assertTrue(len(dataset) > 0)
        sample = dataset[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        self.assertEqual(sample['img'].shape, (6, 3, 64, 64))
        self.assertEqual(sample['gt_masks'].shape, (6, 3, 64, 64))

if __name__ == '__main__':
    unittest.main()
