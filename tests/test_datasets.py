import unittest
import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.factory import build_dataset, build_dataloader
from src.datasets.pusht import PushTMaskHDF5Dataset
from src.datasets.gridshapes import GridShapesDataset


class TestDatasets(unittest.TestCase):
    def test_imports(self):
        """Verify that all retained dataset modules can be imported correctly."""
        self.assertIsNotNone(PushTMaskHDF5Dataset)
        self.assertIsNotNone(GridShapesDataset)
        self.assertIsNotNone(build_dataset)
        self.assertIsNotNone(build_dataloader)

    def test_gridshapes_dataset(self):
        """Test instantiation and output structure of GridShapes synthetic dataset."""
        dataset = build_dataset({'dataset': {'name': 'gridshapes', 'num_samples': 10, 'num_frames': 4}}, split='train')
        self.assertEqual(len(dataset), 10)
        sample = dataset[0]
        self.assertIn('img', sample)
        self.assertIn('gt_masks', sample)
        self.assertEqual(sample['img'].shape, (4, 3, 64, 64))
        self.assertEqual(sample['gt_masks'].shape, (4, 3, 64, 64))

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
