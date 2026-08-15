"""
Tests for the canonical deterministic eval path:
  - greedy_slot_assignments (swap tracking)
  - DeterministicEvaluator (batch-level accumulation + finalize)
  - DeterministicEpisodeEvalDataset multi-episode val coverage
"""

import os
import unittest

import torch

from src.metrics import DeterministicEvaluator, greedy_slot_assignments


def _square_masks(positions):
    """Build [T, K, H, W] binary masks; positions: T-length list of K (x0, y0) tuples."""
    T = len(positions)
    K = len(positions[0])
    masks = torch.zeros(T, K, 64, 64)
    for t in range(T):
        for k, (x0, y0) in enumerate(positions[t]):
            masks[t, k, y0:y0 + 16, x0:x0 + 16] = 1.0
    return masks


class TestGreedySlotAssignments(unittest.TestCase):
    def test_static_assignment_no_swap(self):
        pos = [((8, 8), (40, 40))] * 4  # slots stay put
        pred = _square_masks(pos)
        gt = _square_masks(pos)
        out = greedy_slot_assignments(pred.unsqueeze(0), gt.unsqueeze(0))
        self.assertEqual(out['swap_transitions'], 0)
        self.assertEqual(out['total_transitions'], 3)
        self.assertFalse(out['seq_records'][0]['swapped'])

    def test_swap_detected(self):
        pos = [((8, 8), (40, 40)), ((8, 8), (40, 40)), ((40, 40), (8, 8)), ((40, 40), (8, 8))]
        pred = _square_masks(pos)
        gt = _square_masks([((8, 8), (40, 40))] * 4)
        out = greedy_slot_assignments(pred.unsqueeze(0), gt.unsqueeze(0))
        self.assertEqual(out['swap_transitions'], 1)
        self.assertEqual(out['seq_records'][0]['swap_count'], 1)
        self.assertTrue(out['seq_records'][0]['swapped'])
        self.assertEqual(out['frame_swaps'][2], [1, 1])  # swap enters frame 2 (t=2)

    def test_shapes(self):
        pred = torch.rand(2, 5, 3, 64, 64)
        gt = torch.rand(2, 5, 4, 64, 64)
        out = greedy_slot_assignments(pred, gt)
        self.assertEqual(out['assignments'].shape, (2, 5, 3))
        self.assertEqual(out['iou_matrices'].shape, (2, 5, 3, 4))
        self.assertEqual(out['total_transitions'], 2 * 4)


class TestDeterministicEvaluator(unittest.TestCase):
    def test_update_and_finalize(self):
        evaluator = DeterministicEvaluator(num_classes=3, thresh=0.5)

        pos = [((8, 8), (40, 40)), ((8, 8), (40, 40)), ((40, 40), (8, 8)), ((40, 40), (8, 8))]
        gt = _square_masks([((8, 8), (40, 40))] * 4)
        pred = _square_masks(pos)
        video = torch.rand(2, 4, 3, 64, 64)
        recon = video  # perfect reconstruction → mse 0

        evaluator.update(
            pred_masks=pred.unsqueeze(0).expand(2, -1, -1, -1, -1),
            gt_masks=gt.unsqueeze(0).expand(2, -1, -1, -1, -1),
            recon=recon,
            video=video,
            episode_idx=torch.tensor([101, 102]),
            start_frame=torch.tensor([5, 6]),
        )

        raw = evaluator.finalize()

        self.assertEqual(raw['summary']['total_sequences'], 2)
        self.assertAlmostEqual(raw['summary']['val_mse']['mean'], 0.0, places=6)
        self.assertAlmostEqual(raw['summary']['slot_swapping_rate'], 1.0 / 3.0)  # 1 swap / 3 transitions per seq
        # Per-slot IoU is index-wise (slot k vs GT k): slot 0 matches GT 0 for
        # half the frames (pre-swap) and misses for the other half.
        self.assertAlmostEqual(raw['per_slot']['slot_0']['iou']['mean'], 0.5, places=5)
        self.assertEqual(len(raw['per_frame']), 4)
        records = raw['per_sequence']
        self.assertEqual(records[0]['episode_idx'], 101)
        self.assertEqual(records[0]['start_frame'], 5)
        self.assertEqual(records[1]['episode_idx'], 102)
        self.assertEqual(records[0]['swap_count'], 1)
        self.assertAlmostEqual(records[0]['miou'], 0.5, places=5)

    def test_missing_masks_skips_mask_metrics(self):
        evaluator = DeterministicEvaluator(num_classes=3)
        video = torch.rand(2, 4, 3, 64, 64)
        evaluator.update(pred_masks=None, gt_masks=None, recon=video, video=video)
        raw = evaluator.finalize()
        self.assertEqual(raw['summary']['total_sequences'], 0)
        self.assertEqual(raw['summary']['val_mse']['mean'], 0.0)


class TestDeterministicEpisodeEvalDatasetCoverage(unittest.TestCase):
    @unittest.skipUnless(
        os.path.exists('/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5'),
        "PushT h5 dataset not available",
    )
    def test_val_index_spans_multiple_episodes(self):
        from src.datasets.pusht import DeterministicEpisodeEvalDataset

        ds = DeterministicEpisodeEvalDataset(
            h5_path='/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5',
            split='val',
            resolution=(64, 64),
            n_sample_frames=6,
            clips_per_episode=2,
            base_seed=42,
        )
        episodes = {ep for ep, _ in ds._index}
        self.assertGreater(len(episodes), 1)  # coverage bug fix: index spans all episodes
        self.assertEqual(len(ds._index), len(ds))
        # every clip belongs to a val episode
        self.assertTrue(all(ep in ds._episode_indices for ep, _ in ds._index))
        del ds  # close h5 handle


if __name__ == "__main__":
    unittest.main()
