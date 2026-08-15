"""
Input-validation tests — every public seam raises loudly on wrong types
and shapes instead of failing deep inside with a cryptic broadcast error.
"""

import unittest

import torch

from src.models.savi import SAVi
from src.models.rollout import predict_slot_rollout
from src.models.intact_actor import RobotSlotIntentActionActor
from src.models.factory import build_model
from src.metrics import DeterministicEvaluator, greedy_slot_assignments
from src.losses.sigreg import SIGRegLoss, compute_sigreg_statistic
from src.utils.checkpoint_bootstrap import bootstrap_checkpoint


def _make_savi():
    return SAVi(resolution=(64, 64), clip_len=4, num_slots=2, slot_dim=32, num_iterations=1).eval()


class TestSAViInputValidation(unittest.TestCase):
    def setUp(self):
        self.model = _make_savi()

    def test_forward_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            self.model("not a tensor")

    def test_forward_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 4, 64, 64))  # 4-D, missing channel

    def test_core_forward_requires_img_key(self):
        with self.assertRaises(ValueError):
            self.model.model({"video": torch.randn(2, 4, 3, 64, 64)})

    def test_encode_rejects_wrong_ndim(self):
        with self.assertRaises(TypeError):
            self.model.model.encode(torch.randn(2, 3, 64, 64))

    def test_encode_rejects_bad_prev_slots(self):
        video = torch.randn(2, 4, 3, 64, 64)
        with self.assertRaises(ValueError):
            self.model.model.encode(video, prev_slots=torch.randn(2, 32))  # 2-D
        with self.assertRaises(ValueError):
            self.model.model.encode(video, prev_slots=torch.randn(3, 2, 32))  # B mismatch

    def test_decode_rejects_wrong_ndim(self):
        with self.assertRaises(TypeError):
            self.model.model.decode(torch.randn(2, 2, 2, 32))  # 4-D

    def test_slot_attention_rejects_wrong_ndim(self):
        sa = self.model.model.slot_attention
        with self.assertRaises(TypeError):
            sa(torch.randn(2, 64, 64), torch.randn(2, 2, 2, 32))


class TestRolloutInputValidation(unittest.TestCase):
    def test_video_wrong_ndim(self):
        wrapper = build_model({"model": {"name": "savi", "type": "savi",
                                         "num_slots": 2, "slot_dim": 32, "resolution": [64, 64]}}).eval()
        with self.assertRaises(TypeError):
            predict_slot_rollout(wrapper, torch.randn(2, 4, 64, 64))

    def test_n_cond_frames_bounds(self):
        wrapper = build_model({"model": {"name": "savi", "type": "savi",
                                         "num_slots": 2, "slot_dim": 32, "resolution": [64, 64]}}).eval()
        video = torch.randn(2, 4, 3, 64, 64)
        for bad in (0, 5):
            with self.assertRaises(ValueError):
                predict_slot_rollout(wrapper, video, n_cond_frames=bad)
        with self.assertRaises(TypeError):
            predict_slot_rollout(wrapper, video, n_cond_frames=2.5)


class TestMetricsInputValidation(unittest.TestCase):
    def test_greedy_assignments_shape_mismatch(self):
        """Same-named dims (B/T/H/W) are cross-checked by the annotations."""
        with self.assertRaises(TypeError):
            greedy_slot_assignments(torch.rand(2, 4, 2, 64, 64), torch.rand(3, 4, 2, 64, 64))  # B mismatch
        with self.assertRaises(TypeError):
            greedy_slot_assignments(torch.rand(2, 4, 2, 64, 64), torch.rand(2, 4, 2, 32, 32))  # H/W mismatch

    def test_greedy_assignments_non_tensor(self):
        with self.assertRaises(TypeError):
            greedy_slot_assignments("masks", torch.rand(2, 4, 2, 64, 64))

    def test_evaluator_update_shape_mismatch(self):
        evaluator = DeterministicEvaluator(num_classes=3)
        with self.assertRaises(ValueError):
            evaluator.update(pred_masks=torch.rand(2, 4, 2, 64, 64),
                             gt_masks=torch.rand(2, 5, 2, 64, 64))  # T mismatch
        with self.assertRaises(ValueError):
            evaluator.update(pred_masks=torch.rand(2, 4, 2, 64, 64),
                             gt_masks=torch.rand(2, 4, 2, 64, 64),
                             recon=torch.rand(2, 4, 3, 64, 64),
                             video=torch.rand(3, 4, 3, 64, 64))  # B mismatch


class TestSigRegInputValidation(unittest.TestCase):
    def test_statistic_rejects_wrong_ndim(self):
        with self.assertRaises(TypeError):
            compute_sigreg_statistic(torch.randn(4, 6, 3), num_proj=8)  # 3-D

    def test_loss_rejects_wrong_ndim_tensor(self):
        with self.assertRaises(ValueError):
            SIGRegLoss(num_proj=8)(torch.randn(4, 6, 3))

    def test_loss_still_tolerates_none_and_small_batch(self):
        loss_fn = SIGRegLoss(num_proj=8)
        weighted, info = loss_fn({"post_slots": None})
        self.assertEqual(weighted.item(), 0.0)
        weighted, info = loss_fn(torch.randn(1, 4, 3, 32))
        self.assertEqual(weighted.item(), 0.0)


class TestIntactActorInputValidation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.actor = RobotSlotIntentActionActor(slot_dim=16, action_dim=2,
                                                action_emb_dim=8, hidden_dim=32).eval()

    def test_wrong_ndim_raises(self):
        with self.assertRaises(TypeError):
            self.actor(torch.randn(3, 16), torch.randn(3, 16))  # 2-D slots

    def test_prev_action_batch_mismatch(self):
        with self.assertRaises(ValueError):
            self.actor.extract_features(torch.randn(3, 4, 16), torch.randn(3, 4, 16),
                                        prev_action=torch.randn(2, 2))  # B mismatch


class TestBootstrapInputValidation(unittest.TestCase):
    def test_non_string_ckpt_path(self):
        with self.assertRaises(TypeError):
            bootstrap_checkpoint(12345)

    def test_bad_cli_overrides(self):
        with self.assertRaises(TypeError):
            bootstrap_checkpoint("some/path.pt", cli_overrides="device=cpu")


if __name__ == "__main__":
    unittest.main()
