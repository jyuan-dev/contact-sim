"""
Tests for INTACT (intent-to-action operator).
"""

import unittest

import torch

from src.models.intact import INTACT, Intact, RobotSlotIntentActionActor


class TestINTACT(unittest.TestCase):
    def _make(self, **overrides):
        torch.manual_seed(0)
        params = dict(slot_dim=16, action_dim=2, action_emb_dim=8,
                      robot_slot_idx=0, hidden_dim=32, num_heads=4, depth=2,
                      min_log_std=-5.0, max_log_std=2.0)
        params.update(overrides)
        return INTACT(**params).eval()

    def test_aliases(self):
        self.assertIs(Intact, INTACT)
        self.assertIs(RobotSlotIntentActionActor, INTACT)

    def test_forward_shapes_and_clamp(self):
        actor = self._make(max_log_std=0.5, min_log_std=-2.0)
        z_curr = torch.randn(3, 4, 16)  # [B, K, D]
        z_next = torch.randn(3, 4, 16)
        mean, log_std = actor(z_curr, z_next)

        self.assertEqual(tuple(mean.shape), (3, 2))
        self.assertEqual(tuple(log_std.shape), (3, 2))
        self.assertTrue(torch.all(log_std <= 0.5 + 1e-6))
        self.assertTrue(torch.all(log_std >= -2.0 - 1e-6))

    def test_extract_features_with_prev_action(self):
        actor = self._make()
        z_curr = torch.randn(3, 4, 16)
        z_next = torch.randn(3, 4, 16)
        prev = torch.randn(3, 2)

        feat = actor.extract_features(z_curr, z_next, prev_action=prev)
        self.assertEqual(tuple(feat.shape), (3, 32 + 8))  # hidden + action_emb

        feat_none = actor.extract_features(z_curr, z_next, prev_action=None)
        self.assertEqual(tuple(feat_none.shape), (3, 32 + 8))  # zero-filled emb

    def test_extract_features_without_action_encoder(self):
        actor = self._make(action_emb_dim=0)
        z_curr = torch.randn(3, 4, 16)
        feat = actor.extract_features(z_curr, z_next=torch.randn(3, 4, 16))
        self.assertEqual(tuple(feat.shape), (3, 32))  # hidden only

    def test_shape_mismatch_raises(self):
        actor = self._make()
        with self.assertRaises((ValueError, TypeError)):
            actor.extract_features(torch.randn(3, 4, 16), torch.randn(3, 5, 16))

    def test_action_nll(self):
        actor = self._make()
        z_curr = torch.randn(3, 4, 16)
        z_next = torch.randn(3, 4, 16)
        target = torch.randn(3, 2)

        out = actor.action_nll(z_curr, z_next, target, reduction="mean")
        for key in ("loss", "nll", "mean", "log_std", "action_mae", "action_rmse"):
            self.assertIn(key, out)
        self.assertEqual(out["loss"].shape, torch.Size([]))
        self.assertEqual(tuple(out["nll"].shape), (3,))

        out_none = actor.action_nll(z_curr, z_next, target, reduction="none")
        self.assertEqual(out_none["loss"].shape, torch.Size([3]))

    def test_invalid_reduction_raises(self):
        actor = self._make()
        with self.assertRaises(ValueError):
            actor.action_nll(torch.randn(3, 4, 16), torch.randn(3, 4, 16),
                             torch.randn(3, 2), reduction="bogus")


if __name__ == "__main__":
    unittest.main()
