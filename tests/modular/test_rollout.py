"""
Tests for BaseModelWrapper.rollout — the standardized autoregressive rollout method
implemented across StandardizedSAViWrapper, StandardizedSlotFormerWrapper, and StandardizedLeWMWrapper.
"""

import unittest

import torch
import torch.nn as nn

from src.models.factory import build_model
from src.models.wrappers.savi_wrapper import StandardizedSAViWrapper
from src.models.wrappers.slotformer_wrapper import StandardizedSlotFormerWrapper
from src.models.wrappers.lewm_wrapper import StandardizedLeWMWrapper


def _make_savi_wrapper():
    cfg = {
        "model": {
            "name": "savi",
            "type": "savi",
            "num_slots": 2,
            "slot_dim": 32,
            "resolution": [64, 64],
            "n_sample_frames": 4,
        }
    }
    return build_model(cfg).eval()


def _video(B=2, T=4):
    return torch.randn(B, T, 3, 64, 64)


class _FakeRollouter(nn.Module):
    """Returns zero rollout slots; lets us exercise the stage-2 branch cheaply."""

    def __init__(self, K=2, D=32):
        super().__init__()
        self.K = K
        self.D = D
        self.received_kwargs = None

    def forward(self, cond_slots, pred_len=2, actions=None, goal_slots=None, **kwargs):
        B = cond_slots.shape[0]
        self.received_kwargs = {"actions": actions, "goal_slots": goal_slots}
        return torch.zeros(B, pred_len, self.K, self.D, device=cond_slots.device)


class TestModelWrapperRollout(unittest.TestCase):
    def test_savi_output_contract(self):
        wrapper = _make_savi_wrapper()
        out = wrapper.rollout(_video(), n_cond_frames=2)

        for key in ("input_img", "pred_masks", "recon_img", "post_slots", "is_rollout_mask"):
            self.assertIn(key, out)
        B, T, C, H, W = 2, 4, 3, 64, 64
        self.assertEqual(tuple(out["post_slots"].shape), (B, T, 2, 32))
        self.assertEqual(tuple(out["pred_masks"].shape), (B, T, 2, H, W))
        self.assertEqual(tuple(out["recon_img"].shape), (B, T, C, H, W))

    def test_is_rollout_mask(self):
        wrapper = _make_savi_wrapper()
        out = wrapper.rollout(_video(), n_cond_frames=2)
        self.assertTrue(torch.equal(out["is_rollout_mask"],
                                    torch.tensor([False, False, True, True])))

    def test_conditioned_slots_match_encode(self):
        """Cond frames must equal the canonical StoSAVi.encode output."""
        wrapper = _make_savi_wrapper()
        video = _video()
        out = wrapper.rollout(video, n_cond_frames=2)

        inner = wrapper.inner_savi()
        inner._reset_rnn()
        cond_slots, _ = inner.encode(video[:, :2])
        self.assertTrue(torch.allclose(out["post_slots"][:, :2], cond_slots, atol=1e-6, rtol=1e-6))

    def test_full_conditioning_skips_rollout(self):
        """n_cond_frames == T: no rollout frames, all slots come from encode."""
        wrapper = _make_savi_wrapper()
        video = _video()
        out = wrapper.rollout(video, n_cond_frames=4)

        inner = wrapper.inner_savi()
        inner._reset_rnn()
        full_slots, _ = inner.encode(video)
        self.assertTrue(torch.allclose(out["post_slots"], full_slots, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(out["is_rollout_mask"], torch.tensor([False] * 4)))

    def test_slotformer_stage2_rollouter(self):
        """SlotFormer wrapper rollout unrolls future slots via rollouter."""
        stage1 = _make_savi_wrapper()
        fake_rollouter = _FakeRollouter(K=2, D=32)

        class _FakeSlotFormer(nn.Module):
            def __init__(self, stage1, rollouter):
                super().__init__()
                self.stage1_model = stage1
                self.rollouter = rollouter

            def extract_slots(self, video):
                return self.stage1_model.encode_slots(video)

        container = StandardizedSlotFormerWrapper(_FakeSlotFormer(stage1, fake_rollouter))
        actions = torch.randn(2, 4, 2)
        goal_slots = torch.randn(2, 2, 32)

        out = container.rollout(_video(), n_cond_frames=2, actions=actions, goal_slots=goal_slots)
        self.assertIs(fake_rollouter.received_kwargs["actions"], actions)
        self.assertIs(fake_rollouter.received_kwargs["goal_slots"], goal_slots)

        # Rollout frames (t >= 2) come from the fake rollouter -> zeros
        self.assertTrue(torch.all(out["post_slots"][:, 2:] == 0.0))
        # Cond frames come from the real encode path -> non-zero
        self.assertTrue(out["post_slots"][:, :2].abs().sum() > 0)

    def test_lewm_rollout(self):
        """LeWM wrapper rollout unrolls representations and returns is_rollout_mask."""
        cfg = {
            "model": {
                "type": "lewm",
                "resolution": [32, 32],
                "in_channels": 3,
                "action_dim": 2,
                "embed_dim": 32,
                "hidden_dim": 64,
                "num_frames": 8,
                "predictor": {
                    "depth": 2,
                    "heads": 2,
                    "dim_head": 16,
                    "mlp_dim": 64,
                },
            }
        }
        lewm_wrapper = build_model(cfg).eval()
        self.assertIsInstance(lewm_wrapper, StandardizedLeWMWrapper)

        video = torch.randn(2, 2, 3, 32, 32)
        actions = torch.randn(2, 6, 2)
        out = lewm_wrapper.rollout(video, actions=actions, n_cond_frames=2)

        self.assertIn("post_slots", out)
        self.assertIn("is_rollout_mask", out)
        self.assertEqual(out["post_slots"].shape, (2, 6, 1, 32))
        self.assertTrue(torch.equal(out["is_rollout_mask"],
                                    torch.tensor([False, False, True, True, True, True])))


if __name__ == "__main__":
    unittest.main()
