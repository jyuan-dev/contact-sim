"""
Tests for predict_slot_rollout — the shared autoregressive rollout path
used by scripts/rollout.py, eval_full_sequence_rollout.py,
rollout_full_episodes.py, and eval_multi_swap_rollout.py.
"""

import unittest

import torch
import torch.nn as nn

from src.models.rollout import predict_slot_rollout
from src.models.factory import build_model


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

    def forward(self, cond_slots, pred_len=2):
        B = cond_slots.shape[0]
        return torch.zeros(B, pred_len, self.K, self.D, device=cond_slots.device)


class TestPredictSlotRollout(unittest.TestCase):
    def test_output_contract(self):
        wrapper = _make_savi_wrapper()
        out = predict_slot_rollout(wrapper, _video(), n_cond_frames=2)

        for key in ("input_img", "pred_masks", "recon_img", "post_slots", "is_rollout_mask"):
            self.assertIn(key, out)
        B, T, C, H, W = 2, 4, 3, 64, 64
        self.assertEqual(tuple(out["post_slots"].shape), (B, T, 2, 32))
        self.assertEqual(tuple(out["pred_masks"].shape), (B, T, 2, H, W))
        self.assertEqual(tuple(out["recon_img"].shape), (B, T, C, H, W))

    def test_is_rollout_mask(self):
        wrapper = _make_savi_wrapper()
        out = predict_slot_rollout(wrapper, _video(), n_cond_frames=2)
        self.assertTrue(torch.equal(out["is_rollout_mask"],
                                    torch.tensor([False, False, True, True])))

    def test_conditioned_slots_match_encode(self):
        """Cond frames must equal the canonical StoSAVi.encode output."""
        wrapper = _make_savi_wrapper()
        video = _video()
        out = predict_slot_rollout(wrapper, video, n_cond_frames=2)

        inner = wrapper.inner_savi()
        inner._reset_rnn()
        cond_slots, _ = inner.encode(video[:, :2])
        self.assertTrue(torch.allclose(out["post_slots"][:, :2], cond_slots, atol=1e-6, rtol=1e-6))

    def test_full_conditioning_skips_rollout(self):
        """n_cond_frames == T: no rollout frames, all slots come from encode."""
        wrapper = _make_savi_wrapper()
        video = _video()
        out = predict_slot_rollout(wrapper, video, n_cond_frames=4)

        inner = wrapper.inner_savi()
        inner._reset_rnn()
        full_slots, _ = inner.encode(video)
        self.assertTrue(torch.allclose(out["post_slots"], full_slots, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(out["is_rollout_mask"], torch.tensor([False] * 4)))

    def test_stage2_rollouter_branch(self):
        """Wrapper-of-model with rollouter + stage1: rollout frames come from the rollouter."""
        wrapper = _make_savi_wrapper()

        class _FakeInner(nn.Module):
            def __init__(self, stage1):
                super().__init__()
                self.rollouter = _FakeRollouter(K=2, D=32)
                self.stage1_model = stage1

        class _Stage2Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner

        container = _Stage2Wrapper(_FakeInner(wrapper))
        out = predict_slot_rollout(container, _video(), n_cond_frames=2)

        # Rollout frames (t >= 2) come from the fake rollouter -> zeros
        self.assertTrue(torch.all(out["post_slots"][:, 2:] == 0.0))
        # Cond frames come from the real encode path -> non-zero
        self.assertTrue(out["post_slots"][:, :2].abs().sum() > 0)


if __name__ == "__main__":
    unittest.main()
