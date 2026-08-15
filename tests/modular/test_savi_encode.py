"""
Tests for StoSAVi.encode — the canonical clip→post-slots procedure.

Verifies that the native encode() is numerically equivalent to the two
hand-rolled loop variants it replaced (per-frame encoder loop from rollout.py,
flattened-encoder loop from slotformer.py/pidm.py/extract_slots.py), and that
the two-phase conditioned pattern used by rollout is equivalent to a single
full-clip encode.
"""

import unittest

import torch

from src.models.savi import SAVi


def _build_model():
    return SAVi(
        resolution=(64, 64),
        clip_len=4,
        num_slots=3,
        slot_dim=32,
        num_iterations=2,
        use_encoder_bn=True,
        use_residual_bn=True,
    )


class TestStoSAViEncodeEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _build_model().eval()
        self.inner = self.model.model
        self.video = torch.randn(2, 4, 3, 64, 64)

    def _reference_per_frame_loop(self, video):
        """Old rollout.py conditioning loop: encoder called frame-by-frame."""
        B, T, C, H, W = video.shape
        init_latents = self.inner.init_latents.repeat(B, 1, 1)
        prev_slots = None
        all_slots = []
        for t in range(T):
            enc_out_t = self.inner._get_encoder_out(video[:, t])
            latents = init_latents if prev_slots is None else self.inner.predictor(prev_slots)
            post_slots = self.inner.slot_attention(
                enc_out_t, self.inner._sample_dist(self.inner.kernel_dist_layer(latents)))
            all_slots.append(post_slots)
            prev_slots = post_slots
        return torch.stack(all_slots, dim=1)

    def _reference_flattened_loop(self, video):
        """Old slotformer/pidm/extract_slots loop: encoder over B*T at once."""
        B, T, C, H, W = video.shape
        enc_out_all = self.inner._get_encoder_out(video.flatten(0, 1)).unflatten(0, (B, T))
        init_latents = self.inner.init_latents.repeat(B, 1, 1)
        prev_slots = None
        all_slots = []
        for t in range(T):
            latents = init_latents if prev_slots is None else self.inner.predictor(prev_slots)
            post_slots = self.inner.slot_attention(
                enc_out_all[:, t], self.inner._sample_dist(self.inner.kernel_dist_layer(latents)))
            all_slots.append(post_slots)
            prev_slots = post_slots
        return torch.stack(all_slots, dim=1)

    def test_encode_matches_per_frame_reference(self):
        self.inner._reset_rnn()
        post_slots, _ = self.inner.encode(self.video)
        self.inner._reset_rnn()  # reference must start from a fresh RNN state too
        reference = self._reference_per_frame_loop(self.video)
        self.assertEqual(post_slots.shape, (2, 4, 3, 32))
        self.assertTrue(torch.allclose(post_slots, reference, atol=1e-6, rtol=1e-6))

    def test_encode_matches_flattened_reference(self):
        self.inner._reset_rnn()
        post_slots, _ = self.inner.encode(self.video)
        self.inner._reset_rnn()  # reference must start from a fresh RNN state too
        reference = self._reference_flattened_loop(self.video)
        self.assertTrue(torch.allclose(post_slots, reference, atol=1e-6, rtol=1e-6))

    def test_two_phase_conditioned_encode_matches_full_encode(self):
        """rollout.py pattern: encode(cond) + encode(rest, prev_slots) == encode(full)."""
        self.inner._reset_rnn()
        full, _ = self.inner.encode(self.video)

        self.inner._reset_rnn()
        cond, _ = self.inner.encode(self.video[:, :2])
        rest, _ = self.inner.encode(self.video[:, 2:], prev_slots=cond[:, -1])

        self.assertTrue(torch.allclose(torch.cat([cond, rest], dim=1), full, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
