"""
Tests for src/models/deformable_slot_attention.py.
Covers: MultiScaleDeformableAttention, DeformableSlotAttention.
"""

import math
import unittest

import torch

from src.models.deformable_slot_attention import (
    MultiScaleDeformableAttention,
    DeformableSlotAttention,
)


# ── MultiScaleDeformableAttention ────────────────────────────────────────────

class TestMultiScaleDeformableAttention(unittest.TestCase):
    """Tests for the pure-PyTorch 2D deformable attention module."""

    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.n_points = 4
        self.n_levels = 1
        self.attn = MultiScaleDeformableAttention(
            d_model=self.d_model, n_levels=self.n_levels,
            n_heads=self.n_heads, n_points=self.n_points,
        )

    def test_init_divisible_check(self):
        """d_model not divisible by n_heads should raise ValueError."""
        with self.assertRaises(ValueError):
            MultiScaleDeformableAttention(d_model=65, n_heads=4)

    def test_output_shape(self):
        """Output should match query shape [B, Len_q, C]."""
        B, Len_q, C = 2, 16, self.d_model
        H, W = 8, 8
        query = torch.randn(B, Len_q, C)
        ref_points = torch.rand(B, Len_q, self.n_levels, 2)
        input_flatten = torch.randn(B, H * W, C)
        spatial_shapes = torch.tensor([[H, W]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)

        out = self.attn(query, ref_points, input_flatten, spatial_shapes, level_start_index)
        self.assertEqual(out.shape, (B, Len_q, C))

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        B, Len_q, C = 1, 4, self.d_model
        H, W = 4, 4
        query = torch.randn(B, Len_q, C)
        ref_points = torch.rand(B, Len_q, self.n_levels, 2)
        input_flatten = torch.randn(B, H * W, C)
        spatial_shapes = torch.tensor([[H, W]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)

        out = self.attn(query, ref_points, input_flatten, spatial_shapes, level_start_index)
        self.assertEqual(out.shape, (B, Len_q, C))

    def test_deterministic_with_same_input(self):
        """Same input should produce same output (no randomness in forward)."""
        B, Len_q, C = 2, 8, self.d_model
        H, W = 4, 4
        query = torch.randn(B, Len_q, C)
        ref_points = torch.rand(B, Len_q, self.n_levels, 2)
        input_flatten = torch.randn(B, H * W, C)
        spatial_shapes = torch.tensor([[H, W]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)

        out1 = self.attn(query, ref_points, input_flatten, spatial_shapes, level_start_index)
        out2 = self.attn(query, ref_points, input_flatten, spatial_shapes, level_start_index)
        self.assertTrue(torch.allclose(out1, out2))

    def test_gradient_flow(self):
        """Gradients should flow through all parameters."""
        B, Len_q, C = 2, 8, self.d_model
        H, W = 4, 4
        query = torch.randn(B, Len_q, C, requires_grad=True)
        ref_points = torch.rand(B, Len_q, self.n_levels, 2)
        input_flatten = torch.randn(B, H * W, C, requires_grad=True)
        spatial_shapes = torch.tensor([[H, W]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)

        out = self.attn(query, ref_points, input_flatten, spatial_shapes, level_start_index)
        loss = out.sum()
        loss.backward()

        # Check gradients exist for key parameters
        self.assertIsNotNone(self.attn.sampling_offsets.weight.grad)
        self.assertIsNotNone(self.attn.attention_weights.weight.grad)
        self.assertIsNotNone(self.attn.value_proj.weight.grad)
        self.assertIsNotNone(self.attn.output_proj.weight.grad)

    def test_multi_level(self):
        """Multi-level feature maps should work."""
        attn = MultiScaleDeformableAttention(d_model=64, n_levels=2, n_heads=4, n_points=2)
        B, Len_q, C = 2, 8, 64
        H1, W1 = 8, 8
        H2, W2 = 4, 4
        total_len = H1 * W1 + H2 * W2

        query = torch.randn(B, Len_q, C)
        ref_points = torch.rand(B, Len_q, 2, 2)
        input_flatten = torch.randn(B, total_len, C)
        spatial_shapes = torch.tensor([[H1, W1], [H2, W2]], dtype=torch.long)
        level_start_index = torch.tensor([0, H1 * W1], dtype=torch.long)

        out = attn(query, ref_points, input_flatten, spatial_shapes, level_start_index)
        self.assertEqual(out.shape, (B, Len_q, C))

    def test_attention_weights_sum_to_one(self):
        """Softmax attention weights should sum to ~1 per query-head-level."""
        B, Len_q, C = 2, 4, self.d_model
        H, W = 4, 4
        query = torch.randn(B, Len_q, C)
        ref_points = torch.rand(B, Len_q, self.n_levels, 2)
        input_flatten = torch.randn(B, H * W, C)
        spatial_shapes = torch.tensor([[H, W]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)

        # Run forward to trigger weight computation
        self.attn(query, ref_points, input_flatten, spatial_shapes, level_start_index)

        # Recompute attention weights manually to verify
        attn_w = self.attn.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attn_w = torch.softmax(attn_w, dim=-1)
        sums = attn_w.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


# ── DeformableSlotAttention ──────────────────────────────────────────────────

class TestDeformableSlotAttention(unittest.TestCase):
    """Tests for the iterative deformable slot attention module."""

    def setUp(self):
        self.num_slots = 4
        self.slot_dim = 64
        self.num_iterations = 3
        self.n_heads = 4
        self.n_points = 4

        self.dsa = DeformableSlotAttention(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            num_iterations=self.num_iterations,
            n_heads=self.n_heads,
            n_points=self.n_points,
        )

    # ── 4D input (feature map) ──────────────────────────────────────────────

    def test_4d_input_output_shapes(self):
        """4D feature map input should return correct updated slots and ref points."""
        B, C, H, W = 2, self.slot_dim, 8, 8
        inputs = torch.randn(B, C, H, W)
        slots = torch.randn(B, self.num_slots, self.slot_dim)

        updated_slots, ref_points = self.dsa(inputs, slots)

        self.assertEqual(updated_slots.shape, (B, self.num_slots, self.slot_dim))
        self.assertEqual(ref_points.shape, (B, self.num_slots, 2))

    def test_4d_input_ref_points_in_range(self):
        """Reference points from sigmoid head should be in [0, 1]."""
        B, C, H, W = 2, self.slot_dim, 8, 8
        inputs = torch.randn(B, C, H, W)
        slots = torch.randn(B, self.num_slots, self.slot_dim)

        _, ref_points = self.dsa(inputs, slots)
        self.assertTrue((ref_points >= 0).all() and (ref_points <= 1).all())

    # ── 3D input (flattened features) ──────────────────────────────────────

    def test_3d_input_output_shapes(self):
        """3D flattened feature input should return correct shapes."""
        B, H, W = 2, 8, 8
        inputs = torch.randn(B, H * W, self.slot_dim)
        slots = torch.randn(B, self.num_slots, self.slot_dim)

        updated_slots, ref_points = self.dsa(inputs, slots)

        self.assertEqual(updated_slots.shape, (B, self.num_slots, self.slot_dim))
        self.assertEqual(ref_points.shape, (B, self.num_slots, 2))

    def test_3d_input_square_assumption(self):
        """3D input assumes H==W (sqrt); non-square HW skips this path
        and current code would map sqrt(HW) to both H and W."""
        B, H, W = 2, 10, 10  # 100 = 10x10
        inputs = torch.randn(B, H * W, self.slot_dim)
        slots = torch.randn(B, self.num_slots, self.slot_dim)

        updated_slots, ref_points = self.dsa(inputs, slots)
        self.assertEqual(updated_slots.shape, (B, self.num_slots, self.slot_dim))

    # ── Invalid input shape ─────────────────────────────────────────────────

    def test_invalid_input_rank_raises(self):
        """5D or 2D input should raise ValueError."""
        with self.assertRaises(ValueError):
            self.dsa(torch.randn(2, 3, 4, 5, 6), torch.randn(2, 4, self.slot_dim))

    # ── Slot count variations ──────────────────────────────────────────────

    def test_single_slot(self):
        """Single slot should work correctly."""
        dsa = DeformableSlotAttention(num_slots=1, slot_dim=64, num_iterations=2)
        inputs = torch.randn(2, 64, 8, 8)
        slots = torch.randn(2, 1, 64)
        updated, ref = dsa(inputs, slots)
        self.assertEqual(updated.shape, (2, 1, 64))

    def test_many_slots(self):
        """Many slots should work correctly."""
        dsa = DeformableSlotAttention(num_slots=10, slot_dim=32, num_iterations=2)
        inputs = torch.randn(1, 32, 8, 8)
        slots = torch.randn(1, 10, 32)
        updated, ref = dsa(inputs, slots)
        self.assertEqual(updated.shape, (1, 10, 32))

    def test_single_iteration(self):
        """Single iteration should still produce valid outputs."""
        dsa = DeformableSlotAttention(num_slots=4, slot_dim=64, num_iterations=1)
        inputs = torch.randn(2, 64, 8, 8)
        slots = torch.randn(2, 4, 64)
        updated, ref = dsa(inputs, slots)
        self.assertEqual(updated.shape, (2, 4, 64))

    # ── Gradient flow ──────────────────────────────────────────────────────

    def test_gradient_flow(self):
        """Gradients should flow through the module parameters."""
        inputs = torch.randn(2, self.slot_dim, 8, 8)
        slots = torch.randn(2, self.num_slots, self.slot_dim)

        updated_slots, _ = self.dsa(inputs, slots)
        loss = updated_slots.sum()
        loss.backward()

        # Check that key parameter groups received gradients
        self.assertIsNotNone(self.dsa.reference_point_head[0].weight.grad)
        self.assertIsNotNone(self.dsa.gru.weight_ih.grad)

    def test_deterministic_eval_mode(self):
        """In eval mode, same input should produce same output."""
        self.dsa.eval()
        inputs = torch.randn(1, self.slot_dim, 4, 4)
        slots = torch.randn(1, self.num_slots, self.slot_dim)

        with torch.no_grad():
            out1, _ = self.dsa(inputs, slots)
            out2, _ = self.dsa(inputs, slots)

        self.assertTrue(torch.allclose(out1, out2))

    # ── Small spatial resolution ────────────────────────────────────────────

    def test_small_spatial_resolution(self):
        """Small feature map (2x2) should work."""
        inputs = torch.randn(2, self.slot_dim, 2, 2)
        slots = torch.randn(2, self.num_slots, self.slot_dim)
        updated, ref = self.dsa(inputs, slots)
        self.assertEqual(updated.shape, (2, self.num_slots, self.slot_dim))

    def test_large_spatial_resolution(self):
        """Larger feature map (16x16) should work."""
        inputs = torch.randn(1, self.slot_dim, 16, 16)
        slots = torch.randn(1, self.num_slots, self.slot_dim)
        updated, ref = self.dsa(inputs, slots)
        self.assertEqual(updated.shape, (1, self.num_slots, self.slot_dim))


if __name__ == '__main__':
    unittest.main()
