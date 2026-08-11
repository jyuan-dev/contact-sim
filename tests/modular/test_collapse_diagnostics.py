"""
Unit tests for compute_collapse_diagnostics — verifies correctness with
softmax-normalized masks (sum to 1 per pixel).

Metrics use argmax-based binarization for slot win rates and per-pixel
entropy to measure assignment sharpness.  All tests use synthetic masks
with known properties.
"""

import math
import pytest
import torch
from src.metrics.eval_metrics import compute_collapse_diagnostics


def _make_out(pred_masks=None, post_slots=None):
    return {"pred_masks": pred_masks, "post_slots": post_slots}


# ── Tests: usage_* with argmax ──────────────────────────────────────────────

def test_one_slot_dominates_gives_high_std():
    """When one slot clearly wins most pixels, usage_std is high."""
    masks = torch.zeros(1, 1, 4, 16, 16)
    masks[..., 3, :, :] = 1.0
    masks = masks / masks.sum(dim=2, keepdim=True)
    r = compute_collapse_diagnostics(_make_out(masks))
    assert r["slot_usage_max"] == pytest.approx(1.0, abs=0.01)
    assert r["slot_usage_min"] == pytest.approx(0.0, abs=0.01)
    assert r["slot_usage_std"] > 0.4


def test_spatial_partition_equal_wins_low_entropy():
    """Slots partition the image into quadrants — each wins ~25%,
    assignment is sharp → low entropy, usage_std near 0."""
    masks = torch.zeros(1, 1, 4, 16, 16)
    masks[..., 0, :8, :8] = 1.0
    masks[..., 1, :8, 8:] = 1.0
    masks[..., 2, 8:, :8] = 1.0
    masks[..., 3, 8:, 8:] = 1.0
    masks = masks / masks.sum(dim=2, keepdim=True)
    r = compute_collapse_diagnostics(_make_out(masks))
    assert r["slot_usage_std"] < 0.01
    assert r["mask_entropy"] < 0.01  # sharp = near-zero entropy


# ── Tests: mask_entropy ─────────────────────────────────────────────────────

def test_uniform_masks_have_max_entropy():
    """Uniform [0.25,0.25,0.25,0.25] → max entropy = ln(4) ≈ 1.386."""
    masks = torch.full((2, 3, 4, 8, 8), 0.25)
    r = compute_collapse_diagnostics(_make_out(masks))
    assert r["mask_entropy"] == pytest.approx(math.log(4), abs=0.02)


def test_sharp_masks_have_near_zero_entropy():
    """One-hot-like masks → entropy near 0."""
    masks = torch.zeros(1, 1, 4, 16, 16)
    masks[..., 3, :, :] = 1.0
    masks = masks / masks.sum(dim=2, keepdim=True)  # [0,0,0,1]
    r = compute_collapse_diagnostics(_make_out(masks))
    assert r["mask_entropy"] < 0.01


# ── Tests: edge cases ───────────────────────────────────────────────────────

def test_no_masks_returns_empty():
    """If pred_masks is None, collapse diagnostics return empty dict."""
    r = compute_collapse_diagnostics(_make_out(None))
    assert r == {}


def test_latent_std_decreases_when_slots_collapse():
    """latent_std should approach 0 when all slot vectors are identical."""
    diverse = torch.randn(2, 3, 4, 64)
    r_diverse = compute_collapse_diagnostics(
        _make_out(torch.rand(1, 1, 4, 8, 8), diverse)
    )
    collapsed = torch.ones(2, 3, 4, 64)
    r_collapsed = compute_collapse_diagnostics(
        _make_out(torch.rand(1, 1, 4, 8, 8), collapsed)
    )
    assert r_collapsed["latent_std"] < r_diverse["latent_std"]


def test_6d_masks_are_squeezed():
    """Masks with shape [B, T, K, 1, H, W] are squeezed to [B, T, K, H, W]."""
    masks = torch.rand(2, 3, 4, 1, 8, 8)
    r = compute_collapse_diagnostics(_make_out(masks))
    assert "slot_usage_mean" in r


def test_single_slot_entropy_is_zero():
    """With K=1, softmax always outputs 1.0 → entropy = 0."""
    masks = torch.ones(1, 1, 1, 8, 8)
    r = compute_collapse_diagnostics(_make_out(masks))
    assert "mask_entropy" in r
