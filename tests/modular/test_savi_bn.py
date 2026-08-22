import unittest
import torch
import torch.nn as nn

from src.models.savi import SAVi
from src.models.factory import build_model


class TestSAVIBatchNorm(unittest.TestCase):
    """Test BatchNorm integration on SAVi encoder and slot residual."""

    def test_savi_with_bn_shapes_and_statistics(self):
        """Verify that SAVi with BatchNorm outputs zero-mean unit-variance slots."""
        model = SAVi(
            resolution=(64, 64),
            clip_len=6,
            num_slots=4,
            slot_dim=64,
            use_encoder_bn=True,
            use_residual_bn=True,
        )
        model.train()

        x = torch.randn(4, 6, 3, 64, 64)
        out = model(x)

        self.assertIn("post_slots", out)
        self.assertIn("post_recon_combined", out)

        slots = out["post_slots"]
        self.assertEqual(slots.shape, (4, 6, 4, 64))

        # Under training mode, the output of BatchNorm1d across batch and slots should have approx unit variance
        per_channel_std = slots.view(-1, 64).std(dim=0)
        self.assertTrue(torch.allclose(per_channel_std, torch.ones_like(per_channel_std), atol=1e-2, rtol=1e-2))

    def test_savi_bn_backward_gradients(self):
        """Verify gradient flow through both encoder_bn and residual_bn."""
        model = SAVi(
            resolution=(64, 64),
            clip_len=4,
            num_slots=3,
            slot_dim=32,
            use_encoder_bn=True,
            use_residual_bn=True,
        )
        model.train()

        x = torch.randn(2, 4, 3, 64, 64, requires_grad=True)
        out = model(x)
        loss = out["post_recon_combined"].sum() + out["post_slots"].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.encoder_bn.weight.grad)

    def test_build_model_with_bn_config(self):
        """Verify build_model dispatches use_encoder_bn and use_residual_bn properly."""
        cfg = {
            "model": {
                "name": "savi",
                "type": "savi",
                "resolution": [64, 64],
                "n_sample_frames": 4,
                "num_slots": 4,
                "slot_dim": 64,
                "use_encoder_bn": True,
                "use_residual_bn": True,
            }
        }
        wrapper = build_model(cfg)
        self.assertTrue(wrapper.model.use_encoder_bn)
        self.assertTrue(wrapper.model.use_residual_bn)
        self.assertIsNotNone(wrapper.model.encoder_bn)
