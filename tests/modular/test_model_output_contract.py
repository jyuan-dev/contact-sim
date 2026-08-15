"""
ModelOutput shape contract test — the wrapper guarantees one shape per key;
every consumer (losses, metrics) takes those keys strictly.
"""

import unittest

import torch

from src.models.factory import build_model


class TestModelOutputContract(unittest.TestCase):
    def setUp(self):
        cfg = {
            "model": {
                "name": "savi",
                "type": "savi",
                "num_slots": 3,
                "slot_dim": 32,
                "resolution": [64, 64],
                "n_sample_frames": 4,
                "use_encoder_bn": True,
                "use_residual_bn": True,
            }
        }
        self.wrapper = build_model(cfg).eval()
        self.video = torch.randn(2, 4, 3, 64, 64)

    def test_forward_shapes_and_keys(self):
        out = self.wrapper(self.video)

        for key in ("input_img", "pred_masks", "recon_img", "post_slots"):
            self.assertIn(key, out, f"ModelOutput missing required key: {key}")

        B, T, C, H, W = self.video.shape
        self.assertEqual(tuple(out["input_img"].shape), (B, T, C, H, W))
        self.assertEqual(tuple(out["pred_masks"].shape), (B, T, 3, H, W))  # 5-D, squeezed
        self.assertEqual(tuple(out["recon_img"].shape), (B, T, C, H, W))
        self.assertEqual(tuple(out["post_slots"].shape), (B, T, 3, 32))

    def test_losses_consume_strict_keys(self):
        """Every loss must be able to consume the normalized output."""
        from src.losses import build_loss
        from src.losses.mask_loss import MaskSegmentationLoss

        out = self.wrapper(self.video)
        batch = {"gt_masks": torch.zeros(2, 4, 3, 64, 64)}

        mask_loss, mask_info = MaskSegmentationLoss(weight=1.0)(out, batch)
        self.assertIsInstance(mask_loss, torch.Tensor)
        self.assertIn("mask_bce", mask_info)

        sigreg_loss = build_loss({
            "_target_": "src.losses.composite.CompositeLoss",
            "losses": {"sigreg": {"_target_": "src.losses.sigreg.SIGRegLoss", "weight": 0.01}},
        })
        total, info = sigreg_loss(out, batch)
        self.assertIsInstance(total, torch.Tensor)
        self.assertIn("sigreg_loss", info)


if __name__ == "__main__":
    unittest.main()
