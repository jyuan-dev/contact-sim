"""
Unit and modular tests for LeWorldModel (LeWM).
"""

import unittest
import torch

from src.models.lewm import LeWM, CNNVisualEncoder, ActionEmbedder, ARPredictor
from src.models.wrappers.lewm_wrapper import StandardizedLeWMWrapper
from src.losses.lewm_loss import LeWMLoss
from src.models.factory import build_model, list_models


class TestLeWMCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B, self.T, self.C, self.H, self.W = 2, 4, 3, 64, 64
        self.act_dim = 2
        self.embed_dim = 64
        self.model = LeWM(
            resolution=(self.H, self.W),
            in_channels=self.C,
            action_dim=self.act_dim,
            embed_dim=self.embed_dim,
            hidden_dim=128,
            num_frames=8,
            predictor_depth=2,
            predictor_heads=4,
            predictor_dim_head=16,
            predictor_mlp_dim=128,
            dropout=0.0,
        )

    def test_encode(self):
        video = torch.randn(self.B, self.T, self.C, self.H, self.W)
        actions = torch.randn(self.B, self.T, self.act_dim)

        emb, act_emb = self.model.encode(video, actions)
        self.assertEqual(emb.shape, (self.B, self.T, self.embed_dim))
        self.assertIsNotNone(act_emb)
        self.assertEqual(act_emb.shape, (self.B, self.T, self.embed_dim))

    def test_predict(self):
        emb = torch.randn(self.B, self.T - 1, self.embed_dim)
        act_emb = torch.randn(self.B, self.T - 1, self.embed_dim)

        pred = self.model.predict(emb, act_emb)
        self.assertEqual(pred.shape, (self.B, self.T - 1, self.embed_dim))

    def test_forward(self):
        video = torch.randn(self.B, self.T, self.C, self.H, self.W)
        actions = torch.randn(self.B, self.T, self.act_dim)

        out = self.model(video, actions, n_preds=1)
        self.assertIn("emb", out)
        self.assertIn("pred_emb", out)
        self.assertIn("target_emb", out)
        self.assertIn("pred_loss", out)
        self.assertEqual(out["emb"].shape, (self.B, self.T, self.embed_dim))
        self.assertEqual(out["pred_emb"].shape, (self.B, self.T - 1, self.embed_dim))
        self.assertEqual(out["target_emb"].shape, (self.B, self.T - 1, self.embed_dim))
        self.assertTrue(torch.isfinite(out["pred_loss"]))

    def test_rollout(self):
        video = torch.randn(self.B, 2, self.C, self.H, self.W)
        actions = torch.randn(self.B, 6, self.act_dim)

        rollout_res = self.model.rollout(video, actions, n_cond_frames=2)
        self.assertEqual(rollout_res["emb"].shape, (self.B, 6, self.embed_dim))
        self.assertEqual(rollout_res["cond_emb"].shape, (self.B, 2, self.embed_dim))
        self.assertEqual(len(rollout_res["is_rollout_mask"]), 6)

    def test_backward(self):
        video = torch.randn(self.B, self.T, self.C, self.H, self.W)
        actions = torch.randn(self.B, self.T, self.act_dim)

        out = self.model(video, actions)
        loss = out["pred_loss"]
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Grad is None for {name}")


class TestLeWMLoss(unittest.TestCase):
    def test_loss(self):
        torch.manual_seed(42)
        loss_fn = LeWMLoss(sigreg_weight=0.1, num_proj=128, knots=9)
        model_output = {
            "emb": torch.randn(4, 6, 32),
            "pred_emb": torch.randn(4, 5, 32),
            "target_emb": torch.randn(4, 5, 32),
        }
        res = loss_fn(model_output)
        self.assertIn("loss", res)
        self.assertIn("pred_loss", res)
        self.assertIn("sigreg_loss", res)
        self.assertTrue(torch.isfinite(res["loss"]))
        self.assertTrue(res["loss"] > 0)


class TestStandardizedLeWMWrapper(unittest.TestCase):
    def test_registry_and_build(self):
        self.assertIn("lewm", list_models())
        self.assertIn("leworldmodel", list_models())
        self.assertIn("le_wm", list_models())

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
                "sigreg_weight": 0.05,
            }
        }
        wrapper = build_model(cfg)
        self.assertIsInstance(wrapper, StandardizedLeWMWrapper)

        # Test forward pass with dict
        batch = {
            "img": torch.randn(2, 4, 3, 32, 32),
            "actions": torch.randn(2, 4, 2),
        }
        out = wrapper(batch)
        self.assertIn("input_img", out)
        self.assertIn("post_slots", out)
        self.assertEqual(out["post_slots"].shape, (2, 4, 1, 32))

        # Test compute_loss
        total_loss, loss_dict = wrapper.compute_loss(out, batch)
        self.assertTrue(torch.is_tensor(total_loss))
        self.assertIn("loss", loss_dict)
        self.assertIn("pred_loss", loss_dict)
        self.assertIn("sigreg_loss", loss_dict)

    def test_spatial_resize(self):
        cfg = {
            "model": {
                "type": "lewm",
                "resolution": [64, 64],
                "embed_dim": 32,
                "hidden_dim": 64,
                "predictor": {"depth": 1, "heads": 2, "dim_head": 16, "mlp_dim": 32},
            }
        }
        wrapper = build_model(cfg)
        # Input with different resolution [32, 32]
        batch = {"img": torch.randn(2, 3, 3, 32, 32)}
        out = wrapper(batch)
        self.assertEqual(out["input_img"].shape, (2, 3, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
