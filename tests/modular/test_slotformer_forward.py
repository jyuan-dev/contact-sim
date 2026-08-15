"""
SlotFormerModel forward smoke test — exercises stage-1 slot extraction,
the rollouter, decoding, and calc_train_loss with small dims.
"""

import unittest

import torch

from src.models.factory import build_model
from src.models.slotformer import SlotFormerModel


def _make_stage1():
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
    wrapper = build_model(cfg).eval()
    for p in wrapper.parameters():
        p.requires_grad = False
    return wrapper


class TestSlotFormerModelForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.stage1 = _make_stage1()
        self.model = SlotFormerModel(
            stage1_model=self.stage1,
            history_len=2,
            rollout_len=2,
            d_model=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            use_img_recon_loss=False,
        ).eval()

    def test_extract_slots_matches_encode(self):
        video = torch.randn(2, 4, 3, 64, 64)
        slots = self.model.extract_slots(video)
        self.assertEqual(tuple(slots.shape), (2, 4, 2, 32))

        inner = self.stage1.inner_savi()
        inner._reset_rnn()
        reference, _ = inner.encode(video)
        self.assertTrue(torch.allclose(slots, reference, atol=1e-6, rtol=1e-6))

    def test_forward_outputs(self):
        video = torch.randn(2, 4, 3, 64, 64)
        out = self.model(video)

        for key in ("gt_slots", "pred_slots", "history_slots", "input_img"):
            self.assertIn(key, out)
        self.assertEqual(tuple(out["gt_slots"].shape), (2, 2, 2, 32))
        self.assertEqual(tuple(out["pred_slots"].shape), (2, 2, 2, 32))
        # eval mode -> reconstruction outputs present
        self.assertEqual(tuple(out["recon_img"].shape), (2, 4, 3, 64, 64))
        self.assertEqual(tuple(out["pred_masks"].shape), (2, 4, 2, 64, 64))
        self.assertEqual(tuple(out["post_slots"].shape), (2, 4, 2, 32))

    def test_calc_train_loss(self):
        video = torch.randn(2, 4, 3, 64, 64)
        out = self.model(video)
        loss, loss_dict = self.model.calc_train_loss(out, batch={"img": video})
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIn("slot_mse", loss_dict)
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
