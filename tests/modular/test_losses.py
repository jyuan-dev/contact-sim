import unittest
import torch
from src.losses.recon_loss import ReconstructionMSELoss
from src.losses.mask_loss import MaskSegmentationLoss
from src.losses.sigreg import SIGRegLoss
from src.losses.contrastive import TemporalSlotContrastiveLoss
from src.losses.composite import CompositeLoss
from src.losses import build_loss


# ── ReconstructionMSELoss edge cases ──────────────────────────────────────────

class TestReconstructionMSELossEdgeCases(unittest.TestCase):
    def test_missing_recon_returns_zero(self):
        """When recon_img is None, loss should be 0.0."""
        loss_fn = ReconstructionMSELoss(weight=1.0)
        out = {'recon_img': None, 'input_img': torch.randn(2, 3, 64, 64)}
        weighted, raw = loss_fn(out)
        self.assertEqual(weighted.item(), 0.0)
        self.assertEqual(raw, 0.0)

    def test_missing_input_returns_zero(self):
        """When input_img is None, loss should be 0.0."""
        loss_fn = ReconstructionMSELoss(weight=1.0)
        out = {'recon_img': torch.randn(2, 3, 64, 64), 'input_img': None}
        weighted, raw = loss_fn(out)
        self.assertEqual(weighted.item(), 0.0)
        self.assertEqual(raw, 0.0)

    def test_weight_scaling(self):
        """Weight should linearly scale the loss."""
        out = {'recon_img': torch.zeros(2, 3, 64, 64), 'input_img': torch.ones(2, 3, 64, 64)}
        w1, r1 = ReconstructionMSELoss(weight=1.0)(out)
        w2, r2 = ReconstructionMSELoss(weight=2.0)(out)
        self.assertAlmostEqual(w2.item() / w1.item(), 2.0)
        self.assertAlmostEqual(r1, r2)  # raw loss unchanged


# ── MaskSegmentationLoss edge cases ──────────────────────────────────────────

class TestMaskSegmentationLossEdgeCases(unittest.TestCase):
    def test_missing_pred_masks_returns_zero(self):
        """When pred_masks is None, loss should be 0.0 with both keys."""
        loss_fn = MaskSegmentationLoss(weight=1.0)
        out = {'pred_masks': None}
        batch = {'gt_masks': torch.randn(2, 3, 3, 64, 64)}
        weighted, info = loss_fn(out, batch)
        self.assertEqual(weighted.item(), 0.0)
        self.assertIn('mask_bce', info)
        self.assertIn('mask_dice', info)

    def test_missing_gt_masks_returns_zero(self):
        """When gt_masks is None, loss should be 0.0."""
        loss_fn = MaskSegmentationLoss(weight=1.0)
        out = {'pred_masks': torch.sigmoid(torch.randn(2, 3, 4, 64, 64))}
        batch = {'gt_masks': None}
        weighted, info = loss_fn(out, batch)
        self.assertEqual(weighted.item(), 0.0)

    def test_hungarian_mode(self):
        """Hungarian matching mode should produce valid loss."""
        loss_fn = MaskSegmentationLoss(weight=1.0, match_mode='hungarian')
        out = {'pred_masks': torch.sigmoid(torch.randn(2, 3, 4, 64, 64))}
        batch = {'gt_masks': (torch.randn(2, 3, 3, 64, 64) > 0).float()}
        weighted, info = loss_fn(out, batch)
        self.assertTrue(torch.is_tensor(weighted))
        self.assertIn('mask_bce', info)
        self.assertIn('mask_dice', info)

    def test_fixed_mode_with_more_slots_than_classes(self):
        """Fixed matching with more slots than GT classes should still work."""
        loss_fn = MaskSegmentationLoss(weight=1.0, match_mode='fixed')
        out = {'pred_masks': torch.sigmoid(torch.randn(2, 3, 6, 64, 64))}  # 6 slots
        batch = {'gt_masks': (torch.randn(2, 3, 3, 64, 64) > 0).float()}  # 3 classes
        weighted, info = loss_fn(out, batch)
        self.assertTrue(torch.is_tensor(weighted))

    def test_6d_pred_masks_squeezed(self):
        """6D pred_masks [B,T,K,1,H,W] should be squeezed to 5D."""
        loss_fn = MaskSegmentationLoss(weight=1.0)
        out = {'pred_masks': torch.sigmoid(torch.randn(2, 3, 3, 1, 64, 64))}
        batch = {'gt_masks': (torch.randn(2, 3, 3, 64, 64) > 0).float()}
        weighted, info = loss_fn(out, batch)
        self.assertTrue(torch.is_tensor(weighted))

    def test_size_mismatch_interpolates(self):
        """When pred and GT masks have different spatial sizes, interpolate."""
        loss_fn = MaskSegmentationLoss(weight=1.0)
        out = {'pred_masks': torch.sigmoid(torch.randn(2, 3, 3, 32, 32))}  # 32x32
        batch = {'gt_masks': (torch.randn(2, 3, 3, 64, 64) > 0).float()}  # 64x64
        weighted, info = loss_fn(out, batch)
        self.assertTrue(torch.is_tensor(weighted))


# ── TemporalSlotContrastiveLoss edge cases ───────────────────────────────────

class TestTemporalSlotContrastiveLossEdgeCases(unittest.TestCase):
    def test_single_timestep_returns_zero(self):
        """T=1 should return zero loss (no temporal pairs)."""
        loss_fn = TemporalSlotContrastiveLoss(weight=1.0)
        slots = torch.randn(2, 1, 4, 32)  # B=2, T=1
        weighted, raw = loss_fn(slots)
        self.assertEqual(weighted.item(), 0.0)
        self.assertEqual(raw, 0.0)

    def test_dict_input_extracts_post_slots(self):
        """Should extract post_slots from an output dict."""
        loss_fn = TemporalSlotContrastiveLoss(weight=1.0)
        out = {'post_slots': torch.randn(2, 3, 4, 32)}
        weighted, raw = loss_fn(out)
        self.assertTrue(torch.is_tensor(weighted))
        self.assertGreater(raw, 0.0)

    def test_dict_input_extracts_slots_fallback(self):
        """Should fall back to 'slots' key if 'post_slots' is missing."""
        loss_fn = TemporalSlotContrastiveLoss(weight=1.0)
        out = {'slots': torch.randn(2, 3, 4, 32)}
        weighted, raw = loss_fn(out)
        self.assertTrue(torch.is_tensor(weighted))

    def test_dict_input_no_slots_returns_zero(self):
        """Dict with no slot keys should return zero loss."""
        loss_fn = TemporalSlotContrastiveLoss(weight=1.0)
        out = {'recon_img': torch.randn(2, 3, 64, 64)}
        weighted, raw = loss_fn(out)
        self.assertEqual(weighted.item(), 0.0)

    def test_temperature_scaling(self):
        """Lower temperature should produce higher loss (sharper contrast)."""
        slots = torch.randn(2, 3, 4, 32)
        _, raw_hot = TemporalSlotContrastiveLoss(weight=1.0, temperature=0.05)(slots)
        _, raw_cold = TemporalSlotContrastiveLoss(weight=1.0, temperature=1.0)(slots)
        self.assertGreater(raw_hot, raw_cold)


# ── CompositeLoss edge cases ────────────────────────────────────────────────

class TestCompositeLossEdgeCases(unittest.TestCase):
    def test_empty_losses_returns_zero(self):
        """CompositeLoss with no sub-losses should return zero."""
        composite = CompositeLoss()
        out = {'input_img': torch.randn(2, 3, 64, 64)}
        total, loss_dict = composite(out, {})
        self.assertEqual(total.item(), 0.0)
        self.assertIn('total_loss', loss_dict)

    def test_kwargs_instantiation(self):
        """Keyword arguments should create sub-loss entries."""
        composite = CompositeLoss(
            recon=ReconstructionMSELoss(weight=1.0),
        )
        self.assertIn('recon', composite.losses)

    def test_dict_instantiation(self):
        """Dict argument should create sub-loss entries."""
        composite = CompositeLoss(losses={
            'mask': MaskSegmentationLoss(weight=1.0),
        })
        self.assertIn('mask', composite.losses)

    def test_ignores_non_module_values(self):
        """Non-nn.Module dict values should be silently ignored."""
        composite = CompositeLoss(
            losses={'not_a_module': 42, 'also_not': "hello"},
            recon=ReconstructionMSELoss(weight=1.0),
        )
        self.assertIn('recon', composite.losses)
        self.assertNotIn('not_a_module', composite.losses)
        self.assertNotIn('also_not', composite.losses)

    def test_sub_loss_dict_info_merged(self):
        """Sub-loss dict infos should be merged into flat loss_dict."""
        composite = CompositeLoss(
            recon=ReconstructionMSELoss(weight=1.0),
        )
        out = {'recon_img': torch.randn(2, 3, 64, 64), 'input_img': torch.randn(2, 3, 64, 64)}
        _, loss_dict = composite(out, {})
        self.assertIn('recon_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)


# ── Existing tests (unchanged) ───────────────────────────────────────────────

class TestLosses(unittest.TestCase):
    def test_reconstruction_mse_loss(self):
        """Test ReconstructionMSELoss forward pass."""
        loss_fn = ReconstructionMSELoss(weight=2.0)
        out = {
            'recon_img': torch.randn(2, 3, 64, 64),
            'input_img': torch.randn(2, 3, 64, 64),
        }
        weighted, raw_val = loss_fn(out)
        self.assertTrue(torch.is_tensor(weighted))
        self.assertAlmostEqual(weighted.item(), 2.0 * raw_val, places=4)

    def test_mask_segmentation_loss(self):
        """Test MaskSegmentationLoss forward pass."""
        loss_fn = MaskSegmentationLoss(weight=1.0, match_mode='fixed')
        out = {'pred_masks': torch.sigmoid(torch.randn(2, 3, 4, 64, 64))}
        batch = {'gt_masks': (torch.randn(2, 3, 3, 64, 64) > 0).float()}
        weighted, info = loss_fn(out, batch)
        self.assertTrue(torch.is_tensor(weighted))
        self.assertIn('mask_bce', info)
        self.assertIn('mask_dice', info)

    def test_contrastive_loss_shape(self):
        """Test TemporalSlotContrastiveLoss forward pass and output type."""
        loss_fn = TemporalSlotContrastiveLoss(weight=1.0, temperature=0.1)
        dummy_slots = torch.randn(2, 3, 4, 32)
        weighted, raw_val = loss_fn(dummy_slots)

        self.assertTrue(torch.is_tensor(weighted))
        self.assertEqual(weighted.ndim, 0)
        self.assertTrue(raw_val > 0)

    def test_sigreg_loss_shape(self):
        """Test SIGRegLoss forward pass and output type."""
        loss_fn = SIGRegLoss(weight=1.0, num_proj=16, knots=17)
        dummy_latents = torch.randn(2, 3, 4, 16)  # [B, T, K, D]
        weighted, info = loss_fn(dummy_latents)

        self.assertTrue(torch.is_tensor(weighted))
        self.assertEqual(weighted.ndim, 0)
        self.assertIsInstance(info, dict)
        self.assertIn("sigreg_loss", info)
        self.assertGreater(info["sigreg_loss"], 0.0)

    def test_composite_loss_aggregation(self):
        """Test CompositeLoss aggregates sub-losses dynamically."""
        composite = CompositeLoss(
            recon=ReconstructionMSELoss(weight=1.0),
            mask=MaskSegmentationLoss(weight=1.0),
            sigreg=SIGRegLoss(weight=0.1),
        )
        out = {
            'recon_img': torch.randn(2, 3, 3, 64, 64),
            'input_img': torch.randn(2, 3, 3, 64, 64),
            'pred_masks': torch.sigmoid(torch.randn(2, 3, 4, 64, 64)),
            'post_slots': torch.randn(2, 3, 4, 64),
        }
        batch = {'gt_masks': (torch.randn(2, 3, 3, 64, 64) > 0).float()}

        total_loss, loss_dict = composite(out, batch)
        self.assertTrue(torch.is_tensor(total_loss))
        self.assertIn('recon_loss', loss_dict)
        self.assertIn('mask_bce', loss_dict)
        self.assertIn('sigreg_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)

    def test_build_loss_hydra_instantiation(self):
        """Test build_loss helper with Hydra _target_ config dict."""
        cfg_loss = {
            '_target_': 'src.losses.composite.CompositeLoss',
            'losses': {
                'recon': {'_target_': 'src.losses.recon_loss.ReconstructionMSELoss', 'weight': 1.0},
                'mask': {'_target_': 'src.losses.mask_loss.MaskSegmentationLoss', 'weight': 1.0},
            }
        }
        loss_fn = build_loss(cfg_loss)
        self.assertIsInstance(loss_fn, CompositeLoss)
        self.assertIn('recon', loss_fn.losses)
        self.assertIn('mask', loss_fn.losses)


if __name__ == '__main__':
    unittest.main()
