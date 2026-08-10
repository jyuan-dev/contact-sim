import unittest
import torch
from src.losses.recon_loss import ReconstructionMSELoss
from src.losses.mask_loss import MaskSegmentationLoss
from src.losses.sigreg import SIGRegLoss
from src.losses.contrastive import TemporalSlotContrastiveLoss
from src.losses.composite import CompositeLoss
from src.losses import build_loss


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
        loss_fn = SIGRegLoss(weight=1.0, sketch_dim=16, num_points=10, t_max=3.0)
        dummy_latents = torch.randn(2, 3, 4, 16)
        weighted, raw_val = loss_fn(dummy_latents)

        self.assertTrue(torch.is_tensor(weighted))
        self.assertEqual(weighted.ndim, 0)

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
