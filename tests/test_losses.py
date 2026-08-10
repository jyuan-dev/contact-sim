import unittest
import torch
from src.losses.contrastive import TemporalSlotContrastiveLoss
from src.losses.sigreg import SIGRegLoss


class TestLosses(unittest.TestCase):
    def test_contrastive_loss_shape(self):
        """Test TemporalSlotContrastiveLoss forward pass and output type."""
        loss_fn = TemporalSlotContrastiveLoss(temperature=0.1)
        
        # Dummy slot representation [B, T, K, D] -> [2, 3, 4, 32]
        dummy_slots = torch.randn(2, 3, 4, 32)
        loss = loss_fn(dummy_slots)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)  # Scalar loss
        self.assertTrue(loss.item() > 0)

    def test_contrastive_loss_edge_cases(self):
        """Test TemporalSlotContrastiveLoss with short sequences (T < 2) or invalid shapes."""
        loss_fn = TemporalSlotContrastiveLoss()
        
        # Single frame: should return 0.0 scalar loss
        short_slots = torch.randn(2, 1, 4, 32)
        loss = loss_fn(short_slots)
        self.assertEqual(loss.item(), 0.0)

    def test_sigreg_loss_shape(self):
        """Test SIGRegLoss (isotropic Gaussian regularization) forward pass and output type."""
        loss_fn = SIGRegLoss(sketch_dim=16, num_points=10, t_max=3.0)
        
        # Dummy latent representation [B, T, K, D] -> [2, 3, 4, 16]
        dummy_latents = torch.randn(2, 3, 4, 16)
        loss = loss_fn(dummy_latents)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)  # Scalar loss
        
        # Try 2D input [N, D]
        dummy_latents_2d = torch.randn(10, 16)
        loss_2d = loss_fn(dummy_latents_2d)
        self.assertTrue(torch.is_tensor(loss_2d))
        self.assertEqual(loss_2d.ndim, 0)

    def test_sigreg_loss_single_item(self):
        """Test SIGRegLoss handles single-sample batches gracefully."""
        loss_fn = SIGRegLoss()
        single_latent = torch.randn(1, 8)
        loss = loss_fn(single_latent)
        self.assertEqual(loss.item(), 0.0)

    def test_savi_sigreg_loss_integration(self):
        """Test compute_savi_loss properly computes and logs sigreg_loss term."""
        from src.losses.model_losses import compute_savi_loss
        
        out = {
            'recon_img': torch.randn(2, 3, 3, 64, 64),
            'input_img': torch.randn(2, 3, 3, 64, 64),
            'pred_masks': torch.sigmoid(torch.randn(2, 3, 4, 64, 64)),
            'post_slots': torch.randn(2, 3, 4, 64),
        }
        gt_masks = (torch.randn(2, 3, 3, 64, 64) > 0).float()
        weight_dict = {'recon': 1.0, 'mask': 1.0, 'sigreg': 0.1, 'match_mode': 'fixed'}
        
        total_loss, loss_dict = compute_savi_loss(out, gt_masks, weight_dict=weight_dict)
        self.assertTrue('sigreg_loss' in loss_dict)
        self.assertTrue(loss_dict['sigreg_loss'] > 0.0)


if __name__ == '__main__':
    unittest.main()
