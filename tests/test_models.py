import unittest
import os
import sys
import torch

# Ensure workspace root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.detr import DETR, ResNetBackbone, Transformer
from src.models.slot_attention import StoSAVi

class TestModels(unittest.TestCase):
    def test_detr_forward(self):
        """Test instantiation and a dummy forward pass of the DETR model."""
        backbone = ResNetBackbone(train_backbone=True)
        transformer = Transformer(
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256
        )
        model = DETR(
            backbone=backbone,
            transformer=transformer,
            num_classes=3,
            num_queries=5
        )
        
        # Dummy batch: [B, 3, 64, 64]
        dummy_input = torch.randn(2, 3, 64, 64)
        outputs = model(dummy_input)
        
        self.assertIn('pred_logits', outputs)
        self.assertIn('pred_boxes', outputs)
        
        # pred_logits: [B, Q, num_classes + 1] -> [2, 5, 4]
        self.assertEqual(outputs['pred_logits'].shape, (2, 5, 4))
        # pred_boxes: [B, Q, 4] -> [2, 5, 4]
        self.assertEqual(outputs['pred_boxes'].shape, (2, 5, 4))

    def test_savi_forward(self):
        """Test instantiation and a dummy forward pass of the StoSAVi slot attention model."""
        # Setup minimal config dicts for StoSAVi
        slot_dict = dict(
            num_slots=4,
            slot_size=64,
            slot_mlp_size=128,
            num_iterations=2,
            kernel_mlp=False
        )
        enc_dict = dict(
            enc_channels=(3, 16, 16, 16, 16),
            enc_ks=5,
            enc_out_channels=32,
            enc_norm=''
        )
        dec_dict = dict(
            dec_channels=(64, 16, 16, 16, 16),
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm=''
        )
        pred_dict = dict(
            pred_type='mlp',
            pred_rnn=False,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=256,
            pred_sg_every=None
        )
        loss_dict = dict(
            use_post_recon_loss=True,
            kld_method='var-0.01'
        )
        
        model = StoSAVi(
            resolution=(64, 64),
            clip_len=6,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            pred_dict=pred_dict,
            loss_dict=loss_dict
        )
        
        # Dummy video batch: [B, T, C, H, W] -> [2, 3, 3, 64, 64]
        dummy_input = torch.randn(2, 3, 3, 64, 64)
        outputs = model({'img': dummy_input})
        
        self.assertIn('post_recon_combined', outputs)
        self.assertIn('post_masks', outputs)
        self.assertIn('post_recons', outputs)
        
        loss_dict = model.calc_train_loss({'img': dummy_input}, outputs)
        self.assertIn('kld_loss', loss_dict)
        
        # post_recon_combined: [B, T, C, H, W] -> [2, 3, 3, 64, 64]
        self.assertEqual(outputs['post_recon_combined'].shape, (2, 3, 3, 64, 64))
        # post_masks: [B, T, S, 1, H, W] -> [2, 3, 4, 1, 64, 64]
        self.assertEqual(outputs['post_masks'].shape, (2, 3, 4, 1, 64, 64))

if __name__ == '__main__':
    unittest.main()
