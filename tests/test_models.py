import unittest
import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.detr import DETR, ResNetBackbone, Transformer
from src.models.playslot_savi import PlaySlotSAVi
from src.models.factory import build_model


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
        self.assertEqual(outputs['pred_logits'].shape, (2, 5, 4))
        self.assertEqual(outputs['pred_boxes'].shape, (2, 5, 4))

    def test_savi_forward(self):
        """Test instantiation and a dummy forward pass of the PlaySlotSAVi model."""
        model = PlaySlotSAVi(
            num_slots=4,
            slot_dim=64,
            num_iterations=2
        )
        
        # Dummy video batch: [B, T, C, H, W] -> [2, 3, 3, 64, 64]
        dummy_input = torch.randn(2, 3, 3, 64, 64)
        outputs = model(dummy_input)
        
        self.assertIn('recon_combined', outputs)
        self.assertIn('post_masks', outputs)
        self.assertIn('slots', outputs)

    def test_factory_build_model(self):
        """Test building DETR and SAVi models via unified build_model factory."""
        detr_cfg = {'model': {'name': 'detr', 'num_classes': 3, 'num_queries': 5}}
        detr_model = build_model(detr_cfg)
        self.assertIsNotNone(detr_model)

        savi_cfg = {'model': {'name': 'savi', 'num_slots': 4}}
        savi_model = build_model(savi_cfg)
        self.assertIsNotNone(savi_model)


if __name__ == '__main__':
    unittest.main()
