import unittest
import torch

from src.models.detr import (
    DETR, ResNetBackbone, Transformer, MLP,
    box_cxcywh_to_xyxy, box_iou, generalized_box_iou,
    masks_to_boxes_and_labels, HungarianMatcher, SetCriterion,
)
from src.models.savi import SAVi
from src.models.factory import build_model, StandardizedDETRWrapper, StandardizedSAViWrapper


# ── DETR Components ───────────────────────────────────────────────────────────

class TestResNetBackbone(unittest.TestCase):
    def test_output_shape(self):
        """ResNetBackbone should reduce a 64x64 input to [B, 256, H', W']."""
        backbone = ResNetBackbone(train_backbone=False)
        x = torch.randn(2, 3, 64, 64)
        out = backbone(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 256)  # layer3 output channels

    def test_frozen_backbone_no_grad(self):
        """Frozen backbone parameters should not require grad."""
        backbone = ResNetBackbone(train_backbone=False)
        for p in backbone.parameters():
            self.assertFalse(p.requires_grad)


class TestMLP(unittest.TestCase):
    def test_output_shape(self):
        """MLP should map [B, in] -> [B, out]."""
        mlp = MLP(input_dim=64, hidden_dim=128, output_dim=4, num_layers=3)
        x = torch.randn(5, 64)
        out = mlp(x)
        self.assertEqual(out.shape, (5, 4))

    def test_single_layer(self):
        """Single-layer MLP is a plain linear transform."""
        mlp = MLP(input_dim=16, hidden_dim=16, output_dim=8, num_layers=1)
        x = torch.randn(3, 16)
        out = mlp(x)
        self.assertEqual(out.shape, (3, 8))


class TestBoxUtilities(unittest.TestCase):
    def test_box_cxcywh_to_xyxy_identity(self):
        """A unit box centred at (0.5, 0.5) with w=h=1 should map to [0,0,1,1]."""
        box = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        xyxy = box_cxcywh_to_xyxy(box)
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.assertTrue(torch.allclose(xyxy, expected))

    def test_box_iou_perfect(self):
        """Identical boxes should have IoU=1."""
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        iou, _ = box_iou(boxes, boxes)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)

    def test_box_iou_no_overlap(self):
        """Non-overlapping boxes should have IoU=0."""
        b1 = torch.tensor([[0.0, 0.0, 0.4, 0.4]])
        b2 = torch.tensor([[0.6, 0.6, 1.0, 1.0]])
        iou, _ = box_iou(b1, b2)
        self.assertAlmostEqual(iou.item(), 0.0, places=5)

    def test_generalized_box_iou_range(self):
        """GIoU should be in [-1, 1]."""
        b1 = torch.rand(4, 4)
        b2 = torch.rand(4, 4)
        # Ensure valid xyxy (x2 > x1, y2 > y1)
        b1[:, 2:] = b1[:, :2] + b1[:, 2:].abs() + 0.01
        b2[:, 2:] = b2[:, :2] + b2[:, 2:].abs() + 0.01
        giou = generalized_box_iou(b1, b2)
        self.assertTrue((giou >= -1.0 - 1e-5).all())
        self.assertTrue((giou <= 1.0 + 1e-5).all())


class TestMasksToBoxes(unittest.TestCase):
    def test_output_structure(self):
        """masks_to_boxes_and_labels should return a list of target dicts."""
        # N=2 samples, M=3 objects, H=W=32
        masks = torch.zeros(2, 3, 32, 32)
        masks[0, 0, 4:12, 4:12] = 1.0   # valid object
        targets = masks_to_boxes_and_labels(masks)
        self.assertEqual(len(targets), 2)
        self.assertIn('boxes', targets[0])
        self.assertIn('labels', targets[0])

    def test_empty_mask(self):
        """All-zero masks should yield targets with 0 objects."""
        masks = torch.zeros(1, 3, 32, 32)
        targets = masks_to_boxes_and_labels(masks)
        self.assertEqual(targets[0]['boxes'].shape[0], 0)


class TestHungarianMatcher(unittest.TestCase):
    def test_returns_indices(self):
        """Matcher should return a list of index tuples of correct length."""
        matcher = HungarianMatcher()
        B, Q, num_classes = 2, 5, 3
        outputs = {
            'pred_logits': torch.randn(B, Q, num_classes + 1),
            'pred_boxes': torch.rand(B, Q, 4),
        }
        targets = [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)},
            {'labels': torch.tensor([2]),    'boxes': torch.rand(1, 4)},
        ]
        indices = matcher(outputs, targets)
        self.assertEqual(len(indices), B)
        for row_ind, col_ind in indices:
            self.assertEqual(len(row_ind), len(col_ind))


class TestSetCriterion(unittest.TestCase):
    def _make_criterion(self, num_classes=3):
        matcher = HungarianMatcher()
        return SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
            eos_coef=0.1,
            losses=['labels', 'boxes'],
        )

    def test_loss_keys_present(self):
        """SetCriterion should return loss_ce, loss_bbox, and loss_giou."""
        criterion = self._make_criterion()
        outputs = {
            'pred_logits': torch.randn(2, 5, 4),
            'pred_boxes': torch.rand(2, 5, 4),
        }
        targets = [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)},
            {'labels': torch.tensor([2]),    'boxes': torch.rand(1, 4)},
        ]
        losses = criterion(outputs, targets)
        self.assertIn('loss_ce', losses)
        self.assertIn('loss_bbox', losses)
        self.assertIn('loss_giou', losses)

    def test_losses_are_finite_scalars(self):
        """All computed losses should be finite scalar tensors."""
        criterion = self._make_criterion()
        outputs = {
            'pred_logits': torch.randn(1, 5, 4),
            'pred_boxes': torch.rand(1, 5, 4),
        }
        targets = [{'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}]
        losses = criterion(outputs, targets)
        for name, val in losses.items():
            self.assertTrue(torch.isfinite(val), f"{name} is not finite")
            self.assertEqual(val.ndim, 0)


# ── DETR Full Model ───────────────────────────────────────────────────────────

class TestDETR(unittest.TestCase):
    def _build_detr(self, d_model=128, num_classes=3, num_queries=5):
        backbone = ResNetBackbone(train_backbone=False)
        transformer = Transformer(
            d_model=d_model, nhead=4,
            num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256
        )
        return DETR(backbone=backbone, transformer=transformer,
                    num_classes=num_classes, num_queries=num_queries)

    def test_forward_output_shapes(self):
        """DETR forward pass should produce correctly shaped outputs with layer dim."""
        model = self._build_detr(num_classes=3, num_queries=5)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        # [B, L, Q, C] — L=2 decoder layers by default in _build_detr
        self.assertEqual(out['pred_logits'].shape, (2, 2, 5, 4))  # 2 layers, 3 classes + 1
        self.assertEqual(out['pred_boxes'].shape, (2, 2, 5, 4))

    def test_pred_boxes_sigmoid(self):
        """Predicted boxes should be in [0, 1] (sigmoid-normalised)."""
        model = self._build_detr()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        # pred_boxes is [B, L, Q, 4] — check all layers
        self.assertTrue((out['pred_boxes'] >= 0.0).all())
        self.assertTrue((out['pred_boxes'] <= 1.0).all())


# ── Factory / Wrappers ────────────────────────────────────────────────────────

class TestStandardizedDETRWrapper(unittest.TestCase):
    def _make_wrapper(self):
        backbone = ResNetBackbone(train_backbone=False)
        transformer = Transformer(d_model=128, nhead=4, num_encoder_layers=2,
                                  num_decoder_layers=2, dim_feedforward=256)
        base = DETR(backbone=backbone, transformer=transformer,
                    num_classes=3, num_queries=5)
        return StandardizedDETRWrapper(base)

    def test_output_contract_keys(self):
        """Wrapper output must contain all standardised contract keys."""
        wrapper = self._make_wrapper()
        x = torch.randn(2, 3, 64, 64)
        out = wrapper(x)
        for key in ('pred_boxes', 'pred_masks', 'pred_logits', 'recon_img', 'input_img'):
            self.assertIn(key, out)
        # Contract key pred_logits is last layer only [B, Q, C]
        self.assertEqual(out['pred_logits'].ndim, 3)
        # Full stacked output available for loss
        self.assertIn('pred_logits_all', out)
        self.assertEqual(out['pred_logits_all'].ndim, 4)

    def test_pred_masks_is_none(self):
        """DETR wrapper should always set pred_masks=None."""
        wrapper = self._make_wrapper()
        out = wrapper(torch.randn(1, 3, 64, 64))
        self.assertIsNone(out['pred_masks'])

    def test_video_input_batch(self):
        """5D input [B, T, C, H, W] should be processed without error."""
        wrapper = self._make_wrapper()
        x = torch.randn(2, 3, 3, 64, 64)  # B=2, T=3
        out = wrapper(x)
        self.assertIn('pred_boxes', out)


class TestStandardizedSAViWrapper(unittest.TestCase):
    def _make_savi_wrapper(self):
        from src.models.savi import SAVi
        base = SAVi(resolution=(64, 64), clip_len=3, num_slots=4, slot_dim=64, num_iterations=2)
        return StandardizedSAViWrapper(base)

    def test_output_contract_keys(self):
        """SAVi wrapper output must contain all standardised contract keys."""
        wrapper = self._make_savi_wrapper()
        x = torch.randn(2, 3, 3, 64, 64)
        out = wrapper(x)
        for key in ('pred_boxes', 'pred_masks', 'pred_logits', 'recon_img'):
            self.assertIn(key, out)

    def test_pred_boxes_is_none(self):
        """SAVi wrapper should always set pred_boxes=None."""
        wrapper = self._make_savi_wrapper()
        out = wrapper(torch.randn(2, 3, 3, 64, 64))
        self.assertIsNone(out['pred_boxes'])


class TestBuildModel(unittest.TestCase):
    def test_build_detr(self):
        """build_model with 'detr' config should return a StandardizedDETRWrapper."""
        cfg = {'model': {'name': 'detr', 'num_classes': 3, 'num_queries': 5}}
        model = build_model(cfg)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, StandardizedDETRWrapper)

    def test_build_savi(self):
        """build_model with 'savi' config should return a StandardizedSAViWrapper."""
        cfg = {'model': {'name': 'savi', 'num_slots': 4}}
        model = build_model(cfg)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, StandardizedSAViWrapper)

    def test_unknown_model_raises(self):
        """build_model with an unknown type should raise ValueError."""
        cfg = {'model': {'name': 'nonexistent_model_xyz'}}
        with self.assertRaises(ValueError):
            build_model(cfg)


if __name__ == '__main__':
    unittest.main()

