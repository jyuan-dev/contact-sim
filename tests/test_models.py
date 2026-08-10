"""
Tests for src/models — covering:
  - DETR component unit tests (backbone, MLP, box utilities, matcher, criterion)
  - DETR full model forward shape
  - Model registry (@register_model decorator + build_model dispatch)
  - BaseModelWrapper ABC enforcement
  - Wrapper forward() output contract (ModelOutput keys)
  - ModelOutput required key validation
  - run_epoch() with a mock model and loader (training loop decoupling)
  - TrainConfig.from_cfg construction
"""

import sys
import unittest
import unittest.mock as mock
import tempfile
import os

import torch

from src.models.detr import (
    DETR, ResNetBackbone, Transformer, MLP,
    box_cxcywh_to_xyxy, box_iou, generalized_box_iou,
    masks_to_boxes_and_labels, HungarianMatcher, SetCriterion,
)
from src.models.savi import SAVi
from src.models.factory import build_model, register_model, list_models
from src.models.base import BaseModelWrapper
from src.models.model_output import ModelOutput, validate_model_output, EVAL_OUTPUT_KEYS
from src.models.wrappers import (
    StandardizedDETRWrapper,
    StandardizedSAViWrapper,
)


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
        b1[:, 2:] = b1[:, :2] + b1[:, 2:].abs() + 0.01
        b2[:, 2:] = b2[:, :2] + b2[:, 2:].abs() + 0.01
        giou = generalized_box_iou(b1, b2)
        self.assertTrue((giou >= -1.0 - 1e-5).all())
        self.assertTrue((giou <= 1.0 + 1e-5).all())


class TestMasksToBoxes(unittest.TestCase):
    def test_output_structure(self):
        """masks_to_boxes_and_labels should return a list of target dicts."""
        masks = torch.zeros(2, 3, 32, 32)
        masks[0, 0, 4:12, 4:12] = 1.0
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
        self.assertEqual(out['pred_logits'].shape, (2, 2, 5, 4))
        self.assertEqual(out['pred_boxes'].shape, (2, 2, 5, 4))

    def test_pred_boxes_sigmoid(self):
        """Predicted boxes should be in [0, 1] (sigmoid-normalised)."""
        model = self._build_detr()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        self.assertTrue((out['pred_boxes'] >= 0.0).all())
        self.assertTrue((out['pred_boxes'] <= 1.0).all())


# ── Model Registry Tests ──────────────────────────────────────────────────────

class TestModelRegistry(unittest.TestCase):
    def test_list_models_contains_expected(self):
        """list_models() should return all 4 built-in model types."""
        registered = list_models()
        for name in ("detr", "deformable_detr", "savi", "stosavi", "deformable_savi"):
            self.assertIn(name, registered, f"'{name}' not found in registry: {registered}")

    def test_register_model_adds_to_registry(self):
        """@register_model decorator should add a class under the given name."""
        from src.models.factory import _MODEL_REGISTRY

        # Use a unique name to avoid collision with built-in names
        test_name = "_pytest_dummy_model_zzz"
        self.assertNotIn(test_name, _MODEL_REGISTRY)

        @register_model(test_name)
        class _DummyWrapper(BaseModelWrapper):
            @classmethod
            def build(cls, model_cfg):
                return cls()

            def forward(self, x):
                return {"input_img": x, "pred_boxes": None, "pred_masks": None,
                        "pred_logits": None, "recon_img": None, "post_slots": None}

            def compute_loss(self, out, batch):
                return torch.tensor(0.0), {}

        self.assertIn(test_name, _MODEL_REGISTRY)
        self.assertIs(_MODEL_REGISTRY[test_name], _DummyWrapper)

        # Clean up so other tests are not affected
        del _MODEL_REGISTRY[test_name]

    def test_build_model_dispatches_detr(self):
        """build_model({'model': {'name': 'detr', ...}}) should return a DETRWrapper."""
        cfg = {'model': {'name': 'detr', 'type': 'detr', 'num_classes': 3, 'num_queries': 5}}
        model = build_model(cfg)
        self.assertIsInstance(model, StandardizedDETRWrapper)

    def test_build_model_dispatches_savi(self):
        """build_model({'model': {'name': 'savi', ...}}) should return a SAViWrapper."""
        cfg = {'model': {'name': 'savi', 'type': 'savi', 'num_slots': 4}}
        model = build_model(cfg)
        self.assertIsInstance(model, StandardizedSAViWrapper)

    def test_build_model_unknown_raises_value_error(self):
        """build_model with an unknown type should raise ValueError listing available models."""
        cfg = {'model': {'name': 'nonexistent_model_xyz', 'type': 'nonexistent_model_xyz'}}
        with self.assertRaises(ValueError) as ctx:
            build_model(cfg)
        self.assertIn("nonexistent_model_xyz", str(ctx.exception))

    def test_register_duplicate_raises(self):
        """Registering the same name twice should raise ValueError."""
        from src.models.factory import _MODEL_REGISTRY
        test_name = "_pytest_dup_model_aaa"

        @register_model(test_name)
        class _W1(BaseModelWrapper):
            @classmethod
            def build(cls, c): return cls()
            def forward(self, x): return {"input_img": x, "pred_boxes": None,
                                          "pred_masks": None, "pred_logits": None,
                                          "recon_img": None, "post_slots": None}
            def compute_loss(self, o, b): return torch.tensor(0.0), {}

        with self.assertRaises(ValueError):
            @register_model(test_name)
            class _W2(BaseModelWrapper):
                @classmethod
                def build(cls, c): return cls()
                def forward(self, x): return {}
                def compute_loss(self, o, b): return torch.tensor(0.0), {}

        del _MODEL_REGISTRY[test_name]


# ── BaseModelWrapper ABC Tests ────────────────────────────────────────────────

class TestBaseModelWrapperABC(unittest.TestCase):
    def test_cannot_instantiate_abc_directly(self):
        """BaseModelWrapper is abstract and cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseModelWrapper()

    def test_partial_implementation_raises(self):
        """A subclass missing any abstract method cannot be instantiated."""
        class _Incomplete(BaseModelWrapper):
            # Missing build() and compute_loss()
            def forward(self, x):
                return {"input_img": x, "pred_boxes": None, "pred_masks": None,
                        "pred_logits": None, "recon_img": None, "post_slots": None}

        with self.assertRaises(TypeError):
            _Incomplete()


# ── ModelOutput Contract Tests ────────────────────────────────────────────────

class TestModelOutputContract(unittest.TestCase):
    def test_validate_requires_input_img(self):
        """validate_model_output should raise if 'input_img' is missing."""
        bad_out = {"pred_boxes": None}
        with self.assertRaises(ValueError):
            validate_model_output(bad_out, "test_model")

    def test_validate_passes_with_input_img(self):
        """validate_model_output should not raise when 'input_img' is present."""
        good_out = {"input_img": torch.zeros(1, 3, 64, 64)}
        validate_model_output(good_out, "test_model")  # no exception

    def test_eval_output_keys_set(self):
        """EVAL_OUTPUT_KEYS should contain the core contract keys."""
        for key in ("input_img", "pred_boxes", "pred_masks", "recon_img"):
            self.assertIn(key, EVAL_OUTPUT_KEYS)


# ── Wrapper Forward Shape Tests ───────────────────────────────────────────────

class TestStandardizedDETRWrapper(unittest.TestCase):
    def _make_wrapper(self):
        backbone = ResNetBackbone(train_backbone=False)
        transformer = Transformer(
            d_model=128, nhead=4,
            num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256
        )
        base = DETR(backbone=backbone, transformer=transformer,
                    num_classes=3, num_queries=5)
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=3, matcher=matcher,
            weight_dict={"class": 1.0, "bbox": 5.0, "giou": 2.0},
            eos_coef=0.1, losses=["labels", "boxes"]
        )
        return StandardizedDETRWrapper(base, criterion=criterion,
                                       weight_dict={"class": 1.0, "bbox": 5.0, "giou": 2.0})

    def test_output_contract_keys(self):
        """Wrapper output must contain all standardised contract keys."""
        wrapper = self._make_wrapper()
        x = torch.randn(2, 3, 64, 64)
        out = wrapper(x)
        for key in ('pred_boxes', 'pred_masks', 'pred_logits', 'recon_img', 'input_img'):
            self.assertIn(key, out)
        self.assertEqual(out['pred_logits'].ndim, 3)
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
        x = torch.randn(2, 3, 3, 64, 64)
        out = wrapper(x)
        self.assertIn('pred_boxes', out)


class TestStandardizedSAViWrapper(unittest.TestCase):
    def _make_savi_wrapper(self):
        base = SAVi(resolution=(64, 64), clip_len=3, num_slots=4, slot_dim=64, num_iterations=2)
        return StandardizedSAViWrapper(base)

    def test_output_contract_keys(self):
        """SAVi wrapper output must contain all standardised contract keys."""
        wrapper = self._make_savi_wrapper()
        x = torch.randn(2, 3, 3, 64, 64)
        out = wrapper(x)
        for key in ('pred_boxes', 'pred_masks', 'pred_logits', 'recon_img', 'input_img'):
            self.assertIn(key, out)

    def test_pred_boxes_is_none(self):
        """SAVi wrapper should always set pred_boxes=None."""
        wrapper = self._make_savi_wrapper()
        out = wrapper(torch.randn(2, 3, 3, 64, 64))
        self.assertIsNone(out['pred_boxes'])

    def test_all_eval_keys_present(self):
        """All EVAL_OUTPUT_KEYS must be present in SAVi wrapper output."""
        wrapper = self._make_savi_wrapper()
        out = wrapper(torch.randn(1, 3, 3, 64, 64))
        for key in EVAL_OUTPUT_KEYS:
            self.assertIn(key, out, f"Missing EVAL_OUTPUT_KEY: '{key}'")


# ── run_epoch() Mock Tests ────────────────────────────────────────────────────

class TestRunEpoch(unittest.TestCase):
    """Test run_epoch() in isolation using mock model/loader/trainer."""

    def _make_mock_model(self):
        """Create a mock BaseModelWrapper that returns dummy outputs."""
        m = mock.MagicMock(spec=BaseModelWrapper)
        dummy_out = {
            "input_img": torch.zeros(2, 3, 3, 64, 64),
            "pred_boxes": None, "pred_masks": None,
            "pred_logits": None, "recon_img": None, "post_slots": None,
        }
        m.return_value = dummy_out
        m.__call__ = lambda self_m, x: dummy_out
        m.compute_loss.return_value = (torch.tensor(0.5), {"recon_loss": 0.5})
        m.train = mock.MagicMock()
        m.eval = mock.MagicMock()
        m.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])
        return m

    def _make_mock_loader(self, n_batches=3):
        batch = {"img": torch.randn(2, 3, 3, 64, 64)}
        return [batch] * n_batches

    def test_run_epoch_train_returns_avg_loss(self):
        """run_epoch in train mode should return a non-zero avg_loss."""
        from src.training.train_loop import TrainConfig, run_epoch

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.training.trainer import BaseTrainer
            import sys
            original_stdout = sys.stdout

            try:
                trainer = BaseTrainer(save_dir=tmpdir, experiment_name="test")
                model = self._make_mock_model()

                # Patch the model __call__ to work with MagicMock
                dummy_out = {
                    "input_img": torch.zeros(2, 3, 3, 64, 64),
                    "pred_boxes": None, "pred_masks": None,
                    "pred_logits": None, "recon_img": None, "post_slots": None,
                }
                model.side_effect = lambda x: dummy_out
                model.compute_loss.return_value = (torch.tensor(0.42), {"recon_loss": 0.42})

                optimizer = mock.MagicMock()
                optimizer.zero_grad = mock.MagicMock()
                optimizer.step = mock.MagicMock()

                scaler = mock.MagicMock()
                scaler.scale.return_value = torch.tensor(0.42)
                scaler.scale.return_value.backward = mock.MagicMock()
                scaler.unscale_ = mock.MagicMock()
                scaler.step = mock.MagicMock()
                scaler.update = mock.MagicMock()

                cfg = TrainConfig(dry_run=False, use_amp=False, grad_clip_norm=1.0)
                loader = self._make_mock_loader(n_batches=2)
                flag = [False]

                avg_loss, new_step = run_epoch(
                    model=model, loader=loader, optimizer=optimizer, scaler=scaler,
                    device=torch.device("cpu"), trainer=trainer, global_step=0,
                    cfg=cfg, is_train=True, total_target_steps=10, start_time=0.0,
                    epoch=0, num_epochs=1, stop_training_flag=flag,
                )
            finally:
                trainer.close()
                sys.stdout = original_stdout

        self.assertGreater(avg_loss, 0.0)
        self.assertEqual(new_step, 2)  # 2 batches processed

    def test_run_epoch_val_does_not_call_optimizer(self):
        """Validation mode should not call optimizer.step."""
        from src.training.train_loop import TrainConfig, run_epoch

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.training.trainer import BaseTrainer
            import sys
            original_stdout = sys.stdout

            try:
                trainer = BaseTrainer(save_dir=tmpdir, experiment_name="val_test")
                model = self._make_mock_model()
                dummy_out = {
                    "input_img": torch.zeros(2, 3, 3, 64, 64),
                    "pred_boxes": None, "pred_masks": None,
                    "pred_logits": None, "recon_img": None, "post_slots": None,
                }
                model.side_effect = lambda x: dummy_out
                model.compute_loss.return_value = (torch.tensor(0.3), {"recon_loss": 0.3})

                optimizer = mock.MagicMock()
                cfg = TrainConfig(dry_run=False, use_amp=False)
                loader = self._make_mock_loader(n_batches=2)
                flag = [False]

                run_epoch(
                    model=model, loader=loader, optimizer=None, scaler=None,
                    device=torch.device("cpu"), trainer=trainer, global_step=0,
                    cfg=cfg, is_train=False, total_target_steps=10, start_time=0.0,
                    epoch=0, num_epochs=1, stop_training_flag=flag,
                )
            finally:
                trainer.close()
                sys.stdout = original_stdout

        optimizer.step.assert_not_called()


# ── TrainConfig Tests ─────────────────────────────────────────────────────────

class TestTrainConfig(unittest.TestCase):
    def test_defaults(self):
        """Default TrainConfig should have sensible values."""
        from src.training.train_loop import TrainConfig
        cfg = TrainConfig()
        self.assertEqual(cfg.epochs, 8)
        self.assertAlmostEqual(cfg.lr, 2e-4)
        self.assertFalse(cfg.dry_run)

    def test_from_cfg_dict(self):
        """TrainConfig.from_cfg should correctly parse a config dict."""
        from src.training.train_loop import TrainConfig
        cfg_dict = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 20,
            "batch_size": 64,
            "dry_run": True,
            "use_amp": False,
            "grad_clip_norm": 0.5,
        }
        tc = TrainConfig.from_cfg(cfg_dict, ckpt_dir="/tmp/ckpt", model_name="savi")
        self.assertAlmostEqual(tc.lr, 1e-3)
        self.assertEqual(tc.epochs, 20)
        self.assertTrue(tc.dry_run)
        self.assertFalse(tc.use_amp)
        self.assertAlmostEqual(tc.grad_clip_norm, 0.5)
        self.assertEqual(tc.model_name, "savi")


if __name__ == '__main__':
    unittest.main()
