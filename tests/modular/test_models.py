"""
Tests for src/models — covering:
  - SAVi models
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

from src.models.savi import SAVi
from src.models.factory import build_model, register_model, list_models
from src.models.base import BaseModelWrapper
from src.models.base import ModelOutput
from src.models.wrappers import (
    StandardizedSAViWrapper,
)

# ── Moved from model_output.py (only tests used them) ─────────────────────────

_EVAL_OUTPUT_KEYS: frozenset = frozenset({
    "input_img", "pred_masks", "recon_img", "post_slots",
})


def _validate_model_output(out: dict, model_name: str = "unknown") -> None:
    """Runtime check that a forward-pass output contains the required 'input_img' key."""
    if "input_img" not in out:
        raise ValueError(
            f"Model '{model_name}' forward() output is missing the required "
            f"'input_img' key.  All wrappers must include it."
        )


# ── Model Registry Tests ──────────────────────────────────────────────────────

class TestModelRegistry(unittest.TestCase):
    def test_list_models_contains_expected(self):
        """list_models() should return all built-in model types."""
        registered = list_models()
        for name in ("savi", "stosavi"):
            self.assertIn(name, registered, f"'{name}' not found in registry: {registered}")

    def test_register_model_adds_to_registry(self):
        """@register_model decorator should add a class under the given name."""
        from src.models.factory import _MODEL_REGISTRY

        test_name = "_pytest_dummy_model_zzz"
        self.assertNotIn(test_name, _MODEL_REGISTRY)

        @register_model(test_name)
        class _DummyWrapper(BaseModelWrapper):
            @classmethod
            def build(cls, model_cfg):
                return cls()

            def forward(self, x):
                return {"input_img": x, "pred_masks": None, "recon_img": None, "post_slots": None}

            def compute_loss(self, out, batch):
                return torch.tensor(0.0), {}

        self.assertIn(test_name, _MODEL_REGISTRY)
        self.assertIs(_MODEL_REGISTRY[test_name], _DummyWrapper)

        del _MODEL_REGISTRY[test_name]

    def test_build_model_dispatches_savi(self):
        """build_model({'model': {'name': 'savi', ...}}) should return a SAViWrapper."""
        cfg = {'model': {'name': 'savi', 'type': 'savi', 'num_slots': 4}}
        model = build_model(cfg)
        self.assertIsInstance(model, StandardizedSAViWrapper)

    def test_slotformer_missing_stage1_checkpoint_raises(self):
        """A configured-but-missing stage-1 checkpoint must fail loudly."""
        cfg = {
            "model": {
                "name": "slotformer",
                "type": "slotformer",
                "stage1_ckpt_path": "/nonexistent/stage1_model.pt",
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 4,
                "ffn_dim": 64,
            }
        }
        with self.assertRaises(FileNotFoundError):
            build_model(cfg)

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
            def forward(self, x): return {"input_img": x, "pred_masks": None,
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
            def forward(self, x):
                return {"input_img": x, "pred_masks": None, "recon_img": None, "post_slots": None}

        with self.assertRaises(TypeError):
            _Incomplete()


# ── ModelOutput Contract Tests ────────────────────────────────────────────────

class TestModelOutputContract(unittest.TestCase):
    def test_validate_requires_input_img(self):
        """validate_model_output should raise if 'input_img' is missing."""
        bad_out = {"pred_masks": None}
        with self.assertRaises(ValueError):
            _validate_model_output(bad_out, "test_model")

    def test_validate_passes_with_input_img(self):
        """validate_model_output should not raise when 'input_img' is present."""
        good_out = {"input_img": torch.zeros(1, 3, 64, 64)}
        _validate_model_output(good_out, "test_model")

    def test_eval_output_keys_set(self):
        """EVAL_OUTPUT_KEYS should contain the core contract keys."""
        for key in ("input_img", "pred_masks", "recon_img"):
            self.assertIn(key, _EVAL_OUTPUT_KEYS)


# ── Wrapper Forward Shape Tests ───────────────────────────────────────────────

class TestStandardizedSAViWrapper(unittest.TestCase):
    def _make_savi_wrapper(self):
        base = SAVi(resolution=(64, 64), clip_len=3, num_slots=4, slot_dim=64, num_iterations=2)
        return StandardizedSAViWrapper(base)

    def test_output_contract_keys(self):
        """SAVi wrapper output must contain all standardised contract keys."""
        wrapper = self._make_savi_wrapper()
        x = torch.randn(2, 3, 3, 64, 64)
        out = wrapper(x)
        for key in ('pred_masks', 'recon_img', 'input_img', 'post_slots'):
            self.assertIn(key, out)

    def test_all_eval_keys_present(self):
        """All EVAL_OUTPUT_KEYS must be present in SAVi wrapper output."""
        wrapper = self._make_savi_wrapper()
        out = wrapper(torch.randn(1, 3, 3, 64, 64))
        for key in _EVAL_OUTPUT_KEYS:
            self.assertIn(key, out, f"Missing EVAL_OUTPUT_KEY: '{key}'")


# ── run_epoch() Mock Tests ────────────────────────────────────────────────────

class TestRunEpoch(unittest.TestCase):
    def _make_mock_model(self):
        m = mock.MagicMock(spec=BaseModelWrapper)
        dummy_out = {
            "input_img": torch.zeros(2, 3, 3, 64, 64),
            "pred_masks": None, "recon_img": None, "post_slots": None,
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

                dummy_out = {
                    "input_img": torch.zeros(2, 3, 3, 64, 64),
                    "pred_masks": None, "recon_img": None, "post_slots": None,
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
        self.assertEqual(new_step, 2)

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
                    "pred_masks": None, "recon_img": None, "post_slots": None,
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


# ── OCVP Factorized SlotRollouter Tests ────────────────────────────────────────

class TestOCVPSlotRollouter(unittest.TestCase):
    def test_ocvp_rollouter_forward_shape(self):
        """OCVPSlotRollouter should output correct shape [B, pred_len, K, D]."""
        from src.models.slotformer import OCVPSlotRollouter
        rollouter = OCVPSlotRollouter(
            num_slots=4,
            slot_size=64,
            history_len=2,
            d_model=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
        )
        x = torch.randn(2, 2, 4, 64)
        out = rollouter(x, pred_len=4)
        self.assertEqual(out.shape, (2, 4, 4, 64))

    def test_ocvp_rollouter_gradient_flow(self):
        """Gradients must flow through Temporal and Interactive attention layers."""
        from src.models.slotformer import OCVPSlotRollouter
        rollouter = OCVPSlotRollouter(
            num_slots=4,
            slot_size=32,
            history_len=2,
            d_model=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
        )
        x = torch.randn(2, 2, 4, 32, requires_grad=True)
        out = rollouter(x, pred_len=2)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_build_ocvp_slotformer_wrapper(self):
        """build_model should construct OCVP SlotFormer wrapper from config."""
        from src.models.factory import build_model
        cfg = {
            "model": {
                "name": "ocvp_slotformer",
                "type": "ocvp_slotformer",
                "rollouter_type": "ocvp",
                "stage1_ckpt_path": None,
                "d_model": 32,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 64,
            }
        }
        wrapper = build_model(cfg)
        self.assertEqual(wrapper.model.rollouter.__class__.__name__, "OCVPSlotRollouter")


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

