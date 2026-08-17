"""
Tests for the canonical checkpoint format and the documented
load_checkpoint_state format table.
"""

import os
import tempfile
import unittest

import torch

from src.models.savi import SAVi
from src.models.wrappers import StandardizedSAViWrapper
from src.utils.training_utils import load_checkpoint_state, save_checkpoint


def _make_wrapper():
    """Small SAVi wrapper for fast CPU tests."""
    base = SAVi(resolution=(64, 64), clip_len=3, num_slots=2, slot_dim=32, num_iterations=1)
    return StandardizedSAViWrapper(base)


class TestCanonicalCheckpointFormat(unittest.TestCase):
    def test_save_load_roundtrip(self):
        wrapper = _make_wrapper()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "savi_best.pt")
            save_checkpoint(wrapper, path, epoch=3)

            ckpt = torch.load(path, map_location="cpu")
            self.assertEqual(ckpt["epoch"], 3)
            self.assertEqual(set(ckpt["model_state"].keys()), set(wrapper.state_dict().keys()))

            fresh = _make_wrapper()
            load_checkpoint_state(fresh, path)  # canonical: exact match
            for k in wrapper.state_dict():
                self.assertTrue(torch.equal(wrapper.state_dict()[k], fresh.state_dict()[k]))

    def test_inner_savi_accessor(self):
        from src.models.savi import StoSAVi
        wrapper = _make_wrapper()
        self.assertIsInstance(wrapper.inner_savi(), StoSAVi)

    def test_base_wrapper_without_core(self):
        from src.models.base import BaseModelWrapper

        class NoCore(BaseModelWrapper):
            @classmethod
            def build(cls, cfg): return cls()
            def forward(self, x): return {}
            def compute_loss(self, out, batch): return torch.tensor(0.0), {}

        self.assertFalse(hasattr(NoCore(), "inner_savi"))


class TestLoadCheckpointStateFormats(unittest.TestCase):
    def _load_via(self, state, target=None):
        target = target or _make_wrapper()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save({"model_state": state, "epoch": 0}, path)
            load_checkpoint_state(target, path)
        return target

    def test_legacy_inner_keys_prefix_model_model(self):
        """Bare core StoSAVi keys load into the wrapper (prefix model.model.)."""
        wrapper = _make_wrapper()
        inner_state = wrapper.inner_savi().state_dict()  # bare core keys
        target = self._load_via(inner_state)
        self.assertEqual(set(target.inner_savi().state_dict().keys()),
                         set(wrapper.inner_savi().state_dict().keys()))

    def test_legacy_savi_keys_prefix_model(self):
        """SAVi-level 'model.*' keys load into the wrapper (prefix model.)."""
        wrapper = _make_wrapper()
        savi_state = wrapper.model.state_dict()  # 'model.*' keys
        target = self._load_via(savi_state)
        for k in wrapper.model.state_dict():
            self.assertTrue(torch.equal(wrapper.model.state_dict()[k], target.model.state_dict()[k]))

    def test_legacy_wrapper_keys_strip_model(self):
        """'model.model.*' keys load into a bare SAVi (strip one model.)."""
        wrapper = _make_wrapper()
        wrapper_state = wrapper.state_dict()  # 'model.model.*' keys
        target = SAVi(resolution=(64, 64), clip_len=3, num_slots=2, slot_dim=32, num_iterations=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            torch.save({"model_state": wrapper_state, "epoch": 0}, path)
            load_checkpoint_state(target, path)
        self.assertEqual(set(target.state_dict().keys()), set(wrapper.model.state_dict().keys()))

    def test_ddp_module_strip(self):
        wrapper = _make_wrapper()
        state = {"module." + k: v for k, v in wrapper.state_dict().items()}
        target = self._load_via(state)
        self.assertEqual(set(target.state_dict().keys()), set(wrapper.state_dict().keys()))

    def test_unexpected_key_raises(self):
        wrapper = _make_wrapper()
        state = dict(wrapper.state_dict())
        state["bogus.parameter"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            self._load_via(state)


if __name__ == "__main__":
    unittest.main()
