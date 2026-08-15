"""
Tests for the checkpoint bootstrap module — reconstructing an experiment
(config + model) from a checkpoint.
"""

import os
import tempfile
import unittest
import unittest.mock as mock

import torch


SAVI_YAML = """\
model:
  name: savi
  type: savi
  num_slots: 2
  resolution: [64, 64]
n_sample_frames: 4
"""


class TestSniffSlotFormerArch(unittest.TestCase):
    def test_sniffs_d_model_ffn_layers(self):
        from src.utils.checkpoint_bootstrap import sniff_slotformer_arch

        state = {
            "rollouter.in_proj.weight": torch.zeros(160, 80),
            "rollouter.transformer_encoder.layers.0.linear1.weight": torch.zeros(640, 160),
            "rollouter.transformer_encoder.layers.0.self_attn.out_proj.weight": torch.zeros(160, 160),
            "rollouter.transformer_encoder.layers.1.self_attn.out_proj.weight": torch.zeros(160, 160),
            "rollouter.transformer_encoder.layers.4.linear2.weight": torch.zeros(160, 640),
            "unrelated.key": torch.zeros(4),
        }
        arch = sniff_slotformer_arch(state)
        # num_layers starts at 4 and only grows (ported legacy behavior)
        self.assertEqual(arch, {"d_model": 160, "ffn_dim": 640, "num_layers": 5})

    def test_defaults_without_rollouter_keys(self):
        from src.utils.checkpoint_bootstrap import sniff_slotformer_arch

        arch = sniff_slotformer_arch({"some.other.key": torch.zeros(1)})
        self.assertEqual(arch, {"d_model": 128, "ffn_dim": 512, "num_layers": 4})


class TestBootstrapCheckpoint(unittest.TestCase):
    def _write_checkpoint(self, tmpdir, name="savi_best.pt", state=None):
        ckpt_path = os.path.join(tmpdir, name)
        torch.save({"model_state": state or {}, "epoch": 0}, ckpt_path)
        return ckpt_path

    def test_found_config_builds_model(self):
        from src.utils.checkpoint_bootstrap import bootstrap_checkpoint
        from src.models.wrappers import StandardizedSAViWrapper

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
                f.write(SAVI_YAML)
            ckpt_path = self._write_checkpoint(tmpdir)

            model, cfg_dict = bootstrap_checkpoint(ckpt_path)

            self.assertIsInstance(model, StandardizedSAViWrapper)
            self.assertEqual(cfg_dict["model"]["num_slots"], 2)

    def test_cli_overrides_merge_over_saved_config(self):
        from src.utils.checkpoint_bootstrap import bootstrap_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
                f.write(SAVI_YAML)
            ckpt_path = self._write_checkpoint(tmpdir)

            _, cfg_dict = bootstrap_checkpoint(ckpt_path, cli_overrides={"device": "cpu"})
            self.assertEqual(cfg_dict["device"], "cpu")

    def test_missing_checkpoint_raises(self):
        from src.utils.checkpoint_bootstrap import bootstrap_checkpoint

        with self.assertRaises(FileNotFoundError):
            bootstrap_checkpoint("/nonexistent/path/model.pt")

    def test_parse_error_raises_runtime_error(self):
        from src.utils.checkpoint_bootstrap import bootstrap_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
                f.write("model: [unclosed")
            ckpt_path = self._write_checkpoint(tmpdir)

            with self.assertRaises(RuntimeError):
                bootstrap_checkpoint(ckpt_path)

    def test_missing_config_sniffs_slotformer(self):
        from src.utils.checkpoint_bootstrap import bootstrap_checkpoint

        state = {
            "rollouter.in_proj.weight": torch.zeros(144, 72),
            "rollouter.transformer_encoder.layers.0.linear1.weight": torch.zeros(576, 144),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = self._write_checkpoint(tmpdir, state=state)

            with mock.patch("src.utils.checkpoint_bootstrap.build_model") as fake_build:
                fake_build.return_value = "fake-wrapper"
                model, cfg_dict = bootstrap_checkpoint(ckpt_path)

            fake_build.assert_called_once()
            self.assertEqual(model, "fake-wrapper")
            self.assertEqual(cfg_dict["model"]["name"], "slotformer")
            self.assertEqual(cfg_dict["model"]["d_model"], 144)
            self.assertEqual(cfg_dict["model"]["ffn_dim"], 576)


if __name__ == "__main__":
    unittest.main()
