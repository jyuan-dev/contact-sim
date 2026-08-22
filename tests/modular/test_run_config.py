"""
Tests for the configuration seam (src/config/run_config.py):
golden default resolution, experiment precedence, typo rejection,
canonicalization conflicts, legacy snapshot round-trips, and resume checks.
"""

import os
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config.run_config import (
    ConfigError,
    RunConfig,
    assert_resume_compatible,
    load_snapshot,
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")


def _compose(overrides=None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides or [])
        return OmegaConf.to_container(cfg, resolve=True)


def _norm(d):
    return {k: (_norm(v) if isinstance(v, dict) else v)
            for k, v in sorted(d.items()) if v is not None}


class TestGoldenResolution(unittest.TestCase):
    def test_default_resolution_roundtrips(self):
        raw = _compose()
        emitted = RunConfig.from_dict(raw).to_dict()
        self.assertEqual(_norm(emitted), _norm(raw))

    def test_experiment_keys_take_effect(self):
        run = RunConfig.from_dict(_compose(["experiment=savi_pusht_3slots_sigreg_001"]))
        self.assertEqual(run.train.exp_name, "savi_pusht_3slots_sigreg_001")
        self.assertEqual(run.model.num_slots, 3)

    def test_rollouter_roundtrips(self):
        raw = _compose(["model=ocvp_intact_slotformer", "model.stage1_ckpt_path=''"])
        emitted = RunConfig.from_dict(raw).to_dict()
        self.assertEqual(_norm(emitted), _norm(raw))


class TestValidation(unittest.TestCase):
    def test_top_level_typo_rejected(self):
        raw = _compose()
        raw["lrate"] = 1e-3
        with self.assertRaises(ConfigError) as ctx:
            RunConfig.from_dict(raw)
        self.assertIn("lrate", str(ctx.exception))
        self.assertIn("lr", str(ctx.exception))

    def test_model_section_typo_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            RunConfig.from_dict({"model": {"type": "savi", "num_slot": 3},
                                 "dataset": {"name": "pusht"}})
        self.assertIn("num_slot", str(ctx.exception))

    def test_dataset_section_typo_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            RunConfig.from_dict({"model": {"type": "savi"},
                                 "dataset": {"name": "pusht", "train_fraction": 0.8}})
        self.assertIn("train_fraction", str(ctx.exception))
        self.assertIn("train_frac", str(ctx.exception))

    def test_family_passthrough_allowed(self):
        run = RunConfig.from_dict(
            {"model": {"type": "slotformer", "d_model": 96, "stage1_ckpt_path": "x.pt"},
             "dataset": {"name": "pusht", "h5_path": "y.h5"}})
        self.assertEqual(run.model.extra["d_model"], 96)
        self.assertEqual(run.dataset.extra["h5_path"], "y.h5")

    def test_conflicting_num_slots_rejected(self):
        with self.assertRaises(ConfigError):
            RunConfig.from_dict({"model": {"type": "savi", "num_slots": 4,
                                           "slot_dict": {"num_slots": 3}},
                                 "dataset": {"name": "pusht"}})

    def test_conflicting_resolution_rejected(self):
        with self.assertRaises(ConfigError):
            RunConfig.from_dict({"model": {"type": "savi", "resolution": [64, 64]},
                                 "dataset": {"name": "pusht", "resolution": [32, 32]}})


class TestSnapshotRoundTrip(unittest.TestCase):
    def test_legacy_snapshot_loads_permissively(self):
        run = load_snapshot(os.path.join(CONFIG_DIR, "..", "scratch",
                                         "checkpoints", "savi_pusht"))
        self.assertIsNotNone(run)
        self.assertEqual(run.model.num_slots, 4)
        self.assertEqual(run.train.exp_name, "savi_pusht")

    def test_save_load_roundtrip(self):
        raw = _compose()
        run = RunConfig.from_dict(raw)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            run.save_snapshot(path)
            loaded = load_snapshot(tmpdir)
        self.assertIsNotNone(loaded)
        self.assertEqual(_norm(loaded.to_dict()), _norm(run.to_dict()))

    def test_missing_snapshot_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(load_snapshot(tmpdir))


class TestOverridesAndResume(unittest.TestCase):
    def test_dotted_overrides(self):
        raw = _compose()
        run = RunConfig.from_dict(raw).apply_overrides(
            {"batch_size": 7, "dataset.train_frac": 0.5,
             "loss.losses.sigreg.weight": 0.03})
        self.assertEqual(run.train.batch_size, 7)
        self.assertEqual(run.dataset.train_frac, 0.5)
        self.assertEqual(run.to_dict()["loss"]["losses"]["sigreg"]["weight"], 0.03)

    def test_resume_compat_rejects_topology_mismatch(self):
        snapshot = RunConfig.from_dict(_compose(["experiment=savi_pusht_3slots_sigreg_001"]))
        live = RunConfig.from_dict(_compose())  # default 4 slots
        with self.assertRaises(ConfigError) as ctx:
            assert_resume_compatible(snapshot, live)
        self.assertIn("num_slots", str(ctx.exception))

    def test_resume_compat_passes_identical(self):
        snapshot = RunConfig.from_dict(_compose())
        assert_resume_compatible(snapshot, snapshot)


class TestEvalShim(unittest.TestCase):
    def test_parse_flags_and_keyvalue(self):
        from scripts.eval import _parse_eval_args

        args = _parse_eval_args(["--ckpt_path", "a/b.pt", "mode=visualize", "batch_size=8"])
        self.assertEqual(args.ckpt_path, "a/b.pt")
        self.assertEqual(args.mode, "visualize")
        self.assertEqual(args.batch_size, "8")

    def test_unknown_flag_rejected(self):
        from scripts.eval import _parse_eval_args

        with self.assertRaises(ConfigError):
            _parse_eval_args(["model=slotformer"])


if __name__ == "__main__":
    unittest.main()
