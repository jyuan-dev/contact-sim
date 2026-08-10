"""
End-to-End Pipeline Tests — full train → checkpoint → infer → eval flow.

Uses Hydra compose API (same config resolution as scripts/train.py) and
exercises the pipeline programmatically.  Much faster than subprocess
because it avoids @hydra.main's chdir + output-directory overhead.

Speed optimizations (batch_size=1, T=4 frames, 2 slots, 1 iteration):
~6 s total (down from ~22 s with defaults).

Run:  python -m unittest tests.e2e.test_pipeline -v
"""

import os
import sys
import time as _time
import tempfile
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.training.trainer import BaseTrainer
from src.training.train_loop import TrainConfig, run_training

# Speed-focused overrides: minimal model + synthetic data + single batch
_SPEED_OVERRIDES = [
    "dataset=gridshapes",
    "batch_size=1",
    "num_workers=0",
    "dry_run=true",
    "device=cpu",
    "model.num_slots=2",
    "model.num_iterations=1",
    "model.n_sample_frames=4",
    "dataset.n_sample_frames=4",
    "dataset.train_samples=16",
    "dataset.val_samples=4",
]


def _resolve_cfg(overrides: list[str] | None = None) -> dict:
    """Resolve Hydra config with the given overrides (same as scripts/train.py)."""
    with initialize_config_dir(
        config_dir=os.path.join(REPO_ROOT, "configs"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    return OmegaConf.to_container(cfg, resolve=True)


class TestPipeline(unittest.TestCase):
    """Train once (fast, ~5 s), then test checkpoint / infer / eval."""

    _profile: dict[str, float] = {}

    @classmethod
    def setUpClass(cls):
        t_total = _time.time()

        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.ckpt_dir = cls.tmpdir.name

        # ── Config resolve ──────────────────────────────────────────────
        t0 = _time.time()
        cfg_dict = _resolve_cfg(_SPEED_OVERRIDES)
        cls._profile["config"] = _time.time() - t0
        cls.model_name = cfg_dict["model"]["name"]

        # ── Model build ─────────────────────────────────────────────────
        t0 = _time.time()
        device = torch.device("cpu")
        model = build_model(cfg_dict).to(device)
        cls._profile["model_build"] = _time.time() - t0

        # ── Dataloaders ─────────────────────────────────────────────────
        t0 = _time.time()
        train_loader = build_dataloader(cfg_dict, split="train", batch_size=1, num_workers=0)
        val_loader = build_dataloader(cfg_dict, split="val", batch_size=1, num_workers=0)
        cls._profile["dataloaders"] = _time.time() - t0

        # ── Training ────────────────────────────────────────────────────
        t0 = _time.time()
        train_cfg = TrainConfig(
            dry_run=True, use_amp=False, batch_size=1, epochs=1,
            ckpt_dir=cls.ckpt_dir, model_name=cls.model_name,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        def _save_ckpt(m, path, epoch):
            torch.save({"model_state": m.state_dict(), "epoch": epoch}, path)

        trainer = BaseTrainer(save_dir=cls.ckpt_dir, experiment_name="e2e")
        try:
            run_training(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, scaler=scaler, device=device, cfg=train_cfg,
                trainer=trainer, save_checkpoint_fn=_save_ckpt,
            )
        finally:
            trainer.close()
        cls._profile["training"] = _time.time() - t0

        cls.ckpt_path = os.path.join(cls.ckpt_dir, f"{cls.model_name}_best.pt")
        cls._profile["total"] = _time.time() - t_total
        assert os.path.isfile(cls.ckpt_path), f"No checkpoint at {cls.ckpt_path}"

    @classmethod
    def tearDownClass(cls):
        # Print profile summary after all tests run
        p = cls._profile
        print(f"\n  ⏱  Pipeline profile: config={p['config']:.1f}s  "
              f"build={p['model_build']:.1f}s  data={p['dataloaders']:.1f}s  "
              f"train={p['training']:.1f}s  total={p['total']:.1f}s\n")
        cls.tmpdir.cleanup()

    # ── Training outputs ───────────────────────────────────────────────────

    def test_01_checkpoint_loadable(self):
        """Checkpoint should be valid .pt with model_state + epoch."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        self.assertIn("model_state", ckpt)
        self.assertIn("epoch", ckpt)

    def test_02_checkpoint_roundtrip(self):
        """Load checkpoint state into a fresh model successfully."""
        cfg_dict = _resolve_cfg(_SPEED_OVERRIDES)
        fresh = build_model(cfg_dict)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        fresh.load_state_dict(state["model_state"], strict=False)
        self.assertIsNotNone(fresh)

    # ── Inference ──────────────────────────────────────────────────────────

    def test_03_infer_forward_pass(self):
        """Run a forward pass like scripts/infer.py does and verify outputs."""
        cfg_dict = _resolve_cfg(_SPEED_OVERRIDES)
        model = build_model(cfg_dict)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=False)
        model.eval()

        loader = build_dataloader(cfg_dict, split="val", batch_size=1,
                                  num_workers=0, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(batch["img"])

        self.assertIsNotNone(out.get("recon_img"))
        self.assertIsNotNone(out.get("pred_masks"))

    # ── Eval ───────────────────────────────────────────────────────────────

    def test_04_eval_metrics_pipeline(self):
        """Run eval like scripts/eval.py does: load ckpt, compute metrics."""
        from src.metrics import EvaluationSuite

        cfg_dict = _resolve_cfg(_SPEED_OVERRIDES)
        model = build_model(cfg_dict)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        # Normalize keys (same logic as scripts/eval.py)
        norm_state = {}
        for k, v in state["model_state"].items():
            while k.startswith("module."):
                k = k[len("module."):]
            while k.startswith("model.model."):
                k = k[len("model."):]
            norm_state[k] = v
        model.load_state_dict(norm_state, strict=False)
        model.eval()

        loader = build_dataloader(cfg_dict, split="val", batch_size=1,
                                  num_workers=0, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(batch["img"])

        pred_masks = out["pred_masks"][0].cpu().numpy()
        gt_masks = batch.get("gt_masks")
        gt_dict = {}
        if gt_masks is not None:
            gt_np = gt_masks[0].cpu().numpy()
            gt_dict = {i: gt_np[:, i] for i in range(min(gt_np.shape[1], 3))}

        suite = EvaluationSuite(num_classes=3)
        metrics = suite.evaluate_sequence_masks(pred_masks, gt_dict)
        self.assertIn("overall_mIoU", metrics)
        self.assertIn("total_swap_events", metrics)


if __name__ == "__main__":
    unittest.main()
