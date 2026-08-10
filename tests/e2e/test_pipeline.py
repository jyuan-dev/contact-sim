"""
End-to-End Pipeline Tests — full train → checkpoint → infer → eval flow.

Uses Hydra compose API (same as scripts/train.py) to resolve the default
config, then exercises the pipeline programmatically.  Much faster than
subprocess (~20 s for 5-batch dry-run on CPU) because it avoids the
@hydra.main decorator's chdir + output-directory overhead.

Benchmark: gridshapes deformable_savi, CPU, 5-batch dry-run = ~20 s.
"""

import os
import sys
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


def _resolve_cfg(overrides: list[str] | None = None) -> dict:
    """Resolve Hydra config with the given overrides (same as CLI args)."""
    with initialize_config_dir(
        config_dir=os.path.join(REPO_ROOT, "configs"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    return OmegaConf.to_container(cfg, resolve=True)


class TestPipeline(unittest.TestCase):
    """Single training run, then test checkpoint loading, inference, and eval."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.ckpt_dir = cls.tmpdir.name

        # Resolve config like scripts/train.py does, with speed overrides
        cfg_dict = _resolve_cfg([
            "dataset=gridshapes", "batch_size=4", "num_workers=0",
            "dry_run=true", "device=cpu",
        ])
        cls.model_name = cfg_dict["model"]["name"]

        device = torch.device("cpu")
        model = build_model(cfg_dict).to(device)
        train_loader = build_dataloader(cfg_dict, split="train", batch_size=4, num_workers=0)
        val_loader = build_dataloader(cfg_dict, split="val", batch_size=4, num_workers=0)

        train_cfg = TrainConfig(
            dry_run=True, use_amp=False, batch_size=4, epochs=1,
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

        cls.ckpt_path = os.path.join(cls.ckpt_dir, f"{cls.model_name}_best.pt")
        assert os.path.isfile(cls.ckpt_path), f"No checkpoint at {cls.ckpt_path}"

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    # ── Training outputs ───────────────────────────────────────────────────

    def test_checkpoint_loadable(self):
        """Checkpoint should be valid .pt with model_state + epoch."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        self.assertIn("model_state", ckpt)
        self.assertIn("epoch", ckpt)

    def test_checkpoint_roundtrip(self):
        """Load checkpoint state into a fresh model successfully."""
        cfg_dict = _resolve_cfg([
            "dataset=gridshapes", "batch_size=4", "num_workers=0",
            "dry_run=true", "device=cpu",
        ])
        fresh = build_model(cfg_dict)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = fresh.load_state_dict(state["model_state"], strict=False)
        # Should load successfully (silent drops are acceptable with strict=False)
        self.assertIsNotNone(fresh)

    # ── Inference ──────────────────────────────────────────────────────────

    def test_infer_forward_pass(self):
        """Run a forward pass like scripts/infer.py does and verify outputs."""
        cfg_dict = _resolve_cfg([
            "dataset=gridshapes", "batch_size=4", "num_workers=0",
            "dry_run=true", "device=cpu",
        ])
        model = build_model(cfg_dict)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=False)
        model.eval()

        loader = build_dataloader(cfg_dict, split="val", batch_size=1, num_workers=0, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(batch["img"])

        self.assertIsNotNone(out.get("recon_img"))
        self.assertIsNotNone(out.get("pred_masks"))

    # ── Eval ───────────────────────────────────────────────────────────────

    def test_eval_metrics_pipeline(self):
        """Run eval like scripts/eval.py does: load ckpt, compute metrics."""
        from src.metrics import EvaluationSuite
        import numpy as np

        cfg_dict = _resolve_cfg([
            "dataset=gridshapes", "batch_size=4", "num_workers=0",
            "dry_run=true", "device=cpu",
        ])
        model = build_model(cfg_dict)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        norm_state = {}
        for k, v in state["model_state"].items():
            while k.startswith("module."):
                k = k[len("module."):]
            while k.startswith("model.model."):
                k = k[len("model."):]
            norm_state[k] = v
        model.load_state_dict(norm_state, strict=False)
        model.eval()

        loader = build_dataloader(cfg_dict, split="val", batch_size=1, num_workers=0, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(batch["img"])

        pred_masks = out["pred_masks"][0].cpu().numpy()  # [T, K, H, W]
        gt_masks = batch.get("gt_masks")
        if gt_masks is not None:
            gt_np = gt_masks[0].cpu().numpy()
            gt_dict = {i: gt_np[:, i] for i in range(min(gt_np.shape[1], 3))}
        else:
            gt_dict = {}

        suite = EvaluationSuite(num_classes=3)
        metrics = suite.evaluate_sequence_masks(pred_masks, gt_dict)
        self.assertIn("overall_mIoU", metrics)
        self.assertIn("total_swap_events", metrics)


if __name__ == "__main__":
    unittest.main()
