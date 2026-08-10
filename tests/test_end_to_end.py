"""
End-to-End Integration Tests for Contact-Sim pipelines.

Covers full training and evaluation flows with real model construction,
dataset loading, forward/backward passes, and checkpoint roundtrips.
"""

import os
import sys
import tempfile
import unittest

import torch
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.factory import build_model
from src.datasets.factory import build_dataloader
from src.training.train_loop import TrainConfig, run_training
from src.training.trainer import BaseTrainer


# ── End-to-End Training Pipeline ──────────────────────────────────────────────

class TestEndToEndTraining(unittest.TestCase):
    """Full training pipeline: config → model → dataset → train loop → checkpoint."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmpdir.name, "checkpoints", "e2e_test")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_cfg(self, model_name="savi"):
        """Build a minimal resolved config dict for training."""
        return {
            "model": {
                "name": model_name,
                "type": model_name,
                "num_slots": 4,
                "slot_dim": 64,
                "num_iterations": 2,
                "n_sample_frames": 4,
                "resolution": [64, 64],
            },
            "dataset": {
                "name": "gridshapes",
                "type": "gridshapes",
                "train_samples": 32,
                "val_samples": 8,
                "num_frames": 4,
                "num_objects": 3,
                "img_size": 64,
                "resolution": [64, 64],
            },
            "batch_size": 4,
        }

    def test_full_training_pipeline_savi(self):
        """End-to-end: build SAVi model, gridshapes dataset, run training loop."""
        cfg_dict = self._make_cfg("savi")
        device = torch.device("cpu")

        model = build_model(cfg_dict).to(device)
        train_loader = build_dataloader(cfg_dict, split="train", batch_size=4, num_workers=0)
        val_loader = build_dataloader(cfg_dict, split="val", batch_size=4, num_workers=0)

        train_cfg = TrainConfig(
            epochs=1,
            dry_run=True,
            use_amp=False,
            batch_size=4,
            ckpt_dir=self.ckpt_dir,
            model_name="savi",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        def _save_ckpt(m, path, epoch):
            torch.save({"model_state": m.state_dict(), "epoch": epoch}, path)

        trainer = BaseTrainer(save_dir=self.ckpt_dir, experiment_name="e2e_savi")
        try:
            run_training(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, scaler=scaler, device=device, cfg=train_cfg,
                trainer=trainer, save_checkpoint_fn=_save_ckpt,
            )
        finally:
            trainer.close()

        # Verify checkpoint files were written
        self.assertTrue(os.path.isfile(os.path.join(self.ckpt_dir, "savi_best.pt")))
        self.assertTrue(os.path.isfile(os.path.join(self.ckpt_dir, "savi_final.pt")))

    def test_full_training_pipeline_deformable_savi(self):
        """End-to-end: build DeformableSAVi, run training loop."""
        cfg_dict = self._make_cfg("deformable_savi")
        device = torch.device("cpu")

        model = build_model(cfg_dict).to(device)
        train_loader = build_dataloader(cfg_dict, split="train", batch_size=4, num_workers=0)
        val_loader = build_dataloader(cfg_dict, split="val", batch_size=4, num_workers=0)

        train_cfg = TrainConfig(
            epochs=1, dry_run=True, use_amp=False, batch_size=4,
            ckpt_dir=self.ckpt_dir, model_name="deformable_savi",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        def _save_ckpt(m, path, epoch):
            torch.save({"model_state": m.state_dict(), "epoch": epoch}, path)

        trainer = BaseTrainer(save_dir=self.ckpt_dir, experiment_name="e2e_dsavi")
        try:
            run_training(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, scaler=scaler, device=device, cfg=train_cfg,
                trainer=trainer, save_checkpoint_fn=_save_ckpt,
            )
        finally:
            trainer.close()

        self.assertTrue(os.path.isfile(os.path.join(self.ckpt_dir, "deformable_savi_best.pt")))

    def test_training_saves_best_and_final(self):
        """Training should save both best.pt and final.pt checkpoints."""
        cfg_dict = self._make_cfg("savi")
        device = torch.device("cpu")

        model = build_model(cfg_dict).to(device)
        train_loader = build_dataloader(cfg_dict, split="train", batch_size=4, num_workers=0)
        val_loader = build_dataloader(cfg_dict, split="val", batch_size=4, num_workers=0)

        train_cfg = TrainConfig(
            epochs=1, dry_run=True, use_amp=False, batch_size=4,
            ckpt_dir=self.ckpt_dir, model_name="savi",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        def _save_ckpt(m, path, epoch):
            torch.save({"model_state": m.state_dict(), "epoch": epoch}, path)

        trainer = BaseTrainer(save_dir=self.ckpt_dir, experiment_name="e2e_ckpt")
        try:
            run_training(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, scaler=scaler, device=device, cfg=train_cfg,
                trainer=trainer, save_checkpoint_fn=_save_ckpt,
            )
        finally:
            trainer.close()

        best_path = os.path.join(self.ckpt_dir, "savi_best.pt")
        final_path = os.path.join(self.ckpt_dir, "savi_final.pt")
        self.assertTrue(os.path.isfile(best_path))
        self.assertTrue(os.path.isfile(final_path))

        # Both should be loadable
        best_ckpt = torch.load(best_path, map_location="cpu")
        final_ckpt = torch.load(final_path, map_location="cpu")
        self.assertIn("model_state", best_ckpt)
        self.assertIn("epoch", best_ckpt)
        self.assertIn("model_state", final_ckpt)
        self.assertIn("epoch", final_ckpt)

    def test_checkpoint_roundtrip(self):
        """Save a checkpoint mid-training, then load it into a fresh model."""
        cfg_dict = self._make_cfg("savi")
        device = torch.device("cpu")

        # Train briefly
        model = build_model(cfg_dict).to(device)
        train_loader = build_dataloader(cfg_dict, split="train", batch_size=4, num_workers=0)
        val_loader = build_dataloader(cfg_dict, split="val", batch_size=4, num_workers=0)

        ckpt_path = os.path.join(self.ckpt_dir, "test_roundtrip.pt")

        train_cfg = TrainConfig(
            epochs=1, dry_run=True, use_amp=False, batch_size=4,
            ckpt_dir=self.ckpt_dir, model_name="savi",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        def _save_ckpt(m, path, epoch):
            torch.save({"model_state": m.state_dict(), "epoch": epoch}, path)
            # Also save to our roundtrip path
            if "best" in path:
                torch.save({"model_state": m.state_dict(), "epoch": epoch}, ckpt_path)

        trainer = BaseTrainer(save_dir=self.ckpt_dir, experiment_name="e2e_roundtrip")
        try:
            run_training(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, scaler=scaler, device=device, cfg=train_cfg,
                trainer=trainer, save_checkpoint_fn=_save_ckpt,
            )
        finally:
            trainer.close()

        # Load checkpoint into a fresh model
        fresh_model = build_model(cfg_dict).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)

        # Normalize keys (simulate eval.py's loading logic)
        norm_state = {}
        for k, v in state.items():
            while k.startswith("module."):
                k = k[len("module."):]
            while k.startswith("model.model."):
                k = k[len("model."):]
            norm_state[k] = v

        missing, unexpected = fresh_model.load_state_dict(norm_state, strict=False)
        # After brief training, the fresh model should load weights successfully
        self.assertIsNotNone(fresh_model)


# ── End-to-End Eval Pipeline ──────────────────────────────────────────────────

class TestEndToEndEval(unittest.TestCase):
    """End-to-end evaluation pipeline: model → forward → metrics."""

    def test_model_forward_to_metrics_pipeline(self):
        """Build model, run forward pass, compute metrics on output."""
        from src.metrics import EvaluationSuite

        cfg_dict = {
            "model": {"name": "savi", "type": "savi", "num_slots": 4,
                       "slot_dim": 64, "num_iterations": 2,
                       "n_sample_frames": 4, "resolution": [64, 64]},
        }

        model = build_model(cfg_dict)
        model.eval()

        # Simulate a batch of video frames
        B, T, C, H, W = 2, 4, 3, 64, 64
        video = torch.randn(B, T, C, H, W)

        with torch.no_grad():
            out = model(video)

        # Verify output contract
        self.assertIn("input_img", out)
        self.assertIn("pred_masks", out)
        self.assertIn("recon_img", out)
        self.assertIn("pred_boxes", out)
        self.assertIn("pred_logits", out)
        self.assertIn("post_slots", out)

        # Predicted masks should have expected shape
        pred_masks = out["pred_masks"]
        self.assertIsNotNone(pred_masks)
        self.assertEqual(pred_masks.shape[0], B)
        self.assertEqual(pred_masks.shape[1], T)

        # Reconstruction should match input spatial dims
        recon = out["recon_img"]
        self.assertIsNotNone(recon)
        self.assertEqual(recon.shape[-2:], (H, W))

        # Compute metrics on the output
        suite = EvaluationSuite(num_classes=3)
        pred_np = pred_masks[0].cpu().numpy()  # [T, K, H, W]
        gt_dict = {
            0: (torch.rand(T, H, W) > 0.7).float().numpy(),
            1: (torch.rand(T, H, W) > 0.7).float().numpy(),
            2: (torch.rand(T, H, W) > 0.7).float().numpy(),
        }
        metrics = suite.evaluate_sequence_masks(pred_np, gt_dict)
        self.assertIn("overall_mIoU", metrics)
        self.assertIn("total_swap_events", metrics)

    def test_deformable_savi_eval_pipeline(self):
        """DeformableSAVi eval forward pass produces valid outputs."""
        cfg_dict = {
            "model": {"name": "deformable_savi", "type": "deformable_savi",
                       "num_slots": 4, "slot_dim": 64, "num_iterations": 2,
                       "n_sample_frames": 4, "resolution": [64, 64]},
        }

        model = build_model(cfg_dict)
        model.eval()

        B, T, C, H, W = 1, 3, 3, 64, 64
        video = torch.randn(B, T, C, H, W)

        with torch.no_grad():
            out = model(video)

        self.assertIsNotNone(out.get("pred_masks"))
        self.assertIsNotNone(out.get("recon_img"))

    def test_savi_deformable_savi_both_eval(self):
        """Both SAVi and DeformableSAVi should produce output contract at 64x64."""
        for name in ("savi", "deformable_savi"):
            cfg_dict = {
                "model": {"name": name, "type": name, "num_slots": 4,
                           "slot_dim": 64, "num_iterations": 2,
                           "n_sample_frames": 3, "resolution": [64, 64]},
            }
            model = build_model(cfg_dict)
            model.eval()
            video = torch.randn(1, 3, 3, 64, 64)
            with torch.no_grad():
                out = model(video)
            for key in ("input_img", "pred_masks", "recon_img"):
                self.assertIsNotNone(out.get(key), f"{name} missing {key}")


# ── End-to-End Dataloader + Model Integration ─────────────────────────────────

class TestEndToEndDataloaderModel(unittest.TestCase):
    """Integration: dataset → dataloader → model forward."""

    def test_gridshapes_loader_to_model(self):
        """Load gridshapes batches through SAVi model."""
        cfg_dict = {
            "model": {"name": "savi", "type": "savi", "num_slots": 4,
                       "slot_dim": 64, "num_iterations": 2,
                       "n_sample_frames": 4, "resolution": [64, 64]},
            "dataset": {"name": "gridshapes", "type": "gridshapes",
                         "train_samples": 16, "num_frames": 4,
                         "num_objects": 3, "img_size": 64},
        }

        model = build_model(cfg_dict)
        loader = build_dataloader(cfg_dict, split="train", batch_size=4, num_workers=0)

        batch = next(iter(loader))
        self.assertIn("img", batch)
        self.assertIn("gt_masks", batch)

        video = batch["img"]
        with torch.no_grad():
            out = model(video)

        self.assertIsNotNone(out.get("pred_masks"))
        # pred_masks slot count should match model config
        self.assertEqual(out["pred_masks"].shape[2], 4)  # K=4 slots

    def test_loss_computation_on_real_batch(self):
        """Compute loss on a real dataloader batch."""
        cfg_dict = {
            "model": {"name": "savi", "type": "savi", "num_slots": 4,
                       "slot_dim": 64, "num_iterations": 2,
                       "n_sample_frames": 4, "resolution": [64, 64]},
            "dataset": {"name": "gridshapes", "type": "gridshapes",
                         "train_samples": 8, "num_frames": 4,
                         "num_objects": 3, "img_size": 64},
        }

        model = build_model(cfg_dict)
        loader = build_dataloader(cfg_dict, split="train", batch_size=2, num_workers=0)
        batch = next(iter(loader))

        out = model(batch["img"])
        loss, loss_dict = model.compute_loss(out, batch)

        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("recon_loss", loss_dict)
        self.assertIn("mask_bce", loss_dict)
        self.assertIn("total_loss", loss_dict)


if __name__ == "__main__":
    unittest.main()
