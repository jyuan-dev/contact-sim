"""
Unit tests for PIDM (Predictive Inverse Dynamics Model) modules:
- GoalConditionedSlotRollouter with 'goal_film', 'goal_cross_attn', and 'goal_sum'
- PIDMModel end-to-end forward pass and loss computation
- StandardizedSlotFormerWrapper with pidm_slotformer registry
- plan_action closed-loop deployment interface
"""

import unittest
import torch
import torch.nn as nn

from src.models.pidm import (
    GoalConditionedSlotRollouter,
    PIDMModel,
)
from src.models.factory import build_model, list_models


class DummyStage1Model(nn.Module):
    """Mock Stage 1 Slot Extractor for testing."""

    def __init__(self, num_slots: int = 4, slot_size: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.dummy_linear = nn.Linear(3, slot_size)

    def inner_savi(self):
        return self

    def encode(self, video: torch.Tensor, prev_slots=None):
        B, T, C, H, W = video.shape
        slots = torch.randn(B, T, self.num_slots, self.slot_size, device=video.device)
        return slots, None

    def decode(self, slots_flat: torch.Tensor):
        N = slots_flat.shape[0]
        recon = torch.zeros(N, 3, 64, 64, device=slots_flat.device)
        masks = torch.zeros(N, self.num_slots, 1, 64, 64, device=slots_flat.device)
        return recon, None, masks, None


class TestPIDMModules(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 6
        self.history_len = 2
        self.rollout_len = 4
        self.num_slots = 4
        self.slot_size = 64
        self.d_model = 64

    def test_goal_conditioned_rollouter_film(self):
        """Test GoalConditionedSlotRollouter with goal_film condition mode."""
        rollouter = GoalConditionedSlotRollouter(
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            history_len=self.history_len,
            d_model=self.d_model,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            condition_mode="goal_film",
            goal_slot_idx=2,
        )

        hist_slots = torch.randn(self.B, self.history_len, self.num_slots, self.slot_size)
        goal_slots = torch.randn(self.B, self.num_slots, self.slot_size)

        out = rollouter(hist_slots, pred_len=self.rollout_len, goal_slots=goal_slots)
        self.assertEqual(
            out.shape,
            (self.B, self.rollout_len, self.num_slots, self.slot_size),
            f"Expected shape ({self.B}, {self.rollout_len}, {self.num_slots}, {self.slot_size}), got {out.shape}",
        )

    def test_goal_conditioned_rollouter_cross_attn(self):
        """Test GoalConditionedSlotRollouter with goal_cross_attn mode."""
        rollouter = GoalConditionedSlotRollouter(
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            history_len=self.history_len,
            d_model=self.d_model,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            condition_mode="goal_cross_attn",
        )

        hist_slots = torch.randn(self.B, self.history_len, self.num_slots, self.slot_size)
        goal_slots = torch.randn(self.B, self.num_slots, self.slot_size)

        out = rollouter(hist_slots, pred_len=self.rollout_len, goal_slots=goal_slots)
        self.assertEqual(
            out.shape,
            (self.B, self.rollout_len, self.num_slots, self.slot_size),
        )

    def test_pidm_model_forward_and_loss(self):
        """Test PIDMModel forward and loss computation."""
        stage1 = DummyStage1Model(num_slots=self.num_slots, slot_size=self.slot_size)
        pidm = PIDMModel(
            stage1_model=stage1,
            history_len=self.history_len,
            rollout_len=self.rollout_len,
            d_model=self.d_model,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            condition_mode="goal_film",
            goal_slot_idx=2,
            raw_action_dim=2,
            action_loss_weight=1.0,
            slot_loss_weight=1.0,
            rollout_consistent=True,
        )

        batch = {
            "img": torch.randn(self.B, self.T, 3, 64, 64),
            "action": torch.randn(self.B, self.T - 1, 2),
        }

        out = pidm(batch)
        self.assertIn("pred_slots", out)
        self.assertIn("action_nll_dict", out)
        self.assertEqual(out["pred_slots"].shape, (self.B, self.rollout_len, self.num_slots, self.slot_size))

        loss, loss_dict = pidm.calc_train_loss(out, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("slot_mse", loss_dict)
        self.assertIn("action_nll", loss_dict)
        self.assertIn("gt_idm_loss", loss_dict)
        self.assertIn("pred_idm_loss", loss_dict)

    def test_pidm_plan_action_interface(self):
        """Test PIDM closed-loop action planning interface."""
        stage1 = DummyStage1Model(num_slots=self.num_slots, slot_size=self.slot_size)
        pidm = PIDMModel(
            stage1_model=stage1,
            history_len=self.history_len,
            rollout_len=self.rollout_len,
            d_model=self.d_model,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            condition_mode="goal_film",
            goal_slot_idx=2,
            raw_action_dim=2,
        )

        hist_slots = torch.randn(self.B, self.history_len, self.num_slots, self.slot_size)
        goal_slots = torch.randn(self.B, self.num_slots, self.slot_size)

        action = pidm.plan_action(hist_slots, goal_video_or_slots=goal_slots)
        self.assertEqual(action.shape, (self.B, 2))

    def test_factory_build_pidm_model(self):
        """Test build_model factory dispatch with 'pidm_slotformer'."""
        cfg = {
            "model": {
                "name": "pidm_slotformer",
                "type": "pidm_slotformer",
                "stage1_ckpt_path": "",
                "history_len": 2,
                "rollout_len": 4,
                "d_model": 64,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 128,
                "condition_mode": "goal_film",
            }
        }
        wrapper = build_model(cfg)
        self.assertIsNotNone(wrapper)
        self.assertIn("pidm", list_models())


if __name__ == "__main__":
    unittest.main()
