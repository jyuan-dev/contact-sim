import sys
import os
import unittest
import torch

# Add workspace root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.slot_pidm import SlotPIDMAgent, IterativeSlotInteractionBlock, InverseDynamicsBlock

class TestSlotPIDM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.k_slots = 4
        self.d_model = 128
        self.action_dim = 2

    def test_forward_dynamics_block(self):
        block = IterativeSlotInteractionBlock(
            d_model=self.d_model,
            action_dim=self.action_dim,
            num_heads=4,
            num_iterations=2
        )
        slots_t = torch.randn(self.batch_size, self.k_slots, self.d_model)
        action_t = torch.randn(self.batch_size, self.action_dim)
        
        pred_next = block(slots_t, action_t)
        self.assertEqual(pred_next.shape, (self.batch_size, self.k_slots, self.d_model))
        self.assertFalse(torch.isnan(pred_next).any())

    def test_inverse_dynamics_block(self):
        block = InverseDynamicsBlock(
            d_model=self.d_model,
            action_dim=self.action_dim,
            num_heads=4
        )
        slots_t = torch.randn(self.batch_size, self.k_slots, self.d_model)
        slots_next = torch.randn(self.batch_size, self.k_slots, self.d_model)

        pred_action = block(slots_t, slots_next)
        self.assertEqual(pred_action.shape, (self.batch_size, self.action_dim))
        self.assertFalse(torch.isnan(pred_action).any())

    def test_full_agent(self):
        agent = SlotPIDMAgent(
            d_model=self.d_model,
            action_dim=self.action_dim,
            k_slots=self.k_slots,
            num_heads=4,
            num_iterations=2
        )
        slots_t = torch.randn(self.batch_size, self.k_slots, self.d_model)
        slots_next = torch.randn(self.batch_size, self.k_slots, self.d_model)
        action_gt = torch.randn(self.batch_size, self.action_dim)

        out = agent(slots_t, slots_next, action_gt)
        self.assertIn('pred_action', out)
        self.assertIn('pred_slots_next', out)
        self.assertIn('loss_sigreg', out)
        self.assertIn('loss_total', out)

        self.assertEqual(out['pred_action'].shape, (self.batch_size, self.action_dim))
        self.assertEqual(out['pred_slots_next'].shape, (self.batch_size, self.k_slots, self.d_model))

        # Test backward pass
        out['loss_total'].backward()
        print("SlotPIDMAgent backward pass successful! Loss total:", out['loss_total'].item())

if __name__ == '__main__':
    unittest.main()
