"""
Deformable Slot Attention Module for PyTorch.

Combines 2D Deformable Attention (Zhu et al., ICLR 2021) with Slot Attention
iterative competitive refinement (Locatello et al., NeurIPS 2020) and GRU slot state updates.

Provides:
  - DeformableSlotAttention: Iterative slot attention with learned 2D reference points,
    deformable sampling offsets, and GRU slot state recurrence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.deformable_detr import MultiScaleDeformableAttention


class DeformableSlotAttention(nn.Module):
    """
    Deformable Slot Attention Module.

    Iteratively updates slot representations by sampling local 2D deformable locations
    on spatial feature maps rather than performing dense global cross-attention.

    Args:
        num_slots (int): Number of slots K (default: 4).
        slot_dim (int): Slot feature dimension D (default: 64).
        num_iterations (int): Number of Slot Attention iterations per frame (default: 3).
        n_heads (int): Number of deformable attention heads (default: 4).
        n_points (int): Number of sampling points per head (default: 4).
        n_levels (int): Number of feature levels (default: 1).
        eps (float): Epsilon for numerical stability (default: 1e-8).
    """

    def __init__(
        self,
        num_slots=4,
        slot_dim=64,
        num_iterations=3,
        n_heads=4,
        n_points=4,
        n_levels=1,
        eps=1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.eps = eps

        # LayerNorms
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        # Learned Reference Point Predictor (maps slot vector -> 2D (x, y) center in [0, 1])
        self.reference_point_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, 2),
            nn.Sigmoid(),
        )

        # Deformable Attention Module
        self.deformable_attn = MultiScaleDeformableAttention(
            d_model=slot_dim,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )

        # Recurrent GRU State Update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP Residual Update
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.reference_point_head[-2].weight, 0.0)
        nn.init.constant_(self.reference_point_head[-2].bias, 0.0)

    def forward(self, inputs, slots):
        """
        Args:
            inputs (Tensor): Image feature map [B, C, H, W] or flattened features [B, H*W, C].
            slots (Tensor): Initial slot representations [B, K, slot_dim].

        Returns:
            updated_slots (Tensor): Updated slot representations [B, K, slot_dim].
            ref_points (Tensor): Predicted 2D reference coordinates [B, K, 2] in [0, 1].
        """
        # Normalize and reshape inputs to flattened format
        if inputs.ndim == 4:
            B, C, H, W = inputs.shape
            spatial_shapes = torch.tensor([[H, W]], device=inputs.device, dtype=torch.long)
            inputs_flat = inputs.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        elif inputs.ndim == 3:
            B, HW, C = inputs.shape
            H = W = int(math.sqrt(HW))
            spatial_shapes = torch.tensor([[H, W]], device=inputs.device, dtype=torch.long)
            inputs_flat = inputs
        else:
            raise ValueError(f"Expected 3D or 4D inputs, got shape {inputs.shape}")

        level_start_index = torch.tensor([0], device=inputs.device, dtype=torch.long)
        inputs_flat = self.norm_inputs(inputs_flat)

        B, K, D = slots.shape

        for _ in range(self.num_iterations):
            slots_norm = self.norm_slots(slots)  # [B, K, D]

            # 1. Predict 2D Reference Coordinates (x, y) in [0, 1] per slot
            ref_points = self.reference_point_head(slots_norm)  # [B, K, 2]
            ref_points_expanded = ref_points.unsqueeze(2)       # [B, K, 1, 2]

            # 2. Deformable Attention feature gathering over feature map
            updates = self.deformable_attn(
                query=slots_norm,
                reference_points=ref_points_expanded,
                input_flatten=inputs_flat,
                input_spatial_shapes=spatial_shapes,
                input_level_start_index=level_start_index,
            )  # [B, K, D]

            # 3. GRU State Recurrence Update
            slots_flat = slots.reshape(B * K, D)
            updates_flat = updates.reshape(B * K, D)
            slots = self.gru(updates_flat, slots_flat).reshape(B, K, D)

            # 4. Residual MLP Update
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, ref_points
