"""
Deformable SAVi (Slot Attention for Video with 2D Deformable Attention).

Replaces dense global key-value Slot Attention in SAVi with DeformableSlotAttention,
enabling 2D local sampling, spatial reference point prediction, and fast slot updates.
"""

import torch
import torch.nn as nn
from src.models.savi import SAVi
from src.models.deformable_slot_attention import DeformableSlotAttention


class DeformableSAVi(SAVi):
    """
    Deformable SAVi Model.

    Inherits encoder, decoder, predictor, and loss computation from SAVi, but swaps
    out `model.slot_attention` with `DeformableSlotAttention`.
    """

    def __init__(
        self,
        resolution=(64, 64),
        clip_len=6,
        num_slots=4,
        slot_dim=64,
        num_iterations=3,
        n_heads=4,
        n_points=4,
        in_channels=3,
        **kwargs,
    ):
        super().__init__(
            resolution=resolution,
            clip_len=clip_len,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            in_channels=in_channels,
            **kwargs,
        )

        # Replace standard dense Slot Attention with DeformableSlotAttention wrapper
        deformable_attn_module = DeformableSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            n_heads=n_heads,
            n_points=n_points,
        )

        class _DeformableSlotAttentionWrapper(nn.Module):
            def __init__(self, core_module):
                super().__init__()
                self.core = core_module

            def forward(self, inputs, slots):
                updated_slots, _ = self.core(inputs, slots)
                return updated_slots

        self.model.slot_attention = _DeformableSlotAttentionWrapper(deformable_attn_module)

