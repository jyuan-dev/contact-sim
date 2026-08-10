"""
Deformable Slot Attention Module for PyTorch.

Combines 2D Deformable Attention (Zhu et al., ICLR 2021) with Slot Attention
iterative competitive refinement (Locatello et al., NeurIPS 2020) and GRU slot state updates.

Provides:
  - MultiScaleDeformableAttention: Pure PyTorch 2D Deformable Attention module.
  - DeformableSlotAttention: Iterative slot attention with learned 2D reference points,
    deformable sampling offsets, and GRU slot state recurrence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module (Pure PyTorch implementation).
    Ref: Zhu et al., Deformable DETR (ICLR 2021).
    """

    def __init__(self, d_model=64, n_levels=1, n_heads=4, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Initialize sampling offsets in 4 cardinal directions per head
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)
        grid_init = (grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2
        ).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
    ):
        """
        Args:
            query: [B, Len_q, C]
            reference_points: [B, Len_q, n_levels, 2] in [0, 1]
            input_flatten: [B, Len_in, C]
            input_spatial_shapes: [n_levels, 2] (H, W per level)
            input_level_start_index: [n_levels]
        """
        B, Len_q, C = query.shape
        _, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(B, Len_in, self.n_heads, self.d_head)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], dim=-1
        ).to(query.device, dtype=query.dtype)

        sampling_locations = (
            reference_points.unsqueeze(2).unsqueeze(4)
            + sampling_offsets / offset_normalizer.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        # Bilinear sampling per level
        output = torch.zeros(B, Len_q, self.n_heads, self.d_head, device=query.device, dtype=query.dtype)

        for l_idx in range(self.n_levels):
            H, W = int(input_spatial_shapes[l_idx, 0]), int(input_spatial_shapes[l_idx, 1])
            start_idx = int(input_level_start_index[l_idx])
            end_idx = start_idx + H * W

            value_l = value[:, start_idx:end_idx].view(B, H, W, self.n_heads, self.d_head)
            value_l = value_l.permute(0, 3, 4, 1, 2).reshape(B * self.n_heads, self.d_head, H, W)

            # Sampling locations for level l: [B, Len_q, n_heads, n_points, 2] -> [B*n_heads, Len_q, n_points, 2]
            sampling_locs_l = sampling_locations[:, :, :, l_idx].permute(0, 2, 1, 3, 4).reshape(
                B * self.n_heads, Len_q, self.n_points, 2
            )
            # Map [0, 1] coordinates -> [-1, 1] for grid_sample
            grid = 2.0 * sampling_locs_l - 1.0

            # Sample features: [B*n_heads, d_head, Len_q, n_points]
            sampled_feat = F.grid_sample(
                value_l, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )
            sampled_feat = sampled_feat.view(B, self.n_heads, self.d_head, Len_q, self.n_points)
            sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)  # [B, Len_q, n_heads, n_points, d_head]

            attn_l = attention_weights[:, :, :, l_idx].unsqueeze(-1)  # [B, Len_q, n_heads, n_points, 1]
            output += (sampled_feat * attn_l).sum(dim=3)  # Sum over n_points

        output = output.reshape(B, Len_q, C)
        return self.output_proj(output)


class DeformableSlotAttention(nn.Module):
    """
    Deformable Slot Attention Module.

    Iteratively updates slot representations by sampling local 2D deformable locations
    on spatial feature maps rather than performing dense global cross-attention.
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
