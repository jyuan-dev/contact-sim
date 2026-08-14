"""
Deformable SAVi — Slot Attention for Video with 2D Deformable Attention.

Replaces dense global key-value Slot Attention with DeformableSlotAttention,
enabling 2D local sampling, spatial reference point prediction, and fast slot
updates.

References:
  - Deformable DETR (Zhu et al., ICLR 2021)
  - Slot Attention (Locatello et al., NeurIPS 2020)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.savi import SAVi


# ═══════════════════════════════════════════════════════════════════════════════
# Deformable Attention building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention (pure PyTorch).

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

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index):
        B, Len_q, C = query.shape
        _, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(B, Len_in, self.n_heads, self.d_head)

        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points)

        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], dim=-1
        ).to(query.device, dtype=query.dtype)

        sampling_locations = (
            reference_points.unsqueeze(2).unsqueeze(4)
            + sampling_offsets / offset_normalizer.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        output = torch.zeros(B, Len_q, self.n_heads, self.d_head,
                             device=query.device, dtype=query.dtype)

        for l_idx in range(self.n_levels):
            H, W = int(input_spatial_shapes[l_idx, 0]), int(input_spatial_shapes[l_idx, 1])
            start_idx = int(input_level_start_index[l_idx])
            end_idx = start_idx + H * W

            value_l = value[:, start_idx:end_idx].view(B, H, W, self.n_heads, self.d_head)
            value_l = value_l.permute(0, 3, 4, 1, 2).reshape(B * self.n_heads, self.d_head, H, W)

            sampling_locs_l = sampling_locations[:, :, :, l_idx].permute(0, 2, 1, 3, 4).reshape(
                B * self.n_heads, Len_q, self.n_points, 2)
            grid = 2.0 * sampling_locs_l - 1.0

            sampled_feat = F.grid_sample(
                value_l, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled_feat = sampled_feat.view(B, self.n_heads, self.d_head, Len_q, self.n_points)
            sampled_feat = sampled_feat.permute(0, 3, 1, 4, 2)

            attn_l = attention_weights[:, :, :, l_idx].unsqueeze(-1)
            output += (sampled_feat * attn_l).sum(dim=3)

        output = output.reshape(B, Len_q, C)
        return self.output_proj(output)


class DeformableSlotAttention(nn.Module):
    """Iterative slot attention with learned 2D reference points and GRU recurrence.

    Ref: Locatello et al. (NeurIPS 2020) + Zhu et al. (ICLR 2021).
    """

    def __init__(self, num_slots=4, slot_dim=64, num_iterations=3,
                 n_heads=4, n_points=4, n_levels=1, eps=1e-8, use_residual_bn: bool = False):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        self.eps = eps
        self.use_residual_bn = use_residual_bn

        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.reference_point_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim), nn.ReLU(inplace=True),
            nn.Linear(slot_dim, 2), nn.Sigmoid(),
        )

        self.deformable_attn = MultiScaleDeformableAttention(
            d_model=slot_dim, n_levels=n_levels, n_heads=n_heads, n_points=n_points)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 2, slot_dim))
        self.norm_mlp = nn.LayerNorm(slot_dim)
        if self.use_residual_bn:
            self.residual_bn = nn.BatchNorm1d(slot_dim)
        else:
            self.residual_bn = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.reference_point_head[-2].weight, 0.0)
        nn.init.constant_(self.reference_point_head[-2].bias, 0.0)

    def forward(self, inputs, slots):
        if inputs.ndim == 4:
            B, C, H, W = inputs.shape
            spatial_shapes = torch.tensor([[H, W]], device=inputs.device, dtype=torch.long)
            inputs_flat = inputs.flatten(2).permute(0, 2, 1)
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
            slots_norm = self.norm_slots(slots)
            ref_points = self.reference_point_head(slots_norm)
            ref_points_expanded = ref_points.unsqueeze(2)

            updates = self.deformable_attn(
                query=slots_norm, reference_points=ref_points_expanded,
                input_flatten=inputs_flat, input_spatial_shapes=spatial_shapes,
                input_level_start_index=level_start_index)

            slots = self.gru(updates.reshape(B * K, D), slots.reshape(B * K, D)).reshape(B, K, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
            if self.residual_bn is not None:
                slots = self.residual_bn(slots.view(B * K, D)).view(B, K, D)

        return slots, ref_points


# ═══════════════════════════════════════════════════════════════════════════════
# Deformable SAVi model
# ═══════════════════════════════════════════════════════════════════════════════

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
        use_encoder_bn: bool = False,
        use_residual_bn: bool = False,
        **kwargs,
    ):
        super().__init__(
            resolution=resolution,
            clip_len=clip_len,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            in_channels=in_channels,
            use_encoder_bn=use_encoder_bn,
            use_residual_bn=use_residual_bn,
            **kwargs,
        )

        # Replace standard dense Slot Attention with DeformableSlotAttention wrapper
        deformable_attn_module = DeformableSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            n_heads=n_heads,
            n_points=n_points,
            use_residual_bn=self.use_residual_bn,
        )

        class _DeformableSlotAttentionWrapper(nn.Module):
            def __init__(self, core_module):
                super().__init__()
                self.core = core_module

            def forward(self, inputs, slots):
                updated_slots, _ = self.core(inputs, slots)
                return updated_slots

        self.model.slot_attention = _DeformableSlotAttentionWrapper(deformable_attn_module)

