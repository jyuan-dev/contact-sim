"""
Deformable DETR Model Implementation for PyTorch.

Reference Paper:
  "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
  Zhu et al., ICLR 2021 (https://arxiv.org/abs/2010.04159)

Provides:
  - MultiScaleDeformableAttention: Core 2D deformable attention with learned sampling offsets & attention weights.
  - DeformableTransformerEncoderLayer & DeformableTransformerEncoder: Multi-scale deformable encoder.
  - DeformableTransformerDecoderLayer & DeformableTransformerDecoder: Multi-scale deformable decoder with reference points.
  - DeformableDETR: End-to-end Deformable DETR network architecture supporting single-scale and multi-scale feature maps.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Deformable Attention Module ───────────────────────────────────────────
class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module.

    Calculates sampling offsets and scale-dependent attention weights for target queries
    over multi-scale feature maps using bilinear interpolation.

    Args:
        d_model (int): Hidden dimension size (default: 256).
        n_levels (int): Number of feature map scales (default: 4).
        n_heads (int): Number of attention heads (default: 8).
        n_points (int): Number of sampling points per head per level (default: 4).
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Linear projections for sampling offsets and attention weights
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling offsets: uniform angular spacing per head
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [n_heads, 2]
        grid_init = (grid_init / grid_init.abs().max(dim=-1, keepdim=True).values).view(
            self.n_heads, 1, 1, 2
        ).repeat(1, self.n_levels, self.n_points, 1)

        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        """
        Args:
            query (Tensor): [B, Len_q, d_model] - Query embeddings.
            reference_points (Tensor): [B, Len_q, n_levels, 2] or [B, Len_q, 2] - Normalized reference coordinates in [0, 1].
            input_flatten (Tensor): [B, Len_in, d_model] - Flattened multi-scale feature values.
            input_spatial_shapes (Tensor): [n_levels, 2] - Spatial shapes (H_l, W_l) of each feature level.
            input_level_start_index (Tensor): [n_levels] - Start index of each level in input_flatten.

        Returns:
            output (Tensor): [B, Len_q, d_model] - Deformable attention output.
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # Scale sampling offsets according to spatial dimensions
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], dim=-1
            ).to(query.device)  # [n_levels, 2] (W_l, H_l)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(f"Last dimension of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        # Bilinear sampling over multi-scale feature maps
        output = self._ms_deformable_sampling(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights
        )
        return self.output_proj(output)

    def _ms_deformable_sampling(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        """
        Performs multi-scale bilinear grid sampling.
        """
        N, Len_in, n_heads, head_dim = value.shape
        _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

        # Split value by feature level
        value_list = value.split([H * W for H, W in spatial_shapes], dim=1)
        sampling_grids = 2.0 * sampling_locations - 1.0  # Normalize [0, 1] -> [-1, 1] for grid_sample

        output_levels = []
        for level, (H, W) in enumerate(spatial_shapes):
            # value_l: [N, H*W, n_heads, head_dim] -> [N * n_heads, head_dim, H, W]
            value_l = value_list[level].permute(0, 2, 3, 1).reshape(N * n_heads, head_dim, H, W)

            # grid_l shape after slice [N, Len_q, n_heads, n_points, 2] (5D tensor)
            # We want [N * n_heads, Len_q, n_points, 2] -> permute to (0, 2, 1, 3, 4)
            grid_l = sampling_grids[:, :, :, level, :, :].permute(0, 2, 1, 3, 4).reshape(
                N * n_heads, Len_q, n_points, 2
            )


            # grid_sample: [N * n_heads, head_dim, Len_q, n_points]
            sampling_value_l = F.grid_sample(
                value_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False
            )
            output_levels.append(sampling_value_l)

        # Combine sampled values across levels & points with attention weights
        # output_levels: [n_levels, N * n_heads, head_dim, Len_q, n_points] -> stack level -> [N, n_heads, head_dim, Len_q, n_levels, n_points]
        output = torch.stack(output_levels, dim=-2).view(
            N, n_heads, head_dim, Len_q, n_levels, n_points
        )

        # attention_weights: [N, Len_q, n_heads, n_levels, n_points] -> permute -> [N, n_heads, 1, Len_q, n_levels, n_points]
        attn = attention_weights.permute(0, 2, 1, 3, 4).unsqueeze(2)

        # Weighted sum over levels and points: [N, n_heads, head_dim, Len_q] -> permute -> [N, Len_q, n_heads * head_dim]
        output = (output * attn).sum(dim=(-2, -1)).permute(0, 3, 1, 2).flatten(2)
        return output


# ── 2. Deformable Transformer Encoder ────────────────────────────────────────
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        # Self attention with deformable sampling
        src2 = self.self_attn(
            query=src + pos,
            reference_points=reference_points,
            input_flatten=src,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
        )
        src = self.norm1(src + self.dropout1(src2))

        # Feed forward network
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, spatial_shapes, level_start_index, pos=None):
        output = src
        # Reference points for encoder: normalized 2D grid coordinates for each spatial shape
        reference_points = self._get_reference_points(spatial_shapes, device=src.device)

        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)
        return output

    def _get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for H, W in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1) / H
            ref_x = ref_x.reshape(-1) / W
            ref = torch.stack((ref_x, ref_y), dim=-1)  # [H*W, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, dim=0)  # [Len_in, 2]
        return reference_points.unsqueeze(0).unsqueeze(2)  # [1, Len_in, 1, 2]


# ── 3. Deformable Transformer Decoder ────────────────────────────────────────
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, src, spatial_shapes, level_start_index):
        # 1. Self Attention between object queries
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Deformable Cross Attention over multi-scale encoder feature maps
        tgt2 = self.cross_attn(
            query=tgt + query_pos,
            reference_points=reference_points,
            input_flatten=src,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, query_pos, reference_points, src, spatial_shapes, level_start_index):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, query_pos, reference_points, src, spatial_shapes, level_start_index)
            intermediate.append(output)

        return torch.stack(intermediate, dim=1)  # [B, num_layers, num_queries, d_model]


# ── 4. Main Deformable DETR Architecture ──────────────────────────────────────
class DeformableDETR(nn.Module):
    """
    Deformable DETR End-to-End Object Detection Architecture.

    Args:
        num_classes (int): Number of target object classes (default: 3).
        num_queries (int): Number of object query slots (default: 10).
        d_model (int): Hidden feature channel size (default: 128).
        nhead (int): Number of attention heads (default: 4).
        num_encoder_layers (int): Encoder depth (default: 3).
        num_decoder_layers (int): Decoder depth (default: 3).
        dim_feedforward (int): FFN hidden dimension (default: 512).
        dropout (float): Dropout probability (default: 0.1).
        backbone_name (str): Backbone CNN architecture name (default: 'resnet18').
        train_backbone (bool): Whether to train backbone weights (default: True).
    """

    def __init__(
        self,
        num_classes=3,
        num_queries=10,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        backbone_name='resnet18',
        train_backbone=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        # Backbone & Multi-Scale Feature Extraction
        from src.models.detr import ResNetBackbone, PositionEmbeddingSine
        self.backbone = ResNetBackbone(name=backbone_name, train_backbone=train_backbone)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        # Feature level projection (adapts backbone out channels -> d_model)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)

        # Deformable Transformer
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=d_model, d_ffn=dim_feedforward, dropout=dropout, n_levels=1, n_heads=nhead, n_points=4
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model, d_ffn=dim_feedforward, dropout=dropout, n_levels=1, n_heads=nhead, n_points=4
        )
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Query embeddings & reference point heads
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.reference_points_head = nn.Linear(d_model, 2)

        # Prediction Heads (multi-layer for auxiliary losses)
        self.class_embed = nn.ModuleList([
            nn.Linear(d_model, num_classes + 1) for _ in range(num_decoder_layers)
        ])
        self.bbox_embed = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3) for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.reference_points_head.weight, 0.0)
        nn.init.constant_(self.reference_points_head.bias, 0.0)
        for class_embed in self.class_embed:
            nn.init.xavier_uniform_(class_embed.weight)
            nn.init.constant_(class_embed.bias, 0.0)
        for bbox_embed in self.bbox_embed:
            nn.init.xavier_uniform_(bbox_embed.layers[-1].weight)
            nn.init.constant_(bbox_embed.layers[-1].bias, 0.0)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input images [B, C, H, W]

        Returns:
            dict: {
                'pred_logits': [B, L, Q, num_classes + 1],
                'pred_boxes':  [B, L, Q, 4] in (cx, cy, w, h) normalized coordinates
            }
        """
        B, C, H, W = x.shape

        # 1. Feature Extraction & Positional Embeddings
        feats = self.backbone(x)  # [B, C_backbone, H_feat, W_feat]
        pos = self.pos_embed(feats)  # [B, d_model, H_feat, W_feat]

        src = self.input_proj(feats)  # [B, d_model, H_feat, W_feat]
        H_feat, W_feat = src.shape[-2:]

        spatial_shapes = torch.tensor([[H_feat, W_feat]], device=x.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=x.device, dtype=torch.long)

        # Flatten spatial dimensions: [B, d_model, H_feat, W_feat] -> [B, H_feat*W_feat, d_model]
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos.flatten(2).permute(0, 2, 1)

        # 2. Deformable Transformer Encoder Pass
        memory = self.encoder(src_flat, spatial_shapes, level_start_index, pos=pos_flat)

        # 3. Deformable Transformer Decoder Pass
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, d_model]
        tgt = torch.zeros_like(query_embed)

        # Compute normalized reference points from query embeddings via sigmoid
        reference_points = self.reference_points_head(query_embed).sigmoid().unsqueeze(2)  # [B, Q, 1, 2]

        hs = self.decoder(
            tgt=tgt,
            query_pos=query_embed,
            reference_points=reference_points,
            src=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )  # [B, num_decoder_layers, Q, d_model]

        # 4. Multilayer Classification & Box Prediction Heads
        outputs_class = []
        outputs_coord = []
        for l in range(hs.shape[1]):
            layer_hs = hs[:, l]
            outputs_class.append(self.class_embed[l](layer_hs))
            
            # Reference point refinement (offset delta + sigmoid reference)
            bbox_offsets = self.bbox_embed[l](layer_hs)
            ref_xy = reference_points[:, :, 0]
            cxcy = (bbox_offsets[..., :2] + torch.logit(ref_xy, eps=1e-5)).sigmoid()
            wh = bbox_offsets[..., 2:].sigmoid()
            outputs_coord.append(torch.cat([cxcy, wh], dim=-1))

        outputs_class = torch.stack(outputs_class, dim=1)  # [B, L, Q, num_classes + 1]
        outputs_coord = torch.stack(outputs_coord, dim=1)  # [B, L, Q, 4]

        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
        }


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron (MLP) for DETR heads."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
