"""
LeWorldModel (LeWM) — native, self-contained implementation.

Joint-Embedding Predictive Architecture (JEPA) from raw pixels with
Sketched Isotropic Gaussian Regularization (SIGReg).

Ref:
    Maes, Le Lidec, Scieur, LeCun, Balestriero,
    "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" (2026/2024).
    https://le-wm.github.io/
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from src.utils.tensor_checks import typechecked


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation helper."""
    return x * (1 + scale) + shift


class CNNVisualEncoder(nn.Module):
    """
    CNN visual encoder mapping [B*T, C, H, W] -> latent embedding [B*T, embed_dim].

    Uses convolutional feature extraction followed by an MLP projector with BatchNorm.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 192,
        hidden_dim: int = 2048,
        resolution: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.embed_dim = embed_dim

        # 4-stage convolutional backbone
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),  # -> H/2, W/2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),           # -> H/4, W/4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),          # -> H/8, W/8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),          # -> H/16, W/16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        flat_dim = 256 * 2 * 2
        self.projector = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat_flat = feat.flatten(1)
        return self.projector(feat_flat)


class ActionEmbedder(nn.Module):
    """
    Action sequence embedder mapping [B, T, action_dim] -> [B, T, embed_dim].
    """

    def __init__(
        self,
        action_dim: int = 2,
        embed_dim: int = 192,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(action_dim, embed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(embed_dim, mlp_scale * embed_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, action_dim]
        """
        x = x.float().permute(0, 2, 1)  # [B, action_dim, T]
        x = self.patch_embed(x).permute(0, 2, 1)  # [B, T, embed_dim]
        return self.embed(x)


class FeedForward(nn.Module):
    """FeedForward network used in Transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalAttention(nn.Module):
    """Multi-head attention with causal masking."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        x: [B, T, dim]
        """
        B, T, _ = x.shape
        x_norm = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)

        # Reshape to [B, heads, T, dim_head]
        q, k, v = [
            t.view(B, T, self.heads, self.dim_head).transpose(1, 2)
            for t in qkv
        ]

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, self.heads * self.dim_head)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ARPredictor(nn.Module):
    """Autoregressive Transformer predictor for next-step embedding prediction."""

    def __init__(
        self,
        num_frames: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([
            ConditionalBlock(
                dim=embed_dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, embed_dim]
        c: [B, T, embed_dim]
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, c)
        return self.norm(x)


class LeWM(nn.Module):
    """
    LeWorldModel (LeWM) Core Architecture.

    Joint-Embedding Predictive Architecture with:
      - Visual Encoder & Projector: pixels [B, T, C, H, W] -> state embeddings [B, T, D]
      - Action Embedder: actions [B, T, ActDim] -> action embeddings [B, T, D]
      - Autoregressive Predictor: (emb_{1:T-1}, act_emb_{1:T-1}) -> pred_emb_{2:T}
      - Prediction Projector: pred_emb -> projected_pred
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (64, 64),
        in_channels: int = 3,
        action_dim: int = 2,
        embed_dim: int = 192,
        hidden_dim: int = 2048,
        num_frames: int = 16,
        predictor_depth: int = 6,
        predictor_heads: int = 8,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.resolution = tuple(resolution)
        self.in_channels = in_channels
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # 1. Visual Encoder
        self.encoder = CNNVisualEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            resolution=resolution,
        )

        # 2. Action Embedder
        self.action_encoder = ActionEmbedder(
            action_dim=action_dim,
            embed_dim=embed_dim,
        )

        # 3. Autoregressive Predictor
        self.predictor = ARPredictor(
            num_frames=num_frames,
            embed_dim=embed_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            dropout=dropout,
        )

        # 4. Prediction Projector
        self.pred_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    @typechecked
    def encode(
        self,
        video: Float[torch.Tensor, "B T C H W"],
        actions: Float[torch.Tensor, "B T ActDim"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B T D"], Float[torch.Tensor, "B T D"] | None]:
        """
        Encode raw video pixels and actions into latent embeddings.

        Returns:
            (emb [B, T, embed_dim], act_emb [B, T, embed_dim] or None)
        """
        B, T, C, H, W = video.shape
        flat_video = video.flatten(0, 1)  # [B*T, C, H, W]
        emb_flat = self.encoder(flat_video)  # [B*T, embed_dim]
        emb = emb_flat.unflatten(0, (B, T))  # [B, T, embed_dim]

        act_emb = None
        if actions is not None:
            act_emb = self.action_encoder(actions)  # [B, T, embed_dim]

        return emb, act_emb

    @typechecked
    def predict(
        self,
        emb: Float[torch.Tensor, "B T D"],
        act_emb: Float[torch.Tensor, "B T D"],
    ) -> Float[torch.Tensor, "B T D"]:
        """
        Predict next-step state embeddings from context embeddings and actions.
        """
        B, T, D = emb.shape
        preds = self.predictor(emb, act_emb)  # [B, T, embed_dim]
        preds_flat = preds.flatten(0, 1)
        preds_proj_flat = self.pred_proj(preds_flat)
        return preds_proj_flat.unflatten(0, (B, T))

    @typechecked
    def forward(
        self,
        video: Float[torch.Tensor, "B T C H W"],
        actions: Float[torch.Tensor, "B T ActDim"] | None = None,
        n_preds: int = 1,
    ) -> dict[str, Any]:
        """
        Full LeWM forward pass for training.
        """
        B, T, C, H, W = video.shape

        if actions is None:
            actions = torch.zeros(B, T, self.action_dim, device=video.device, dtype=video.dtype)

        emb, act_emb = self.encode(video, actions)

        # Context frames 0 .. T - n_preds - 1
        ctx_len = T - n_preds
        if ctx_len < 1:
            raise ValueError(f"Sequence length T={T} must be > n_preds={n_preds}")

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]

        # Target embeddings to predict: frames n_preds .. T - 1
        target_emb = emb[:, n_preds:]
        pred_emb = self.predict(ctx_emb, ctx_act)

        # Prediction loss in latent embedding space
        pred_loss = F.mse_loss(pred_emb, target_emb)

        return {
            "emb": emb,
            "act_emb": act_emb,
            "pred_emb": pred_emb,
            "target_emb": target_emb,
            "pred_loss": pred_loss,
            "video": video,
            "actions": actions,
        }

    @typechecked
    def rollout(
        self,
        video: Float[torch.Tensor, "B T_cond C H W"],
        actions: Float[torch.Tensor, "B T_total ActDim"],
        n_cond_frames: int = 2,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive rollout in latent embedding space.

        Args:
            video: Conditioning video frames [B, T_cond, C, H, W]
            actions: Full action sequence [B, T_total, ActDim]
            n_cond_frames: Number of context frames to initialize rollout
        """
        B, T_cond, C, H, W = video.shape
        T_total = actions.shape[1]

        if n_cond_frames > T_cond:
            raise ValueError(f"n_cond_frames={n_cond_frames} cannot exceed video T_cond={T_cond}")

        # Encode conditioning frames
        cond_video = video[:, :n_cond_frames]
        cond_actions = actions[:, :n_cond_frames]
        emb, _ = self.encode(cond_video, cond_actions)  # [B, n_cond_frames, D]
        all_act_emb = self.action_encoder(actions)  # [B, T_total, D]

        cur_emb = emb
        for t in range(n_cond_frames, T_total):
            cur_act_emb = all_act_emb[:, :t]
            pred_next = self.predict(cur_emb, cur_act_emb)[:, -1:]  # [B, 1, D]
            cur_emb = torch.cat([cur_emb, pred_next], dim=1)  # [B, t+1, D]

        return {
            "emb": cur_emb,
            "cond_emb": emb,
            "is_rollout_mask": torch.tensor(
                [False] * n_cond_frames + [True] * (T_total - n_cond_frames),
                device=video.device,
            ),
        }
