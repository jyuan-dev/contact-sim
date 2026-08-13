"""
INTACT RobotSlotIntentActionActor: Robot-Grounded Intent-to-Action Operator.

Maps current slot state z_t and motion/goal intent m_t = z_{t+1} - z_t
to robot action Gaussian parameters (mean, log_std) using per-slot 4-slot grammar
([z_k, m_k, z_k * m_k]) and robot-anchored inter-slot cross-attention.

Reference:
  INTACT: Isomorphic Intent-to-Action Learning (Sun et al., 2026)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobotSlotIntentActionActor(nn.Module):
    """
    Robot-Grounded Intent-to-Action Actor.

    Computes per-slot 4-slot feature grammar:
        g_k = [z_{t,k}, m_{t,k}, z_{t,k} * m_{t,k}]
    and applies inter-slot multi-head attention where the robot slot (default idx 0)
    attends to object slots to infer contact-conditioned robot control actions.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        action_dim: int = 2,
        action_emb_dim: int = 64,
        robot_slot_idx: int = 0,
        hidden_dim: int = 256,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.action_dim = action_dim
        self.action_emb_dim = action_emb_dim
        self.robot_slot_idx = robot_slot_idx
        self.hidden_dim = hidden_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Grammar feature size per slot: [z, m, z * m] -> 3 * slot_dim
        self.grammar_dim = 3 * slot_dim

        self.grammar_proj = nn.Linear(self.grammar_dim, hidden_dim)

        # Previous action embedder
        if action_emb_dim > 0:
            self.prev_action_encoder = nn.Sequential(
                nn.Linear(action_dim, action_emb_dim),
                nn.SiLU(),
                nn.Linear(action_emb_dim, action_emb_dim),
            )
        else:
            self.prev_action_encoder = None

        # Inter-slot interaction attention block
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_slot = nn.LayerNorm(hidden_dim)

        # MLP actor head
        in_actor_dim = hidden_dim + (action_emb_dim if action_emb_dim > 0 else 0)
        actor_layers: list[nn.Module] = [
            nn.Linear(in_actor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            actor_layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        actor_layers.append(nn.Linear(hidden_dim, 2 * action_dim))
        self.actor_net = nn.Sequential(*actor_layers)

    def extract_features(
        self,
        z_curr: torch.Tensor,
        z_next: torch.Tensor,
        prev_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z_curr: Current slot tokens [B, K, D]
            z_next: Next/Goal slot tokens [B, K, D]
            prev_action: Previous action [B, action_dim] or None

        Returns:
            Concatenated actor feature vector [B, hidden_dim + action_emb_dim]
        """
        if z_curr.shape != z_next.shape:
            raise ValueError(f"z_curr and z_next must have same shape, got {z_curr.shape} and {z_next.shape}")

        intent = z_next - z_curr  # [B, K, D]
        grammar = torch.cat([z_curr, intent, z_curr * intent], dim=-1)  # [B, K, 3D]

        slot_feats = self.grammar_proj(grammar)  # [B, K, hidden_dim]

        # Multi-head attention across slots
        attn_out, _ = self.slot_attn(slot_feats, slot_feats, slot_feats)
        slot_feats = self.norm_slot(slot_feats + attn_out)  # [B, K, hidden_dim]

        # Extract robot slot feature
        robot_idx = min(self.robot_slot_idx, slot_feats.size(1) - 1)
        robot_feat = slot_feats[:, robot_idx]  # [B, hidden_dim]

        # Encode previous action if present
        if self.prev_action_encoder is not None:
            if prev_action is None:
                prev_act_emb = torch.zeros(
                    z_curr.size(0), self.action_emb_dim, device=z_curr.device, dtype=z_curr.dtype
                )
            else:
                prev_act_emb = self.prev_action_encoder(prev_action)
            return torch.cat([robot_feat, prev_act_emb], dim=-1)

        return robot_feat

    def forward(
        self,
        z_curr: torch.Tensor,
        z_next: torch.Tensor,
        prev_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Gaussian action mean and log_std given z_curr and z_next (or z_goal).
        Returns: (mean, log_std) each [B, action_dim]
        """
        feat = self.extract_features(z_curr, z_next, prev_action)
        params = self.actor_net(feat)
        mean, log_std = params.chunk(2, dim=-1)
        return mean, log_std.clamp(self.min_log_std, self.max_log_std)

    def action_nll(
        self,
        z_curr: torch.Tensor,
        z_next: torch.Tensor,
        target_action: torch.Tensor,
        prev_action: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """
        Calculate Gaussian Negative Log-Likelihood (NLL) Loss against ground-truth actions.
        """
        mean, log_std = self(z_curr, z_next, prev_action)
        # NLL formula: 0.5 * [ ((a - mu)^2 / exp(2*log_std)) + 2*log_std ]
        var = torch.exp(2 * log_std)
        nll = 0.5 * (((target_action - mean).square() / var) + 2 * log_std).mean(dim=-1)

        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "none":
            loss = nll
        else:
            raise ValueError(f"Unsupported reduction: '{reduction}'")

        mae = (mean - target_action).abs().mean()
        rmse = (mean - target_action).square().mean().sqrt()

        return {
            "loss": loss,
            "nll": nll,
            "mean": mean,
            "log_std": log_std,
            "action_mae": mae,
            "action_rmse": rmse,
        }
