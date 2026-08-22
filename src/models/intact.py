"""
INTACT RobotSlotIntentActionActor: Robot-Grounded Intent-to-Action Operator.

Maps current slot state z_t and motion/goal intent m_t = z_{next} - z_t
to robot action Gaussian parameters (mean, log_std) using per-slot 4-slot grammar
([z_k, m_k, z_k * m_k, a_{t-1}]) and robot-anchored inter-slot cross-attention.

Reference:
  INTACT: Isomorphic Intent-to-Action Learning (Sun et al., 2026)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from src.utils.tensor_checks import typechecked


class INTACT(nn.Module):
    """
    INTACT: Robot-Grounded Intent-to-Action Operator.

    Computes per-slot 4-slot feature grammar:
        g_k = [z_{t,k}, m_{t,k}, z_{t,k} * m_{t,k}]
    and applies inter-slot multi-head attention where the robot slot (default idx 0)
    attends to object slots to infer contact-conditioned robot control actions.

    Supports dual-intent learning:
      - Attached Local Physical Transition: m_local = z_{t+1} - z_t
      - Detached Deployment Goal Intent:    m_goal  = sg(z_g) - z_t
    """

    def __init__(
        self,
        slot_dim: int = 64,
        action_dim: int = 2,
        action_emb_dim: int = 64,
        robot_slot_idx: int = 0,
        robot_only_action: bool = True,
        hidden_dim: int = 256,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.0,
        chunk_size: int = 1,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.raw_action_dim = action_dim
        self.chunk_size = chunk_size
        self.total_action_dim = action_dim * chunk_size
        self.action_emb_dim = action_emb_dim
        self.robot_slot_idx = int(robot_slot_idx)
        self.robot_only_action = bool(robot_only_action)
        self.hidden_dim = hidden_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Grammar feature size per slot: [z, m, z * m] -> 3 * slot_dim
        self.grammar_dim = 3 * slot_dim
        self.grammar_proj = nn.Linear(self.grammar_dim, hidden_dim)

        # Previous action embedder (embeds raw action dimension)
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

        # MLP actor head outputs Gaussian parameters for the action chunk
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
        actor_layers.append(nn.Linear(hidden_dim, 2 * self.total_action_dim))
        self.actor_net = nn.Sequential(*actor_layers)

    @typechecked
    def extract_features(
        self,
        z_curr: Float[torch.Tensor, "B K D"],
        z_next: Float[torch.Tensor, "B K D"],
        prev_action: Float[torch.Tensor, "B ActDim"] | None = None,
    ) -> Float[torch.Tensor, "..."]:
        """
        Args:
            z_curr: Current slot tokens [B, K, D]
            z_next: Next/Goal slot tokens [B, K, D]
            prev_action: Previous action [B, raw_action_dim] or None

        Returns:
            Actor feature vector [B, hidden_dim + action_emb_dim] if robot_only_action=True,
            or [B, K, hidden_dim + action_emb_dim] if robot_only_action=False.
        """
        intent = z_next - z_curr  # [B, K, D]
        grammar = torch.cat([z_curr, intent, z_curr * intent], dim=-1)  # [B, K, 3D]

        slot_feats = self.grammar_proj(grammar)  # [B, K, hidden_dim]

        # Multi-head attention across slots
        attn_out, _ = self.slot_attn(slot_feats, slot_feats, slot_feats)
        slot_feats = self.norm_slot(slot_feats + attn_out)  # [B, K, hidden_dim]

        # Encode previous action if present
        if self.prev_action_encoder is not None:
            if prev_action is None:
                prev_act_emb = torch.zeros(
                    z_curr.size(0), self.action_emb_dim, device=z_curr.device, dtype=z_curr.dtype
                )
            else:
                # If prev_action is a chunk, slice to the most recent step
                if prev_action.dim() > 2:
                    prev_act_slice = prev_action[:, -1]
                elif prev_action.shape[-1] > self.raw_action_dim:
                    prev_act_slice = prev_action[:, :self.raw_action_dim]
                else:
                    prev_act_slice = prev_action
                prev_act_emb = self.prev_action_encoder(prev_act_slice)
        else:
            prev_act_emb = None

        if self.robot_only_action:
            # Extract ONLY the robot slot feature
            robot_idx = min(self.robot_slot_idx, slot_feats.size(1) - 1)
            robot_feat = slot_feats[:, robot_idx]  # [B, hidden_dim]
            if prev_act_emb is not None:
                return torch.cat([robot_feat, prev_act_emb], dim=-1)
            return robot_feat
        else:
            # All slots predict action
            if prev_act_emb is not None:
                prev_act_emb_k = prev_act_emb.unsqueeze(1).repeat(1, slot_feats.size(1), 1)  # [B, K, action_emb_dim]
                return torch.cat([slot_feats, prev_act_emb_k], dim=-1)  # [B, K, hidden_dim + action_emb_dim]
            return slot_feats  # [B, K, hidden_dim]

    @typechecked
    def forward(
        self,
        z_curr: Float[torch.Tensor, "B K D"],
        z_next: Float[torch.Tensor, "B K D"],
        prev_action: Float[torch.Tensor, "B ActDim"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B ActDimOut"], Float[torch.Tensor, "B ActDimOut"]]:
        """
        Predict Gaussian action mean and log_std given z_curr and z_next (or z_goal).
        Returns: (mean, log_std) each [B, total_action_dim]
        """
        feat = self.extract_features(z_curr, z_next, prev_action)
        if self.robot_only_action:
            params = self.actor_net(feat)  # [B, 2 * total_action_dim]
        else:
            # All slots predict action hypotheses, aggregated across all slots
            all_params = self.actor_net(feat)  # [B, K, 2 * total_action_dim]
            params = all_params.mean(dim=1)    # [B, 2 * total_action_dim]

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
        
        # Flatten target action if provided as [B, chunk, dim]
        if target_action.dim() > 2:
            target_flat = target_action.flatten(1, -1)
        else:
            target_flat = target_action

        # Handle size alignment
        if target_flat.shape[-1] != self.total_action_dim:
            if target_flat.shape[-1] < self.total_action_dim:
                # Pad or repeat if necessary
                target_flat = target_flat.repeat(1, self.chunk_size)[:, :self.total_action_dim]
            else:
                target_flat = target_flat[:, :self.total_action_dim]

        var = torch.exp(2 * log_std)
        nll = 0.5 * (((target_flat - mean).square() / var) + 2 * log_std).mean(dim=-1)

        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "none":
            loss = nll
        else:
            raise ValueError(f"Unsupported reduction: '{reduction}'")

        mae = (mean - target_flat).abs().mean()
        rmse = (mean - target_flat).square().mean().sqrt()

        return {
            "loss": loss,
            "nll": nll,
            "mean": mean,
            "log_std": log_std,
            "action_mae": mae,
            "action_rmse": rmse,
        }

    def action_nll_dual(
        self,
        z_curr: torch.Tensor,
        z_local_next: torch.Tensor,
        z_goal: torch.Tensor,
        target_action_local: torch.Tensor,
        target_action_goal: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
        lambda_inv: float = 1.0,
        lambda_goal: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Isomorphic Dual-Intent Loss (Eq. 14 in Sun et al., 2026):
          L_I2A = lambda_inv * NLL(a_t | z_t, z_{t+1}) + lambda_goal * NLL(a_t | z_t, sg(z_g))
        """
        # 1. Local physical transition (attached gradient)
        res_local = self.action_nll(
            z_curr=z_curr,
            z_next=z_local_next,
            target_action=target_action_local,
            prev_action=prev_action,
            reduction="mean",
        )

        # 2. Deployment goal intent (stop-gradient anchor)
        target_goal_act = target_action_goal if target_action_goal is not None else target_action_local
        res_goal = self.action_nll(
            z_curr=z_curr,
            z_next=z_goal.detach(),  # sg(z_g) stop-gradient deployment anchor
            target_action=target_goal_act,
            prev_action=prev_action,
            reduction="mean",
        )

        total_loss = lambda_inv * res_local["loss"] + lambda_goal * res_goal["loss"]

        return {
            "loss": total_loss,
            "loss_local": res_local["loss"],
            "loss_goal": res_goal["loss"],
            "mean_local": res_local["mean"],
            "mean_goal": res_goal["mean"],
            "log_std_local": res_local["log_std"],
            "log_std_goal": res_goal["log_std"],
            "action_mae": 0.5 * (res_local["action_mae"] + res_goal["action_mae"]),
            "action_rmse": 0.5 * (res_local["action_rmse"] + res_goal["action_rmse"]),
        }


# Aliases for backwards compatibility and ergonomics
Intact = INTACT
RobotSlotIntentActionActor = INTACT
