"""
PIDM: Predictive Inverse Dynamics Model over Latent Object Slots.

Decouples visual slot trajectory prediction from low-level action generation:
1. Goal-Conditioned Slot Rollouter: Autoregressively plans future slot states
   z_{t+1:t+H} conditioned on history slots z_{1:t} and target/goal slots z_goal.
2. Inverse Dynamics Model (IDM): Contact-grounded RobotSlotIntentActionActor
   mapping (z_t, z_{t+1}) state transitions to continuous robot actions.
3. Rollout-Consistent Training: Multi-step joint training over both ground-truth
   and rollouter-predicted transitions.
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from src.utils.tensor_checks import typechecked
from src.models.intact import RobotSlotIntentActionActor
from src.models.slotformer import (
    build_pos_enc,
    extract_stage1_slots,
    decode_stage1_slots,
    TemporalSelfAttention,
    InteractiveSelfAttention,
    SlotTransitionMLP,
)


class GoalConditionedRollouterLayer(nn.Module):
    """
    Transformer Rollouter Layer supporting Goal Slot Conditioning via
    FiLM modulation or Cross-Attention.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        condition_mode: str = "goal_film",
        goal_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.condition_mode = condition_mode.lower()
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = TemporalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.interactive_attn = InteractiveSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.slot_transition = SlotTransitionMLP(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)

        if self.condition_mode in ("goal_film", "film"):
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(goal_emb_dim, 6 * d_model),
            )
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)
        elif self.condition_mode in ("goal_cross_attn", "cross_attn"):
            self.cross_attn_norm = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.modulation = None

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, "B T K D"],
        goal_emb: Float[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "B T K D"]:
        """
        Args:
            x: Slot tokens [B, T, K, D]
            goal_emb: Goal conditioning tensor
                - If goal_film: [B, D_goal] or [B, 1, 1, D_goal]
                - If goal_cross_attn: [B, K_goal, D]
        """
        B, T, K, D = x.shape

        if self.condition_mode in ("goal_film", "film") and goal_emb is not None:
            if goal_emb.ndim == 2:
                goal_emb = goal_emb.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D_goal]
            elif goal_emb.ndim == 3:
                goal_emb = goal_emb.unsqueeze(1)  # [B, 1, K, D_goal]

            assert self.modulation is not None, "modulation layer not built for this condition mode"
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(goal_emb).chunk(6, dim=-1)
            x = x + gate_a * self.temporal_attn(self.norm1(x) * (1 + scale_a) + shift_a)
            x = x + self.interactive_attn(self.norm2(x))
            x = x + gate_m * self.slot_transition(self.norm3(x) * (1 + scale_m) + shift_m)
        elif self.condition_mode in ("goal_cross_attn", "cross_attn") and goal_emb is not None:
            x = x + self.temporal_attn(self.norm1(x))
            x = x + self.interactive_attn(self.norm2(x))
            # Cross-attention against goal slots
            x_flat = x.view(B, T * K, D)
            norm_x = self.cross_attn_norm(x_flat)
            attn_out, _ = self.cross_attn(norm_x, goal_emb, goal_emb)
            x = (x_flat + attn_out).view(B, T, K, D)
            x = x + self.slot_transition(self.norm3(x))
        else:
            x = x + self.temporal_attn(self.norm1(x))
            x = x + self.interactive_attn(self.norm2(x))
            x = x + self.slot_transition(self.norm3(x))

        return x


class GoalConditionedSlotRollouter(nn.Module):
    """
    Goal-Conditioned Slot Rollouter Transformer.
    Predicts future slot trajectories z_{t+1:t+H} conditioned on history z_{1:t}
    and goal representation z_goal.
    """

    def __init__(
        self,
        num_slots: int = 4,
        slot_size: int = 64,
        history_len: int = 2,
        t_pe: str = "sin",
        slots_pe: str = "",
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        condition_mode: str = "goal_film",
        goal_slot_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.history_len = history_len
        self.condition_mode = condition_mode.lower()
        self.goal_slot_idx = goal_slot_idx
        self.d_model = d_model

        self.in_proj = nn.Linear(slot_size, d_model)

        # Goal encoder / projector
        if self.condition_mode in ("goal_film", "film"):
            # If single slot index or flattened slots
            in_goal_dim = slot_size if goal_slot_idx is not None else num_slots * slot_size
            self.goal_encoder = nn.Sequential(
                nn.Linear(in_goal_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            goal_emb_dim = d_model
        elif self.condition_mode in ("goal_cross_attn", "cross_attn"):
            self.goal_encoder = nn.Linear(slot_size, d_model)
            goal_emb_dim = d_model
        elif self.condition_mode in ("goal_sum", "sum"):
            in_goal_dim = slot_size if goal_slot_idx is not None else num_slots * slot_size
            self.goal_encoder = nn.Sequential(
                nn.Linear(in_goal_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            goal_emb_dim = d_model
        else:
            self.goal_encoder = None
            goal_emb_dim = 0

        self.layers = nn.ModuleList([
            GoalConditionedRollouterLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                condition_mode=self.condition_mode,
                goal_emb_dim=goal_emb_dim,
            )
            for _ in range(num_layers)
        ])

        self.enc_t_pe = build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = build_pos_enc(slots_pe, num_slots, d_model)
        self.out_proj = nn.Linear(d_model, slot_size)

    def extract_goal_feature(self, goal_slots: torch.Tensor) -> torch.Tensor | None:
        """
        Extract and project goal conditioning vector/sequence.
        Args:
            goal_slots: [B, K, D] or [B, D]
        """
        if self.goal_encoder is None or goal_slots is None:
            return None

        if self.condition_mode in ("goal_cross_attn", "cross_attn"):
            if goal_slots.ndim == 2:
                goal_slots = goal_slots.unsqueeze(1)  # [B, 1, D]
            return self.goal_encoder(goal_slots)      # [B, K_goal, D_model]

        # For FiLM or Sum mode
        if goal_slots.ndim == 3:
            if self.goal_slot_idx is not None:
                idx = min(self.goal_slot_idx, goal_slots.size(1) - 1)
                g_in = goal_slots[:, idx]             # [B, D]
            else:
                g_in = goal_slots.flatten(1)          # [B, K*D]
        else:
            g_in = goal_slots                         # [B, D]

        return self.goal_encoder(g_in)                # [B, D_model]

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, "B H K D"],
        pred_len: int,
        goal_slots: Float[torch.Tensor, "..."] | None = None,
        actions: Float[torch.Tensor, "B T_act ActDim"] | None = None,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "B P K D"]:
        """
        Args:
            x: History slot sequence [B, history_len, num_slots, slot_size]
            pred_len: Number of future timesteps to rollout
            goal_slots: Target goal slots [B, num_slots, slot_size] or [B, slot_size]

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.history_len, f"Expected history_len={self.history_len}, got {x.shape[1]}"

        goal_emb = self.extract_goal_feature(goal_slots) if goal_slots is not None else None

        curr_x = x
        pred_out = []

        for _ in range(pred_len):
            curr_t_len = curr_x.shape[1]
            proj_x = self.in_proj(curr_x)  # [B, curr_t_len, K, D_model]

            if self.condition_mode in ("goal_sum", "sum") and goal_emb is not None:
                # Add goal embedding broadcast across time and slots
                proj_x = proj_x + goal_emb.unsqueeze(1).unsqueeze(2)

            if self.enc_t_pe is not None:
                t_pe = self.enc_t_pe.unsqueeze(2).to(x.device)
                proj_x = proj_x + t_pe[:, :curr_t_len]

            if self.enc_slots_pe is not None:
                slots_pe = self.enc_slots_pe.unsqueeze(1).to(x.device)
                proj_x = proj_x + slots_pe

            layer_out = proj_x
            for layer in self.layers:
                layer_out = layer(layer_out, goal_emb=goal_emb)

            last_timestep_tokens = layer_out[:, -1]
            pred_slots = self.out_proj(last_timestep_tokens)
            pred_out.append(pred_slots)

            # Shift window: drop oldest frame, append predicted frame
            curr_x = torch.cat([curr_x[:, 1:], pred_slots.unsqueeze(1)], dim=1)

        return torch.stack(pred_out, dim=1)  # [B, pred_len, K, slot_size]


class PIDMModel(nn.Module):
    """
    Predictive Inverse Dynamics Model (PIDM).

    Jointly trains:
    1. Goal-Conditioned Slot Rollouter for multi-step slot trajectory prediction.
    2. RobotSlotIntentActionActor (IDM) for contact-grounded action inference.
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        history_len: int = 2,
        rollout_len: int = 4,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        t_pe: str = "sin",
        slots_pe: str = "",
        loss_decay_factor: float = 1.0,
        use_img_recon_loss: bool = False,
        condition_mode: str = "goal_film",
        goal_slot_idx: int = 2,
        raw_action_dim: int = 2,
        action_embed_dim: int = 64,
        action_loss_weight: float = 1.0,
        slot_loss_weight: float = 1.0,
        robot_slot_idx: int = 0,
        rollout_consistent: bool = True,
    ) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        self.history_len = history_len
        self.rollout_len = rollout_len
        self.loss_decay_factor = loss_decay_factor
        self.use_img_recon_loss = use_img_recon_loss
        self.condition_mode = condition_mode
        self.goal_slot_idx = goal_slot_idx
        self.action_loss_weight = action_loss_weight
        self.slot_loss_weight = slot_loss_weight
        self.robot_slot_idx = robot_slot_idx
        self.rollout_consistent = rollout_consistent

        inner_savi = self.stage1_model.inner_savi()
        self.slot_size = inner_savi.slot_size
        self.num_slots = inner_savi.num_slots

        # Goal-Conditioned Slot Rollouter
        self.rollouter = GoalConditionedSlotRollouter(
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            history_len=history_len,
            t_pe=t_pe,
            slots_pe=slots_pe,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            condition_mode=condition_mode,
            goal_slot_idx=goal_slot_idx,
        )

        # Inverse Dynamics Model (IDM)
        self.idm_actor = RobotSlotIntentActionActor(
            slot_dim=self.slot_size,
            action_dim=raw_action_dim,
            action_emb_dim=action_embed_dim,
            robot_slot_idx=robot_slot_idx,
            hidden_dim=256,
            num_heads=4,
            depth=2,
        )

        # Freeze Stage 1 model parameters
        for p in self.stage1_model.parameters():
            p.requires_grad = False
        self.stage1_model.eval()

    @classmethod
    def from_config(cls, model_cfg: dict, stage1_wrapper: nn.Module) -> "PIDMModel":
        """
        Construct a PIDMModel from a flat model-config dict and an already-built
        Stage 1 wrapper. Owns all PIDM-specific kwarg defaults/aliases so callers
        (e.g. ``StandardizedSlotFormerWrapper.build``) don't need to know them.
        """
        return cls(
            stage1_model=stage1_wrapper,
            history_len=model_cfg.get("history_len", 2),
            rollout_len=model_cfg.get("rollout_len", 4),
            d_model=model_cfg.get("d_model", 128),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            ffn_dim=model_cfg.get("ffn_dim", 512),
            t_pe=model_cfg.get("t_pe", "sin"),
            slots_pe=model_cfg.get("slots_pe", ""),
            loss_decay_factor=model_cfg.get("loss_decay_factor", 1.0),
            use_img_recon_loss=model_cfg.get("use_img_recon_loss", False),
            condition_mode=model_cfg.get("condition_mode", "goal_film"),
            goal_slot_idx=model_cfg.get("goal_slot_idx", 2),
            raw_action_dim=model_cfg.get("raw_action_dim", model_cfg.get("action_dim", 2)),
            action_embed_dim=model_cfg.get("action_embed_dim", model_cfg.get("action_emb_dim", 64)),
            action_loss_weight=model_cfg.get("action_loss_weight", 1.0),
            slot_loss_weight=model_cfg.get("slot_loss_weight", 1.0),
            robot_slot_idx=model_cfg.get("robot_slot_idx", 0),
            rollout_consistent=model_cfg.get("rollout_consistent", True),
        )

    def train(self, mode: bool = True) -> "PIDMModel":
        super().train(mode)
        # Always maintain Stage 1 in eval mode
        self.stage1_model.eval()
        return self

    @typechecked
    def extract_slots(self, video: Float[torch.Tensor, "B T C H W"]) -> Float[torch.Tensor, "B T K D"]:
        """
        Extract per-frame slots [B, T, K, D] using frozen Stage 1 model.
        """
        return extract_stage1_slots(self.stage1_model, video)

    def forward(self, batch: dict | Float[torch.Tensor, "B T C H W"]) -> dict[str, Any]:
        """
        Forward pass for PIDM training and rollout inference.
        """
        if isinstance(batch, torch.Tensor):
            video = batch
            actions = None
            goal_slots_in = None
        else:
            video = batch["img"]
            actions = batch.get("action", None)
            goal_slots_in = batch.get("goal_slots", None)

        B, T = video.shape[:2]

        with torch.no_grad():
            gt_all_slots = self.extract_slots(video)  # [B, T, K, D]

        history_slots = gt_all_slots[:, :self.history_len]
        gt_rollout_slots = gt_all_slots[:, self.history_len:self.history_len + self.rollout_len]

        # Determine goal slot representation (final frame of clip or explicit goal)
        if goal_slots_in is not None:
            goal_slots = goal_slots_in
        else:
            # Default to slots from final available frame
            goal_slots = gt_all_slots[:, -1]  # [B, K, D]

        # Goal-Conditioned Slot Rollout
        pred_rollout_slots = self.rollouter(
            history_slots, pred_len=self.rollout_len, goal_slots=goal_slots
        )

        out_dict = {
            "gt_slots": gt_rollout_slots,
            "pred_slots": pred_rollout_slots,
            "history_slots": history_slots,
            "gt_all_slots": gt_all_slots,
            "goal_slots": goal_slots,
            "input_img": video,
        }

        # Multi-Step Rollout-Consistent Inverse Dynamics Action Inference
        if actions is not None and actions.shape[1] >= 1:
            total_act_steps = min(actions.shape[1], T - 1)
            act_dim = actions.shape[-1]

            # 1. Ground Truth Transitions (Fully Batched)
            z_curr_gt = gt_all_slots[:, :total_act_steps].reshape(B * total_act_steps, self.num_slots, self.slot_size)
            z_next_gt = gt_all_slots[:, 1:total_act_steps + 1].reshape(B * total_act_steps, self.num_slots, self.slot_size)
            target_act_gt = actions[:, :total_act_steps].reshape(B * total_act_steps, act_dim)

            prev_act_gt = torch.cat([
                torch.zeros(B, 1, act_dim, device=actions.device, dtype=actions.dtype),
                actions[:, :total_act_steps - 1]
            ], dim=1).reshape(B * total_act_steps, act_dim)

            gt_res = self.idm_actor.action_nll(
                z_curr=z_curr_gt,
                z_next=z_next_gt,
                target_action=target_act_gt,
                prev_action=prev_act_gt,
            )
            gt_loss_t = gt_res["loss"]

            # 2. Predicted Transitions (Fully Batched Rollout-Consistent Training)
            if self.rollout_consistent and self.rollout_len >= 1:
                full_pred_trajectory = torch.cat([history_slots, pred_rollout_slots], dim=1)
                max_rollout_steps = min(total_act_steps, full_pred_trajectory.shape[1] - 1)
                num_pred_steps = max_rollout_steps - (self.history_len - 1)

                if num_pred_steps > 0:
                    start_t = self.history_len - 1
                    end_t = max_rollout_steps
                    z_curr_pred = full_pred_trajectory[:, start_t:end_t].reshape(B * num_pred_steps, self.num_slots, self.slot_size)
                    z_next_pred = full_pred_trajectory[:, start_t + 1:end_t + 1].reshape(B * num_pred_steps, self.num_slots, self.slot_size)
                    target_act_pred = actions[:, start_t:end_t].reshape(B * num_pred_steps, act_dim)
                    prev_act_pred = actions[:, start_t - 1:end_t - 1].reshape(B * num_pred_steps, act_dim)

                    pred_res = self.idm_actor.action_nll(
                        z_curr=z_curr_pred,
                        z_next=z_next_pred,
                        target_action=target_act_pred,
                        prev_action=prev_act_pred,
                    )
                    pred_loss_t = pred_res["loss"]
                    joint_idm_loss = 0.5 * (gt_loss_t + pred_loss_t)
                else:
                    pred_loss_t = torch.tensor(0.0, device=video.device)
                    joint_idm_loss = gt_loss_t
            else:
                pred_loss_t = torch.tensor(0.0, device=video.device)
                joint_idm_loss = gt_loss_t

            pred_means = gt_res["mean"].view(B, total_act_steps, act_dim)
            act_mae = gt_res["action_mae"]
            act_rmse = gt_res["action_rmse"]

            out_dict["action_nll_dict"] = {
                "loss": joint_idm_loss,
                "gt_idm_loss": gt_loss_t,
                "pred_idm_loss": pred_loss_t,
                "action_mae": act_mae,
                "action_rmse": act_rmse,
                "pred_actions": pred_means,
            }

        # Visual reconstructions if requested
        if self.use_img_recon_loss or not self.training:
            full_slots = torch.cat([history_slots, pred_rollout_slots], dim=1)
            recon_img, pred_masks = decode_stage1_slots(self.stage1_model, full_slots, B)

            out_dict["recon_img"] = recon_img
            out_dict["pred_masks"] = pred_masks
            out_dict["post_slots"] = full_slots

        return out_dict

    def calc_train_loss(self, out_dict: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate Stage 2 PIDM Joint Loss (Slot MSE + Multi-step IDM Action NLL)."""
        gt_slots = out_dict["gt_slots"]      # [B, rollout_len, K, D]
        pred_slots = out_dict["pred_slots"]  # [B, rollout_len, K, D]

        slots_loss = F.mse_loss(pred_slots, gt_slots, reduction="none")

        if self.loss_decay_factor < 1.0:
            w = self.loss_decay_factor ** torch.arange(gt_slots.shape[1], device=gt_slots.device)
            w = w / w.sum() * gt_slots.shape[1]
            slots_loss = slots_loss * w[None, :, None, None]

        slot_recon_loss = slots_loss.mean()
        total_loss = self.slot_loss_weight * slot_recon_loss

        loss_metrics = {
            "loss": total_loss.item(),
            "slot_mse": slot_recon_loss.item(),
        }

        if "action_nll_dict" in out_dict:
            act_dict = out_dict["action_nll_dict"]
            act_loss = act_dict["loss"]
            total_loss = total_loss + self.action_loss_weight * act_loss
            loss_metrics["action_nll"] = act_loss.item()
            loss_metrics["gt_idm_loss"] = act_dict["gt_idm_loss"].item()
            loss_metrics["pred_idm_loss"] = act_dict["pred_idm_loss"].item()
            loss_metrics["action_mae"] = act_dict["action_mae"].item()
            loss_metrics["action_rmse"] = act_dict["action_rmse"].item()
            loss_metrics["loss"] = total_loss.item()

        if self.use_img_recon_loss and "recon_img" in out_dict:
            video = out_dict["input_img"]
            rollout_recon = out_dict["recon_img"][:, self.history_len:self.history_len + self.rollout_len]
            gt_video_rollout = video[:, self.history_len:self.history_len + self.rollout_len]
            img_loss = F.mse_loss(rollout_recon, gt_video_rollout)
            total_loss = total_loss + img_loss
            loss_metrics["img_mse"] = img_loss.item()
            loss_metrics["loss"] = total_loss.item()

        return total_loss, loss_metrics

    @torch.no_grad()
    def plan_action(
        self,
        history_video_or_slots: torch.Tensor,
        goal_video_or_slots: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Closed-loop execution interface for simulation / real-world deployment.

        Args:
            history_video_or_slots: Either [B, history_len, C, H, W] frames
                                    or [B, history_len, K, D] slots.
            goal_video_or_slots: Optional goal image [B, C, H, W] or goal slots [B, K, D].
            prev_action: Optional previous action [B, action_dim].

        Returns:
            Executed robot action [B, action_dim]
        """
        if history_video_or_slots.ndim == 5:
            # Video frames [B, T_hist, C, H, W]
            hist_slots = self.extract_slots(history_video_or_slots)
        else:
            hist_slots = history_video_or_slots

        if goal_video_or_slots is not None:
            if goal_video_or_slots.ndim == 4:
                # Single goal frame [B, C, H, W] -> make [B, 1, C, H, W]
                goal_slots = self.extract_slots(goal_video_or_slots.unsqueeze(1))[:, 0]
            elif goal_video_or_slots.ndim == 5:
                goal_slots = self.extract_slots(goal_video_or_slots)[:, -1]
            else:
                goal_slots = goal_video_or_slots
        else:
            goal_slots = None

        # 1. Rollout 1 step into the future
        pred_next_slots = self.rollouter(hist_slots, pred_len=1, goal_slots=goal_slots)[:, 0]  # [B, K, D]

        # 2. Extract action via IDM
        z_curr = hist_slots[:, -1]
        action_mean, _ = self.idm_actor(z_curr=z_curr, z_next=pred_next_slots, prev_action=prev_action)
        return action_mean
