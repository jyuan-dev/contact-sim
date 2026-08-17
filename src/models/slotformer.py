"""
SlotFormer: Transformer-based Autoregressive Dynamics Model over Slots (Stage 2).

References:
  - SlotFormer: Slot-Based Visual Reasoning and Prediction (Wu et al., NeurIPS 2022)
"""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from src.utils.tensor_checks import typechecked


def get_sin_pos_enc(seq_len: int, d_model: int) -> Float[torch.Tensor, "1 seq_len d_model"]:
    """Sinusoid temporal positional encoding [1, seq_len, d_model]."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).to(dtype=inv_freq.dtype, device=inv_freq.device)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, D]


def build_pos_enc(pos_enc_type: str, length: int, d_model: int) -> nn.Parameter | None:
    """Build positional encoding parameter."""
    if not pos_enc_type:
        return None
    if pos_enc_type == "learnable":
        return nn.Parameter(torch.zeros(1, length, d_model))
    elif "sin" in pos_enc_type:
        return nn.Parameter(get_sin_pos_enc(length, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f"Unsupported pos enc type: '{pos_enc_type}'")


class SlotRollouter(nn.Module):
    """
    Transformer-based Autoregressive Rollouter for Slot Latents.

    Takes past slot tokens [B, history_len, num_slots, slot_size],
    applies spatial-temporal position encodings, and autoregressively predicts
    future slot tokens for pred_len steps.
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
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.history_len = history_len

        self.in_proj = nn.Linear(slot_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_layers,
        )

        self.enc_t_pe = build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = build_pos_enc(slots_pe, num_slots, d_model)
        self.out_proj = nn.Linear(d_model, slot_size)

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, "B H K D"],
        pred_len: int,
        actions: Float[torch.Tensor, "B T_act ActDim"] | None = None,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "B P K D"]:
        """
        Args:
            x: [B, history_len, num_slots, slot_size]
            pred_len: Number of future timesteps to rollout

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.history_len, f"Expected history_len={self.history_len}, got {x.shape[1]}"
        B = x.shape[0]

        in_x = x.flatten(1, 2)  # [B, history_len * num_slots, slot_size]

        # Temporal PE: [1, T, D] -> [B, T, N, D] -> [B, T * N, D]
        assert self.enc_t_pe is not None, "rollouter built without a temporal PE"
        enc_pe = self.enc_t_pe.unsqueeze(2).repeat(B, 1, self.num_slots, 1).flatten(1, 2).to(x.device)

        if self.enc_slots_pe is not None:
            slots_pe = self.enc_slots_pe.unsqueeze(1).repeat(B, self.history_len, 1, 1).flatten(1, 2).to(x.device)
            enc_pe = slots_pe + enc_pe

        pred_out = []
        for _ in range(pred_len):
            proj_x = self.in_proj(in_x)
            proj_x = proj_x + enc_pe
            trans_out = self.transformer_encoder(proj_x)

            # Predict next step slots from the last N slot tokens
            last_tokens = trans_out[:, -self.num_slots:]
            pred_slots = self.out_proj(last_tokens)  # [B, N, slot_size]
            pred_out.append(pred_slots)

            # Shift sequence window: drop oldest frame slots and append newly predicted slots
            in_x = torch.cat([in_x[:, self.num_slots:], pred_slots], dim=1)

        return torch.stack(pred_out, dim=1)  # [B, pred_len, N, slot_size]


# ── OCVP Factorized SlotRollouter Variant ──────────────────────────────────────

class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention modeling motion dynamics over sequence length T
    for each slot token independently.
    Input: [B, T, K, D] -> Reshape [B * K, T, D] -> MultiHeadAttention -> Reshape [B, T, K, D].
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "B T K D"]) -> Float[torch.Tensor, "B T K D"]:
        B, T, K, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * K, T, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        return attn_out.reshape(B, K, T, D).permute(0, 2, 1, 3)


class InteractiveSelfAttention(nn.Module):
    """
    Interactive Self-Attention modeling inter-object relationships across K slots
    at each timestep independently.
    Input: [B, T, K, D] -> Reshape [B * T, K, D] -> MultiHeadAttention -> Reshape [B, T, K, D].
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

    @typechecked
    def forward(self, x: Float[torch.Tensor, "B T K D"]) -> Float[torch.Tensor, "B T K D"]:
        B, T, K, D = x.shape
        x_flat = x.reshape(B * T, K, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        return attn_out.reshape(B, T, K, D)


class SlotTransitionMLP(nn.Module):
    """
    Per-slot latent state transition MLP.
    Updates each slot token's feature representation after temporal motion
    and inter-slot interaction reasoning.
    """
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    @typechecked
    def forward(self, x: Float[torch.Tensor, "B T K D"]) -> Float[torch.Tensor, "B T K D"]:
        return self.mlp(x)


class OCVPRollouterLayer(nn.Module):
    """
    OCVP Transformer Layer combining:
      1. LayerNorm -> TemporalSelfAttention -> Residual
      2. LayerNorm -> InteractiveSelfAttention -> Residual
      3. LayerNorm -> SlotTransitionMLP -> Residual
    Supports optional FiLM conditioning on action embeddings.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        condition_mode: str = "none",
        action_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.condition_mode = condition_mode.lower()
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = TemporalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.interactive_attn = InteractiveSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.slot_transition = SlotTransitionMLP(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)

        if self.condition_mode == "film":
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(action_emb_dim, 6 * d_model),
            )
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, "B T K D"],
        action_emb: Float[torch.Tensor, "B T K D_act"] | None = None,
    ) -> Float[torch.Tensor, "B T K D"]:
        if self.condition_mode == "film" and action_emb is not None:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(action_emb).chunk(6, dim=-1)
            x = x + gate_a * self.temporal_attn(self.norm1(x) * (1 + scale_a) + shift_a)
            x = x + self.interactive_attn(self.norm2(x))
            x = x + gate_m * self.slot_transition(self.norm3(x) * (1 + scale_m) + shift_m)
        else:
            x = x + self.temporal_attn(self.norm1(x))
            x = x + self.interactive_attn(self.norm2(x))
            x = x + self.slot_transition(self.norm3(x))
        return x


class OCVPSlotRollouter(nn.Module):
    """
    OCVP Factorized Autoregressive Rollouter for Slot Latents.
    Supports Action-Conditioned OCVP (cOCVP) with 'sum', 'concat', and 'film' modes.
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
        raw_action_dim: int = 0,
        action_embed_dim: int = 64,
        condition_mode: str = "none",
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.history_len = history_len
        self.raw_action_dim = raw_action_dim
        self.action_embed_dim = action_embed_dim
        self.condition_mode = condition_mode.lower()

        if self.raw_action_dim > 0 and self.condition_mode != "none":
            if self.condition_mode in ("sum", "film"):
                self.action_encoder = nn.Sequential(
                    nn.Linear(raw_action_dim, action_embed_dim),
                    nn.SiLU(),
                    nn.Linear(action_embed_dim, d_model if self.condition_mode == "sum" else action_embed_dim),
                )
                self.in_proj = nn.Linear(slot_size, d_model)
            elif self.condition_mode == "concat":
                self.action_encoder = nn.Sequential(
                    nn.Linear(raw_action_dim, action_embed_dim),
                    nn.SiLU(),
                    nn.Linear(action_embed_dim, action_embed_dim),
                )
                self.in_proj = nn.Linear(slot_size + action_embed_dim, d_model)
            else:
                raise ValueError(f"Unsupported condition_mode: '{condition_mode}'")
        else:
            self.action_encoder = None
            self.in_proj = nn.Linear(slot_size, d_model)

        self.layers = nn.ModuleList([
            OCVPRollouterLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                condition_mode=self.condition_mode,
                action_emb_dim=action_embed_dim,
            )
            for _ in range(num_layers)
        ])

        self.enc_t_pe = build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = build_pos_enc(slots_pe, num_slots, d_model)
        self.out_proj = nn.Linear(d_model, slot_size)

    @typechecked
    def forward(
        self,
        x: Float[torch.Tensor, "B H K D"],
        pred_len: int,
        actions: Float[torch.Tensor, "B T_act ActDim"] | None = None,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "B P K D"]:
        """
        Args:
            x: [B, history_len, num_slots, slot_size]
            pred_len: Number of future timesteps to rollout
            actions: Action sequence [B, T_act, raw_action_dim] or None

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.history_len, f"Expected history_len={self.history_len}, got {x.shape[1]}"
        B = x.shape[0]

        act_emb_seq = None
        if self.action_encoder is not None and actions is not None:
            act_emb_seq = self.action_encoder(actions)  # [B, T_act, D_act]

        curr_x = x  # [B, history_len, num_slots, slot_size]
        pred_out = []

        for i in range(pred_len):
            curr_t_len = curr_x.shape[1]
            # The window slides by one frame per step, so the actions that
            # produced the frames in the current window are the slice
            # [i, i + curr_t_len) — not a constant prefix.
            if act_emb_seq is not None and act_emb_seq.shape[1] >= i + curr_t_len:
                a_sub = act_emb_seq[:, i:i + curr_t_len]  # [B, curr_t_len, D_act]
                a_sub_slots = a_sub.unsqueeze(2).repeat(1, 1, self.num_slots, 1)  # [B, curr_t_len, K, D_act]

                if self.condition_mode == "sum":
                    proj_x = self.in_proj(curr_x) + a_sub_slots
                    film_act_emb = None
                elif self.condition_mode == "concat":
                    proj_x = self.in_proj(torch.cat([curr_x, a_sub_slots], dim=-1))
                    film_act_emb = None
                elif self.condition_mode == "film":
                    proj_x = self.in_proj(curr_x)
                    film_act_emb = a_sub_slots
                else:
                    proj_x = self.in_proj(curr_x)
                    film_act_emb = None
            else:
                proj_x = self.in_proj(curr_x)
                film_act_emb = None

            if self.enc_t_pe is not None:
                t_pe = self.enc_t_pe.unsqueeze(2).to(x.device)
                proj_x = proj_x + t_pe[:, :curr_t_len]

            if self.enc_slots_pe is not None:
                slots_pe = self.enc_slots_pe.unsqueeze(1).to(x.device)
                proj_x = proj_x + slots_pe

            layer_out = proj_x
            for layer in self.layers:
                layer_out = layer(layer_out, action_emb=film_act_emb)

            last_timestep_tokens = layer_out[:, -1]
            pred_slots = self.out_proj(last_timestep_tokens)
            pred_out.append(pred_slots)

            curr_x = torch.cat([curr_x[:, 1:], pred_slots.unsqueeze(1)], dim=1)

        return torch.stack(pred_out, dim=1)  # [B, pred_len, K, slot_size]


class SlotFormerModel(nn.Module):
    """
    Combined Stage 2 SlotFormer Model wrapping a frozen Stage 1 slot extractor/decoder
    and a SlotRollouter Transformer (Standard or Action-Conditioned OCVP cOCVP).
    Supports optional INTACT RobotSlotIntentActionActor joint loss training.
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
        rollouter_type: str = "standard",
        raw_action_dim: int = 0,
        action_embed_dim: int = 64,
        condition_mode: str = "none",
        use_intact_actor: bool = False,
        action_loss_weight: float = 1.0,
        robot_slot_idx: int = 0,
    ) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        self.history_len = history_len
        self.rollout_len = rollout_len
        self.loss_decay_factor = loss_decay_factor
        self.use_img_recon_loss = use_img_recon_loss
        self.rollouter_type = rollouter_type.lower()
        self.use_intact_actor = use_intact_actor
        self.action_loss_weight = action_loss_weight

        inner_savi = stage1_model.inner_savi()
        num_slots = inner_savi.num_slots
        slot_dim = inner_savi.slot_size

        self.num_slots = num_slots
        self.slot_dim = slot_dim

        if self.rollouter_type in ("ocvp", "factorized", "ocvp_slotformer", "cocvp"):
            self.rollouter = OCVPSlotRollouter(
                num_slots=num_slots,
                slot_size=slot_dim,
                history_len=history_len,
                t_pe=t_pe,
                slots_pe=slots_pe,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                raw_action_dim=raw_action_dim,
                action_embed_dim=action_embed_dim,
                condition_mode=condition_mode,
            )
        else:
            self.rollouter = SlotRollouter(
                num_slots=num_slots,
                slot_size=slot_dim,
                history_len=history_len,
                t_pe=t_pe,
                slots_pe=slots_pe,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
            )

        if use_intact_actor and raw_action_dim > 0:
            from src.models.intact import INTACT
            self.intact_actor = INTACT(
                slot_dim=slot_dim,
                action_dim=raw_action_dim,
                action_emb_dim=action_embed_dim,
                robot_slot_idx=robot_slot_idx,
            )
        else:
            self.intact_actor = None

        # Freeze Stage 1 model parameters
        for p in self.stage1_model.parameters():
            p.requires_grad = False
        self.stage1_model.eval()

    def train(self, mode: bool = True) -> "SlotFormerModel":
        super().train(mode)
        # Keep Stage 1 model in eval mode always
        self.stage1_model.eval()
        return self

    @typechecked
    def extract_slots(self, video: Float[torch.Tensor, "B T C H W"]) -> Float[torch.Tensor, "B T K D"]:
        """
        Extract per-frame slots for full video [B, T, C, H, W] using Stage 1 model.
        Returns: [B, T, K, D]
        """
        if hasattr(self.stage1_model, "encode_slots"):
            return self.stage1_model.encode_slots(video)

        inner_savi = self.stage1_model.inner_savi()
        if hasattr(inner_savi, "_reset_rnn"):
            inner_savi._reset_rnn()
        post_slots, _ = inner_savi.encode(video)
        return post_slots  # [B, T, K, D]

    def forward(self, batch: dict | Float[torch.Tensor, "B T C H W"]) -> dict[str, Any]:
        """
        Forward pass for Stage 2 training or evaluation.
        """
        if isinstance(batch, torch.Tensor):
            video = batch
            actions = None
        else:
            video = batch["img"]
            actions = batch.get("action", None)

        B, T = video.shape[:2]

        with torch.no_grad():
            gt_all_slots = self.extract_slots(video)  # [B, T, K, D]

        history_slots = gt_all_slots[:, :self.history_len]
        gt_rollout_slots = gt_all_slots[:, self.history_len:self.history_len + self.rollout_len]

        pred_rollout_slots = self.rollouter(history_slots, pred_len=self.rollout_len, actions=actions)

        out_dict = {
            "gt_slots": gt_rollout_slots,
            "pred_slots": pred_rollout_slots,
            "history_slots": history_slots,
            "input_img": video,
        }

        # Calculate INTACT RobotSlotIntentActionActor outputs if enabled
        if self.intact_actor is not None and actions is not None and actions.shape[1] >= 1:
            # Predict action a_0 given transition z_0 (history_slots[:, 0]) -> z_1 (history_slots[:, 1])
            z_curr = gt_all_slots[:, 0]
            z_next = gt_all_slots[:, 1]
            target_act = actions[:, 0]  # Action a_0 that drives z_0 -> z_1
            prev_act = None             # No previous action at t=0

            act_loss_dict = self.intact_actor.action_nll(
                z_curr=z_curr,
                z_next=z_next,
                target_action=target_act,
                prev_action=prev_act,
            )
            out_dict["action_nll_dict"] = act_loss_dict

        if self.use_img_recon_loss or not self.training:
            full_slots = torch.cat([history_slots, pred_rollout_slots], dim=1)
            if hasattr(self.stage1_model, "decode_slots"):
                recon_img, pred_masks = self.stage1_model.decode_slots(full_slots)
            else:
                inner_savi = self.stage1_model.inner_savi()
                slots_flat = full_slots.flatten(0, 1)
                recon_img_flat, _, masks_flat, _ = inner_savi.decode(slots_flat)
                recon_img = recon_img_flat.unflatten(0, (B, full_slots.shape[1]))
                pred_masks = masks_flat.squeeze(2).unflatten(0, (B, full_slots.shape[1]))

            out_dict["recon_img"] = recon_img
            out_dict["pred_masks"] = pred_masks
            out_dict["post_slots"] = full_slots

        return out_dict

    def calc_train_loss(self, out_dict: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate Stage 2 Loss (Slot MSE + optional Action NLL)."""
        from src.losses.slot_losses import SlotMSELoss

        loss_module = SlotMSELoss(
            decay_factor=self.loss_decay_factor,
            action_loss_weight=self.action_loss_weight,
        )
        loss_dict = loss_module(out_dict, batch)
        total_loss = loss_dict["loss"]

        loss_metrics = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}

        if self.use_img_recon_loss and "recon_img" in out_dict:
            video = out_dict["input_img"]
            rollout_recon = out_dict["recon_img"][:, self.history_len:self.history_len + self.rollout_len]
            gt_video_rollout = video[:, self.history_len:self.history_len + self.rollout_len]
            img_loss = F.mse_loss(rollout_recon, gt_video_rollout)
            total_loss = total_loss + img_loss
            loss_metrics["img_mse"] = img_loss.item()
            loss_metrics["loss"] = total_loss.item()

        return total_loss, loss_metrics

