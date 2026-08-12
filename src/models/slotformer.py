"""
SlotFormer: Transformer-based Autoregressive Dynamics Model over Slots (Stage 2).

References:
  - SlotFormer: Slot-Based Visual Reasoning and Prediction (Wu et al., NeurIPS 2022)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inner_savi(wrapper_model: nn.Module) -> nn.Module:
    """Helper to unwrap nested model wrappers to obtain core StoSAVi model."""
    model = getattr(wrapper_model, "model", wrapper_model)
    while hasattr(model, "model"):
        model = getattr(model, "model")
    return model


def get_sin_pos_enc(seq_len: int, d_model: int) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, pred_len: int) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class OCVPRollouterLayer(nn.Module):
    """
    OCVP Transformer Layer combining:
      1. LayerNorm -> TemporalSelfAttention -> Residual
      2. LayerNorm -> InteractiveSelfAttention -> Residual
      3. LayerNorm -> SlotTransitionMLP -> Residual
    """
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = TemporalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.interactive_attn = InteractiveSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.slot_transition = SlotTransitionMLP(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Temporal Motion Self-Attention (over T)
        x = x + self.temporal_attn(self.norm1(x))
        # 2. Inter-Slot Interactive Self-Attention (over K)
        x = x + self.interactive_attn(self.norm2(x))
        # 3. Slot Latent State Transition MLP
        x = x + self.slot_transition(self.norm3(x))
        return x


class OCVPSlotRollouter(nn.Module):
    """
    OCVP Factorized Autoregressive Rollouter for Slot Latents.

    Takes past slot tokens [B, history_len, num_slots, slot_size],
    applies spatial-temporal position encodings, and iterates through
    N stacked OCVPRollouterLayers (Temporal Attention -> Interactive Attention -> Slot Transition MLP)
    to autoregressively predict future slot tokens for pred_len steps.
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
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.history_len = history_len

        self.in_proj = nn.Linear(slot_size, d_model)

        self.layers = nn.ModuleList([
            OCVPRollouterLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(num_layers)
        ])

        self.enc_t_pe = build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = build_pos_enc(slots_pe, num_slots, d_model)
        self.out_proj = nn.Linear(d_model, slot_size)

    def forward(self, x: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        Args:
            x: [B, history_len, num_slots, slot_size]
            pred_len: Number of future timesteps to rollout

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.history_len, f"Expected history_len={self.history_len}, got {x.shape[1]}"
        B = x.shape[0]

        curr_x = x  # [B, history_len, num_slots, slot_size]
        pred_out = []

        for _ in range(pred_len):
            proj_x = self.in_proj(curr_x)  # [B, T, K, d_model]

            if self.enc_t_pe is not None:
                t_pe = self.enc_t_pe.unsqueeze(2).to(x.device)  # [1, T, 1, d_model]
                proj_x = proj_x + t_pe

            if self.enc_slots_pe is not None:
                slots_pe = self.enc_slots_pe.unsqueeze(1).to(x.device)  # [1, 1, K, d_model]
                proj_x = proj_x + slots_pe

            layer_out = proj_x
            for layer in self.layers:
                layer_out = layer(layer_out)  # [B, T, K, d_model]

            # Predict next step slots from the last timestep tokens
            last_timestep_tokens = layer_out[:, -1]  # [B, K, d_model]
            pred_slots = self.out_proj(last_timestep_tokens)  # [B, K, slot_size]
            pred_out.append(pred_slots)

            # Shift sequence window: drop oldest frame slots and append newly predicted slots
            curr_x = torch.cat([curr_x[:, 1:], pred_slots.unsqueeze(1)], dim=1)

        return torch.stack(pred_out, dim=1)  # [B, pred_len, K, slot_size]


class SlotFormerModel(nn.Module):
    """
    Combined Stage 2 SlotFormer Model wrapping a frozen Stage 1 slot extractor/decoder
    and a SlotRollouter Transformer (Standard or OCVP Factorized).
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
    ) -> None:
        super().__init__()
        self.stage1_model = stage1_model
        self.history_len = history_len
        self.rollout_len = rollout_len
        self.loss_decay_factor = loss_decay_factor
        self.use_img_recon_loss = use_img_recon_loss
        self.rollouter_type = rollouter_type.lower()

        inner_savi = get_inner_savi(stage1_model)
        if hasattr(inner_savi, "num_slots"):
            num_slots = inner_savi.num_slots
            slot_dim = getattr(inner_savi, "slot_size", getattr(inner_savi, "slot_dim", 64))
        else:
            num_slots = 4
            slot_dim = 64

        self.num_slots = num_slots
        self.slot_dim = slot_dim

        if self.rollouter_type in ("ocvp", "factorized", "ocvp_slotformer"):
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

        # Freeze Stage 1 model parameters
        for p in self.stage1_model.parameters():
            p.requires_grad = False
        self.stage1_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep Stage 1 model in eval mode always
        self.stage1_model.eval()
        return self

    def extract_slots(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame slots for full video [B, T, C, H, W] using Stage 1 model.
        Optimized to batch-encode all T frames through the CNN backbone in a single GPU pass.
        Returns: [B, T, K, D]
        """
        inner_savi = get_inner_savi(self.stage1_model)
        B, T, C, H, W = video.shape

        if hasattr(inner_savi, "_reset_rnn"):
            inner_savi._reset_rnn()

        # Batch-encode all T frames at once to maximize GPU parallelism
        video_flat = video.flatten(0, 1)  # [B*T, C, H, W]
        enc_out_all = inner_savi._get_encoder_out(video_flat)  # [B*T, HW, enc_channels]
        enc_out_all = enc_out_all.unflatten(0, (B, T))  # [B, T, HW, enc_channels]

        init_latents = inner_savi.init_latents.repeat(B, 1, 1)
        prev_slots = None
        all_slots = []

        for t in range(T):
            enc_out_t = enc_out_all[:, t]
            if prev_slots is None:
                latents = init_latents
            else:
                latents = inner_savi.predictor(prev_slots)

            kernel_dist = inner_savi.kernel_dist_layer(latents)
            kernels = inner_savi._sample_dist(kernel_dist)
            post_slots = inner_savi.slot_attention(enc_out_t, kernels)
            all_slots.append(post_slots)
            prev_slots = post_slots

        return torch.stack(all_slots, dim=1)  # [B, T, K, D]

    def forward(self, batch: dict | torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass for Stage 2 training or evaluation.
        """
        if isinstance(batch, torch.Tensor):
            video = batch
        else:
            video = batch["img"]

        B, T = video.shape[:2]

        with torch.no_grad():
            gt_all_slots = self.extract_slots(video)  # [B, T, K, D]

        history_slots = gt_all_slots[:, :self.history_len]
        gt_rollout_slots = gt_all_slots[:, self.history_len:self.history_len + self.rollout_len]

        pred_rollout_slots = self.rollouter(history_slots, pred_len=self.rollout_len)

        out_dict = {
            "gt_slots": gt_rollout_slots,
            "pred_slots": pred_rollout_slots,
            "history_slots": history_slots,
            "input_img": video,
        }

        if self.use_img_recon_loss or not self.training:
            full_slots = torch.cat([history_slots, pred_rollout_slots], dim=1)
            inner_savi = get_inner_savi(self.stage1_model)
            slots_flat = full_slots.flatten(0, 1)
            recon_img_flat, _, masks_flat, _ = inner_savi.decode(slots_flat)

            out_dict["recon_img"] = recon_img_flat.unflatten(0, (B, full_slots.shape[1]))
            out_dict["pred_masks"] = masks_flat.squeeze(2).unflatten(0, (B, full_slots.shape[1]))
            out_dict["post_slots"] = full_slots

        return out_dict

    def calc_train_loss(self, out_dict: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate Stage 2 Slot MSE Loss with temporal decay."""
        gt_slots = out_dict["gt_slots"]      # [B, rollout_len, K, D]
        pred_slots = out_dict["pred_slots"]  # [B, rollout_len, K, D]

        slots_loss = F.mse_loss(pred_slots, gt_slots, reduction="none")

        if self.loss_decay_factor < 1.0:
            w = self.loss_decay_factor ** torch.arange(gt_slots.shape[1], device=gt_slots.device)
            w = w / w.sum() * gt_slots.shape[1]
            slots_loss = slots_loss * w[None, :, None, None]

        slot_recon_loss = slots_loss.mean()
        total_loss = slot_recon_loss

        loss_metrics = {"loss": total_loss.item(), "slot_mse": slot_recon_loss.item()}

        if self.use_img_recon_loss and "recon_img" in out_dict:
            video = out_dict["input_img"]
            rollout_recon = out_dict["recon_img"][:, self.history_len:self.history_len + self.rollout_len]
            gt_video_rollout = video[:, self.history_len:self.history_len + self.rollout_len]
            img_loss = F.mse_loss(rollout_recon, gt_video_rollout)
            total_loss = total_loss + img_loss
            loss_metrics["img_mse"] = img_loss.item()
            loss_metrics["loss"] = total_loss.item()

        return total_loss, loss_metrics
