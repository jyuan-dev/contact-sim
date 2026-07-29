"""
Slot-PIDM: Slot Attention + Predictive Inverse Dynamics Model
with Iterative Relational Interaction Architecture.

Architecture:
- StoSAVi / Slot Attention extracts object slots S_t in R^{K x D_slot} (D_slot = 128).
- Direct Slot Input: D_model = D_slot = 128 (no linear projector needed).
- Slot 0: Explicitly bound to Agent (Pusher / End-Effector).
- Forward Dynamics Module: Iterative Relational Interaction (L=2 iterations)
  - Self-Dynamics & Action Fusion on Agent Slot 0
  - Inter-Slot Collision Cross-Attention (Agent Slot 0 <-> Passive Object Slots)
- Inverse Dynamics Module:
  - Predicts action a_t from Agent Slot transition delta (s_{t+1, 0} - s_{t, 0})
    conditioned on target object slot context via Cross-Attention.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── StoSAVi & Slot Attention Imports ─────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLOTFORMER = os.path.join(REPO_ROOT, 'third_party', 'slotformer')

for p in [REPO_ROOT, SLOTFORMER, os.path.join(SLOTFORMER, 'slotformer', 'base_slots', 'models')]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from slotformer.base_slots.models.savi import StoSAVi
except ImportError:
    StoSAVi = None


# ── 1. Iterative Relational Interaction Block (Forward Dynamics) ──────────────
class IterativeSlotInteractionBlock(nn.Module):
    """
    Refines object slots across L iterations by alternating:
    1. Kinematic / Self-Dynamics update (integrating action a_t into Agent Slot 0).
    2. Inter-Slot Collision Cross-Attention (computing contact forces between slots).
    """
    def __init__(
        self,
        d_model: int = 128,
        action_dim: int = 2,
        num_heads: int = 4,
        num_iterations: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.num_iterations = num_iterations

        # Action fusion MLP for Agent Slot 0
        self.action_fusion = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Self-Dynamics Multi-Head Attention (over slots)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(d_model)

        # Inter-Slot Cross-Attention (Contact & Collision Physics)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)

        # FeedForward Refinement Network
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, slots_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots_t: Tensor of shape [B, K, D_model] or [B, 1, K, D_model]
            action_t: Tensor of shape [B, action_dim]
        Returns:
            slots_next_pred: Tensor of shape [B, K, D_model]
        """
        if slots_t.ndim == 4:
            slots_t = slots_t.squeeze(1)

        B, K, D = slots_t.shape

        slots_curr = slots_t.clone()

        # Step 1: Kinematic / Action Update on Agent Slot 0
        agent_slot = slots_curr[:, 0, :]  # [B, D]
        agent_action_cat = torch.cat([agent_slot, action_t], dim=-1)  # [B, D + action_dim]
        agent_updated = agent_slot + self.action_fusion(agent_action_cat)
        
        # Reconstruct updated slots state for iteration 0
        slots_curr = slots_curr.clone()
        slots_curr[:, 0, :] = agent_updated

        # Step 2: Iterative Self & Cross-Attention Interaction Loop (L passes)
        for _ in range(self.num_iterations):
            # A. Self-Attention Pass (Kinematics & Inertia update across slots)
            norm_s = self.norm_self(slots_curr)
            self_out, _ = self.self_attn(norm_s, norm_s, norm_s)
            slots_curr = slots_curr + self_out

            # B. Inter-Slot Collision Cross-Attention (Contact interaction)
            norm_c = self.norm_cross(slots_curr)
            cross_out, _ = self.cross_attn(query=norm_c, key=norm_c, value=norm_c)
            slots_curr = slots_curr + cross_out

            # C. FeedForward Refinement & Residual Update
            slots_curr = slots_curr + self.mlp(self.norm_mlp(slots_curr))

        return slots_curr


# ── 2. Inverse Dynamics Module ────────────────────────────────────────────────
class InverseDynamicsBlock(nn.Module):
    """
    Predicts robot action a_t from the Agent Slot transition delta (s_{t+1, 0} - s_{t, 0})
    conditioned on target object slot states S_t via Cross-Attention.
    """
    def __init__(
        self,
        d_model: int = 128,
        action_dim: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # Cross-Attention over object slot context
        self.context_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ctx = nn.LayerNorm(d_model)

        # Action Prediction MLP
        self.action_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim)
        )

    def forward(self, slot_t: torch.Tensor, slot_next: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_t: [B, K, D_model] or [B, T, K, D_model]
            slot_next: [B, K, D_model] or [B, T, K, D_model]
        Returns:
            pred_action: [B, action_dim]
        """
        while slot_t.ndim > 3:
            slot_t = slot_t.select(1, -1)
        while slot_next.ndim > 3:
            slot_next = slot_next.select(1, -1)

        B, K, D = slot_t.shape



        # Agent transition delta
        agent_delta = (slot_next[:, 0:1, :] - slot_t[:, 0:1, :])  # [B, 1, D]

        # Cross-attend agent transition delta over target object slot context
        norm_delta = self.norm_ctx(agent_delta)
        norm_context = self.norm_ctx(slot_t)
        ctx_out, _ = self.context_cross_attn(query=norm_delta, key=norm_context, value=norm_context)

        # Predict action
        action_feat = (agent_delta + ctx_out).squeeze(1)  # [B, D]
        pred_action = self.action_predictor(action_feat)
        return pred_action


# ── 3. Sketch Isotropic Gaussian Regularizer (SIGReg) ─────────────────────────
class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (SIGReg) from LeWorldModel (leWM).
    Encourages predicted slot latents to follow an isotropic Gaussian distribution,
    preventing latent representation collapse during forward dynamics training.
    """
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: Tensor of shape [B, K, D] or [B, D]
        Returns:
            statistic: scalar loss value
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)  # [1, B, D]
        elif proj.dim() == 3:
            B, K, D = proj.shape
            proj = proj.reshape(1, B * K, D)

        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ── 4. CNN Slot Attention Encoder & Spatial Decoder (GPU Visual Processing) ────
class CNNSlotEncoder(nn.Module):
    """
    CNN Feature Extractor + Soft Spatial Position Embedding + Slot Attention Module.
    Extracts K object slots of dimension D_model from raw image tensors [B, C, H, W] on GPU.
    Uses downsampling strides (64x64 -> 16x16) to optimize CUDA memory memory usage.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 128, k_slots: int = 4, num_iters: int = 3):
        super().__init__()
        self.d_model = d_model
        self.k_slots = k_slots
        self.num_iters = num_iters

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),          # 32x32 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),          # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=5, stride=1, padding=2),     # 16x16 -> 16x16
            nn.ReLU()
        )

        self.slots_mu = nn.Parameter(torch.randn(1, 1, d_model))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.norm_inputs = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

        self.project_k = nn.Linear(d_model, d_model, bias=False)
        self.project_v = nn.Linear(d_model, d_model, bias=False)
        self.project_q = nn.Linear(d_model, d_model, bias=False)

        self.gru = nn.GRUCell(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feats = self.cnn(x)  # [B, D, 16, 16]
        feats = feats.permute(0, 2, 3, 1).flatten(1, 2)  # [B, 256, D]
        inputs = self.norm_inputs(feats)

        mu = self.slots_mu.expand(B, self.k_slots, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.k_slots, -1)
        slots = mu + sigma * torch.randn(B, self.k_slots, self.d_model, device=x.device)

        k = self.project_k(inputs)
        v = self.project_v(inputs)

        for _ in range(self.num_iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)

            attn = torch.bmm(q, k.transpose(1, 2)) * (self.d_model ** -0.5)
            attn = F.softmax(attn, dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.bmm(attn, v)

            slots = self.gru(updates.reshape(-1, self.d_model), slots_prev.reshape(-1, self.d_model))
            slots = slots.reshape(B, self.k_slots, self.d_model)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots



class SlotSpatialDecoder(nn.Module):
    """
    Decodes spatial segmentation masks [B, K, H, W] from slot representations [B, K, D] on GPU.
    """
    def __init__(self, d_model: int = 128, out_res: tuple = (64, 64)):
        super().__init__()
        self.d_model = d_model
        self.out_res = out_res
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        B, K, D = slots.shape
        slots_flat = slots.reshape(B * K, D, 1, 1).expand(-1, -1, 16, 16)
        masks = self.decoder(slots_flat)  # [B*K, 1, 64, 64]
        return masks.reshape(B, K, self.out_res[0], self.out_res[1])


# ── 5. Full Slot-PIDM Agent Architecture ──────────────────────────────────────
class SlotPIDMAgent(nn.Module):
    """
    Complete Slot-PIDM model integrating:
    1. Visual Slot Attention Encoder & Spatial Mask Decoder (GPU Visual Processing).
    2. Forward Dynamics (IterativeSlotInteractionBlock).
    3. Inverse Dynamics (InverseDynamicsBlock).
    4. Latent Slot Regularization (SIGReg from leWM).
    """
    def __init__(
        self,
        d_model: int = 128,
        action_dim: int = 2,
        k_slots: int = 4,
        num_heads: int = 4,
        num_iterations: int = 2,
        savi_config: dict = None,
        weight_action_loss: float = 1.0,
        weight_slot_loss: float = 1.0,
        weight_sigreg_loss: float = 0.01,
        weight_mask_loss: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.k_slots = k_slots
        self.weight_action_loss = weight_action_loss
        self.weight_slot_loss = weight_slot_loss
        self.weight_sigreg_loss = weight_sigreg_loss
        self.weight_mask_loss = weight_mask_loss

        # Visual Slot Encoder & Mask Decoder (GPU-native)
        self.slot_encoder = CNNSlotEncoder(in_channels=3, d_model=d_model, k_slots=k_slots)
        self.mask_decoder = SlotSpatialDecoder(d_model=d_model, out_res=(64, 64))

        # Forward Dynamics Module
        self.forward_dynamics = IterativeSlotInteractionBlock(
            d_model=d_model,
            action_dim=action_dim,
            num_heads=num_heads,
            num_iterations=num_iterations
        )

        # Inverse Dynamics Module
        self.inverse_dynamics = InverseDynamicsBlock(
            d_model=d_model,
            action_dim=action_dim,
            num_heads=num_heads
        )

        # SIGReg Latent Regularizer
        self.sigreg = SIGReg()

    def extract_slots(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract object slots from image tensor [B, C, H, W] on GPU.
        """
        if imgs.dim() == 5:
            imgs = imgs[:, -1]
        return self.slot_encoder(imgs)

    def forward(
        self,
        slots_t: torch.Tensor = None,
        slots_next: torch.Tensor = None,
        action_gt: torch.Tensor = None,
        img_t: torch.Tensor = None,
        img_next: torch.Tensor = None,
        gt_masks: torch.Tensor = None
    ):
        """
        Forward pass with support for direct image pair GPU processing or latent slots.
        """
        loss_mask = torch.tensor(0.0, device=action_gt.device if action_gt is not None else torch.device('cpu'))

        # If raw images provided, extract slots on GPU via CNN Slot Encoder
        if img_t is not None and img_next is not None:
            slots_t = self.slot_encoder(img_t)
            slots_next = self.slot_encoder(img_next)

            # Optional mask decoding loss on GPU
            if gt_masks is not None:
                pred_masks_t = self.mask_decoder(slots_t)  # [B, K, H, W]
                # Match against gt_masks [B, M, H, W]
                target_m = gt_masks[:, :self.k_slots]
                if target_m.shape[1] < self.k_slots:
                    pad = torch.zeros(
                        target_m.shape[0], self.k_slots - target_m.shape[1],
                        target_m.shape[2], target_m.shape[3], device=target_m.device
                    )
                    target_m = torch.cat([target_m, pad], dim=1)
                if target_m.shape[-2:] != pred_masks_t.shape[-2:]:
                    target_m = F.interpolate(target_m, size=pred_masks_t.shape[-2:], mode='nearest')
                loss_mask = F.binary_cross_entropy(pred_masks_t, target_m)

        # 1. Forward & Inverse Dynamics Prediction
        if action_gt is None:
            pred_action = self.inverse_dynamics(slots_t, slots_next)
            action_input = pred_action
        else:
            action_input = action_gt
            pred_action = self.inverse_dynamics(slots_t, slots_next)

        pred_slots_next = self.forward_dynamics(slots_t, action_input)

        # 2. Loss Computation
        loss_slot = F.mse_loss(pred_slots_next, slots_next)
        loss_action = torch.tensor(0.0, device=slots_t.device)
        if action_gt is not None:
            loss_action = F.mse_loss(pred_action, action_gt)

        # 3. SIGReg Regularization on Predicted Future Slots
        loss_sigreg = self.sigreg(pred_slots_next)

        loss_total = (
            self.weight_action_loss * loss_action +
            self.weight_slot_loss * loss_slot +
            self.weight_sigreg_loss * loss_sigreg +
            self.weight_mask_loss * loss_mask
        )

        return {
            'pred_action': pred_action,
            'pred_slots_next': pred_slots_next,
            'loss_action': loss_action,
            'loss_slot': loss_slot,
            'loss_sigreg': loss_sigreg,
            'loss_mask': loss_mask,
            'loss_total': loss_total
        }

