"""
SIGReg (Sketched Isotropic Gaussian Regularization).

Ref: Garrido, Balestriero et al., "Learning to Act without Actions" (Le-WM, 2024).
     https://le-wm.github.io/

Uses Cramér-Wold random 1-D projections and the Epps-Pulley empirical
characteristic function (ECF) test against N(0, I) to prevent latent collapse.
"""

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) Loss.

    Computes the empirical characteristic function (ECF) distance to N(0, I)
    over random 1D Cramér-Wold projections.

    Supports arbitrary input shapes:
      - 4D (B, T, K, D): Evaluates ECF per slot across B*T samples
      - 3D (B, K, D):    Evaluates ECF per slot across B samples
      - 2D (N, D):       Evaluates ECF across N samples
      - dict:            Extracts 'post_slots', 'slots', 'z', or 'latents'
    """

    def __init__(
        self,
        weight: float = 1.0,
        num_proj: int = 512,
        knots: int = 17,
        t_max: float = 3.0,
    ):
        super().__init__()
        self.weight = weight
        self.num_proj = num_proj
        self.knots = knots
        self.t_max = t_max

        # ── Quadrature knots + trapezoidal weights with Gaussian window ──
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        w[[0, -1]] = dt                               # trapezoidal endpoints
        phi = torch.exp(-0.5 * t.square())            # N(0, 1) characteristic function

        self.register_buffer("_t", t)
        self.register_buffer("_phi", phi)
        self.register_buffer("_weights", w * phi)

    def forward(self, out, batch=None):
        if isinstance(out, dict):
            z = out.get("post_slots", out.get("slots", out.get("z", out.get("latents"))))
        else:
            z = out

        if z is None or not torch.is_tensor(z) or z.numel() == 0 or z.ndim < 2:
            device = z.device if torch.is_tensor(z) else "cpu"
            return torch.tensor(0.0, device=device), {"sigreg_loss": 0.0}

        z = z.float()
        D = z.shape[-1]

        # Standardize shape to (K, N, D)
        if z.ndim == 4:
            B, T, K, _ = z.shape
            z_slots = z.permute(2, 0, 1, 3).reshape(K, B * T, D)
        elif z.ndim == 3:
            B, K, _ = z.shape
            z_slots = z.permute(1, 0, 2)  # (K, B, D)
        else:
            K = 1
            z_slots = z.reshape(1, -1, D)

        K_slots, N, _ = z_slots.shape
        if N < 2:
            if z.numel() // D >= 2:
                z_slots = z.reshape(1, -1, D)
                K_slots, N, _ = z_slots.shape
            else:
                return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        # ── Cramer-Wold 1D Random Projections ─────────────────────────────
        A = torch.randn(D, self.num_proj, device=z.device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp(min=1e-8)

        proj = z_slots @ A                  # (K, N, M)
        x_t = proj.unsqueeze(-1) * self._t  # (K, N, M, Kt)

        # ── Empirical Characteristic Function vs Standard Normal ──────────
        err = (
            (x_t.cos().mean(dim=1) - self._phi).square()
            + x_t.sin().mean(dim=1).square()
        )  # (K, M, Kt)

        # ── Integrate over knots & average over projections ────────────────
        statistic = err @ self._weights     # (K, M)
        per_slot = statistic.mean(dim=-1)   # (K,)

        raw_loss = per_slot.mean()
        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        weighted_loss = self.weight * raw_loss

        # ── Return info dict ──────────────────────────────────────────────
        info = {"sigreg_loss": raw_loss.item()}
        if K_slots > 1:
            for k in range(K_slots):
                info[f"sigreg_slot{k}"] = per_slot[k].item()

        return weighted_loss, info
