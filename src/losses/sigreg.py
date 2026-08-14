"""
SIGReg (Sketched Isotropic Gaussian Regularization) — aligned with Le-WorldModel.

Ref: Garrido, Balestriero et al., "Learning to Act without Actions" (Le-WM, 2024).
     https://le-wm.github.io/

Evaluates Cramér-Wold random 1-D projections and the Epps-Pulley empirical
characteristic function (ECF) test against N(0, I) per slot independently.
"""

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer for Slot Latents [B, T, K, D].

    Maps slot latents (B, T, K, D) -> (K, T, B, D), keeping each slot's
    distribution independent while applying LeWM's ECF test across batch B.
    """

    def __init__(
        self,
        weight: float = 1.0,
        num_proj: int = 1024,
        knots: int = 17,
        t_max: float = 3.0,
    ):
        super().__init__()
        self.weight = weight
        self.num_proj = num_proj
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, out, batch=None):
        if isinstance(out, dict):
            z = out.get("post_slots", out.get("slots", out.get("z")))
        else:
            z = out

        if z is None or not torch.is_tensor(z) or z.numel() == 0:
            return torch.tensor(0.0), {"sigreg_loss": 0.0}

        # Canonical slot latents (B, T, K, D) -> (K, T, B, D)
        B, T, K, D = z.shape
        proj = z.float().permute(2, 1, 0, 3)  # (K, T, B, D)

        if B < 2:
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        # Sample random unit projections
        A = torch.randn(D, self.num_proj, device=z.device)
        A = A / A.norm(p=2, dim=0).clamp_min(1e-8)

        # Compute Epps-Pulley empirical characteristic function statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (K, T, B, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # (K, T, num_proj, knots)
        statistic = (err @ self.weights) * B  # (K, T, num_proj)

        per_slot = statistic.mean(dim=(-2, -1))  # (K,) - average over time T and projections
        raw_loss = per_slot.mean()
        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        weighted_loss = self.weight * raw_loss

        # Per-slot diagnostic breakdown
        info = {"sigreg_loss": raw_loss.item()}
        for k in range(K):
            info[f"sigreg_slot{k}"] = per_slot[k].item()

        return weighted_loss, info
