"""
SIGReg (Sketched Isotropic Gaussian Regularization) — aligned with Le-WorldModel.

Ref: Garrido, Balestriero et al., "Learning to Act without Actions" (Le-WM, 2024).
     https://le-wm.github.io/
"""

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer for Slot Latents [B, T, K, D].

    Maps canonical (B, T, K, D) latents to LeWM (T, B*K, D) and computes the
    Epps-Pulley empirical characteristic function test against N(0, I).
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

        # Fixed slot latent shape: (B, T, K, D) -> (T, B*K, D) matching LeWM
        if z.ndim == 4:
            B, T, K, D = z.shape
            proj = z.float().permute(1, 0, 2, 3).reshape(T, B * K, D)
        elif z.ndim == 3:
            B, K, D = z.shape
            proj = z.float().permute(1, 0, 2)  # (K, B, D)
        elif z.ndim == 2:
            proj = z.float().unsqueeze(0)      # (1, B, D)
        else:
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        if proj.size(-2) < 2:
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        # 1D Random unit projections
        A = torch.randn(proj.size(-1), self.num_proj, device=z.device)
        A = A / A.norm(p=2, dim=0).clamp_min(1e-8)

        # Epps-Pulley empirical characteristic function test
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (T, B*K, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)

        raw_loss = statistic.mean()
        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        weighted_loss = self.weight * raw_loss
        return weighted_loss, {"sigreg_loss": raw_loss.item()}
