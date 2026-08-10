"""
SIGReg (Sketched Isotropic Gaussian Regularization) Loss Module from LeWorldModel (Le-WM).
Ref: Garrido et al., 2024 / LeCun et al. (https://le-wm.github.io/)

Uses Cramér-Wold 1D random projections and Epps-Pulley empirical characteristic function testing
against standard Gaussian N(0, I) to prevent latent representation collapse in self-supervised learning.
"""

import math
import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """
    SIGReg Loss Module.
    """

    def __init__(self, weight: float = 1.0, sketch_dim: int = 64, num_points: int = 17, t_max: float = 5.0):
        super().__init__()
        self.weight = weight
        self.sketch_dim = sketch_dim
        self.num_points = num_points
        self.t_max = t_max
        self._initialized = False

    def _lazy_init(self, D: int, device, dtype):
        A = torch.randn(D, self.sketch_dim, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
        self.register_buffer('_A', A)

        t = torch.linspace(-self.t_max, self.t_max, self.num_points, device=device, dtype=dtype)
        self.register_buffer('_t', t)
        exp_f = torch.exp(-0.5 * (t ** 2))
        self.register_buffer('_exp_f', exp_f)

        self._initialized = True

    def forward(self, out, batch=None):
        if isinstance(out, dict):
            z = out.get("post_slots", out.get("slots"))
            if z is None:
                device = out.get("recon_img", out.get("input_img")).device if out else "cpu"
                return torch.tensor(0.0, device=device), 0.0
        else:
            z = out

        if z.ndim > 2:
            z = z.reshape(-1, z.shape[-1])

        N, D = z.shape
        if N <= 1:
            raw_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            return self.weight * raw_loss, 0.0

        if not self._initialized or self._A.shape[0] != D:
            self._lazy_init(D, z.device, z.dtype)

        mean = z.mean(dim=0, keepdim=True)
        var = torch.var(z, dim=0, unbiased=False, keepdim=True)
        std = torch.sqrt(var + 1e-5)
        z_norm = (z - mean) / std

        proj = z_norm @ self._A

        args = proj.unsqueeze(2) * self._t.view(1, 1, -1)
        ecf_real = torch.cos(args).mean(dim=0)
        ecf_imag = torch.sin(args).mean(dim=0)

        diff_sq = (ecf_real - self._exp_f.unsqueeze(0)) ** 2 + (ecf_imag) ** 2
        err = diff_sq * self._exp_f.unsqueeze(0)

        raw_loss = torch.trapz(err, self._t, dim=1).mean() * N
        weighted_loss = self.weight * raw_loss
        return weighted_loss, raw_loss.item()
