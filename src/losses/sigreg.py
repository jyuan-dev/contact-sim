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
    def __init__(self, sketch_dim: int = 64, num_points: int = 17, t_max: float = 5.0):
        super().__init__()
        self.sketch_dim = sketch_dim
        self.num_points = num_points
        self.t_max = t_max
        # Buffers lazily initialized on first forward (D is unknown until then).
        self._initialized = False

    def _lazy_init(self, D: int, device, dtype):
        """Register the random projection matrix and integration grid as persistent buffers."""
        # Cramér-Wold random projection matrix [D, sketch_dim]
        A = torch.randn(D, self.sketch_dim, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
        self.register_buffer('_A', A)

        # Integration grid and theoretical Gaussian CF
        t = torch.linspace(-self.t_max, self.t_max, self.num_points, device=device, dtype=dtype)
        self.register_buffer('_t', t)
        exp_f = torch.exp(-0.5 * (t ** 2))
        self.register_buffer('_exp_f', exp_f)

        self._initialized = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape [N, D] or [B, T, K, D]
        Returns:
            sigreg_loss: Scalar PyTorch tensor
        """
        if z.ndim > 2:
            z = z.reshape(-1, z.shape[-1])  # [N, D]

        N, D = z.shape
        if N <= 1:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        if not self._initialized or self._A.shape[0] != D:
            self._lazy_init(D, z.device, z.dtype)

        # Normalize latents per dimension
        z_norm = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6)

        # 1D Projections: [N, sketch_dim]
        proj = z_norm @ self._A

        # Empirical Characteristic Function
        # args: [N, sketch_dim, T_pts]
        args = proj.unsqueeze(2) * self._t.view(1, 1, -1)
        ecf_real = torch.cos(args).mean(dim=0)  # [sketch_dim, T_pts]
        ecf_imag = torch.sin(args).mean(dim=0)

        # Epps-Pulley Weighted Distance
        diff_sq = (ecf_real - self._exp_f.unsqueeze(0)) ** 2 + (ecf_imag) ** 2
        err = diff_sq * self._exp_f.unsqueeze(0)

        # Integrate via trapezoidal rule
        loss = torch.trapz(err, self._t, dim=1) * N
        return loss.mean()
