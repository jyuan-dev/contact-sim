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

        # Normalize latents per dimension
        z_norm = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6)

        # 1. Cramér-Wold Random Projections A: [D, sketch_dim]
        A = torch.randn(D, self.sketch_dim, device=z.device, dtype=z.dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

        # 2. 1D Projections: proj: [N, sketch_dim]
        proj = z_norm @ A

        # 3. Integration Grid t
        t = torch.linspace(-self.t_max, self.t_max, self.num_points, device=z.device, dtype=z.dtype)

        # 4. Theoretical Gaussian Characteristic Function: exp(-0.5 * t^2)
        exp_f = torch.exp(-0.5 * (t ** 2))  # [T_pts]

        # 5. Empirical Characteristic Function (ECF)
        # args: [N, sketch_dim, T_pts]
        args = proj.unsqueeze(2) * t.view(1, 1, -1)
        ecf_real = torch.cos(args).mean(dim=0)  # [sketch_dim, T_pts]
        ecf_imag = torch.sin(args).mean(dim=0)  # [sketch_dim, T_pts]

        # 6. Epps-Pulley Weighted Distance
        diff_sq = (ecf_real - exp_f.unsqueeze(0)) ** 2 + (ecf_imag) ** 2
        err = diff_sq * exp_f.unsqueeze(0)

        # 7. Integrate via trapezoidal rule
        loss = torch.trapz(err, t, dim=1) * N
        return loss.mean()
