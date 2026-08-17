"""
SIGReg (Sketched Isotropic Gaussian Regularization) — aligned with Le-WorldModel.

Ref: Garrido, Balestriero et al., "Learning to Act without Actions" (Le-WM, 2024).
     https://le-wm.github.io/

Evaluates Cramér-Wold random 1-D projections and the Epps-Pulley empirical
characteristic function (ECF) test against N(0, I) per slot independently.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Union
from jaxtyping import Float

from src.utils.tensor_checks import check_tensor_shape, typechecked


@typechecked
def compute_sigreg_statistic(z: Float[torch.Tensor, "B T K D"], num_proj: int,
                             knots: int = 17, t_max: float = 3.0,
                             seed: Optional[int] = None) -> Float[torch.Tensor, "K"]:
    """
    Per-slot Epps-Pulley ECF statistic — the single SIGReg core.

    One input convention: slot latents ``[B, T, K, D]``. Returns the per-slot
    statistic as a ``(K,)`` tensor (lower = closer to N(0, I)). ``seed`` pins
    the random projections for reproducibility (metrics / tests); ``None``
    resamples them per call.
    """
    B, T, K, D = z.shape
    proj = z.float().permute(2, 1, 0, 3)  # (K, T, B, D)

    t = torch.linspace(0, t_max, knots, dtype=torch.float32, device=z.device)
    dt = t_max / (knots - 1)
    weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=z.device)
    weights[[0, -1]] = dt
    phi = torch.exp(-t.square() / 2.0)

    # The statistic is defined in fp32 (Le-WM reference); pin it so the
    # surrounding train-loop autocast cannot silently run it in fp16.
    with torch.autocast(z.device.type, enabled=False):
        # Sample random unit projections
        generator = None
        if seed is not None:
            generator = torch.Generator(device=z.device).manual_seed(seed)
        A = torch.randn(D, num_proj, device=z.device, generator=generator)
        A = A / A.norm(p=2, dim=0).clamp_min(1e-8)

        # Epps-Pulley empirical characteristic function statistic
        x_t = (proj @ A).unsqueeze(-1) * t  # (K, T, B, num_proj, knots)
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()  # (K, T, num_proj, knots)
        statistic = (err @ (weights * phi)) * B  # (K, T, num_proj)

        return statistic.mean(dim=(-2, -1))  # (K,) - mean over time and projections


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
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.num_proj = num_proj
        self.knots = knots
        self.t_max = t_max
        self.seed = seed
        self._warned_small_batch = False

    def forward(self, out: Union[dict, torch.Tensor, None], batch: Any = None) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(out, dict):
            z = out["post_slots"]
        else:
            z = out

        if z is None:
            return torch.tensor(0.0), {"sigreg_loss": 0.0}

        check_tensor_shape(z, "post_slots", ndim=4)
        if z.numel() == 0:
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}
        if z.shape[0] < 2:
            if not self._warned_small_batch:
                self._warned_small_batch = True
                print(f"[SIGReg] Warning: batch_size={z.shape[0]} < 2 — "
                      "the statistic is undefined; loss set to 0.0.")
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        per_slot = compute_sigreg_statistic(
            z, self.num_proj, knots=self.knots, t_max=self.t_max, seed=self.seed)
        raw_loss = per_slot.mean()
        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            return torch.tensor(0.0, device=z.device), {"sigreg_loss": 0.0}

        weighted_loss = self.weight * raw_loss

        # Per-slot diagnostic breakdown
        info = {"sigreg_loss": raw_loss.item()}
        for k in range(z.shape[2]):
            info[f"sigreg_slot{k}"] = per_slot[k].item()

        return weighted_loss, info
