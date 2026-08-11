"""
SIGReg (Sketched Isotropic Gaussian Regularization) — aligned with Le-WM.

Ref: Garrido, Balestriero et al., LeWorldModel (2024) / LeCun et al.
     https://le-wm.github.io/

Uses Cramér-Wold random 1-D projections and the Epps-Pulley empirical
characteristic function test against N(0, I) to prevent latent collapse.

Implementation follows the reference Le-WM code exactly:
  - Random unit-norm projections resampled fresh every forward pass.
  - ECF mean is taken over the *time* axis, so each (batch, slot) item
    gets its own per-trajectory ECF estimate.
  - Trapezoidal quadrature over [0, 3] with Gaussian window.
  - No per-channel normalization — caller is responsible for ensuring
    latents are approximately zero-mean unit-variance (e.g. via BatchNorm
    or LayerNorm upstream).
"""

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """Sketched Isotropic Gaussian Regularizer.

    Input ``z`` with shape ``(B, T, K, D)`` is rearranged to
    ``(T, B*K, D)`` so that each slot-trajectory is an independent
    "batch element" whose temporal distribution is tested for
    Gaussianity — matching the Le-WM pattern where the ECF mean is
    taken over the time axis.
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

        # ── Quadrature knots + trapezoidal weights with Gaussian window ──
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32)
        w[[0, -1]] = dt                              # trapezoidal endpoints
        window = torch.exp(-0.5 * t.square())         # exp(-t²/2)
        self.register_buffer("_t", t)
        self.register_buffer("_phi", window)
        self.register_buffer("_weights", w * window)

    def forward(self, out, batch=None):
        # ── Extract latents ──────────────────────────────────────────────
        if isinstance(out, dict):
            z = out.get("post_slots", out.get("slots"))
        else:
            z = out

        if z is None:
            return torch.tensor(0.0), 0.0

        if z.ndim < 2:
            return torch.tensor(0.0, device=z.device), 0.0

        with torch.amp.autocast(device_type="cuda", enabled=False):
            z = z.float()

            # Normalise shape to  (T, B, K, D) for per-slot SIGReg.
            # Canonical input from SAVi wrappers:  (B, T, K, D).
            # The metric / ad-hoc callers may pass 2-D or 3-D tensors.
            if z.ndim == 2:
                # [N, D] — treat as (T=N, B=1, K=1)
                z = z.unsqueeze(1).unsqueeze(-1)        # (T, 1, 1, D)
            elif z.ndim == 3:
                # [B, N, D] — treat N as time, K=1
                z = z.permute(1, 0, 2).unsqueeze(2)     # (N, B, 1, D)
            elif z.ndim == 4:
                # [B, T, K, D] — canonical
                z = z.permute(1, 0, 2, 3)               # (T, B, K, D)
            else:
                # Flatten everything except last dim → (T=1, B=1, K=N, D)
                z = z.reshape(-1, z.shape[-1]).unsqueeze(0).unsqueeze(0)
                                                          # (1, 1, N, D)

            T = z.shape[0]
            B = z.shape[1]
            K = z.shape[2]
            D = z.shape[3]

            if T <= 1:
                return torch.tensor(0.0, device=z.device), 0.0

            # ── Per-slot SIGReg ───────────────────────────────────────────
            # Reshape so each slot is an independent batch for projection:
            #   (T, B, K, D) → (T, B*K, D) for the projection
            # then restore K in the ECF computation so each slot's
            # statistic is computed independently.
            z_proj = z.reshape(T, B * K, D)             # pool B*K for projection

            A = torch.randn(D, self.num_proj, device=z.device, dtype=z.dtype)
            A = A.div_(A.norm(p=2, dim=0, keepdim=True))

            proj = z_proj @ A                           # (T, B*K, M)
            proj = proj.view(T, B, K, self.num_proj)    # (T, B, K, M)

            # x_t:  (T, B, K, M, Kt)
            x_t = proj.unsqueeze(-1) * self._t

            # .mean(-4) = mean over B → per-timestep, per-slot ECF
            #   (T, B, K, M, Kt) → (T, K, M, Kt)
            err = (
                (x_t.cos().mean(-4) - self._phi).square()
                + x_t.sin().mean(-4).square()
            )  # (T, K, M, Kt)

            # Integrate over knots, scale by batch size
            #   (T, K, M, Kt) @ (Kt,) → (T, K, M)
            statistic = (err @ self._weights) * B       # (T, K, M)

            # Per-slot breakdown: mean over T and M for each slot
            per_slot = statistic.mean(dim=(0, 2))        # (K,)

            raw_loss = per_slot.mean()                   # scalar: mean over K

            if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                return torch.tensor(0.0, device=z.device), 0.0

            weighted = self.weight * raw_loss

            # Build per-slot info dict for TensorBoard logging
            info = {"sigreg_loss": raw_loss.item()}
            for k in range(K):
                info[f"sigreg_slot{k}"] = per_slot[k].item()

            return weighted, info
