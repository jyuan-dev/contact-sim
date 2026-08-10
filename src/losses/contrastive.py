"""
Temporal Slot Contrastive Loss (InfoNCE across video frames).
Enforces slot representation consistency across timesteps (t -> t+1) while repelling other slots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSlotContrastiveLoss(nn.Module):
    """
    Temporal Slot Contrastive Loss Module.
    """

    def __init__(self, weight: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(self, out, batch=None):
        if isinstance(out, dict):
            post_slots = out.get("post_slots", out.get("slots"))
            if post_slots is None:
                device = out.get("recon_img", out.get("input_img")).device if out else "cpu"
                return torch.tensor(0.0, device=device), 0.0
        else:
            post_slots = out

        if post_slots.ndim != 4 or post_slots.shape[1] < 2:
            raw_loss = torch.tensor(0.0, device=post_slots.device, dtype=post_slots.dtype)
            return self.weight * raw_loss, 0.0

        B, T, K, D = post_slots.shape
        slots_norm = F.normalize(post_slots, p=2, dim=-1)

        slots_t = slots_norm[:, :-1].reshape(-1, D)
        slots_next = slots_norm[:, 1:].reshape(-1, D)

        pos_sim = (slots_t * slots_next).sum(dim=-1, keepdim=True) / self.temperature
        all_sim = (slots_t @ slots_next.T) / self.temperature

        raw_loss = (-pos_sim + torch.logsumexp(all_sim, dim=-1, keepdim=True)).mean()
        weighted_loss = self.weight * raw_loss
        return weighted_loss, raw_loss.item()
