"""
Temporal Slot Contrastive Loss (InfoNCE across video frames).
Enforces slot representation consistency across timesteps (t -> t+1) while repelling other slots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSlotContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, post_slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            post_slots: Tensor of shape [B, T, K, D]
        Returns:
            contrast_loss: Scalar PyTorch tensor
        """
        if post_slots.ndim != 4 or post_slots.shape[1] < 2:
            return torch.tensor(0.0, device=post_slots.device, dtype=post_slots.dtype)

        B, T, K, D = post_slots.shape
        # Normalize slot vectors
        slots_norm = F.normalize(post_slots, p=2, dim=-1)  # [B, T, K, D]

        # Extract adjacent pairs t and t+1
        slots_t = slots_norm[:, :-1].reshape(-1, D)    # [B*(T-1)*K, D]
        slots_next = slots_norm[:, 1:].reshape(-1, D)  # [B*(T-1)*K, D]

        # Positive pair cosine similarities
        pos_sim = (slots_t * slots_next).sum(dim=-1, keepdim=True) / self.temperature  # [N, 1]

        # Similarity matrix against all slots in slots_next
        all_sim = (slots_t @ slots_next.T) / self.temperature  # [N, N]

        # InfoNCE loss
        loss = -pos_sim + torch.logsumexp(all_sim, dim=-1, keepdim=True)
        return loss.mean()
