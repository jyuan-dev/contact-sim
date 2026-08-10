"""
Reconstruction MSE Loss Module.
Computes MSE between reconstructed images and input images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionMSELoss(nn.Module):
    """
    Reconstruction Mean Squared Error (MSE) Loss.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, out: dict, batch: dict = None) -> tuple[torch.Tensor, float]:
        recon_img = out.get("recon_img")
        input_img = out.get("input_img")

        if recon_img is None or input_img is None:
            device = recon_img.device if recon_img is not None else (input_img.device if input_img is not None else "cpu")
            return torch.tensor(0.0, device=device), 0.0

        raw_loss = F.mse_loss(recon_img, input_img)
        weighted_loss = self.weight * raw_loss
        return weighted_loss, raw_loss.item()
