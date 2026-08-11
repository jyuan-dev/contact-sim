"""
Evaluation Metrics Suite for Unsupervised Object-Centric Slot Models.

Provides quantitative evaluation metrics:
- PSNR (Peak Signal-to-Noise Ratio in dB) [Higher is better ^]
- SSIM (Structural Similarity Index Measure) [Higher is better ^]
- FG-ARI (Foreground Adjusted Rand Index for slot segmentation quality) [Higher is better ^]
- Latent Std (Average slot feature standard deviation across channels, detects representation collapse) [Non-zero / Stable ^]
- SIGReg Stat (Epps-Pulley empirical characteristic function statistic matching standard Gaussian N(0, I)) [Lower is better v]
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score


def compute_psnr(pred_img: torch.Tensor, gt_img: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.

    Formula:
        PSNR = 20 * log10(MAX_VAL) - 10 * log10(MSE)

    Interpretation:
        - Higher values (in dB) indicate better image reconstruction fidelity.
        - Typical values range from 15 dB (early training) to 30+ dB (high visual reconstruction fidelity).

    Args:
        pred_img: Reconstructed tensor of shape [..., C, H, W] in range [0, 1]
        gt_img: Ground-truth tensor of shape [..., C, H, W] in range [0, 1]
        max_val: Maximum dynamic range of images (default 1.0)
    """
    mse = F.mse_loss(pred_img, gt_img).item()
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def compute_ssim(pred_img: torch.Tensor, gt_img: torch.Tensor) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM) between reconstructed and ground truth frames.

    Formula:
        SSIM(x, y) = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

    Interpretation:
        - Evaluates luminance, contrast, and structural correlation between frames.
        - Bounded in range [0.0, 1.0], where 1.0 represents perfect structural match.
    """
    if pred_img.ndim > 4:
        pred_img = pred_img.reshape(-1, *pred_img.shape[-3:])
        gt_img = gt_img.reshape(-1, *gt_img.shape[-3:])

    # Luminance / structural statistics constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(pred_img, kernel_size=3, stride=1, padding=1)
    mu2 = F.avg_pool2d(gt_img, kernel_size=3, stride=1, padding=1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred_img ** 2, kernel_size=3, stride=1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(gt_img ** 2, kernel_size=3, stride=1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(pred_img * gt_img, kernel_size=3, stride=1, padding=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def compute_fg_ari(pred_masks: torch.Tensor, gt_masks: torch.Tensor, bg_threshold: float = 0.05) -> float:
    """
    Computes Foreground Adjusted Rand Index (FG-ARI) evaluating slot object segmentation quality against GT object masks.

    Formula:
        FG-ARI = (Index - ExpectedIndex) / (MaxIndex - ExpectedIndex)

    Interpretation:
        - Standard benchmark metric in unsupervised object-centric learning (Slot Attention / SAVi / DINOSAUR).
        - Evaluates how accurately predicted slot alpha masks partition object pixels, excluding background.
        - Output ranges from 0.0 (0%) to 1.0 (100%), where higher values indicate better object separation.
    """
    # Squeeze channel dimension if 6D [B, T, K, 1, H, W]
    if pred_masks.ndim == 6 and pred_masks.shape[3] == 1:
        pred_masks = pred_masks.squeeze(3)

    if pred_masks.ndim == 5:
        B, T, K, H, W = pred_masks.shape
        pred_masks = pred_masks.reshape(B * T, K, H, W)

    if gt_masks.ndim == 5:
        B, T, M, H, W = gt_masks.shape
        gt_masks = gt_masks.reshape(B * T, M, H, W)

    pred_masks_np = pred_masks.detach().cpu().numpy()
    gt_masks_np = gt_masks.detach().cpu().numpy()

    ari_scores = []
    N_samples = min(pred_masks_np.shape[0], gt_masks_np.shape[0])

    for i in range(N_samples):
        p_mask = pred_masks_np[i]  # [K, H, W]
        g_mask = gt_masks_np[i]    # [M, H, W]

        pred_labels = np.argmax(p_mask, axis=0).flatten()  # [H*W]
        gt_labels = np.argmax(g_mask, axis=0).flatten()    # [H*W]

        # Filter background pixels (where GT background / sum across M objects is 0)
        fg_mask = (g_mask.sum(axis=0).flatten() > bg_threshold)
        if fg_mask.sum() <= 1:
            continue

        ari = adjusted_rand_score(gt_labels[fg_mask], pred_labels[fg_mask])
        ari_scores.append(ari)

    return float(np.mean(ari_scores)) if ari_scores else 0.0


def compute_latent_std(post_slots: torch.Tensor) -> float:
    """
    Computes average standard deviation across slot latent feature dimensions.

    Formula:
        Latent Std = mean_d( std( S[:, :, d] ) )

    Interpretation:
        - Detects representation collapse in self-supervised learning.
        - If all slots collapse to a single constant vector S = c, Latent Std drops to 0.0.
        - Healthy target: non-zero stable value (~0.10 to 1.00).
    """
    if post_slots.ndim > 2:
        post_slots = post_slots.reshape(-1, post_slots.shape[-1])
    std = post_slots.std(dim=0).mean().item()
    return float(std)


def compute_sigreg_stat(post_slots: torch.Tensor, sketch_dim: int = 64) -> float:
    """
    Computes Epps-Pulley empirical characteristic function statistic matching standard
    Gaussian N(0, I) (Le-WM / Garrido et al., 2024).

    Interpretation:
        - Lower values indicate slot representations are more isotropic / Gaussian.
    """
    from src.losses.sigreg import SIGRegLoss
    _, info = SIGRegLoss(num_proj=sketch_dim)(post_slots)
    return float(info["sigreg_loss"]) if isinstance(info, dict) else float(info)


