"""
SlotFormer Autoregressive Future Slot Rollout Module.

Given an input video clip [B, T, C, H, W] and a condition length T_cond (e.g. 2 frames):
1. Runs conditioned visual slot extraction on the first T_cond frames to extract slots z_1, ..., z_{T_cond}.
2. Autoregressively rolls out future slots z_{T_cond+1}, ..., z_T using the SAVi predictor:
     z_{t+1} = predictor(z_t)
   without feeding future image frames into the image encoder.
3. Decodes all slots (conditioned + rolled-out) through inner_savi.decode to produce predicted RGB frames
   x_hat_1, ..., x_hat_T and predicted segmentation masks m_hat_1, ..., m_hat_T.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_slot_rollout(
    wrapper_model: nn.Module,
    video: torch.Tensor,
    n_cond_frames: int = 2,
) -> dict[str, torch.Tensor]:
    """
    Perform autoregressive future slot rollout.

    Args:
        wrapper_model: StandardizedSAViWrapper or StandardizedDeformableSAViWrapper instance.
        video: [B, T, C, H, W] video clip tensor.
        n_cond_frames: Number of initial context frames to condition on (e.g. 2).

    Returns:
        Dict containing:
            'input_img': [B, T, C, H, W]
            'pred_masks': [B, T, K, H, W]
            'recon_img': [B, T, C, H, W]
            'post_slots': [B, T, K, D]
            'is_rollout_mask': [T] boolean tensor (True for rollout frames t >= n_cond_frames)
    """
    model = getattr(wrapper_model, "model", wrapper_model)
    inner_savi = getattr(model, "model", model)

    device = video.device
    B, T, C, H, W = video.shape
    n_cond_frames = min(n_cond_frames, T)

    # 1. Conditioned encoding for frames 0 .. n_cond_frames - 1
    cond_video = video[:, :n_cond_frames]
    all_slots = []
    
    # Handle slot initialization & RNN reset
    inner_savi._reset_rnn()
    init_latents = inner_savi.init_latents.repeat(B, 1, 1)  # [B, K, D]
    prev_slots = None

    # Conditioned phase: encode images and update slots via slot attention
    for t in range(n_cond_frames):
        img_t = cond_video[:, t]  # [B, C, H, W]
        enc_out_t = inner_savi._get_encoder_out(img_t)  # [B, H*W, enc_channels]
        
        if prev_slots is None:
            latents = init_latents
        else:
            latents = inner_savi.predictor(prev_slots)

        kernel_dist = inner_savi.kernel_dist_layer(latents)
        kernels = inner_savi._sample_dist(kernel_dist)
        
        post_slots = inner_savi.slot_attention(enc_out_t, kernels)
        all_slots.append(post_slots)
        prev_slots = post_slots

    # 2. Autoregressive Rollout phase: for frames n_cond_frames .. T - 1
    # No image encoder! Future slots are predicted purely from previous slots via predictor.
    for t in range(n_cond_frames, T):
        rollout_latents = inner_savi.predictor(prev_slots)
        all_slots.append(rollout_latents)
        prev_slots = rollout_latents

    # Stack slots: [B, T, K, D]
    slots_stacked = torch.stack(all_slots, dim=1)  # [B, T, K, D]

    # 3. Decode slots back into RGB reconstructions & segmentation masks using inner_savi.decode
    slots_flat = slots_stacked.flatten(0, 1)  # [B*T, K, D]
    post_recon_img, _, post_masks, _ = inner_savi.decode(slots_flat)

    recon_img = post_recon_img.unflatten(0, (B, T))  # [B, T, C, H, W]
    pred_masks = post_masks.squeeze(2).unflatten(0, (B, T))  # [B, T, K, H, W]

    is_rollout_mask = torch.tensor(
        [t >= n_cond_frames for t in range(T)], device=device, dtype=torch.bool
    )

    return {
        "input_img": video,
        "pred_masks": pred_masks,
        "recon_img": recon_img,
        "post_slots": slots_stacked,
        "is_rollout_mask": is_rollout_mask,
    }
