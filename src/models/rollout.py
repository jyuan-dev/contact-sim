"""
SlotFormer Autoregressive Future Slot Rollout Module.

Supports both Stage 1 SAVi predictor fallback and Stage 2 SlotFormer Transformer Rollouter.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from src.utils.tensor_checks import typechecked


@typechecked
def predict_slot_rollout(
    wrapper_model: nn.Module,
    video: Float[torch.Tensor, "B T C H W"],
    n_cond_frames: int = 2,
    actions: Float[torch.Tensor, "B Tact ActDim"] | None = None,
    goal_slots: Float[torch.Tensor, "B K D"] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Perform autoregressive future slot rollout.

    Args:
        wrapper_model: StandardizedSAViWrapper or StandardizedSlotFormerWrapper
                       instance.
        video: [B, T, C, H, W] video clip tensor.
        n_cond_frames: Number of initial context frames to condition on (e.g. 2).
        actions: action sequence for action-conditioned rollouters (cOCVP/PIDM),
            [B, T_act, action_dim] or None.
        goal_slots: goal conditioning slots for goal-conditioned rollouts
            (PIDM), [B, num_slots, slot_size] or None.

    Returns:
        Dict containing:
            'input_img': [B, T, C, H, W]
            'pred_masks': [B, T, K, H, W]
            'recon_img': [B, T, C, H, W]
            'post_slots': [B, T, K, D]
            'is_rollout_mask': [T] boolean tensor (True for rollout frames t >= n_cond_frames)
    """
    # Value bounds — the annotations cover type and shape, not the range
    # (beartype treats bool as an int, hence the explicit bool guard).
    if isinstance(n_cond_frames, bool) or not 1 <= n_cond_frames <= video.shape[1]:
        raise ValueError(
            f"n_cond_frames must be in [1, T={video.shape[1]}], got {n_cond_frames!r}")

    model = getattr(wrapper_model, "model", wrapper_model)

    # Check if Stage 2 SlotFormerModel is present
    if hasattr(model, "rollouter") and hasattr(model, "stage1_model"):
        inner_savi = model.stage1_model.inner_savi()
        rollouter = model.rollouter
    else:
        inner_savi = wrapper_model.inner_savi()
        rollouter = None

    device = video.device
    B, T, C, H, W = video.shape
    n_cond_frames = min(n_cond_frames, T)
    rollout_len = T - n_cond_frames

    # 1. Conditioned encoding for frames 0 .. n_cond_frames - 1
    if hasattr(inner_savi, "_reset_rnn"):
        inner_savi._reset_rnn()

    cond_slots_tensor, _ = inner_savi.encode(video[:, :n_cond_frames])  # [B, n_cond_frames, K, D]
    prev_slots = cond_slots_tensor[:, -1]

    # 2. Autoregressive Rollout phase: for frames n_cond_frames .. T - 1
    if rollout_len > 0:
        if rollouter is not None:
            # Use Stage 2 Transformer Rollouter (SlotFormer). Pass the
            # conditioning through when the rollouter accepts it, so
            # action/goal-conditioned models are not silently evaluated
            # on their unconditioned branch.
            roll_kwargs: dict[str, Any] = {"pred_len": rollout_len}
            if hasattr(rollouter.forward, "__code__"):
                varnames = rollouter.forward.__code__.co_varnames
                if actions is not None and "actions" in varnames:
                    roll_kwargs["actions"] = actions
                if goal_slots is not None and "goal_slots" in varnames:
                    roll_kwargs["goal_slots"] = goal_slots
            rollout_slots_tensor = rollouter(cond_slots_tensor, **roll_kwargs)
        else:
            # Fallback to Stage 1 SAVi GRU predictor
            rollout_slots = []
            for t in range(n_cond_frames, T):
                rollout_latents = inner_savi.predictor(prev_slots)
                rollout_slots.append(rollout_latents)
                prev_slots = rollout_latents
            rollout_slots_tensor = torch.stack(rollout_slots, dim=1)

        slots_stacked = torch.cat([cond_slots_tensor, rollout_slots_tensor], dim=1)
    else:
        slots_stacked = cond_slots_tensor

    # 3. Decode all slots back into RGB reconstructions & segmentation masks
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
