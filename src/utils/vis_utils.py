"""
Visualization & Image Rendering Utilities.
Provides standard color palettes and overlay functions for slot mask visualizations.
"""

import os
import cv2
import numpy as np
from PIL import Image

SLOT_COLORS_RGB = {
    0: (255, 40, 40),     # Slot 0: Red (Agent)
    1: (40, 220, 40),     # Slot 1: Green (T-Block)
    2: (40, 120, 255),    # Slot 2: Blue (Goal Target)
    3: (255, 210, 0),     # Slot 3: Yellow
    4: (230, 40, 230)     # Slot 4: Magenta
}

GT_COLORS_RGB = {
    0: (255, 140, 0),    # Orange
    1: (0, 230, 115),    # Green
    2: (0, 128, 255)     # Blue
}


def render_slot_overlay_frame(
    frame_rgb: np.ndarray,
    pred_masks_t: np.ndarray,
    gt_masks_t: np.ndarray | None = None,
    banner_text: str = "",
) -> np.ndarray:
    """
    Renders a side-by-side composite frame:
      - Left: Original RGB frame with GT contours overlaid.
      - Right: Color-coded slot mask overlay.

    Args:
        frame_rgb: RGB frame array of shape [H, W, 3] in uint8 [0, 255].
        pred_masks_t: Predicted slot masks of shape [K, H, W] float in [0, 1].
        gt_masks_t: Ground-truth masks of shape [M, H, W] float/bool or None.
        banner_text: Text string for overlay top banner.

    Returns:
        combined_large: Resized 480x240 RGB composite frame in uint8.
    """
    if frame_rgb.shape[:2] != (64, 64):
        frame_rgb = cv2.resize(frame_rgb, (64, 64), interpolation=cv2.INTER_LINEAR)

    # 1. Left Panel: Ground Truth Outlines
    p_gt = frame_rgb.copy()
    if gt_masks_t is not None:
        for m_idx in range(gt_masks_t.shape[0]):
            m_bin = gt_masks_t[m_idx] > 0.5
            if m_bin.any():
                contours, _ = cv2.findContours(m_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color_bgr = GT_COLORS_RGB.get(m_idx, (255, 255, 255))
                cv2.drawContours(p_gt, contours, -1, color_bgr, 1)

    # 2. Right Panel: Color-Coded Slot Mask Overlay
    p_slots = frame_rgb.copy().astype(np.float32)
    K = pred_masks_t.shape[0]

    slot_map = np.zeros((64, 64, 3), dtype=np.float32)
    weight_sum = np.zeros((64, 64, 1), dtype=np.float32)
    for k in range(K):
        m_k = np.clip(pred_masks_t[k], 0, 1)[..., None]
        color_k = np.array(SLOT_COLORS_RGB[k % len(SLOT_COLORS_RGB)], dtype=np.float32)
        slot_map += m_k * color_k
        weight_sum += m_k

    weight_sum = np.maximum(weight_sum, 1e-6)
    slot_composite = slot_map / weight_sum
    active_mask = (weight_sum > 0.15)
    alpha = 0.60
    p_slots[active_mask[:, :, 0]] = (1.0 - alpha) * p_slots[active_mask[:, :, 0]] + alpha * slot_composite[active_mask[:, :, 0]]
    p_slots_uint8 = np.clip(p_slots, 0, 255).astype(np.uint8)

    combined = np.hstack([p_gt, p_slots_uint8])
    combined_large = cv2.resize(combined, (480, 240), interpolation=cv2.INTER_NEAREST)

    if banner_text:
        cv2.rectangle(combined_large, (0, 0), (combined_large.shape[1], 18), (30, 140, 220), -1)
        cv2.putText(combined_large, banner_text, (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return combined_large


def save_frames_to_gif(frames: list[np.ndarray], out_gif_path: str, fps: int = 7) -> None:
    """Save list of uint8 RGB frames to infinite looping animated GIF."""
    os.makedirs(os.path.dirname(os.path.abspath(out_gif_path)), exist_ok=True)
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    if pil_frames:
        duration = int(1000 / fps)
        pil_frames[0].save(out_gif_path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0)
