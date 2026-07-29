"""
src/datasets/replay/mujoco_render.py
====================================

Standalone MuJoCo 3D rendering and segmentation mask helpers.
Supports single-pass depth-sorted rendering, unoccluded rendering,
isolated category depth testing, and morphological edge refinement.
"""

from __future__ import annotations

import numpy as np


def render_segmentation(
    env, model, data, height: int, width: int,
    renderer=None,
    opaque_geom_ids: set[int] | None = None,
) -> np.ndarray:
    """Render a per-pixel geom-ID segmentation map (H, W) int32.

    Parameters
    ----------
    renderer : mujoco.Renderer, optional
        A pre-allocated renderer to reuse.
    opaque_geom_ids : set[int], optional
        Geom IDs (such as semi-transparent target blocks) to temporarily force
        to 100% opaque (rgba[3] = 1.0) so OpenGL depth-sorting correctly resolves them.

    Returns
    -------
    geom_ids : (H, W) int32
        Geom ID per pixel. -1 for sky/background pixels.
    """
    import mujoco

    saved_alpha = {}
    if opaque_geom_ids:
        for gid in opaque_geom_ids:
            saved_alpha[gid] = float(model.geom(gid).rgba[3])
            model.geom(gid).rgba[3] = 1.0

    try:
        _own_renderer = renderer is None
        if _own_renderer:
            renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera="front_pixels")
        renderer.enable_segmentation_rendering()
        seg = renderer.render()  # (H, W, 2) int32: [geom_id, type_id]
        renderer.disable_segmentation_rendering()
        if _own_renderer:
            renderer.close()
    finally:
        if saved_alpha:
            for gid, alpha in saved_alpha.items():
                model.geom(gid).rgba[3] = alpha

    return seg[:, :, 0].copy()


def render_unoccluded_mask(
    env, model, data, height: int, width: int,
    target_geom_ids: set[int],
    occluder_geom_ids: set[int],
    renderer=None,
) -> np.ndarray:
    """Render a binary mask for *target_geom_ids* after hiding *occluder_geom_ids*.

    Returns
    -------
    mask : (H, W) uint8   (255 = target geom visible, 0 = else)
    """
    import mujoco

    saved_alpha = {}
    for gid in occluder_geom_ids:
        saved_alpha[gid] = float(model.geom(gid).rgba[3])
        model.geom(gid).rgba[3] = 0.0

    try:
        _own_renderer = renderer is None
        if _own_renderer:
            renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera="front_pixels")
        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        if _own_renderer:
            renderer.close()
    finally:
        for gid, alpha in saved_alpha.items():
            model.geom(gid).rgba[3] = alpha

    geom_map = seg[:, :, 0]
    mask = np.zeros((height, width), dtype=np.uint8)
    for gid in target_geom_ids:
        mask[geom_map == gid] = 255
    return mask


def render_isolated_mask(
    env, model, data, height: int, width: int,
    target_geom_ids: set[int],
    hide_geom_ids: set[int] | None = None,
    renderer=None,
    clean_edges: bool = False,
) -> np.ndarray:
    """Render an isolated binary mask for *target_geom_ids*.

    Z-FIGHTING FIX:
    When objects rest on the floor, Z_bottom == Z_floor == 0.0m. A tiny +1mm (+0.001m)
    Z-bias is temporarily applied during mask rendering to lift bottom surfaces infinitesimally.
    """
    import mujoco

    saved_rgba = {}
    saved_pos = {}
    hide_geom_ids = hide_geom_ids or set()

    for gid in target_geom_ids:
        saved_rgba[gid] = list(model.geom(gid).rgba)
        saved_pos[gid] = model.geom_pos[gid].copy()
        model.geom(gid).rgba[3] = 1.0
        model.geom_pos[gid][2] += 0.001  # +1mm Z-bias

    for gid in hide_geom_ids:
        if gid not in saved_rgba:
            saved_rgba[gid] = list(model.geom(gid).rgba)
        model.geom(gid).rgba[3] = 0.0

    try:
        _own_renderer = renderer is None
        if _own_renderer:
            renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera="front_pixels")
        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        if _own_renderer:
            renderer.close()
    finally:
        for gid, rgba in saved_rgba.items():
            model.geom(gid).rgba = rgba
        for gid, pos in saved_pos.items():
            model.geom_pos[gid] = pos

    geom_map = seg[:, :, 0]
    mask = np.zeros((height, width), dtype=np.uint8)
    for gid in target_geom_ids:
        mask[geom_map == gid] = 255

    if clean_edges:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def render_depth_tested_masks(
    env, model, data, height: int, width: int,
    category_geom_dict: dict[str, set[int]],
    renderer=None,
    clean_edges: bool = False,
) -> dict[str, np.ndarray]:
    """Segment multiple categories in isolation and resolve pixel ownership via 3D depth testing."""
    import mujoco

    _own_renderer = renderer is None
    if _own_renderer:
        renderer = mujoco.Renderer(model, height=height, width=width)

    all_geoms = set(range(model.ngeom))
    categories = list(category_geom_dict.keys())
    depth_maps = []

    try:
        for cat in categories:
            target_gids = category_geom_dict[cat]
            hide_gids = all_geoms - target_gids

            orig_pos = {}
            saved_rgba = {}
            for gid in target_gids:
                saved_rgba[gid] = float(model.geom(gid).rgba[3])
                model.geom(gid).rgba[3] = 1.0
                orig_pos[gid] = model.geom_pos[gid].copy()
                model.geom_pos[gid][2] += 0.001  # +1mm Z-bias
            for gid in hide_gids:
                saved_rgba[gid] = float(model.geom(gid).rgba[3])
                model.geom(gid).rgba[3] = 0.0

            try:
                renderer.update_scene(data, camera="front_pixels")
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()

                renderer.enable_segmentation_rendering()
                seg = renderer.render()[:, :, 0].copy()
                renderer.disable_segmentation_rendering()
            finally:
                for gid, pos in orig_pos.items():
                    model.geom_pos[gid] = pos
                for gid, alpha in saved_rgba.items():
                    model.geom(gid).rgba[3] = alpha

            cat_mask = np.isin(seg, list(target_gids))
            depth[~cat_mask] = np.inf
            depth_maps.append(depth)
    finally:
        if _own_renderer:
            renderer.close()

    depth_stack = np.stack(depth_maps, axis=0)
    min_idx = np.argmin(depth_stack, axis=0)
    min_val = np.min(depth_stack, axis=0)

    masks = {}
    for idx, cat in enumerate(categories):
        m = (min_idx == idx) & (min_val < np.inf)
        mask_uint8 = m.astype(np.uint8) * 255
        if clean_edges:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m_open = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            mask_uint8 = cv2.morphologyEx(m_open, cv2.MORPH_CLOSE, kernel)
        masks[cat] = mask_uint8
    return masks


def seg_to_mask(seg: np.ndarray, model, geom_id_set: set) -> np.ndarray:
    """Convert a segmentation map to a binary uint8 mask (255/0)."""
    mask = np.zeros(seg.shape, dtype=np.uint8)
    for g_id in geom_id_set:
        mask[seg == g_id] = 255
    return mask
