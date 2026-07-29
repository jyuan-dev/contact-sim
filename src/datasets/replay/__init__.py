"""
src/datasets/replay package
===========================

Reusable replay engines and utilities for contact-sim datasets.

Replayers available:
- ``BaseReplayer``       (base.py)
- ``EpisodeData``        (base.py)
- ``PushTReplayer``      (pusht.py)
- ``OGBenchReplayer``    (ogbench.py)
- ``LiberoReplayer``     (libero.py)

Rendering helpers:
- ``render_segmentation``
- ``render_unoccluded_mask``
- ``render_isolated_mask``
- ``render_depth_tested_masks``
- ``seg_to_mask``
"""

from .base import BaseReplayer, EpisodeData
from .pusht import PushTReplayer
from .ogbench import OGBenchReplayer
from .libero import LiberoReplayer
from .mujoco_render import (
    render_segmentation,
    render_unoccluded_mask,
    render_isolated_mask,
    render_depth_tested_masks,
    seg_to_mask,
)

__all__ = [
    "BaseReplayer",
    "EpisodeData",
    "PushTReplayer",
    "OGBenchReplayer",
    "LiberoReplayer",
    "render_segmentation",
    "render_unoccluded_mask",
    "render_isolated_mask",
    "render_depth_tested_masks",
    "seg_to_mask",
]
