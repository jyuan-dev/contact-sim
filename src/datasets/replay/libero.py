"""
src/datasets/replay/libero.py
==============================

Replayer stub for LIBERO datasets (schema TBD).
"""

from __future__ import annotations

import numpy as np
from .base import BaseReplayer, EpisodeData


class LiberoReplayer(BaseReplayer):
    """Replayer stub for LIBERO datasets."""

    def _load_episode_raw(
        self, ep_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("LIBERO raw episode loading not yet implemented.")

    def _replay_episode_physics(
        self,
        ep_idx: int,
        frames: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> EpisodeData:
        raise NotImplementedError("LIBERO physics replay not yet implemented.")

    def _load_episode_enriched(self, ep_idx: int) -> EpisodeData:
        raise NotImplementedError("LIBERO enriched episode loading not yet implemented.")
