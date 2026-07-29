"""
src/datasets/replay/base.py
===========================

Base data structures and abstract replayer interface for contact-sim dataset replayers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Iterator

import h5py
import hdf5plugin
import numpy as np

# ── Locate third-party packages relative to this file ────────────────────────
_REPLAY_DIR   = os.path.dirname(os.path.abspath(__file__))
_DATASETS_DIR = os.path.dirname(_REPLAY_DIR)
_SRC_DIR      = os.path.dirname(_DATASETS_DIR)
_REPO_ROOT    = os.path.dirname(_SRC_DIR)
_GYM_PUSHT    = os.path.join(_REPO_ROOT, "third_party", "gym-pusht")
_SWM_ROOT     = os.path.join(_REPO_ROOT, "third_party", "stable-worldmodel")


@dataclass
class EpisodeData:
    """All quantities for one replayed episode.

    Shapes
    ------
    frames           : (T, H, W, 3)    uint8
    states           : (T, state_dim)  float32
    actions          : (T, action_dim) float32
    contact_pos      : (T, 2)          float32  — NaN rows indicate no contact
    normal_force     : (T, 2)          float32
    frictional_force : (T, 2)          float32
    masks            : dict  keyed by object name, each (T, mask_H, mask_W) uint8
    """

    episode_idx:      int
    frames:           np.ndarray
    states:           np.ndarray
    actions:          np.ndarray
    contact_pos:      np.ndarray
    normal_force:     np.ndarray
    frictional_force: np.ndarray
    masks:            dict = field(default_factory=dict)


class BaseReplayer(ABC):
    """Abstract base class for dataset replayers.

    Subclasses must implement:
    - ``_load_episode_raw``       — read raw (pixels/state/action) for one episode
    - ``_replay_episode_physics`` — run simulator for one episode → EpisodeData
    - ``_load_episode_enriched``  — read pre-computed enriched data → EpisodeData

    Parameters
    ----------
    h5_path : str
        Path to the primary data file/directory (HDF5 or Lance).
    run_physics : bool
        If True, re-run the simulator to derive contact/force/mask data.
        If False, read those quantities from pre-existing keys in the file.
    num_workers : int
        Number of parallel workers (Pool size). Use 1 for serial execution.
    episodes : list[int] | None
        Explicit episode indices to replay. None = all episodes in the file.
    """

    def __init__(
        self,
        h5_path: str,
        run_physics: bool = True,
        num_workers: int = 1,
        episodes: list[int] | None = None,
        unoccluded_masks: bool = False,
    ) -> None:
        self.h5_path          = h5_path
        self.run_physics      = run_physics
        self.num_workers      = num_workers
        self.unoccluded_masks = unoccluded_masks

        n_episodes = self._read_episode_count()
        self.episode_indices: list[int] = (
            list(range(n_episodes)) if episodes is None else list(episodes)
        )

    def _read_episode_count(self) -> int:
        """Return total number of episodes in the dataset."""
        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, "r") as f:
            return int(f["ep_len"].shape[0])

    @abstractmethod
    def _load_episode_raw(
        self, ep_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (frames [T,H,W,3], states [T,D], actions [T,A]) for ep_idx."""

    @abstractmethod
    def _replay_episode_physics(
        self,
        ep_idx: int,
        frames: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> EpisodeData:
        """Run the simulator and return a fully populated EpisodeData."""

    @abstractmethod
    def _load_episode_enriched(self, ep_idx: int) -> EpisodeData:
        """Read pre-computed enriched keys from the HDF5 and return EpisodeData."""

    def iter_episodes(self) -> Iterator[EpisodeData]:
        """Yield one EpisodeData per episode, in order."""
        if self.num_workers > 1:
            yield from self._iter_parallel()
        else:
            yield from self._iter_serial()

    def _process_one(self, ep_idx: int) -> EpisodeData:
        if self.run_physics:
            frames, states, actions = self._load_episode_raw(ep_idx)
            return self._replay_episode_physics(ep_idx, frames, states, actions)
        else:
            return self._load_episode_enriched(ep_idx)

    def _iter_serial(self) -> Iterator[EpisodeData]:
        for ep_idx in self.episode_indices:
            yield self._process_one(ep_idx)

    def _iter_parallel(self) -> Iterator[EpisodeData]:
        args = [
            (type(self), self.h5_path, self.run_physics, ep_idx)
            for ep_idx in self.episode_indices
        ]
        with Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(type(self), self.h5_path, self.run_physics),
        ) as pool:
            yield from pool.imap(_worker_process, args)


_worker_replayer: BaseReplayer | None = None


def _worker_init(replayer_cls, h5_path: str, run_physics: bool) -> None:
    """Initialise one persistent replayer per worker process."""
    global _worker_replayer
    _worker_replayer = replayer_cls(
        h5_path=h5_path, run_physics=run_physics, num_workers=1
    )


def _worker_process(args: tuple) -> EpisodeData:
    _cls, _path, _phys, ep_idx = args
    assert _worker_replayer is not None, "Worker not initialised."
    return _worker_replayer._process_one(ep_idx)
