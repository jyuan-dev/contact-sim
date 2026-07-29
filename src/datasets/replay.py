"""
src/datasets/replay.py
======================

Reusable replay engine for contact-sim datasets.

Supported datasets
------------------
* **PushT** — HDF5 file, pymunk physics, full implementation.
* **OGBench Cube** — Lance directory, MuJoCo physics, full implementation.
* **LIBERO** — stub (schema TBD).

Each replayer loads episodes and either:
  (a) re-runs the simulator to derive contacts / forces / masks (run_physics=True), or
  (b) reads those quantities from pre-computed enriched keys already in the file
      (run_physics=False).

Usage example (PushT, re-running physics)::

    from src.datasets.replay import PushTReplayer

    replayer = PushTReplayer(
        h5_path="/path/to/pusht_expert_train.h5",
        run_physics=True,
        num_workers=8,
    )
    for ep in replayer.iter_episodes():
        print(ep.episode_idx, ep.frames.shape, ep.contact_pos.shape)

Usage example (OGBench Cube, reading pre-computed data)::

    from src.datasets.replay import OGBenchReplayer

    replayer = OGBenchReplayer(
        lance_path="/path/to/cube_single_multiview_expert.lance",
        run_physics=False,
    )
    for ep in replayer.iter_episodes():
        print(ep.masks['cube_0'].shape)   # (T, H, W)
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Iterator

import h5py
import hdf5plugin
import numpy as np

# ── Locate third-party packages relative to this file ────────────────────────
_DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR      = os.path.dirname(_DATASETS_DIR)
_REPO_ROOT    = os.path.dirname(_SRC_DIR)
_GYM_PUSHT    = os.path.join(_REPO_ROOT, "third_party", "gym-pusht")
_SWM_ROOT     = os.path.join(_REPO_ROOT, "third_party", "stable-worldmodel")


# ════════════════════════════════════════════════════════════════════════════
# EpisodeData
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
# BaseReplayer
# ════════════════════════════════════════════════════════════════════════════

class BaseReplayer(ABC):
    """Abstract base class for dataset replayers.

    Subclasses must implement:
    - ``_load_episode_raw``       — read raw (pixels/state/action) for one episode
    - ``_replay_episode_physics`` — run simulator for one episode → EpisodeData
    - ``_load_episode_enriched``  — read pre-computed enriched data → EpisodeData

    Subclasses may also override:
    - ``_read_episode_count``     — return the number of episodes from any source
                                    (default reads ``ep_len`` from HDF5).

    Parameters
    ----------
    h5_path : str
        Path to the primary data file/directory (HDF5 or Lance).
        Pass an empty string ``""`` when the subclass overrides ``__init__``
        and manages ``episode_indices`` itself.
    run_physics : bool
        If True, re-run the simulator to derive contact/force/mask data.
        If False, read those quantities from pre-existing keys in the file.
    num_workers : int
        Number of parallel workers (Pool size). Use 1 for serial execution.
    episodes : list[int] | None
        Explicit episode indices to replay.  None = all episodes in the file.
    """

    def __init__(
        self,
        h5_path: str,
        run_physics: bool = True,
        num_workers: int = 1,
        episodes: list[int] | None = None,
        unoccluded_masks: bool = False,
    ) -> None:
        self.h5_path     = h5_path
        self.run_physics  = run_physics
        self.num_workers  = num_workers
        self.unoccluded_masks = unoccluded_masks

        n_episodes = self._read_episode_count()
        self.episode_indices: list[int] = (
            list(range(n_episodes)) if episodes is None else list(episodes)
        )

    # ── Extension hook ────────────────────────────────────────────────────

    def _read_episode_count(self) -> int:
        """Return total number of episodes in the dataset.

        Default implementation reads ``ep_len`` from the HDF5 at
        ``self.h5_path``.  Override in subclasses that use a different
        storage format (e.g. Lance).
        """
        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, "r") as f:
            return int(f["ep_len"].shape[0])

    # ── Abstract interface ────────────────────────────────────────────────

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

    # ── Public API ────────────────────────────────────────────────────────

    def iter_episodes(self) -> Iterator[EpisodeData]:
        """Yield one :class:`EpisodeData` per episode, in order.

        Uses ``multiprocessing.Pool`` when ``num_workers > 1``, otherwise
        runs serially (safer for debugging or when the simulator cannot be
        pickled across processes).
        """
        if self.num_workers > 1:
            yield from self._iter_parallel()
        else:
            yield from self._iter_serial()

    # ── Internal helpers ──────────────────────────────────────────────────

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
        # Build picklable arg list; each worker reconstructs its own replayer
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


# ── Module-level worker helpers (must be top-level for pickling) ──────────────

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


# ════════════════════════════════════════════════════════════════════════════
# PushTReplayer
# ════════════════════════════════════════════════════════════════════════════

class PushTReplayer(BaseReplayer):
    """Replayer for the PushT HDF5 dataset (gym_pusht/PushT-v0).

    HDF5 layout expected (raw or enriched input)
    --------------------------------------------
    ep_len    : (N,)            int
    ep_offset : (N,)            int
    pixels    : (total, H, W, 3) uint8
    state     : (total, 7)       float32
    action    : (total, 2)       float32

    Additional keys read when run_physics=False (pre-enriched file)
    ---------------------------------------------------------------
    block_masks      : (total, mask_H, mask_W) uint8
    agent_masks      : (total, mask_H, mask_W) uint8
    goal_masks       : (total, mask_H, mask_W) uint8
    contact_pos      : (total, 2)  float32
    normal_force     : (total, 2)  float32
    frictional_force : (total, 2)  float32

    Parameters
    ----------
    mask_size : tuple[int, int]
        (width, height) of rendered segmentation masks.  Default: (224, 224).
    """

    def __init__(
        self,
        h5_path: str,
        run_physics: bool = True,
        num_workers: int = 1,
        episodes: list[int] | None = None,
        mask_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__(h5_path, run_physics, num_workers, episodes)
        self.mask_size = mask_size

        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(h5_path, "r") as f:
            self._ep_lens = f["ep_len"][:].tolist()
            self._ep_offs = f["ep_offset"][:].tolist()

        # Simulator and tracker are created lazily (or in _worker_init for parallel)
        self._env     = None
        self._tracker = None
        if run_physics and num_workers <= 1:
            self._setup_env()

    # ── Simulator bootstrap ───────────────────────────────────────────────

    def _setup_env(self) -> None:
        """Initialise gym-pusht and the contact tracker (once per process)."""
        import pygame
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

        if _GYM_PUSHT not in sys.path:
            sys.path.insert(0, _GYM_PUSHT)

        import gymnasium as gym
        import gym_pusht  # noqa: F401 — registers the environment

        env = gym.make("gym_pusht/PushT-v0", obs_type="state", block_cog=(0, 0))
        env.reset()
        raw_env = env.unwrapped

        for shape in raw_env.agent.shapes:
            shape.friction = 1.0
        for shape in raw_env.block.shapes:
            shape.friction = 1.0

        tracker = _PushTContactTracker(raw_env.agent, raw_env.block)
        handler = raw_env.space.add_collision_handler(0, 0)
        handler.post_solve = tracker.post_solve

        self._env     = env
        self._tracker = tracker

    # ── Raw data loading ──────────────────────────────────────────────────

    def _load_episode_raw(
        self, ep_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset = int(self._ep_offs[ep_idx])
        length = int(self._ep_lens[ep_idx])
        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, "r") as f:
            frames  = f["pixels"][offset : offset + length]
            states  = f["state"][offset : offset + length].astype(np.float32)
            actions = f["action"][offset : offset + length].astype(np.float32)
        return frames, states, actions

    # ── Physics replay ────────────────────────────────────────────────────

    def _replay_episode_physics(
        self,
        ep_idx: int,
        frames: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> EpisodeData:
        if self._env is None:
            self._setup_env()

        T    = len(states)
        w, h = self.mask_size

        contact_pos      = np.full((T, 2), np.nan, dtype=np.float32)
        normal_force     = np.zeros((T, 2), dtype=np.float32)
        frictional_force = np.zeros((T, 2), dtype=np.float32)
        block_masks      = np.zeros((T, h, w), dtype=np.uint8)
        agent_masks      = np.zeros((T, h, w), dtype=np.uint8)
        goal_masks       = np.zeros((T, h, w), dtype=np.uint8)

        for t in range(T):
            # 1. Step physics and collect impulses
            if t < T - 1:
                _pusht_step_physics(self._env, states[t], actions[t], self._tracker)
                if self._tracker.contact_exists and self._tracker.contact_pts:
                    pts             = self._tracker.contact_pts
                    cx              = sum(p.x for p in pts) / len(pts)
                    cy              = sum(p.y for p in pts) / len(pts)
                    contact_pos[t]  = [cx, cy]
                fn_vec               = -self._tracker.total_normal_impulse / 10.0
                ft_vec               = -self._tracker.total_tangential_impulse / 10.0
                normal_force[t]      = [fn_vec.x, fn_vec.y]
                frictional_force[t]  = [ft_vec.x, ft_vec.y]

            # 2. Position objects and render masks
            raw_env = self._env.unwrapped
            raw_env.block.center_of_gravity = (0, 0)
            raw_env.agent.position = list(states[t][:2])
            raw_env.block.position = list(states[t][2:4])
            raw_env.block.angle    = float(states[t][4])
            raw_env.space.step(raw_env.dt)

            block_masks[t] = _pusht_render_mask(self._env, "block", w, h)
            agent_masks[t] = _pusht_render_mask(self._env, "agent", w, h)
            
            g_mask = _pusht_render_mask(self._env, "goal",  w, h)
            if not self.unoccluded_masks:
                # If camera-visible only (default), the moving block and the agent
                # occlude the goal target underneath them on the screen.
                g_mask_clean = g_mask.copy()
                g_mask_clean[block_masks[t] > 0] = 0
                g_mask_clean[agent_masks[t] > 0] = 0
                goal_masks[t] = g_mask_clean
            else:
                goal_masks[t] = g_mask

        return EpisodeData(
            episode_idx      = ep_idx,
            frames           = frames,
            states           = states,
            actions          = actions,
            contact_pos      = contact_pos,
            normal_force     = normal_force,
            frictional_force = frictional_force,
            masks = {"block": block_masks, "agent": agent_masks, "goal": goal_masks},
        )

    # ── Enriched data loading ─────────────────────────────────────────────

    def _load_episode_enriched(self, ep_idx: int) -> EpisodeData:
        offset = int(self._ep_offs[ep_idx])
        length = int(self._ep_lens[ep_idx])
        sl     = slice(offset, offset + length)

        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, "r") as f:
            frames           = f["pixels"][sl]
            states           = f["state"][sl].astype(np.float32)
            actions          = f["action"][sl].astype(np.float32)
            contact_pos      = f["contact_pos"][sl].astype(np.float32)
            normal_force     = f["normal_force"][sl].astype(np.float32)
            frictional_force = f["frictional_force"][sl].astype(np.float32)
            block_masks      = f["block_masks"][sl]
            agent_masks      = f["agent_masks"][sl]
            goal_masks       = f["goal_masks"][sl]

        return EpisodeData(
            episode_idx      = ep_idx,
            frames           = frames,
            states           = states,
            actions          = actions,
            contact_pos      = contact_pos,
            normal_force     = normal_force,
            frictional_force = frictional_force,
            masks = {"block": block_masks, "agent": agent_masks, "goal": goal_masks},
        )


# ── PushT physics helpers (module-level so Pool can pickle them) ──────────────

class _PushTContactTracker:
    """Accumulates agent–block collision impulses within one physics step group."""

    def __init__(self, agent_body, block_body) -> None:
        self.agent_body = agent_body
        self.block_body = block_body
        self.reset()

    def reset(self) -> None:
        import pymunk
        self.contact_exists           = False
        self.total_normal_impulse     = pymunk.Vec2d(0, 0)
        self.total_tangential_impulse = pymunk.Vec2d(0, 0)
        self.contact_pts: list        = []

    def post_solve(self, arb, space, data) -> None:
        body_a = arb.shapes[0].body
        body_b = arb.shapes[1].body
        is_ab  = (
            (body_a == self.agent_body and body_b == self.block_body)
            or (body_a == self.block_body  and body_b == self.agent_body)
        )
        if not is_ab:
            return

        self.contact_exists = True
        normal      = arb.normal
        impulse_vec = arb.total_impulse
        normal_mag  = impulse_vec.dot(normal)
        normal_imp  = normal * normal_mag
        tang_imp    = impulse_vec - normal_imp

        self.total_normal_impulse     += normal_imp
        self.total_tangential_impulse += tang_imp
        for p in arb.contact_point_set.points:
            self.contact_pts.append((p.point_a + p.point_b) / 2.0)


def _pusht_step_physics(
    env,
    state:   np.ndarray,
    action:  np.ndarray,
    tracker: _PushTContactTracker,
) -> None:
    """Set simulator state, apply action via 10 sub-steps, accumulate impulses."""
    import pymunk

    raw_env = env.unwrapped
    tracker.reset()

    raw_env.block.center_of_gravity = (0, 0)
    raw_env.agent.position          = list(state[:2])
    raw_env.block.position          = list(state[2:4])
    raw_env.block.angle             = float(state[4])
    raw_env.agent.velocity          = pymunk.Vec2d(float(state[5]), float(state[6]))

    target_pos  = state[:2] + action * 30.0
    agent_body  = raw_env.agent

    for _ in range(10):
        acc = (
            raw_env.k_p * (target_pos - agent_body.position)
            + raw_env.k_v * (pymunk.Vec2d(0, 0) - agent_body.velocity)
        )
        agent_body.velocity += acc * 0.01
        raw_env.space.step(0.01)


def _pusht_render_mask(
    env,
    target: str,
    width:  int,
    height: int,
) -> np.ndarray:
    """Render a binary segmentation mask for *target* in {'block','agent','goal'}.

    Returns a (height, width) uint8 array with values 0 or 255.
    """
    import cv2
    import pygame
    import pymunk.pygame_util

    from gym_pusht.envs.pymunk_override import DrawOptions

    raw_env = env.unwrapped
    raw_env.block.center_of_gravity = (0, 0)

    # Save original shape colours
    orig_colors = {shape: shape.color for shape in raw_env.space.shapes}

    screen = pygame.Surface((512, 512))
    screen.fill((0, 0, 0))
    draw_options = DrawOptions(screen)

    if target == "goal":
        goal_body = raw_env.get_goal_pose_body(raw_env.goal_pose)
        for shape in raw_env.block.shapes:
            pts = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            pts = [pymunk.pygame_util.to_pygame(p, screen) for p in pts]
            pts.append(pts[0])
            pygame.draw.polygon(screen, pygame.Color("white"), pts)
    else:
        for shape in raw_env.space.shapes:
            if   target == "agent" and shape.body == raw_env.agent:
                shape.color = pygame.Color("white")
            elif target == "block" and shape.body == raw_env.block:
                shape.color = pygame.Color("white")
            else:
                shape.color = pygame.Color("black")
        raw_env.space.debug_draw(draw_options)

    # Restore colours
    for shape, color in orig_colors.items():
        shape.color = color

    img  = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2))
    img  = cv2.resize(img, (width, height))
    mask = (img[:, :, 0] > 127).astype(np.uint8) * 255
    return mask


# ════════════════════════════════════════════════════════════════════════════
# OGBenchReplayer
# ════════════════════════════════════════════════════════════════════════════

class OGBenchReplayer(BaseReplayer):
    """Replayer for the OGBench Cube dataset stored in Lance format.

    Data source
    -----------
    The OGBench cube dataset is stored as a Lance directory written by
    ``stable_worldmodel.data.formats.lance.LanceWriter``.  Each row is one
    timestep; contiguous rows sharing the same ``episode_idx`` value form an
    episode.  Key columns:

    * ``pixels``         — JPEG-encoded front-camera frame (224×224 RGB)
    * ``pixels_side``    — JPEG-encoded side-camera frame (multiview only)
    * ``state``          — qpos ‖ qvel concatenation, shape ``(nq+nv,)``
    * ``action``         — 7-DoF end-effector delta action, shape ``(7,)``
    * ``episode_idx``    — monotonically increasing episode counter
    * ``step_idx``       — step counter within each episode

    Physics (MuJoCo)
    ----------------
    When ``run_physics=True`` the replayer:

    1. Creates a ``CubeEnv`` from the ``stable-worldmodel`` third-party package.
    2. For every timestep, sets ``qpos = state[:nq]`` and ``qvel = state[nq:]``
       then calls ``mujoco.mj_forward`` to re-derive contact data.
    3. Scans ``mjData.contact[0:ncon]`` and identifies contacts whose pair of
       geoms involves at least one cube body.  Per-contact forces are extracted
       via ``mujoco.mj_contactForce``.
    4. Renders segmentation masks for each cube, the gripper, and the background
       using MuJoCo's ``mjtFramebuffer.mjFB_OFFSCREEN`` + segmentation flag.

    Enriched data
    -------------
    When ``run_physics=False`` the replayer reads pre-computed keys from a
    second Lance directory (``lance_path`` with ``_enriched`` suffix by
    convention) or an HDF5 file, depending on the pipeline configuration.
    Currently the enriched reader expects an HDF5 file at ``h5_path``.

    Parameters
    ----------
    lance_path : str
        Path to the ``.lance`` directory (or containing directory).
    h5_path : str | None
        Path to an HDF5 file for enriched reads (``run_physics=False``).
        Unused when ``run_physics=True``.
    env_type : str
        Cube environment type: ``'single'``, ``'double'``, …  Defaults to
        ``'single'`` to match the shipped ``cube_single_expert`` dataset.
    run_physics : bool
        See :class:`BaseReplayer`.
    num_workers : int
        See :class:`BaseReplayer`.
    episodes : list[int] | None
        See :class:`BaseReplayer`.
    """

    # Geom-name fragments that belong to the robot arm / gripper.
    _GRIPPER_GEOM_FRAGMENTS = (
        "ur5e", "robotiq", "finger", "gripper", "hand", "pad",
    )

    def __init__(
        self,
        h5_path: str,
        run_physics: bool = True,
        num_workers: int = 1,
        episodes: list[int] | None = None,
        env_type: str = "single",
        lance_path: str | None = None,
        unoccluded_masks: bool = False,
    ) -> None:
        # Ensure stable-worldmodel is importable
        if _SWM_ROOT not in sys.path:
            sys.path.insert(0, _SWM_ROOT)

        self.lance_path = lance_path or h5_path
        self.h5_path    = h5_path
        self.env_type   = env_type
        self.run_physics = run_physics
        self.num_workers = num_workers
        self.unoccluded_masks = unoccluded_masks

        # Load the dataset to discover episode structure.
        self._ds = self._open_dataset(self.lance_path)
        n_episodes = int(len(self._ds.lengths))

        self.episode_indices: list[int] = (
            list(range(n_episodes)) if episodes is None else list(episodes)
        )

        # MuJoCo env (created lazily on first physics call, once per process)
        self._env = None

    # ── Dataset helper ───────────────────────────────────────────────────

    @staticmethod
    def _open_dataset(path: str):
        """Return either a LanceDataset or HDF5Dataset depending on path."""
        import os
        is_h5 = os.path.isfile(path) and (path.endswith(".h5") or path.endswith(".hdf5"))
        if is_h5:
            from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
            return HDF5Dataset(
                path=path,
                frameskip=1,
                num_steps=1,
            )
        else:
            from stable_worldmodel.data.formats.lance import LanceDataset
            # Load all columns; image decoding is handled inside LanceDataset
            return LanceDataset(
                path=path,
                frameskip=1,
                num_steps=1,
            )

    # ── Lazy MuJoCo env ───────────────────────────────────────────────────

    def _get_env(self):
        """Return (and lazily create) the CubeEnv for this process."""
        if self._env is not None:
            return self._env

        import os as _os
        _os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless
        _os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
        env = CubeEnv(
            env_type   = self.env_type,
            ob_type    = "states",
            multiview  = False,
            height     = 224,
            width      = 224,
            mode       = "data_collection",
        )
        env.reset()
        self._env = env
        return env

    # ── Abstract interface ────────────────────────────────────────────────

    def _load_episode_raw(
        self, ep_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (frames, states, actions) for episode *ep_idx* from Lance.

        Returns
        -------
        frames  : (T, H, W, 3)  uint8   — decoded JPEG pixels (front camera)
        states  : (T, nq+nv)    float32 — qpos‖qvel simulator state
        actions : (T, 7)        float32 — 7-DoF end-effector delta actions
        """
        episode = self._ds.load_episode(ep_idx)
        # ``pixels`` may be a torch.Tensor (C,H,W) float or a (H,W,C) uint8
        # depending on LanceDataset's image decoding path.
        import torch
        pix = episode.get("pixels") if isinstance(episode, dict) else None
        if pix is None:
            raise KeyError(
                f"'pixels' column missing in Lance episode {ep_idx}. "
                f"Available keys: {list(episode.keys())}"
            )
        # Normalise to (T, H, W, 3) uint8
        if isinstance(pix, torch.Tensor):
            pix = pix.numpy()
        if pix.ndim == 4 and pix.shape[1] == 3:
            # (T, C, H, W) → (T, H, W, C)
            pix = np.transpose(pix, (0, 2, 3, 1))
        if pix.dtype != np.uint8:
            pix = (pix * 255).clip(0, 255).astype(np.uint8)

        if "state" in episode:
            states = np.asarray(episode["state"], dtype=np.float32)
        elif "proprio" in episode:
            states = np.asarray(episode["proprio"], dtype=np.float32)
        elif "qpos" in episode and "qvel" in episode:
            # Concatenate qpos and qvel to form the state vector
            qpos_vals = np.asarray(episode["qpos"], dtype=np.float32)
            qvel_vals = np.asarray(episode["qvel"], dtype=np.float32)
            if qpos_vals.ndim == 1:
                qpos_vals = qpos_vals[np.newaxis]
            if qvel_vals.ndim == 1:
                qvel_vals = qvel_vals[np.newaxis]
            states = np.concatenate([qpos_vals, qvel_vals], axis=-1)
        else:
            raise KeyError(f"No state or qpos/qvel keys found in episode {ep_idx}. Keys: {list(episode.keys())}")

        if hasattr(states, "numpy"):
            states = states.numpy().astype(np.float32)
        if states.ndim == 1:
            states = states[np.newaxis]

        actions = np.asarray(episode["action"], dtype=np.float32)
        if hasattr(actions, "numpy"):
            actions = actions.numpy().astype(np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis]

        return pix, states, actions

    def _replay_episode_physics(
        self,
        ep_idx: int,
        frames: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> EpisodeData:
        """Run MuJoCo forward-dynamics for each timestep and extract contacts.

        For each step:
        * Set ``qpos = state[:nq]``, ``qvel = state[nq:]`` then call
          ``mujoco.mj_forward`` to sync positions and detect contacts.
        * Scan ``mjData.contact[0:ncon]`` and keep contacts where at least one
          geom belongs to a cube body.  Sum all such contact forces.
        * Render per-cube and gripper binary masks via MuJoCo segmentation.

        Returns
        -------
        EpisodeData with contact_pos, normal_force, frictional_force, masks.
        """
        import mujoco

        env = self._get_env()
        model = env._model
        data  = env._data
        nq    = model.nq
        T     = len(states)

        # Load target positions and yaws to place target/goal block correctly.
        # Fallback to zeros if keys are missing (e.g. in unit tests).
        episode = self._ds.load_episode(ep_idx)
        if "privileged/target_block_pos" in episode:
            target_pos_seq = np.asarray(episode["privileged/target_block_pos"], dtype=np.float32)
        else:
            target_pos_seq = np.zeros((T, 3), dtype=np.float32)

        if "privileged/target_block_yaw" in episode:
            target_yaw_seq = np.asarray(episode["privileged/target_block_yaw"], dtype=np.float32)
        else:
            target_yaw_seq = np.zeros((T, 1), dtype=np.float32)

        # Identify geom IDs belonging to each cube body, target body, and to the gripper.
        num_cubes    = env._num_cubes
        cube_geom_id_sets = [
            set(geom_ids) for geom_ids in env._cube_geom_ids_list
        ]
        target_geom_id_sets = [
            set(geom_ids) for geom_ids in env._cube_target_geom_ids_list
        ]
        gripper_geom_ids  = self._find_gripper_geom_ids(model)

        # Output buffers
        contact_pos       = np.full((T, 3), np.nan, dtype=np.float32)
        normal_force      = np.zeros((T, 3), dtype=np.float32)
        frictional_force  = np.zeros((T, 3), dtype=np.float32)
        # One mask per cube + one per target + one for gripper, shape (T, H, W)
        H, W = 224, 224
        cube_masks    = [np.zeros((T, H, W), dtype=np.uint8) for _ in range(num_cubes)]
        target_masks  = [np.zeros((T, H, W), dtype=np.uint8) for _ in range(num_cubes)]
        gripper_masks = np.zeros((T, H, W), dtype=np.uint8)

        force_buf = np.zeros(6, dtype=np.float64)  # mj_contactForce output

        for t in range(T):
            # ── Set state ─────────────────────────────────────────────────
            state = states[t].astype(np.float64)
            if len(state) >= nq:
                data.qpos[:] = state[:nq]
                data.qvel[:] = state[nq: nq + model.nv]
            else:
                data.qpos[:len(state)] = state

            # ── Set target mocap pos / quat ───────────────────────────────
            if hasattr(env, "_cube_target_mocap_ids") and env._cube_target_mocap_ids:
                from ogbench.manipspace import lie
                target_pos = target_pos_seq[t]
                target_yaw = target_yaw_seq[t][0]
                target_quat = lie.SO3.from_z_radians(target_yaw).wxyz.tolist()

                target_block_idx = getattr(env, "_target_block", 0)
                if isinstance(target_block_idx, int) and 0 <= target_block_idx < len(env._cube_target_mocap_ids):
                    mocap_id = env._cube_target_mocap_ids[target_block_idx]
                    data.mocap_pos[mocap_id] = target_pos
                    data.mocap_quat[mocap_id] = target_quat

                    # Enable visual target geom visibility
                    if target_block_idx < len(env._cube_target_geom_ids_list):
                        for gid in env._cube_target_geom_ids_list[target_block_idx]:
                            model.geom(gid).rgba[3] = 0.2

            mujoco.mj_forward(model, data)

            # ── Extract contacts ──────────────────────────────────────────
            # Accumulate forces from contacts involving any cube geom.
            step_normal     = np.zeros(3, dtype=np.float64)
            step_frictional = np.zeros(3, dtype=np.float64)
            step_pos        = np.full(3, np.nan)
            n_cube_contacts = 0

            for c_idx in range(data.ncon):
                contact = data.contact[c_idx]
                g1, g2  = int(contact.geom1), int(contact.geom2)

                # Check if this contact involves a cube geom
                cube_involved = None
                for ci, geom_set in enumerate(cube_geom_id_sets):
                    if g1 in geom_set or g2 in geom_set:
                        cube_involved = ci
                        break
                if cube_involved is None:
                    continue  # not a cube contact

                # Get 6-DoF contact force in contact frame
                mujoco.mj_contactForce(model, data, c_idx, force_buf)
                # force_buf layout: [normal, frictional_x, frictional_y,
                #                    torsional, rolling_x, rolling_y]
                step_normal     += force_buf[:3]
                step_frictional += force_buf[3:6]
                step_pos         = contact.pos.copy()  # last contact pos
                n_cube_contacts += 1

            if n_cube_contacts > 0:
                contact_pos[t]      = step_pos.astype(np.float32)
                normal_force[t]     = step_normal.astype(np.float32)
                frictional_force[t] = step_frictional.astype(np.float32)

            # ── Render segmentation masks ─────────────────────────────────
            seg = self._render_segmentation(env, model, data, H, W)
            for ci in range(num_cubes):
                if self.unoccluded_masks:
                    cube_masks[ci][t] = self._render_unoccluded_mask(
                        env, model, data, H, W,
                        target_geom_ids=cube_geom_id_sets[ci],
                        occluder_geom_ids=gripper_geom_ids,
                    )
                else:
                    cube_masks[ci][t] = self._seg_to_mask(
                        seg, model, cube_geom_id_sets[ci]
                    )
                target_masks[ci][t] = self._seg_to_mask(
                    seg, model, target_geom_id_sets[ci]
                )
            
            gripper_mask_raw = self._seg_to_mask(seg, model, gripper_geom_ids)
            # Ensure grasped cube is clean from gripper mask
            gripper_mask_clean = gripper_mask_raw.copy()
            gripper_mask_clean[cube_masks[0][t] > 0] = 0
            gripper_masks[t] = gripper_mask_clean

        masks = {f"cube_{i}": cube_masks[i] for i in range(num_cubes)}
        for i in range(num_cubes):
            masks[f"target_{i}"] = target_masks[i]
        masks["gripper"] = gripper_masks

        return EpisodeData(
            episode_idx      = ep_idx,
            frames           = frames,
            states           = states,
            actions          = actions,
            contact_pos      = contact_pos,
            normal_force     = normal_force,
            frictional_force = frictional_force,
            masks            = masks,
        )

    def _load_episode_enriched(self, ep_idx: int) -> EpisodeData:
        """Read pre-computed enriched data from an HDF5 file.

        Expects an HDF5 file at ``self.h5_path`` with the following layout:
        * ``ep_len``         : (N,)  int
        * ``ep_offset``      : (N,)  int
        * ``pixels``         : (total, H, W, 3)  uint8
        * ``state``          : (total, nq+nv)    float32
        * ``action``         : (total, 7)        float32
        * ``contact_pos``    : (total, 3)        float32  — NaN = no contact
        * ``normal_force``   : (total, 3)        float32
        * ``frictional_force``: (total, 3)       float32
        * ``cube_0_masks``   : (total, H, W)     uint8   — repeat per cube
        * ``target_0_masks`` : (total, H, W)     uint8   — repeat per target
        * ``gripper_masks``  : (total, H, W)     uint8
        """
        if not self.h5_path:
            raise RuntimeError(
                "OGBenchReplayer._load_episode_enriched(): no h5_path provided. "
                "Pass h5_path= to the constructor, or use run_physics=True."
            )
        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, "r") as f:
            ep_len = int(f["ep_len"][ep_idx])
            off    = int(f["ep_offset"][ep_idx])
            sl     = slice(off, off + ep_len)

            frames           = f["pixels"][sl][:]
            states           = f["state"][sl][:].astype(np.float32)
            actions          = f["action"][sl][:].astype(np.float32)
            contact_pos      = f["contact_pos"][sl][:].astype(np.float32)
            normal_force     = f["normal_force"][sl][:].astype(np.float32)
            frictional_force = f["frictional_force"][sl][:].astype(np.float32)

            masks: dict[str, np.ndarray] = {}
            for key in f.keys():
                if key.endswith("_masks"):
                    masks[key[: -len("_masks")]] = f[key][sl][:]

        return EpisodeData(
            episode_idx      = ep_idx,
            frames           = frames,
            states           = states,
            actions          = actions,
            contact_pos      = contact_pos,
            normal_force     = normal_force,
            frictional_force = frictional_force,
            masks            = masks,
        )

    # ── MuJoCo helpers ────────────────────────────────────────────────────

    def _find_gripper_geom_ids(self, model, gripper_only: bool = False) -> set[int]:
        """Return the set of MuJoCo geom IDs that belong to the gripper/arm.
        
        Parameters
        ----------
        gripper_only : bool, default False
            If True, matches only end-effector gripper geoms (robotiq, finger, pad, gripper),
            excluding upper arm/forearm links ('ur5e').
        """
        ids: set[int] = set()
        fragments = ("robotiq", "finger", "gripper", "hand", "pad") if gripper_only else self._GRIPPER_GEOM_FRAGMENTS
        for g_id in range(model.ngeom):
            name = model.geom(g_id).name.lower()
            if any(frag in name for frag in fragments):
                ids.add(g_id)
        return ids

    @staticmethod
    def _render_segmentation(
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

        # Channel 0 already contains geom IDs with -1 for background.
        return seg[:, :, 0].copy()

    @staticmethod
    def _render_unoccluded_mask(
        env, model, data, height: int, width: int,
        target_geom_ids: set[int],
        occluder_geom_ids: set[int],
        renderer=None,
    ) -> np.ndarray:
        """Render a binary mask for *target_geom_ids* after hiding *occluder_geom_ids*.

        This temporarily sets the alpha of occluder geoms to 0 so that
        objects behind them (e.g. a cube grasped by the gripper) become
        fully visible in the segmentation.  Original alpha values are
        restored after rendering.

        Returns
        -------
        mask : (H, W) uint8   (255 = target geom visible, 0 = else)
        """
        import mujoco

        # Save and hide occluder geoms
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
            seg = renderer.render()  # (H, W, 2) int32
            renderer.disable_segmentation_rendering()
            if _own_renderer:
                renderer.close()
        finally:
            # Restore occluder geom alpha
            for gid, alpha in saved_alpha.items():
                model.geom(gid).rgba[3] = alpha

        # Build mask from target geom IDs
        geom_map = seg[:, :, 0]
        mask = np.zeros((height, width), dtype=np.uint8)
        for gid in target_geom_ids:
            mask[geom_map == gid] = 255
        return mask

    @staticmethod
    def _render_isolated_mask(
        env, model, data, height: int, width: int,
        target_geom_ids: set[int],
        hide_geom_ids: set[int] | None = None,
        renderer=None,
        clean_edges: bool = True,
    ) -> np.ndarray:
        """Render an isolated binary mask for *target_geom_ids*.

        Forces target geoms to be 100% opaque (rgba[3] = 1.0) and hides
        *hide_geom_ids* (rgba[3] = 0.0) during the render pass to eliminate
        subpixel edge bleeding, alpha blending, and Z-buffer misclassifications.

        Z-FIGHTING FIX:
        When objects rest on the floor, their bottom face lies at Z=0, coplanar with the floor.
        To prevent ground-level Z-fighting and contact penetration artifacts in OpenGL z-buffer
        rendering, a tiny +1mm (+0.001m) Z-bias is temporarily applied during mask rendering.

        Returns
        -------
        mask : (H, W) uint8   (255 = target geom visible, 0 = else)
        """
        import mujoco

        saved_rgba = {}
        saved_pos = {}
        hide_geom_ids = hide_geom_ids or set()

        # Save and set target geom alpha to opaque (1.0), with +1mm Z-bias to fix ground Z-fighting
        for gid in target_geom_ids:
            saved_rgba[gid] = list(model.geom(gid).rgba)
            saved_pos[gid] = model.geom_pos[gid].copy()
            model.geom(gid).rgba[3] = 1.0
            model.geom_pos[gid][2] += 0.001  # +1mm Z-bias

        # Save and hide occluder / conflicting geoms (0.0)
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
            seg = renderer.render()  # (H, W, 2) int32
            renderer.disable_segmentation_rendering()
            if _own_renderer:
                renderer.close()
        finally:
            # Restore all modified geom RGBA and position values
            for gid, rgba in saved_rgba.items():
                model.geom(gid).rgba = rgba
            for gid, pos in saved_pos.items():
                model.geom_pos[gid] = pos

        # Build clean binary mask
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

    @staticmethod
    def _render_depth_tested_masks(
        env, model, data, height: int, width: int,
        category_geom_dict: dict[str, set[int]],
        renderer=None,
        clean_edges: bool = True,
    ) -> dict[str, np.ndarray]:
        """Segment multiple categories in isolation and resolve pixel ownership via 3D depth testing.

        For each category, renders its isolated segmentation map and depth map.
        Per pixel, the category with the closest camera depth (smallest z-value > 0)
        wins the pixel ownership.

        Z-FIGHTING & NOISE EDGE FIX:
        When objects rest on the floor, Z_bottom == Z_floor == 0.0m (or slight solver penetration),
        causing OpenGL z-buffer Z-fighting at ground contact boundaries. We apply a +1mm (+0.001m) Z-bias
        to target objects during mask rendering to lift bottom surfaces infinitesimally above the floor.
        Additionally, morphological opening and closing (clean_edges=True) removes isolated single-pixel
        rasterization specks and seals micro-gaps along object contours.

        Parameters
        ----------
        category_geom_dict : dict[str, set[int]]
            Mapping of category names (e.g. 'cube', 'gripper', 'target') to geom ID sets.
        renderer : mujoco.Renderer, optional
        clean_edges : bool, default True
            Apply 3x3 morphological opening and closing to remove edge noise specks and pinholes.

        Returns
        -------
        masks : dict[str, np.ndarray]
            Mapping of category names to (H, W) uint8 binary masks (255/0).
        """
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
                    model.geom_pos[gid][2] += 0.001  # +1mm Z-bias to resolve floor Z-fighting
                for gid in hide_gids:
                    if gid not in orig_pos:
                        orig_pos[gid] = model.geom_pos[gid].copy()
                    model.geom_pos[gid] = [999.0, 999.0, 999.0]

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

        depth_stack = np.stack(depth_maps, axis=0)  # (N_cat, H, W)
        min_idx = np.argmin(depth_stack, axis=0)
        min_val = np.min(depth_stack, axis=0)

        masks = {}
        for idx, cat in enumerate(categories):
            m = (min_idx == idx) & (min_val < np.inf)
            mask_uint8 = m.astype(np.uint8) * 255
            if clean_edges:
                import cv2
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # MORPH_OPEN removes isolated noise specks; MORPH_CLOSE seals internal micro-gaps
                m_open = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
                mask_uint8 = cv2.morphologyEx(m_open, cv2.MORPH_CLOSE, kernel)
            masks[cat] = mask_uint8
        return masks

    @staticmethod
    def _seg_to_mask(
        seg: np.ndarray, model, geom_id_set: set
    ) -> np.ndarray:
        """Convert a segmentation map to a binary uint8 mask (255/0)."""
        mask = np.zeros(seg.shape, dtype=np.uint8)
        for g_id in geom_id_set:
            mask[seg == g_id] = 255
        return mask


# ════════════════════════════════════════════════════════════════════════════
# LiberoReplayer  (skeleton — implement once schema is confirmed)
# ════════════════════════════════════════════════════════════════════════════

class LiberoReplayer(BaseReplayer):
    """Replayer for the LIBERO dataset.

    .. note::
        Not yet implemented.  Fill in the three abstract methods once the
        LIBERO HDF5 schema and simulator (RoboSuite / MuJoCo) are integrated.

    Expected HDF5 layout (TBD)
    --------------------------
    ep_len    : (N,)   int
    ep_offset : (N,)   int
    pixels    : (total, H, W, 3) uint8
    state     : (total, state_dim) float32
    action    : (total, action_dim) float32
    """

    def _load_episode_raw(self, ep_idx: int):
        raise NotImplementedError(
            "LiberoReplayer._load_episode_raw(): fill in LIBERO HDF5 key names."
        )

    def _replay_episode_physics(self, ep_idx, frames, states, actions):
        raise NotImplementedError(
            "LiberoReplayer._replay_episode_physics(): integrate RoboSuite / "
            "MuJoCo to derive contacts and forces."
        )

    def _load_episode_enriched(self, ep_idx: int):
        raise NotImplementedError(
            "LiberoReplayer._load_episode_enriched(): map LIBERO enriched "
            "HDF5 keys to EpisodeData fields."
        )
