"""
src/datasets/replay/pusht.py
============================

Replayer for the PushT HDF5 dataset (gym_pusht/PushT-v0).
"""

from __future__ import annotations

import os
import sys
import h5py
import hdf5plugin
import numpy as np

from .base import BaseReplayer, EpisodeData, _GYM_PUSHT


class PushTReplayer(BaseReplayer):
    """Replayer for the PushT HDF5 dataset (gym_pusht/PushT-v0).

    HDF5 layout expected (raw or enriched input)
    --------------------------------------------
    ep_len    : (N,)            int
    ep_offset : (N,)            int
    pixels    : (total, H, W, 3) uint8
    state     : (total, 7)       float32
    action    : (total, 2)       float32

    Parameters
    ----------
    mask_size : tuple[int, int]
        (width, height) of rendered segmentation masks. Default: (224, 224).
    """

    def __init__(
        self,
        h5_path: str,
        run_physics: bool = True,
        num_workers: int = 1,
        episodes: list[int] | None = None,
        mask_size: tuple[int, int] = (224, 224),
        unoccluded_masks: bool = False,
    ) -> None:
        super().__init__(h5_path, run_physics, num_workers, episodes, unoccluded_masks=unoccluded_masks)
        self.mask_size = mask_size

        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(h5_path, "r") as f:
            self._ep_lens = f["ep_len"][:].tolist()
            self._ep_offs = f["ep_offset"][:].tolist()

        self._env     = None
        self._tracker = None
        if run_physics and num_workers <= 1:
            self._setup_env()

    def _setup_env(self) -> None:
        """Initialise gym-pusht and the contact tracker (once per process)."""
        import pygame
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

        if _GYM_PUSHT not in sys.path:
            sys.path.insert(0, _GYM_PUSHT)

        import gymnasium as gym
        import gym_pusht  # noqa: F401

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
    """Render a binary segmentation mask for *target* in {'block','agent','goal'}."""
    import cv2
    import pygame
    import pymunk.pygame_util

    from gym_pusht.envs.pymunk_override import DrawOptions

    raw_env = env.unwrapped
    raw_env.block.center_of_gravity = (0, 0)

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

    for shape, color in orig_colors.items():
        shape.color = color

    img  = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), (1, 0, 2))
    img  = cv2.resize(img, (width, height))
    mask = (img[:, :, 0] > 127).astype(np.uint8) * 255
    return mask
