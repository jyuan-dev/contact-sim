"""
src/datasets/replay/ogbench.py
==============================

Replayer for the OGBench Cube dataset stored in Lance format.
"""

from __future__ import annotations

import os
import sys
import h5py
import hdf5plugin
import numpy as np

from .base import BaseReplayer, EpisodeData, _SWM_ROOT
from .mujoco_render import (
    render_segmentation,
    render_unoccluded_mask,
    render_isolated_mask,
    render_depth_tested_masks,
    seg_to_mask,
)


class OGBenchReplayer(BaseReplayer):
    """Replayer for the OGBench Cube dataset stored in Lance format.

    Parameters
    ----------
    lance_path : str | None
        Path to the ``.lance`` directory (or containing directory).
    h5_path : str
        Path to an HDF5 file or Lance directory.
    env_type : str
        Cube environment type: ``'single'``, ``'double'``, etc.
    """

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
        if _SWM_ROOT not in sys.path:
            sys.path.insert(0, _SWM_ROOT)

        self.lance_path = lance_path or h5_path
        self.h5_path    = h5_path
        self.env_type   = env_type
        self.run_physics = run_physics
        self.num_workers = num_workers
        self.unoccluded_masks = unoccluded_masks

        self._ds = self._open_dataset(self.lance_path)
        n_episodes = int(len(self._ds.lengths))

        self.episode_indices: list[int] = (
            list(range(n_episodes)) if episodes is None else list(episodes)
        )

        self._env = None

    @staticmethod
    def _open_dataset(path: str):
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
            return LanceDataset(
                path=path,
                frameskip=1,
                num_steps=1,
            )

    def _get_env(self):
        if self._env is not None:
            return self._env

        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

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

    def _load_episode_raw(
        self, ep_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode = self._ds.load_episode(ep_idx)
        import torch
        pix = episode.get("pixels") if isinstance(episode, dict) else None
        if pix is None:
            raise KeyError(
                f"'pixels' column missing in Lance episode {ep_idx}. "
                f"Available keys: {list(episode.keys())}"
            )
        if isinstance(pix, torch.Tensor):
            pix = pix.numpy()
        if pix.ndim == 4 and pix.shape[1] == 3:
            pix = np.transpose(pix, (0, 2, 3, 1))
        if pix.dtype != np.uint8:
            pix = (pix * 255).clip(0, 255).astype(np.uint8)

        if "state" in episode:
            states = np.asarray(episode["state"], dtype=np.float32)
        elif "proprio" in episode:
            states = np.asarray(episode["proprio"], dtype=np.float32)
        elif "qpos" in episode and "qvel" in episode:
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
        import mujoco

        env = self._get_env()
        model = env._model
        data  = env._data
        nq    = model.nq
        T     = len(states)

        episode = self._ds.load_episode(ep_idx)
        if "privileged/target_block_pos" in episode:
            target_pos_seq = np.asarray(episode["privileged/target_block_pos"], dtype=np.float32)
        else:
            target_pos_seq = np.zeros((T, 3), dtype=np.float32)

        if "privileged/target_block_yaw" in episode:
            target_yaw_seq = np.asarray(episode["privileged/target_block_yaw"], dtype=np.float32)
        else:
            target_yaw_seq = np.zeros((T, 1), dtype=np.float32)

        num_cubes    = env._num_cubes
        cube_geom_id_sets = [
            set(geom_ids) for geom_ids in env._cube_geom_ids_list
        ]
        target_geom_id_sets = [
            set(geom_ids) for geom_ids in env._cube_target_geom_ids_list
        ]
        gripper_geom_ids  = self._find_gripper_geom_ids(model)

        contact_pos       = np.full((T, 3), np.nan, dtype=np.float32)
        normal_force      = np.zeros((T, 3), dtype=np.float32)
        frictional_force  = np.zeros((T, 3), dtype=np.float32)
        H, W = 224, 224
        cube_masks    = [np.zeros((T, H, W), dtype=np.uint8) for _ in range(num_cubes)]
        target_masks  = [np.zeros((T, H, W), dtype=np.uint8) for _ in range(num_cubes)]
        gripper_masks = np.zeros((T, H, W), dtype=np.uint8)

        force_buf = np.zeros(6, dtype=np.float64)

        for t in range(T):
            state = states[t].astype(np.float64)
            if len(state) >= nq:
                data.qpos[:] = state[:nq]
                data.qvel[:] = state[nq: nq + model.nv]
            else:
                data.qpos[:len(state)] = state

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

                    if target_block_idx < len(env._cube_target_geom_ids_list):
                        for gid in env._cube_target_geom_ids_list[target_block_idx]:
                            model.geom(gid).rgba[3] = 0.2

            mujoco.mj_forward(model, data)

            step_normal     = np.zeros(3, dtype=np.float64)
            step_frictional = np.zeros(3, dtype=np.float64)
            step_pos        = np.full(3, np.nan)
            n_cube_contacts = 0

            for c_idx in range(data.ncon):
                contact = data.contact[c_idx]
                g1, g2  = int(contact.geom1), int(contact.geom2)

                cube_involved = None
                for ci, geom_set in enumerate(cube_geom_id_sets):
                    if g1 in geom_set or g2 in geom_set:
                        cube_involved = ci
                        break
                if cube_involved is None:
                    continue

                mujoco.mj_contactForce(model, data, c_idx, force_buf)
                step_normal     += force_buf[:3]
                step_frictional += force_buf[3:6]
                step_pos         = contact.pos.copy()
                n_cube_contacts += 1

            if n_cube_contacts > 0:
                contact_pos[t]      = step_pos.astype(np.float32)
                normal_force[t]     = step_normal.astype(np.float32)
                frictional_force[t] = step_frictional.astype(np.float32)

            seg = render_segmentation(env, model, data, H, W)
            for ci in range(num_cubes):
                if self.unoccluded_masks:
                    cube_masks[ci][t] = render_unoccluded_mask(
                        env, model, data, H, W,
                        target_geom_ids=cube_geom_id_sets[ci],
                        occluder_geom_ids=gripper_geom_ids,
                    )
                else:
                    cube_masks[ci][t] = seg_to_mask(
                        seg, model, cube_geom_id_sets[ci]
                    )
                target_masks[ci][t] = seg_to_mask(
                    seg, model, target_geom_id_sets[ci]
                )
            
            gripper_mask_raw = seg_to_mask(seg, model, gripper_geom_ids)
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
        if not self.h5_path:
            raise RuntimeError(
                "OGBenchReplayer._load_episode_enriched(): no h5_path provided."
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

    def _find_gripper_geom_ids(self, model, gripper_only: bool = False) -> set[int]:
        ids: set[int] = set()
        fragments = ("robotiq", "finger", "gripper", "hand", "pad") if gripper_only else self._GRIPPER_GEOM_FRAGMENTS
        for g_id in range(model.ngeom):
            name = model.geom(g_id).name.lower()
            if any(frag in name for frag in fragments):
                ids.add(g_id)
        return ids

    # ── Backward compatibility static methods ──────────────────────────────

    _render_segmentation     = staticmethod(render_segmentation)
    _render_unoccluded_mask  = staticmethod(render_unoccluded_mask)
    _render_isolated_mask    = staticmethod(render_isolated_mask)
    _render_depth_tested_masks = staticmethod(render_depth_tested_masks)
    _seg_to_mask             = staticmethod(seg_to_mask)
