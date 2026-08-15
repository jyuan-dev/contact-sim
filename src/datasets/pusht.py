import os
from typing import Any, Optional

import numpy as np
import h5py

try:
    import hdf5plugin
    HDF5_PLUGIN_PATH = getattr(hdf5plugin, 'PLUGINS_PATH', None)
except ImportError:
    HDF5_PLUGIN_PATH = None

# Set HDF5 plugin path once at module level.
if HDF5_PLUGIN_PATH and os.path.exists(HDF5_PLUGIN_PATH):
    os.environ["HDF5_PLUGIN_PATH"] = HDF5_PLUGIN_PATH

import torch
from torch.utils.data import Dataset

# ── ImageNet Normalization constants ─────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Global zero-copy IPC shared memory cache across DataLoader workers
_SHARED_PIXELS_CACHE = {}


def get_shared_pixels(h5_path: str) -> torch.Tensor:
    if h5_path not in _SHARED_PIXELS_CACHE:
        print(f"[PushTMaskHDF5Dataset] Preloading {h5_path} (1.4 GB) into shared RAM IPC memory...")
        with h5py.File(h5_path, 'r') as f:
            pix = torch.from_numpy(np.asarray(f['pixels']))
            pix.share_memory_()
            _SHARED_PIXELS_CACHE[h5_path] = pix
    return _SHARED_PIXELS_CACHE[h5_path]


# ── Mask-Supervised Dataset (for DETR box / mask tracking) ────────────────────
class PushTMaskHDF5Dataset(Dataset):
    MASK_KEYS = ['agent_masks', 'block_masks', 'goal_masks']

    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        resolution: tuple[int, int] = (64, 64),
        n_sample_frames: int = 6,
        frame_offset: int = 1,
        train_frac: float = 0.9,
        seed: int = 42,
        load_masks: bool = False,
        preload_ram: bool = True,
        include_goal_mask: bool = True,
    ) -> None:
        assert split in ('train', 'val')
        self.h5_path = h5_path
        self.split = split
        self.resolution = resolution
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset
        self.load_masks = load_masks
        self.preload_ram = preload_ram
        self.include_goal_mask = include_goal_mask
        self._h5 = None  # lazy-opened per-worker

        if self.preload_ram:
            get_shared_pixels(self.h5_path)

        with h5py.File(h5_path, 'r') as f:
            self._ep_lens = np.asarray(f['ep_len']).tolist()
            self._ep_offs = np.asarray(f['ep_offset']).tolist()
        n_episodes = len(self._ep_lens)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * train_frac)

        if split == 'train':
            self._episode_indices = sorted(idx[:n_train].tolist())
        else:
            self._episode_indices = sorted(idx[n_train:].tolist())

        self._index = self._build_index()
        print(f"[PushTMaskHDF5Dataset] {split}: {len(self._episode_indices)} episodes, "
              f"{len(self._index)} clips (load_masks={load_masks}, preload_ram={preload_ram})")

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def _build_index(self) -> list[tuple[int, int]]:
        clip_len = (self.n_sample_frames - 1) * self.frame_offset + 1
        index = []
        for ep in self._episode_indices:
            ep_len = self._ep_lens[ep]
            for start in range(0, ep_len - clip_len + 1):
                index.append((ep, start))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t * self.frame_offset
                      for t in range(self.n_sample_frames)]

        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]

        if self.preload_ram:
            pixels = get_shared_pixels(self.h5_path)
            img = (pixels[abs_idxs].float() / 127.5) - 1.0  # Zero-copy IPC slice
            img = img.permute(0, 3, 1, 2)
        else:
            # Direct h5 fancy-indexing — wrapping in np.asarray would
            # materialize the whole 1.4GB dataset per item (~300x slower).
            # The annotation narrows the static type; the value is an ndarray.
            frames_h5: np.ndarray = self.h5['pixels'][abs_idxs]  # type: ignore[assignment]
            video = (frames_h5.astype(np.float32) / 127.5) - 1.0
            img = torch.from_numpy(video.transpose(0, 3, 1, 2))

        item = {
            'data_idx': idx,
            'img': img,
        }

        if 'action' in self.h5:
            act: np.ndarray = self.h5['action'][abs_idxs]  # type: ignore[assignment]
            item['action'] = torch.from_numpy(act.astype(np.float32))

        if self.load_masks:
            masks: dict[str, np.ndarray] = {k: self.h5[k][abs_idxs] for k in self.MASK_KEYS}  # type: ignore[assignment]
            agent_m = torch.from_numpy((masks['agent_masks'] > 127).astype(np.float32))
            block_m = torch.from_numpy((masks['block_masks'] > 127).astype(np.float32))
            if self.include_goal_mask:
                goal_m = torch.from_numpy((masks['goal_masks'] > 127).astype(np.float32))
                goal_visible = torch.clamp(goal_m - torch.maximum(block_m, agent_m), 0.0, 1.0)
                gt_masks = torch.stack([agent_m, block_m, goal_visible], dim=1)
            else:
                gt_masks = torch.stack([agent_m, block_m], dim=1)
            item['gt_masks'] = gt_masks

        return item

    def __del__(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


class DeterministicEpisodeEvalDataset(Dataset):
    """
    Evaluation dataset visiting 100% of validation episodes,
    deterministically sampling fixed clips per episode using per-episode seed.
    """
    MASK_KEYS = ['agent_masks', 'block_masks', 'goal_masks']

    def __init__(
        self,
        h5_path: str,
        split: str = 'val',
        resolution: tuple[int, int] = (64, 64),
        n_sample_frames: int = 6,
        clips_per_episode: int = 2,
        seed: int = 42,
        base_seed: Optional[int] = None,
        train_frac: float = 0.8,
        include_goal_mask: bool = True,
    ) -> None:
        self.h5_path = h5_path
        self.split = split
        self.resolution = resolution
        self.n_sample_frames = n_sample_frames
        self.clips_per_episode = clips_per_episode
        self.train_frac = train_frac
        self.include_goal_mask = include_goal_mask
        self._h5 = None
        if base_seed is not None:
            seed = base_seed
        self.seed = seed

        with h5py.File(h5_path, 'r') as f:
            self._ep_lens = np.asarray(f['ep_len']).tolist()
            self._ep_offs = np.asarray(f['ep_offset']).tolist()
        n_episodes = len(self._ep_lens)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * self.train_frac)

        if split == 'train':
            self._episode_indices = sorted(idx[:n_train].tolist())
        else:
            self._episode_indices = sorted(idx[n_train:].tolist())

        self._index = self._build_index()

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def _build_index(self) -> list[tuple[int, int]]:
        clip_len = self.n_sample_frames
        index = []
        for ep in self._episode_indices:
            ep_len = self._ep_lens[ep]
            if ep_len < clip_len:
                continue
            max_start = ep_len - clip_len
            # Per-episode seed derived from the base seed (reproducible across
            # runs and code changes); sample distinct starts without
            # replacement so duplicate clips are never counted.
            ep_rng = np.random.RandomState(self.seed + ep * 10007)
            n_valid_starts = max_start + 1
            n_clips = min(self.clips_per_episode, n_valid_starts)
            starts = ep_rng.choice(n_valid_starts, size=n_clips,
                                   replace=False).tolist()

            for start in starts:
                index.append((ep, start))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx, start_frame = self._index[idx]
        frame_idxs = list(range(start_frame, start_frame + self.n_sample_frames))
        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]

        # Direct h5 fancy-indexing — np.asarray would materialize the whole
        # dataset per item (~300x slower). Annotations narrow the static type.
        frames_h5: np.ndarray = self.h5['pixels'][abs_idxs]  # type: ignore[assignment]
        masks: dict[str, np.ndarray] = {k: self.h5[k][abs_idxs] for k in self.MASK_KEYS}  # type: ignore[assignment]

        video = (frames_h5.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(video.transpose(0, 3, 1, 2))

        agent_m = (masks['agent_masks'] > 127).astype(np.float32)
        block_m = (masks['block_masks'] > 127).astype(np.float32)
        if self.include_goal_mask:
            goal_m = (masks['goal_masks'] > 127).astype(np.float32)
            goal_visible = np.clip(goal_m - np.maximum(block_m, agent_m), 0.0, 1.0)
            gt_masks = np.stack([agent_m, block_m, goal_visible], axis=1)
        else:
            gt_masks = np.stack([agent_m, block_m], axis=1)
        gt_masks = torch.from_numpy(gt_masks).float()

        return {
            'data_idx': idx,
            'img': img,
            'gt_masks': gt_masks,
            'episode_idx': episode_idx,
            'start_frame': start_frame,
        }

    def __del__(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
