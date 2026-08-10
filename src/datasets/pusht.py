import os
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


# ── Mask-Supervised Dataset (for DETR box / mask tracking) ────────────────────
class PushTMaskHDF5Dataset(Dataset):
    MASK_KEYS = ['agent_masks', 'block_masks', 'goal_masks']


    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        resolution=(64, 64),
        n_sample_frames: int = 6,
        frame_offset: int = 1,
        train_frac: float = 0.9,
        seed: int = 42,
    ):
        assert split in ('train', 'val')
        self.h5_path = h5_path
        self.split = split
        self.resolution = resolution
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset
        self._h5 = None  # lazy-opened per-worker

        with h5py.File(h5_path, 'r') as f:
            ep_lens = f['ep_len'][:]
            ep_offs = f['ep_offset'][:]

        self._ep_lens = ep_lens.tolist()
        self._ep_offs = ep_offs.tolist()
        n_episodes = len(ep_lens)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * train_frac)

        if split == 'train':
            self._episode_indices = sorted(idx[:n_train].tolist())
        else:
            self._episode_indices = sorted(idx[n_train:].tolist())

        self._index = self._build_index()
        print(f"[PushTMaskHDF5Dataset] {split}: {len(self._episode_indices)} episodes, "
              f"{len(self._index)} clips")

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def _build_index(self):
        clip_len = (self.n_sample_frames - 1) * self.frame_offset + 1
        index = []
        for ep in self._episode_indices:
            ep_len = self._ep_lens[ep]
            for start in range(0, ep_len - clip_len + 1):
                index.append((ep, start))
        return index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t * self.frame_offset
                      for t in range(self.n_sample_frames)]

        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]

        frames = self.h5['pixels'][abs_idxs]
        masks = {k: self.h5[k][abs_idxs] for k in self.MASK_KEYS}

        video = (frames.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(video.transpose(0, 3, 1, 2))

        # Disentangle mask overlap: Subtract Block and Agent masks from Goal mask (Occlusion removal)
        agent_m = (masks['agent_masks'] > 127).astype(np.float32)
        block_m = (masks['block_masks'] > 127).astype(np.float32)
        goal_m = (masks['goal_masks'] > 127).astype(np.float32)

        # Visible Goal = Goal AND NOT (Block OR Agent)
        goal_visible = np.clip(goal_m - np.maximum(block_m, agent_m), 0.0, 1.0)

        gt_masks = np.stack([agent_m, block_m, goal_visible], axis=1)
        gt_masks = torch.from_numpy(gt_masks).float()

        return {
            'data_idx': idx,
            'img': img,
            'gt_masks': gt_masks,
        }

    def __del__(self):
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
        resolution=(64, 64),
        n_sample_frames: int = 6,
        clips_per_episode: int = 2,
        train_frac: float = 0.9,
        base_seed: int = 42,
    ):
        self.h5_path = h5_path
        self.resolution = resolution
        self.n_sample_frames = n_sample_frames
        self.clips_per_episode = clips_per_episode
        self._h5 = None

        with h5py.File(h5_path, 'r') as f:
            ep_lens = f['ep_len'][:]
            ep_offs = f['ep_offset'][:]

        self._ep_lens = ep_lens.tolist()
        self._ep_offs = ep_offs.tolist()
        n_episodes = len(ep_lens)

        rng = np.random.RandomState(base_seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * train_frac)

        if split == 'train':
            self._episode_indices = sorted(idx[:n_train].tolist())
        else:
            self._episode_indices = sorted(idx[n_train:].tolist())

        self._index = []
        clip_len = n_sample_frames
        for ep_i, ep_idx in enumerate(self._episode_indices):
            ep_len = self._ep_lens[ep_idx]
            max_start = ep_len - clip_len
            if max_start < 0:
                continue

            ep_seed = (base_seed + ep_idx * 10007) & 0xFFFFFFFF
            ep_rng = np.random.RandomState(ep_seed)
            valid_starts = list(range(0, max_start + 1))

            if len(valid_starts) <= clips_per_episode:
                chosen_starts = valid_starts
            else:
                chosen_starts = sorted(ep_rng.choice(valid_starts, size=clips_per_episode, replace=False).tolist())

            for start in chosen_starts:
                self._index.append((ep_idx, start))

        print(f"[DeterministicEpisodeEvalDataset] Visited 100% of {len(self._episode_indices)} {split} episodes. "
              f"Deterministically sampled {len(self._index)} clips ({clips_per_episode} clips/episode, seed={base_seed}).")

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, start_frame = self._index[idx]
        ep_off = int(self._ep_offs[ep_idx])
        abs_idxs = [ep_off + start_frame + t for t in range(self.n_sample_frames)]

        frames = self.h5['pixels'][abs_idxs]
        masks = {k: self.h5[k][abs_idxs] for k in self.MASK_KEYS}

        video = (frames.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(video.transpose(0, 3, 1, 2))

        agent_m = (masks['agent_masks'] > 127).astype(np.float32)
        block_m = (masks['block_masks'] > 127).astype(np.float32)
        goal_m = (masks['goal_masks'] > 127).astype(np.float32)

        goal_visible = np.clip(goal_m - np.maximum(block_m, agent_m), 0.0, 1.0)
        gt_masks = torch.from_numpy(np.stack([agent_m, block_m, goal_visible], axis=1)).float()

        return {
            'data_idx': idx,
            'img': img,
            'gt_masks': gt_masks,
            'ep_idx': ep_idx,
            'start_frame': start_frame,
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

