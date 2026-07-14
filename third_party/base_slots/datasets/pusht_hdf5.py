"""
HDF5-based PushT dataset for training StoSAVi (C-JEPA slotformer).
Loads pixel frames directly from pusht_expert_train.h5 without MP4 conversion.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import h5py
import hdf5plugin


class PushTHDF5Dataset(Dataset):
    """
    Reads pixel frames from pusht_expert_train.h5.
    
    The HDF5 file has shape (N_episodes, T, H, W, 3) or stacked episodes.
    We load clips of n_sample_frames consecutive frames per episode.
    
    Args:
        h5_path:        path to pusht_expert_train.h5
        split:          'train' or 'val'
        resolution:     (H, W) output image size
        n_sample_frames: number of frames per clip
        frame_offset:   stride between frames in a clip (1 = consecutive)
        train_frac:     fraction of episodes used for train (rest for val)
        seed:           random seed for train/val split
    """

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

        # Transforms: resize + normalize to [-1, 1] (same as original pusht.py)
        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(resolution, antialias=True),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        # Read episode lengths and offsets from HDF5
        with h5py.File(h5_path, 'r') as f:
            self._all_episode_lengths = self._read_episode_lengths(f)
            if 'ep_offset' in f:
                self._episode_offsets = np.array(f['ep_offset']).tolist()
            else:
                self._episode_offsets = None
        
        n_episodes = len(self._all_episode_lengths)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * train_frac)
        
        if split == 'train':
            self._episode_indices = sorted(idx[:n_train].tolist())
        else:
            self._episode_indices = sorted(idx[n_train:].tolist())

        # Build (episode_idx, start_frame) index
        self._index = self._build_index()
        print(f"[PushTHDF5Dataset] {split}: {len(self._episode_indices)} episodes, "
              f"{len(self._index)} clips")

    def _read_episode_lengths(self, f):
        """Infer per-episode lengths from HDF5."""
        if 'ep_len' in f:
            return np.array(f['ep_len']).tolist()
        # Case 1: grouped by episode  (f['episode_0/pixels'], ...)
        if 'episode_0' in f or '0' in f:
            keys = [k for k in f.keys() if 'episode' in k or k.isdigit()]
            return [f[k]['pixels'].shape[0] for k in sorted(keys, key=lambda x: int(x.replace('episode_', '')))]
        # Case 2: flat array (f['pixels'] shape [N, T, H, W, 3])
        pix = f['pixels']
        if pix.ndim == 5:
            return [pix.shape[1]] * pix.shape[0]
        # Case 3: flat sequence (f['pixels'] shape [N_total, H, W, 3]) with episode_ends
        if 'episode_ends' in f:
            ends = np.array(f['episode_ends'])
            starts = np.concatenate([[0], ends[:-1]])
            return (ends - starts).tolist()
        raise RuntimeError("Cannot determine episode structure from HDF5 file. "
                           f"Keys: {list(f.keys())}")

    def _build_index(self):
        """Build list of (episode_idx, start_frame) valid clip starts."""
        clip_len = (self.n_sample_frames - 1) * self.frame_offset + 1
        index = []
        for ep in self._episode_indices:
            ep_len = self._all_episode_lengths[ep]
            for start in range(0, ep_len - clip_len + 1):
                index.append((ep, start))
        return index

    def _read_frames(self, episode_idx, start_frame):
        """Load n_sample_frames from HDF5."""
        frame_idxs = [start_frame + t * self.frame_offset
                      for t in range(self.n_sample_frames)]
        with h5py.File(self.h5_path, 'r') as f:
            pix = f['pixels']
            if pix.ndim == 5:
                # Shape: [N_episodes, T, H, W, 3]
                frames = pix[episode_idx, frame_idxs]  # (T, H, W, 3) uint8
            elif pix.ndim == 4:
                # Flat, use ep_offset to offset
                if self._episode_offsets is not None:
                    offset = self._episode_offsets[episode_idx]
                else:
                    offset = sum(self._all_episode_lengths[:episode_idx])
                abs_idxs = [offset + i for i in frame_idxs]
                frames = pix[abs_idxs]  # (T, H, W, 3)
            else:
                raise RuntimeError(f"Unexpected pixels ndim: {pix.ndim}")
        
        # frames: numpy (T, H, W, 3) uint8 → tensor (T, 3, H, W) float [-1,1]
        video = []
        for frame in frames:
            video.append(self.transform(frame))
        return torch.stack(video, dim=0)  # (T, 3, H, W)

    def get_video(self, episode_idx):
        """Load a full episode for validation visualization."""
        with h5py.File(self.h5_path, 'r') as f:
            pix = f['pixels']
            if pix.ndim == 5:
                frames = pix[episode_idx]  # (T, H, W, 3)
            else:
                if self._episode_offsets is not None:
                    offset = self._episode_offsets[episode_idx]
                else:
                    offset = sum(self._all_episode_lengths[:episode_idx])
                ep_len = self._all_episode_lengths[episode_idx]
                frames = pix[offset:offset + ep_len]

        video = torch.stack([self.transform(f) for f in frames], dim=0)
        return {'video': video, 'data_idx': episode_idx}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        episode_idx, start_frame = self._index[idx]
        img = self._read_frames(episode_idx, start_frame)
        return {'data_idx': idx, 'img': img}


def build_pusht_hdf5_dataset(h5_path, resolution=(64, 64), n_sample_frames=6,
                              frame_offset=1, train_frac=0.9, seed=42):
    """Convenience factory returning (train_dataset, val_dataset)."""
    train_ds = PushTHDF5Dataset(h5_path, split='train', resolution=resolution,
                                 n_sample_frames=n_sample_frames,
                                 frame_offset=frame_offset,
                                 train_frac=train_frac, seed=seed)
    val_ds   = PushTHDF5Dataset(h5_path, split='val',   resolution=resolution,
                                 n_sample_frames=n_sample_frames,
                                 frame_offset=frame_offset,
                                 train_frac=train_frac, seed=seed)
    return train_ds, val_ds
