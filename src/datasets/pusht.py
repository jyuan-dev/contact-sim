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

# ── ImageNet Normalization utilities ──────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_img(img_tensor):
    """Normalize a [C,H,W] tensor from [0,1] range to ImageNet zero-mean/unit-std."""
    return (img_tensor - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_img(img_tensor):
    """Denormalize a [C,H,W] tensor from ImageNet stats back to [0,1] range."""
    return img_tensor * IMAGENET_STD.to(img_tensor.device) + IMAGENET_MEAN.to(img_tensor.device)


def augment_background(img_np, bg_threshold=240):
    """Replace white background pixels with a random color in a uint8 HWC image."""
    bg_mask = np.all(img_np > bg_threshold, axis=-1)
    rand_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    img_aug = img_np.copy()
    img_aug[bg_mask] = rand_color
    return img_aug


# ── Mask-Supervised Dataset (for DETR box / mask tracking) ────────────────────
class PushTMaskHDF5Dataset(Dataset):
    MASK_KEYS = ['agent_masks', 'block_masks']


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

        gt_masks = np.stack([masks[k] for k in self.MASK_KEYS], axis=1)
        gt_masks = torch.from_numpy(gt_masks).float() / 255.0

        return {
            'data_idx': idx,
            'img': img,
            'gt_masks': gt_masks,
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
