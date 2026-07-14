import os
import sys
import time
import torch
import numpy as np
import h5py
import hdf5plugin
import cv2
import torchvision.transforms.v2 as Tv2
from torch.utils.data import Dataset, DataLoader

# Path setup
REPO = '/home/jyuan/jyuan-ws/contact-sim'
sys.path.insert(0, REPO)

class PushTMaskHDF5DatasetTv2(Dataset):
    MASK_KEYS = ['block_masks', 'agent_masks', 'goal_masks']
    def __init__(self, h5_path, resolution=(64,64)):
        self.h5_path = h5_path
        self.resolution = resolution
        self.mask_transform = Tv2.Compose([
            Tv2.ToImage(),
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Resize(resolution, antialias=True),
        ])
        with h5py.File(h5_path, 'r') as f:
            self._ep_offs = np.array(f['ep_offset']).tolist()
        self._index = [(0, s) for s in range(0, 100)]
        self.f = None
    def __len__(self): return len(self._index)
    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.h5_path, 'r')
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t for t in range(6)]
        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]
        masks = {k: self.f[k][abs_idxs] for k in self.MASK_KEYS}
        
        gt_masks = []
        for k in self.MASK_KEYS:
            m_seq = []
            for t_mask in masks[k]:
                m_seq.append(self.mask_transform(t_mask[:, :, np.newaxis]))
            gt_masks.append(torch.stack(m_seq, dim=0))
        gt_masks = torch.cat(gt_masks, dim=1)
        return gt_masks

class PushTMaskHDF5DatasetCV2(Dataset):
    MASK_KEYS = ['block_masks', 'agent_masks', 'goal_masks']
    def __init__(self, h5_path, resolution=(64,64)):
        self.h5_path = h5_path
        self.resolution = resolution
        with h5py.File(h5_path, 'r') as f:
            self._ep_offs = np.array(f['ep_offset']).tolist()
        self._index = [(0, s) for s in range(0, 100)]
        self.f = None
    def __len__(self): return len(self._index)
    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.h5_path, 'r')
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t for t in range(6)]
        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]
        masks = {k: self.f[k][abs_idxs] for k in self.MASK_KEYS}
        
        gt_masks = []
        for k in self.MASK_KEYS:
            resized_frames = [cv2.resize(m, self.resolution, interpolation=cv2.INTER_AREA) for m in masks[k]]
            gt_masks.append(np.stack(resized_frames, axis=0))
        gt_masks = np.stack(gt_masks, axis=1) # (6, 3, 64, 64)
        return torch.from_numpy(gt_masks).float() / 255.0

h5 = '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5'
ds_tv2 = PushTMaskHDF5DatasetTv2(h5)
ds_cv2 = PushTMaskHDF5DatasetCV2(h5)

# warmup
_ = ds_tv2[0]
_ = ds_cv2[0]

t0 = time.time()
for i in range(100):
    _ = ds_tv2[i]
t_tv2 = time.time() - t0
print(f"Tv2 transform took: {t_tv2:.4f}s")

t0 = time.time()
for i in range(100):
    _ = ds_cv2[i]
t_cv2 = time.time() - t0
print(f"CV2 transform took: {t_cv2:.4f}s")
print(f"Speedup factor: {t_tv2 / t_cv2:.2f}x")
