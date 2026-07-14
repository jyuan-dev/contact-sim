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

class PushTImageDatasetTv2(Dataset):
    def __init__(self, h5_path, resolution=(64,64)):
        self.h5_path = h5_path
        self.resolution = resolution
        self.img_transform = Tv2.Compose([
            Tv2.ToImage(),
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Resize(resolution, antialias=True),
            Tv2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
        frames = self.f['pixels'][abs_idxs]
        
        video = []
        for frame in frames:
            video.append(self.img_transform(frame))
        return torch.stack(video, dim=0)

class PushTImageDatasetCV2(Dataset):
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
        frames = self.f['pixels'][abs_idxs]
        
        video = []
        for frame in frames:
            # Resize
            resized = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
            # Normalize to [-1, 1]
            normalized = (resized.astype(np.float32) / 127.5) - 1.0
            # HWC -> CHW
            chw = normalized.transpose(2, 0, 1)
            video.append(chw)
        video = np.stack(video, axis=0) # (6, 3, 64, 64)
        return torch.from_numpy(video)

h5 = '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5'
ds_tv2 = PushTImageDatasetTv2(h5)
ds_cv2 = PushTImageDatasetCV2(h5)

# warmup
_ = ds_tv2[0]
_ = ds_cv2[0]

t0 = time.time()
for i in range(100):
    _ = ds_tv2[i]
t_tv2 = time.time() - t0
print(f"Tv2 Image transform took: {t_tv2:.4f}s")

t0 = time.time()
for i in range(100):
    _ = ds_cv2[i]
t_cv2 = time.time() - t0
print(f"CV2 Image transform took: {t_cv2:.4f}s")
print(f"Speedup factor: {t_tv2 / t_cv2:.2f}x")
