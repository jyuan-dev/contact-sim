import os
import sys
import time
import torch
import numpy as np
import h5py
import hdf5plugin
import torchvision.transforms.v2 as Tv2
from torch.utils.data import Dataset, DataLoader

# Path setup
REPO = '/home/jyuan/jyuan-ws/contact-sim'
sys.path.insert(0, REPO)

class PushTMaskHDF5DatasetOld(Dataset):
    MASK_KEYS = ['block_masks', 'agent_masks', 'goal_masks']
    def __init__(self, h5_path, resolution=(64,64)):
        self.h5_path = h5_path
        self.resolution = resolution
        self.img_transform = Tv2.Compose([Tv2.ToImage(), Tv2.ToDtype(torch.float32, scale=True), Tv2.Resize(resolution, antialias=True)])
        self.mask_transform = Tv2.Compose([Tv2.ToImage(), Tv2.ToDtype(torch.float32, scale=True), Tv2.Resize(resolution, antialias=True)])
        with h5py.File(h5_path, 'r') as f:
            self._ep_lens = np.array(f['ep_len']).tolist()
            self._ep_offs = np.array(f['ep_offset']).tolist()
        self._index = [(0, s) for s in range(0, 100)] # 100 samples
    def __len__(self): return len(self._index)
    def __getitem__(self, idx):
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t for t in range(6)]
        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]
        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, 'r') as f:
            frames = f['pixels'][abs_idxs]
            masks = {k: f[k][abs_idxs] for k in self.MASK_KEYS}
        img = torch.stack([self.img_transform(frame) for frame in frames], dim=0)
        gt_masks = torch.cat([torch.stack([self.mask_transform(masks[k][t][:,:,np.newaxis]) for t in range(6)], dim=0) for k in self.MASK_KEYS], dim=1)
        return {'img': img, 'gt_masks': gt_masks}

class PushTMaskHDF5DatasetLazy(Dataset):
    MASK_KEYS = ['block_masks', 'agent_masks', 'goal_masks']
    def __init__(self, h5_path, resolution=(64,64)):
        self.h5_path = h5_path
        self.resolution = resolution
        self.img_transform = Tv2.Compose([Tv2.ToImage(), Tv2.ToDtype(torch.float32, scale=True), Tv2.Resize(resolution, antialias=True)])
        self.mask_transform = Tv2.Compose([Tv2.ToImage(), Tv2.ToDtype(torch.float32, scale=True), Tv2.Resize(resolution, antialias=True)])
        with h5py.File(h5_path, 'r') as f:
            self._ep_lens = np.array(f['ep_len']).tolist()
            self._ep_offs = np.array(f['ep_offset']).tolist()
        self._index = [(0, s) for s in range(0, 100)] # 100 samples
        self.f = None
    def __len__(self): return len(self._index)
    def __getitem__(self, idx):
        if self.f is None:
            os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
            self.f = h5py.File(self.h5_path, 'r')
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t for t in range(6)]
        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]
        
        frames = self.f['pixels'][abs_idxs]
        masks = {k: self.f[k][abs_idxs] for k in self.MASK_KEYS}
        img = torch.stack([self.img_transform(frame) for frame in frames], dim=0)
        gt_masks = torch.cat([torch.stack([self.mask_transform(masks[k][t][:,:,np.newaxis]) for t in range(6)], dim=0) for k in self.MASK_KEYS], dim=1)
        return {'img': img, 'gt_masks': gt_masks}

h5 = '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5'
ds_old = PushTMaskHDF5DatasetOld(h5)
ds_lazy = PushTMaskHDF5DatasetLazy(h5)

t0 = time.time()
for i in range(50):
    _ = ds_old[i]
t_old = time.time() - t0
print(f"Old approach took: {t_old:.4f}s")

t0 = time.time()
for i in range(50):
    _ = ds_lazy[i]
t_lazy = time.time() - t0
print(f"Lazy approach took: {t_lazy:.4f}s")
print(f"Speedup factor: {t_old / t_lazy:.2f}x")
