import os
import sys
import time
import torch
import numpy as np
import h5py
import hdf5plugin
import cv2
from torch.utils.data import Dataset, DataLoader

# Path setup
REPO = '/home/jyuan/jyuan-ws/contact-sim'
sys.path.insert(0, REPO)

class PushTMaskHDF5DatasetNoGoalRead(Dataset):
    MASK_KEYS = ['block_masks', 'agent_masks']  # No goal_masks here
    def __init__(self, h5_path, resolution=(64,64)):
        self.h5_path = h5_path
        self.resolution = resolution
        with h5py.File(h5_path, 'r') as f:
            self._ep_offs = np.array(f['ep_offset']).tolist()
            # Cache the static goal mask once at startup
            raw_goal = f['goal_masks'][0]  # (224, 224)
            self.static_goal_mask = cv2.resize(raw_goal, resolution, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            
        self._index = [(0, s) for s in range(0, 100)]
    def __len__(self): return len(self._index)
    def __getitem__(self, idx):
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t for t in range(6)]
        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]
        
        with h5py.File(self.h5_path, 'r') as f:
            frames = f['pixels'][abs_idxs]
            # Only read block and agent masks from HDF5
            masks = {k: f[k][abs_idxs] for k in self.MASK_KEYS}
            
        # 1. Video
        video = []
        for frame in frames:
            resized = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
            video.append((resized.astype(np.float32) / 127.5) - 1.0)
        img = torch.from_numpy(np.stack(video, axis=0).transpose(0, 3, 1, 2))
        
        # 2. Masks
        gt_masks = []
        for k in self.MASK_KEYS:
            resized_m = [cv2.resize(m, self.resolution, interpolation=cv2.INTER_AREA) for m in masks[k]]
            gt_masks.append(np.stack(resized_m, axis=0))
            
        # Replicate static goal mask for the 6 frames
        goal_mask_seq = np.stack([self.static_goal_mask] * 6, axis=0)
        gt_masks.append(goal_mask_seq)
        
        gt_masks = np.stack(gt_masks, axis=1) # (6, 3, 64, 64)
        # Note: block/agent are scaled by / 255, goal is already scaled
        gt_masks_tensor = torch.from_numpy(gt_masks).float()
        gt_masks_tensor[:, :2] /= 255.0
        
        return img, gt_masks_tensor

h5 = '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5'
ds = PushTMaskHDF5DatasetNoGoalRead(h5)
_ = ds[0]

t0 = time.time()
for i in range(100):
    _ = ds[i]
print(f"Time without goal disk seeks: {time.time() - t0:.4f}s")
