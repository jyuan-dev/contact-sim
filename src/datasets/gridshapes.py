"""
GridShapes Dataset for Slot-Worldmodel & Object-Centric Learning Baselines.

Generates 2D multi-object moving shape video sequences with ground truth pixel masks.

Conforms to standard dataset dict contract:
  {
      'data_idx': int,
      'img': torch.Tensor,        # [T, C, H, W] normalized to [-1, 1]
      'gt_masks': torch.Tensor,   # [T, K, H, W] binary object masks [0, 1]
  }
"""

from typing import Any

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

COLOR_RGB_DICT = {
    "red": (255, 0, 0),
    "cyan": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "white": (255, 255, 255),
    "brown": (165, 42, 42)
}


class GridShapesDataset(Dataset):
    """
    GridShapes Synthetic Multi-Object Video Dataset.

    Generates dynamic sequences of 2D moving shapes (balls, squares, triangles)
    with fixed or random colors and tracks ground-truth object masks.
    """

    COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    SHAPES = ["ball", "square", "triangle"]

    def __init__(self, num_samples: int = 1000, num_frames: int = 16, num_objects: int = 3, img_size: int = 64,
                 shape_size: int = 15, seed: int = 42) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_objects = num_objects
        self.img_size = img_size
        self.shape_size = shape_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _render_shape(self, shape_name: str, color_rgb: tuple[int, int, int], size: int) -> tuple[np.ndarray, np.ndarray]:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        if shape_name == "ball":
            center = (size // 2, size // 2)
            radius = size // 2 - 1
            cv2.circle(canvas, center, radius, color_rgb, -1)
            cv2.circle(mask, center, radius, 1, -1)
        elif shape_name == "square":
            canvas[:, :] = color_rgb
            mask[:, :] = 1
        elif shape_name == "triangle":
            pts = np.array([[size // 2, 1], [1, size - 2], [size - 2, size - 2]], np.int32)
            cv2.drawContours(canvas, [pts], 0, color_rgb, -1)
            cv2.drawContours(mask, [pts], 0, 1, -1)

        return canvas, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.RandomState(self.seed + idx)

        frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        gt_masks = np.zeros((self.num_frames, self.num_objects, self.img_size, self.img_size), dtype=np.float32)

        # Initialize Object Properties
        objects = []
        for k in range(self.num_objects):
            c_name = self.COLORS[k % len(self.COLORS)]
            c_rgb = COLOR_RGB_DICT[c_name]
            s_name = self.SHAPES[k % len(self.SHAPES)]
            patch, p_mask = self._render_shape(s_name, c_rgb, self.shape_size)

            x = rng.randint(0, self.img_size - self.shape_size)
            y = rng.randint(0, self.img_size - self.shape_size)
            dx = rng.choice([-2, -1, 1, 2])
            dy = rng.choice([-2, -1, 1, 2])

            objects.append({
                'patch': patch,
                'p_mask': p_mask,
                'x': float(x),
                'y': float(y),
                'dx': float(dx),
                'dy': float(dy)
            })

        # Render Frames
        for t in range(self.num_frames):
            frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            for k, obj in enumerate(objects):
                # Update Position & Bounce on Walls
                obj['x'] += obj['dx']
                obj['y'] += obj['dy']

                if obj['x'] <= 0 or obj['x'] >= self.img_size - self.shape_size:
                    obj['dx'] *= -1
                    obj['x'] = np.clip(obj['x'], 0, self.img_size - self.shape_size)

                if obj['y'] <= 0 or obj['y'] >= self.img_size - self.shape_size:
                    obj['dy'] *= -1
                    obj['y'] = np.clip(obj['y'], 0, self.img_size - self.shape_size)

                ix, iy = int(obj['x']), int(obj['y'])
                s = self.shape_size

                # Draw Object on Frame
                m_sub = obj['p_mask'] > 0
                frame[iy:iy+s, ix:ix+s][m_sub] = obj['patch'][m_sub]
                gt_masks[t, k, iy:iy+s, ix:ix+s] = obj['p_mask'].astype(np.float32)

            frames[t] = frame

        # Convert to PyTorch Tensors
        video = (frames.astype(np.float32) / 127.5) - 1.0 # [T, H, W, C] -> [-1, 1]
        img_tensor = torch.from_numpy(video.transpose(0, 3, 1, 2)) # [T, C, H, W]
        gt_masks_tensor = torch.from_numpy(gt_masks)               # [T, K, H, W]

        return {
            'data_idx': idx,
            'img': img_tensor,
            'gt_masks': gt_masks_tensor
        }
