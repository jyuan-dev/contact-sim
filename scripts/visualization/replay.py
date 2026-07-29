"""
Unified Dataset Trajectory Visualizer & Replay tool for PushT and OGBench.

Usage:
    # Replay PushT dataset:
    python scripts/visualization/replay.py --dataset pusht --h5-path scratch/pusht_expert_train_test_enriched.h5 --ep-idx 0

    # Replay OGBench dataset:
    python scripts/visualization/replay.py --dataset ogbench --h5-path scratch/ogbench_cube.h5 --ep-idx 0
"""

import sys
import os
import argparse
import h5py
import hdf5plugin
import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.data_utils import find_dataset_path


def replay_pusht(h5_path, ep_idx, save_video=False):
    h5_path = find_dataset_path(h5_path)
    if not os.path.exists(h5_path):
        print(f"[Replay PushT] Dataset '{h5_path}' not found.")
        return

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        keys = list(data.keys())
        if ep_idx >= len(keys):
            print(f"[Replay PushT] Invalid episode index {ep_idx}. Total episodes: {len(keys)}")
            return

        ep_key = keys[ep_idx]
        imgs = data[ep_key]['img'][:]
        print(f"[Replay PushT] Episode '{ep_key}' loaded. Frames: {len(imgs)}, Resolution: {imgs.shape[1:3]}")


def replay_ogbench(h5_path, ep_idx):
    h5_path = find_dataset_path(h5_path)
    if not os.path.exists(h5_path):
        print(f"[Replay OGBench] Dataset '{h5_path}' not found.")
        return

    print(f"[Replay OGBench] Replaying episode {ep_idx} from {h5_path}...")


def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Replay & Visualization")
    parser.add_argument("--dataset", type=str, choices=["pusht", "ogbench"], default="pusht")
    parser.add_argument("--h5-path", type=str, default="scratch/pusht_expert_train_test_enriched.h5")
    parser.add_argument("--ep-idx", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--save-video", action="store_true", help="Save visualization to video/gif file")
    args = parser.parse_args()

    if args.dataset == "pusht":
        replay_pusht(args.h5_path, args.ep_idx, args.save_video)
    elif args.dataset == "ogbench":
        replay_ogbench(args.h5_path, args.ep_idx)

if __name__ == "__main__":
    main()
