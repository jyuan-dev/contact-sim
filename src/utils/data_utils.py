"""
Dataset loading and path resolution utilities.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_dataset_path(h5_path: str, default_filename: str = "pusht_expert_train_test_enriched.h5") -> str:
    """
    Search for dataset HDF5 file across common workspace and scratch paths if h5_path is missing.
    """
    if h5_path and os.path.exists(h5_path):
        return h5_path
    
    alt_paths = [
        os.path.join("scratch", default_filename),
        os.path.join("scratch", "pusht_cworld_with_masks.h5"),
        os.path.join(REPO_ROOT, "scratch", default_filename),
        os.path.expanduser(f"~/.stable-wm/{default_filename}"),
        os.path.expanduser(f"~/.stable-wm/pusht_expert_train_test_enriched.h5")
    ]
    for p in alt_paths:
        if os.path.exists(p):
            return p
    return h5_path


def get_dataset(
    dataset_name: str,
    h5_path: str,
    split: str = 'train',
    resolution: tuple = (64, 64),
    n_sample_frames: int = 16,
    frame_offset: int = 1,
    train_frac: float = 0.95
):
    """
    Instantiate appropriate dataset class based on dataset_name.
    """
    dataset_name = dataset_name.lower()
    h5_path = find_dataset_path(h5_path)

    if dataset_name == 'pusht':
        from src.datasets.pusht import PushTMaskHDF5Dataset
        return PushTMaskHDF5Dataset(
            h5_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            train_frac=train_frac
        )
    elif dataset_name == 'ogbench':
        from src.datasets.ogbench import OGBenchCubeDataset
        return OGBenchCubeDataset(
            data_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames
        )
    elif dataset_name == 'libero':
        from src.datasets.libero import LiberoDataset
        return LiberoDataset(
            data_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames
        )
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. Supported: 'pusht', 'ogbench', 'libero'.")
