"""
Dataset loading and path resolution utilities.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_dataset_path(h5_path: str, default_filename: str = "pusht_expert_train_64x64.h5") -> str:
    """
    Search for dataset HDF5 file across common workspace and scratch paths if h5_path is missing.
    """
    if h5_path and os.path.exists(h5_path):
        return h5_path
    
    alt_paths = [
        os.path.expanduser(f"~/.stable-wm/{default_filename}"),
        os.path.join("scratch", default_filename),
        os.path.join(REPO_ROOT, "scratch", default_filename),
    ]
    for p in alt_paths:
        if os.path.exists(p):
            return p
    return h5_path


def get_dataset(
    dataset_name: str,
    h5_path: str = None,
    split: str = 'train',
    resolution: tuple = (64, 64),
    n_sample_frames: int = 16,
    frame_offset: int = 1,
    train_frac: float = 0.9
):
    """
    Convenience wrapper forwarding to unified build_dataset factory.
    """
    from src.datasets.factory import build_dataset
    cfg = {
        'dataset': {
            'name': dataset_name,
            'h5_path': find_dataset_path(h5_path) if h5_path else None,
            'resolution': resolution,
            'n_sample_frames': n_sample_frames,
            'frame_offset': frame_offset,
            'train_frac': train_frac
        }
    }
    return build_dataset(cfg, split=split)
