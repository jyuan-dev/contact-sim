"""
Dataset loading and path resolution utilities.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_dataset_path(h5_path: str, default_filename: str = "pusht_expert_train_64x64.h5") -> str:
    """
    Resolve the dataset HDF5 path, probing known workspace and scratch
    locations when ``h5_path`` is missing. Raises when nothing is found.
    """
    if h5_path and os.path.exists(h5_path):
        return h5_path

    probed = [os.path.expanduser(f"~/.stable-wm/{default_filename}"),
              os.path.join("scratch", default_filename),
              os.path.join(REPO_ROOT, "scratch", default_filename)]
    for p in probed:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Dataset file not found (requested: {h5_path!r}). "
        f"Probed: {probed}"
    )


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
