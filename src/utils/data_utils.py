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
