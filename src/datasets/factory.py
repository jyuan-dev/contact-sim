"""
Dataset Factory for Contact-Sim / Slot-Worldmodel Baselines.

Provides unified factory functions:
  - build_dataset(cfg, split='train') -> PyTorch Dataset
  - build_dataloader(cfg, split='train', batch_size=None, num_workers=4) -> PyTorch DataLoader
"""

import torch
from torch.utils.data import DataLoader

from src.datasets.pusht import PushTMaskHDF5Dataset
from src.datasets.gridshapes import GridShapesDataset


def build_dataset(cfg, split: str = 'train'):
    """
    Constructs a dataset instance based on configuration dictionary.

    Args:
        cfg (dict or DictConfig): Configuration containing dataset specifications.
        split (str): Dataset split ('train' or 'val').

    Returns:
        torch.utils.data.Dataset
    """
    try:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    ds_cfg = cfg.get('dataset', cfg)
    if isinstance(cfg, dict):
        merged_cfg = {**cfg, **ds_cfg}
    else:
        merged_cfg = ds_cfg

    ds_name = merged_cfg.get('name', merged_cfg.get('type', 'pusht')).lower()

    if 'pusht' in ds_name:
        from src.utils.data_utils import find_dataset_path
        h5_path = find_dataset_path(
            merged_cfg.get('h5_path', '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5'),
            default_filename='pusht_expert_train_64x64.h5'
        )

        resolution = tuple(merged_cfg.get('resolution', (64, 64)))
        n_sample_frames = merged_cfg.get('n_sample_frames', merged_cfg.get('seq_len', 6))
        frame_offset = merged_cfg.get('frame_offset', 1)
        seed = merged_cfg.get('seed', 42)
        train_frac = merged_cfg.get('train_frac', 0.9)
        load_masks = merged_cfg.get('load_masks', False)
        preload_ram = merged_cfg.get('preload_ram', True)
        include_goal_mask = merged_cfg.get('include_goal_mask', True)

        return PushTMaskHDF5Dataset(
            h5_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            train_frac=train_frac,
            seed=seed,
            load_masks=load_masks,
            preload_ram=preload_ram,
            include_goal_mask=include_goal_mask,
        )

    elif 'gridshapes' in ds_name or 'grid_shapes' in ds_name:
        train_samples = merged_cfg.get('train_samples', 1000)
        val_samples = merged_cfg.get('val_samples', 200)
        num_samples = merged_cfg.get('num_samples', train_samples if split == 'train' else val_samples)
        num_frames = merged_cfg.get('num_frames', merged_cfg.get('n_sample_frames', 16))
        num_objects = merged_cfg.get('num_objects', 3)
        img_size = merged_cfg.get('img_size', merged_cfg.get('resolution', [64, 64]))
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        seed = merged_cfg.get('seed', 42) + (0 if split == 'train' else 9999)

        return GridShapesDataset(
            num_samples=num_samples,
            num_frames=num_frames,
            num_objects=num_objects,
            img_size=img_size,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown dataset name: '{ds_name}'. Supported: 'pusht', 'gridshapes'.")


def build_dataloader(cfg, split: str = 'train', batch_size: int = None, num_workers: int = 4, shuffle: bool = None):
    """
    Constructs a PyTorch DataLoader for the specified dataset split with GPU pin_memory and prefetching optimizations.
    """
    dataset = build_dataset(cfg, split=split)
    
    if batch_size is None:
        batch_size = cfg.get('batch_size', cfg.get('training', {}).get('batch_size', 16))

    if shuffle is None:
        shuffle = (split == 'train')

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (num_workers > 0),
        "drop_last": (split == 'train'),
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs)
