"""
Script to train SAVi on PushT dataset using SOLD autoencoder training framework.
"""
import os
import sys
import h5py
import torch
import hydra
from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
from omegaconf import DictConfig
# Mock write_video in torchvision.io if missing (needed for custom torchvision builds)
# We do this at the very beginning before any other package imports torchvision
import torchvision.io
if not hasattr(torchvision.io, 'write_video'):
    def dummy_write_video(filename, video_array, fps, video_codec='libx264', options=None):
        import cv2
        import numpy as np
        T, H, W, C = video_array.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        out = cv2.VideoWriter(filename, fourcc, fps, (W, H))
        for t in range(T):
            frame = video_array[t]
            if torch.is_tensor(frame):
                frame = frame.cpu().numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            if C == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()
    torchvision.io.write_video = dummy_write_video

# Setup paths for SOLD and base_slots
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLD_ROOT = os.path.join(REPO_ROOT, 'third_party', 'sold')
SOLD_SRC  = os.path.join(SOLD_ROOT, 'sold')
SLOTFORMER = os.path.join(REPO_ROOT, 'third_party', 'cjepa', 'src', 'third_party', 'slotformer')

# Prevent namespace collision on 'datasets' package with HuggingFace datasets library
# by pre-loading SOLD's datasets modules and registering them directly in sys.modules.
import importlib.util
try:
    import datasets
except ImportError:
    pass

SOLD_DATASETS_DIR = os.path.join(SOLD_SRC, 'datasets')
modules = {}
for mod_name in ['ring_buffer', 'utils', 'info', 'image']:
    full_mod_name = f'datasets.{mod_name}'
    file_path = os.path.join(SOLD_DATASETS_DIR, f'{mod_name}.py')
    if os.path.exists(file_path):
        spec = importlib.util.spec_from_file_location(full_mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_mod_name] = mod
        modules[full_mod_name] = (mod, spec)

for name in ['datasets.info', 'datasets.ring_buffer', 'datasets.utils', 'datasets.image']:
    if name in modules:
        mod, spec = modules[name]
        spec.loader.exec_module(mod)

# Add SOLD first to prevent namespace collision on 'datasets'
for p in [SOLD_SRC, SOLD_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Add other paths after
for p in [REPO_ROOT, SLOTFORMER]:
    if p not in sys.path:
        sys.path.append(p)

# Patch ExtendedTensorBoardLogger in utils.logging to handle step=None gracefully
import utils.logging
original_log_metrics = utils.logging.ExtendedTensorBoardLogger.log_metrics
def patched_log_metrics(self, metrics, step=None):
    if step is None:
        step = self.current_step if hasattr(self, 'current_step') else 0
    original_log_metrics(self, metrics, step)
utils.logging.ExtendedTensorBoardLogger.log_metrics = patched_log_metrics

from base_slots.datasets.pusht_hdf5 import PushTHDF5Dataset
from train_autoencoder import AutoencoderModule
from utils.instantiate import instantiate_trainer, fill_in_missing
from utils.training import set_seed

# Patch AutoencoderModule.__init__ to remove non-serializable hyperparameters
original_ae_init = AutoencoderModule.__init__
def patched_ae_init(self, autoencoder, optimizer, scheduler=None):
    original_ae_init(self, autoencoder, optimizer, scheduler)
    for key in ['autoencoder', 'optimizer']:
        if hasattr(self, '_hparams') and key in self._hparams:
            del self._hparams[key]
        if hasattr(self, '_hparams_initial') and key in self._hparams_initial:
            del self._hparams_initial[key]
AutoencoderModule.__init__ = patched_ae_init
from termcolor import colored


class PushTSoldDataset(PushTHDF5Dataset):
    def __init__(self, h5_path: str, split: str = 'train', resolution=(64, 64),
                 sequence_length: int = 6, frame_offset: int = 1,
                 train_frac: float = 0.9, seed: int = 42):
        super().__init__(h5_path=h5_path, split=split, resolution=resolution,
                         n_sample_frames=sequence_length, frame_offset=frame_offset,
                         train_frac=train_frac, seed=seed)
        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(resolution, antialias=True),
        ])

    def _read_actions(self, episode_idx, start_frame):
        frame_idxs = [start_frame + t * self.frame_offset
                      for t in range(self.n_sample_frames)]
        with h5py.File(self.h5_path, 'r') as f:
            act = f['action']
            if act.ndim == 3:
                actions = act[episode_idx, frame_idxs]
            else:
                if self._episode_offsets is not None:
                    offset = self._episode_offsets[episode_idx]
                else:
                    offset = sum(self._all_episode_lengths[:episode_idx])
                abs_idxs = [offset + i for i in frame_idxs]
                actions = act[abs_idxs]
        return torch.tensor(actions, dtype=torch.float32)

    @property
    def dataset_infos(self) -> Dict[str, Any]:
        return {
            "image_size": [self.resolution[0], self.resolution[1]],
            "action_dim": 2,
        }

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        episode_idx, start_frame = self._index[idx]
        images = self._read_frames(episode_idx, start_frame)
        actions = self._read_actions(episode_idx, start_frame)
        return images, actions


@hydra.main(config_path="../third_party/sold/configs", config_name="train_autoencoder", version_base=None)
def train(cfg: DictConfig):
    set_seed(cfg.seed)
    trainer = instantiate_trainer(cfg)
    
    # Adjust batch size for distributed training
    cfg.dataset.batch_size = cfg.dataset.batch_size // trainer.world_size

    # Hardcode/Get dataset loading parameters
    h5_path = cfg.dataset.get("h5_path", "/data/.stable-wm/pusht_expert_train.h5")
    resolution = tuple(cfg.dataset.get("resolution", [64, 64]))
    sequence_length = cfg.dataset.get("sequence_length", 6)
    frame_offset = cfg.dataset.get("frame_offset", 1)
    batch_size = cfg.dataset.get("batch_size", 64)
    num_workers = cfg.dataset.get("num_workers", 8)
    train_frac = cfg.dataset.get("train_frac", 0.9)
    seed = cfg.seed

    print(f"Instantiating customized PushT HDF5 loader from {h5_path}...")
    train_dataset = PushTSoldDataset(
        h5_path=h5_path, split='train', resolution=resolution,
        sequence_length=sequence_length, frame_offset=frame_offset,
        train_frac=train_frac, seed=seed
    )
    val_dataset = PushTSoldDataset(
        h5_path=h5_path, split='val', resolution=resolution,
        sequence_length=sequence_length, frame_offset=frame_offset,
        train_frac=train_frac, seed=seed
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_infos = train_dataset.dataset_infos

    fill_in_missing(cfg, dataset_infos)
    savi = hydra.utils.instantiate(cfg.model)

    print(colored('Output dir:', 'magenta', attrs=['bold']), hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    if cfg.logger.log_to_wandb:
        import wandb
        wandb.init(project="sold", name=cfg.experiment, config=dict(cfg), sync_tensorboard=True)
    
    trainer.fit(savi, train_dataloader, val_dataloader, 
                ckpt_path=os.path.abspath(cfg.checkpoint) if cfg.checkpoint else None)
    
    if cfg.logger.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
