"""
src.training — Training infrastructure for Contact-Sim baselines.

Modules:
  - trainer:     BaseTrainer (TensorBoard logging, file logging, checkpoint saving)
  - train_loop:  TrainConfig, run_epoch, run_training (Hydra-agnostic training loop)
"""

from src.training.trainer import BaseTrainer, TeeLogger
from src.training.train_loop import TrainConfig, run_epoch, run_training

__all__ = [
    "BaseTrainer",
    "TeeLogger",
    "TrainConfig",
    "run_epoch",
    "run_training",
]
