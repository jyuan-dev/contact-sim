"""
Base Trainer class encapsulating TensorBoard logging, checkpoint saving/loading, and common training loop state.
"""

import os
import torch
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, save_dir: str, experiment_name: str = "exp"):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(self.save_dir, exist_ok=True)

        tb_log_dir = os.path.join(self.save_dir, 'tb_logs')
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[{self.experiment_name}] TensorBoard logs pointing to: {tb_log_dir}")

    def log_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_image(self, tag: str, img_tensor: torch.Tensor, global_step: int):
        self.writer.add_image(tag, img_tensor, global_step)

    def save_checkpoint(self, state_dict: dict, filename: str = "checkpoint.pt") -> str:
        ckpt_path = os.path.join(self.save_dir, filename)
        torch.save(state_dict, ckpt_path)
        print(f"[{self.experiment_name}] Checkpoint saved -> {ckpt_path}")
        return ckpt_path

    def close(self):
        if self.writer:
            self.writer.close()
