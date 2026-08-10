"""
Base Trainer class encapsulating TensorBoard logging, file logging, checkpoint saving/loading, and common training loop state.
"""

import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

class TeeLogger:
    """
    Duplicates stdout stream to both the terminal console and a log file on disk.

    Args:
        filepath (str): Absolute or relative path to the target log file.
        mode (str): File open mode for writing log messages.
            Candidate Options:
                - 'a' (Append, Default): Appends new log messages to the end of an existing log file.
                - 'w' (Overwrite): Overwrites and truncates any existing log file at startup.
    """
    def __init__(self, filepath: str, mode: str = 'a'):
        self.terminal = sys.stdout
        self.log_file = open(filepath, mode, buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


class BaseTrainer:
    """
    Base Trainer class encapsulating TensorBoard logging, dedicated file logging,
    checkpoint management, and experiment workspace state.

    Args:
        save_dir (str): Directory where logs and checkpoints will be saved.
        experiment_name (str): Unique name of the experiment.
        mode (str): Log file opening mode for stdout redirection.
            Candidate Options:
                - 'a' (Append, Default): Appends stdout to an existing train.log file.
                - 'w' (Overwrite): Truncates train.log at startup to start a fresh log.
    """
    def __init__(self, save_dir: str, experiment_name: str = "exp", mode: str = 'a'):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(self.save_dir, exist_ok=True)

        tb_log_dir = os.path.join(self.save_dir, 'tb_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # Setup dedicated file logging in save_dir
        self.log_path = os.path.join(self.save_dir, 'train.log')
        self.tb_log_path = os.path.join(tb_log_dir, 'train.log')
        
        self.tee = TeeLogger(self.log_path, mode=mode)
        sys.stdout = self.tee

        print(f"[{self.experiment_name}] TensorBoard logs: {tb_log_dir}", flush=True)
        print(f"[{self.experiment_name}] Dedicated train log: {self.log_path}", flush=True)
        print(f"💡 HINT: To monitor real-time training progress, view the tail of the log file:\n   tail -f {self.log_path}\n", flush=True)

    def log_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_image(self, tag: str, img_tensor: torch.Tensor, global_step: int):
        self.writer.add_image(tag, img_tensor, global_step)

    def save_checkpoint(self, state_dict: dict, filename: str = "checkpoint.pt") -> str:
        ckpt_path = os.path.join(self.save_dir, filename)
        torch.save(state_dict, ckpt_path)
        print(f"[{self.experiment_name}] Checkpoint saved -> {ckpt_path}", flush=True)
        return ckpt_path

    def close(self):
        print(f"\n💡 HINT: Training completed! Inspect the tail of the log file using:\n   tail -n 50 {self.log_path}\n", flush=True)
        if self.writer:
            self.writer.close()
        if hasattr(self, 'tee') and self.tee:
            # Sync train.log into tb_logs/train.log
            try:
                import shutil
                shutil.copy(self.log_path, self.tb_log_path)
            except Exception:
                pass
            sys.stdout = self.tee.terminal
            self.tee.close()

