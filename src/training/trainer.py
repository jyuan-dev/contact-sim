"""
Base Trainer class encapsulating TensorBoard logging, file logging, checkpoint saving/loading, and common training loop state.
"""

import os
import sys
import torch
from typing import Optional
from torch.utils.tensorboard import SummaryWriter  # type: ignore[reportPrivateImportUsage]

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
    def __init__(self, filepath: str, mode: str = 'a') -> None:
        self.terminal = sys.stdout
        self.log_file = open(filepath, mode, buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()


class BaseTrainer:
    """
    Base Trainer class encapsulating TensorBoard logging, WandB logging, dedicated file logging,
    checkpoint management, and experiment workspace state.

    Args:
        save_dir (str): Directory where logs and checkpoints will be saved.
        experiment_name (str): Unique name of the experiment.
        mode (str): Log file opening mode for stdout redirection.
        use_wandb (bool): Whether to enable WandB experiment tracking.
        wandb_project (str): WandB project name.
        cfg_dict (dict): Configuration dictionary to log to WandB.
    """
    def __init__(
        self,
        save_dir: str,
        experiment_name: str = "exp",
        mode: str = 'a',
        use_wandb: bool = True,
        wandb_project: str = "pusht-contact-sim",
        cfg_dict: Optional[dict] = None,
    ) -> None:
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.wandb_run = None
        os.makedirs(self.save_dir, exist_ok=True)

        tb_log_dir = os.path.join(self.save_dir, 'tb_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # Optional WandB initialization
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=cfg_dict,
                    dir=save_dir,
                )
                print(f"[{self.experiment_name}] WandB initialized: {self.wandb_run.url}", flush=True)
            except Exception as e:
                print(f"[{self.experiment_name}] Warning: Failed to initialize WandB ({e}). Continuing with TensorBoard only.", flush=True)
                self.use_wandb = False

        # Setup dedicated file logging in save_dir
        self.log_path = os.path.join(self.save_dir, 'train.log')
        self.tb_log_path = os.path.join(tb_log_dir, 'train.log')
        
        self.tee = TeeLogger(self.log_path, mode=mode)
        sys.stdout = self.tee

        print(f"[{self.experiment_name}] TensorBoard logs: {tb_log_dir}", flush=True)
        print(f"[{self.experiment_name}] Dedicated train log: {self.log_path}", flush=True)
        print(f"💡 HINT: To monitor real-time training progress, view the tail of the log file:\n   tail -f {self.log_path}\n", flush=True)

    def log_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.writer.add_scalar(tag, scalar_value, global_step)
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.log({tag: scalar_value}, step=global_step)
            except Exception:
                pass

    def log_image(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        self.writer.add_image(tag, img_tensor, global_step)

    def close(self) -> None:
        print(f"\n💡 HINT: Training completed! Inspect the tail of the log file using:\n   tail -n 50 {self.log_path}\n", flush=True)
        if self.writer:
            self.writer.close()
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if hasattr(self, 'tee') and self.tee:
            # Sync train.log into tb_logs/train.log
            try:
                import shutil
                shutil.copy(self.log_path, self.tb_log_path)
            except Exception:
                pass
            sys.stdout = self.tee.terminal
            self.tee.close()

