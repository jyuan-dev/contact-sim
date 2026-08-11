"""
Training helper functions: learning rate scheduling, random seed initialization, and hardware device setup.
"""

import math
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cosine_anneal_with_warmup(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float = 1e-5) -> float:
    """
    Calculate learning rate with linear warmup (from min_lr to lr) and cosine annealing decay.
    """
    if step < warmup_steps:
        return min_lr + (lr - min_lr) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def get_device(requested_device: str = None) -> torch.device:
    """Get PyTorch device (cuda/cpu)."""
    if requested_device:
        return torch.device(requested_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str,
                          device: torch.device = None) -> None:
    """
    Load a checkpoint into *model*, validating that keys match.

    Uses ``strict=False`` to discover missing and unexpected keys, then
    **raises** ``ValueError`` if either set is non-empty.  This catches
    wrapper-level prefix mismatches, DDP ``module.`` leftovers, and
    model-vs-checkpoint architecture drift instead of silently loading
    garbage.
    """
    ckpt = torch.load(ckpt_path, map_location=device or "cpu")
    state = ckpt.get("model_state", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("loss_fn.")]

    if missing:
        raise ValueError(
            f"Checkpoint is missing {len(missing)} key(s) required by the model. "
            f"First 5: {missing[:5]}\n"
            f"Checkpoint: {ckpt_path}"
        )
    if unexpected:
        raise ValueError(
            f"Checkpoint has {len(unexpected)} key(s) not present in the model. "
            f"First 5: {unexpected[:5]}\n"
            f"Checkpoint: {ckpt_path}"
        )
