"""
Training helper functions: learning rate scheduling, random seed initialization, and hardware device setup.
"""

import math
import os
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


# ── Checkpoint saving / loading ──────────────────────────────────────────────

def save_checkpoint(model: torch.nn.Module, path: str, epoch: int) -> None:
    """
    Save a checkpoint in the canonical format: the model's own state dict
    (no unwrapping) plus the epoch.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint_state(model: torch.nn.Module, ckpt_path: str,
                          device: torch.device = None) -> None:
    """
    Load a checkpoint into *model*, validating that keys match.

    Recognized state-dict formats (documented table):

    ========================  =====================================
    format                    adaptation
    ========================  =====================================
    canonical                 model's own keys — exact match
    legacy inner              bare core keys — prefix ``model.model.``
    legacy savi               ``model.*`` — prefix ``model.``
    legacy wrapper            ``model.model.*`` — strip one ``model.``
    DDP                       ``module.`` — strip
    ========================  =====================================

    Uses ``strict=False`` to discover missing and unexpected keys, then
    **raises** ``ValueError`` if either set is non-empty — catching wrapper
    prefix mismatches and model-vs-checkpoint architecture drift instead of
    silently loading garbage.
    """
    ckpt = torch.load(ckpt_path, map_location=device or "cpu")
    state = ckpt.get("model_state", ckpt)

    model_keys = set(model.state_dict().keys())
    adapted_state = {}
    for k, v in state.items():
        if k in model_keys:
            adapted_state[k] = v
        elif f"model.model.{k}" in model_keys:
            adapted_state[f"model.model.{k}"] = v
        elif f"model.{k}" in model_keys:
            adapted_state[f"model.{k}"] = v
        elif k.startswith("model.model.") and k[6:] in model_keys:
            adapted_state[k[6:]] = v
        elif k.startswith("model.") and k[6:] in model_keys:
            adapted_state[k[6:]] = v
        elif k.startswith("module.") and k[7:] in model_keys:
            adapted_state[k[7:]] = v
        else:
            adapted_state[k] = v

    missing, unexpected = model.load_state_dict(adapted_state, strict=False)

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

