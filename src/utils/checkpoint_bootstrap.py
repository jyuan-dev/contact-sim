"""
Checkpoint bootstrap — reconstruct an experiment from a checkpoint.

The one implementation of "given a checkpoint path, reconstruct the
experiment": discover the saved training config (``config.yaml``, then
``.hydra/config.yaml``, hard fail on parse errors), apply optional CLI
overrides, and build the model wrapper. Weight loading stays separate —
``load_checkpoint_state`` in :mod:`src.utils.training_utils`.

Last resort for configless checkpoints (pre-config-snapshot legacy runs):
SlotFormer architecture sniffing from ``rollouter.*`` state-dict key shapes.
"""

import os

import torch
from omegaconf import OmegaConf

from src.models.factory import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _discover_config(ckpt_dir: str, ckpt_path: str):
    """Load the saved training config; None if neither candidate exists."""
    candidates = [
        os.path.join(ckpt_dir, "config.yaml"),
        os.path.join(ckpt_dir, ".hydra", "config.yaml"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                saved_cfg = OmegaConf.load(cand)
                print(f"[bootstrap] Loaded training configuration from: {cand}")
                return saved_cfg
            except Exception as e:
                raise RuntimeError(f"Failed to load training config file from '{cand}': {e}") from e
    return None


def sniff_slotformer_arch(state_dict) -> dict:
    """Infer SlotFormer architecture from ``rollouter.*`` state-dict key shapes.

    Last resort for checkpoints saved before config snapshots existed.
    """
    d_model, ffn_dim, num_layers = 128, 512, 4
    for k, v in state_dict.items():
        if "rollouter.in_proj.weight" in k:
            d_model = v.shape[0]
        elif "rollouter.transformer_encoder.layers.0.linear1.weight" in k:
            ffn_dim = v.shape[0]
        elif "rollouter.transformer_encoder.layers." in k:
            parts = k.split(".")
            for p in parts:
                if p.isdigit():
                    num_layers = max(num_layers, int(p) + 1)
    return {"d_model": d_model, "ffn_dim": ffn_dim, "num_layers": num_layers}


def bootstrap_checkpoint(ckpt_path: str, cli_overrides: dict | None = None):
    """
    Reconstruct an experiment from a checkpoint.

    Args:
        ckpt_path: path to the checkpoint (relative paths resolve against the repo root).
        cli_overrides: top-level config keys to override the saved config with
            (e.g. ``{'device': 'cpu', 'batch_size': 64}``).

    Returns:
        (model_wrapper, cfg_dict) — the built wrapper (weights NOT loaded) and
        the resolved config dict.
    """
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_dir = os.path.dirname(ckpt_path)
    saved_cfg = _discover_config(ckpt_dir, ckpt_path)

    if saved_cfg is not None:
        cfg_dict = OmegaConf.to_container(saved_cfg, resolve=True)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        arch = sniff_slotformer_arch(ckpt.get("model_state", ckpt))
        print(f"[bootstrap] No config found alongside checkpoint; "
              f"detected SlotFormer architecture: {arch}")
        cfg_dict = {
            "model": {
                "name": "slotformer",
                "type": "slotformer",
                "num_heads": 8,
                "stage1_ckpt_path": "scratch/checkpoints/savi_pusht/savi_best.pt",
                **arch,
            }
        }

    if cli_overrides:
        for key, value in cli_overrides.items():
            cfg_dict[key] = value

    model = build_model(cfg_dict)
    return model, cfg_dict
