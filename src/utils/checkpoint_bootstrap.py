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
from typing import Any

import torch
from src.config.run_config import RunConfig, load_snapshot
from src.models.factory import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def bootstrap_checkpoint(ckpt_path: str, cli_overrides: dict | None = None) -> tuple[Any, dict]:
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
    if not isinstance(ckpt_path, (str, os.PathLike)):
        raise TypeError(f"ckpt_path must be a path string, got {type(ckpt_path).__name__}")
    if cli_overrides is not None and not isinstance(cli_overrides, dict):
        raise TypeError(f"cli_overrides must be a dict or None, got {type(cli_overrides).__name__}")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_dir = os.path.dirname(ckpt_path)
    run_cfg = load_snapshot(ckpt_dir)

    if run_cfg is not None:
        cfg_dict = run_cfg.to_dict()
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
                "stage1_ckpt_path": "scratch/checkpoints/savi_pusht_default_4ep/savi_best.pt",
                **arch,
            }
        }

    if cli_overrides:
        # Dotted-path overrides (device, batch_size, dataset.train_frac, ...)
        # applied through the validated config seam.
        cfg_dict = RunConfig.from_dict(cfg_dict, permissive=True) \
            .apply_overrides(cli_overrides).to_dict()

    model = build_model(cfg_dict)
    return model, cfg_dict
