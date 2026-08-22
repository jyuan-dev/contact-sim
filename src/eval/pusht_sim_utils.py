"""
Shared helpers for closed-loop PushT gym-simulator evaluation scripts.

Extracted from ``scripts/eval_pusht_sim.py`` so that ``scripts/eval_subgoal_pusht.py``
and ``scripts/eval_world_model_pusht.py`` no longer import from another runnable
script (which coupled their behavior to that script's CLI/``main()``).
"""

from __future__ import annotations

import os
import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf

from src.models.factory import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(ckpt_path: str, device: torch.device):
    """Load model wrapper and restore checkpoint weights."""
    abs_ckpt_path = os.path.join(REPO_ROOT, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    if not os.path.exists(abs_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {abs_ckpt_path}")

    print(f"[PushT Eval] Loading checkpoint: {abs_ckpt_path}")
    cfg_dir = os.path.dirname(abs_ckpt_path)
    config_yaml = os.path.join(cfg_dir, "config.yaml")

    if os.path.exists(config_yaml):
        cfg = OmegaConf.load(config_yaml)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = {
            "model": {
                "name": "ocvp_intact_slotformer",
                "type": "ocvp_intact_slotformer",
                "rollouter_type": "cocvp",
                "stage1_ckpt_path": "scratch/checkpoints/savi_pusht_default_4ep/savi_best.pt",
                "history_len": 2,
                "rollout_len": 4,
                "d_model": 128,
                "num_layers": 4,
                "num_heads": 8,
                "ffn_dim": 512,
                "raw_action_dim": 2,
                "action_embed_dim": 64,
                "condition_mode": "film",
                "use_intact_actor": True,
                "action_loss_weight": 1.0,
                "robot_slot_idx": 0,
            }
        }

    model_wrapper = build_model(cfg_dict).to(device)
    ckpt = torch.load(abs_ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)

    model_keys = set(model_wrapper.state_dict().keys())
    adapted_state = {}
    for k, v in state.items():
        if k in model_keys:
            adapted_state[k] = v
        elif f"model.{k}" in model_keys:
            adapted_state[f"model.{k}"] = v
        elif k.startswith("module.") and k[7:] in model_keys:
            adapted_state[k[7:]] = v
        elif k.startswith("model.") and k[6:] in model_keys:
            adapted_state[k[6:]] = v
        else:
            adapted_state[k] = v

    model_wrapper.load_state_dict(adapted_state, strict=False)
    model_wrapper.eval()
    print("[PushT Eval] Model state loaded successfully.")
    return model_wrapper


def preprocess_obs(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert gym observation image (64, 64, 3) uint8 to model input tensor [1, 1, 3, 64, 64] float in [-1, 1].
    """
    img_float = (obs.astype(np.float32) / 127.5) - 1.0
    img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor


def get_canonical_goal_slots(model_wrapper, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    """
    Generate ground-truth canonical goal slots (z_goal) by setting the T-block directly
    to the target goal pose (100% coverage) and encoding with Stage 1 SAVi.
    """
    goal_env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=64, observation_height=64)
    target_obs, _ = goal_env.reset(seed=42)
    pusht_inner = goal_env.unwrapped

    # Position the T-block precisely at the goal pose [256, 256, pi/4]
    goal_pose = pusht_inner.goal_pose
    pusht_inner.block.angle = float(goal_pose[2])
    pusht_inner.block.position = (float(goal_pose[0]), float(goal_pose[1]))
    pusht_inner.agent.position = (256.0, 100.0)
    pusht_inner.space.step(1e-4)

    goal_coverage = pusht_inner._get_coverage()
    solved_obs = pusht_inner.get_obs()
    goal_env.close()

    print(f"[PushT Eval] Canonical goal image rendered with coverage: {goal_coverage * 100:.1f}%")
    goal_tensor = preprocess_obs(solved_obs, device)

    inner_savi = (
        model_wrapper.model.stage1_model.inner_savi()
        if hasattr(model_wrapper.model, "stage1_model")
        else model_wrapper.model.inner_savi()
    )
    with torch.no_grad():
        if hasattr(inner_savi, "_reset_rnn"):
            inner_savi._reset_rnn()
        goal_slots, _ = inner_savi.encode(goal_tensor)  # [1, 1, K, D]

    return goal_slots[:, 0], solved_obs  # [1, K, D], np.ndarray


def decode_slot_image(stage1_model, slots: torch.Tensor) -> np.ndarray:
    """
    Decode slot latents [1, K, D] into an RGB image (64, 64, 3) uint8 using SAVi spatial decoder.
    """
    inner_savi = stage1_model.inner_savi() if hasattr(stage1_model, "inner_savi") else stage1_model
    with torch.no_grad():
        recon_flat, _, masks_flat, _ = inner_savi.decode(slots)
        recon_img = (recon_flat[0].clamp(-1.0, 1.0) + 1.0) * 127.5
        recon_np = recon_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return recon_np
