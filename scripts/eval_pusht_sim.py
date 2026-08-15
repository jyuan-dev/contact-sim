#!/usr/bin/env python3
"""
CLOSED-LOOP PUSHT SIMULATOR EVALUATION SCRIPT

Evaluates object-centric visual learning models (SAVi slot perception + INTACT Intent-to-Action Actor + SlotFormer)
in the interactive PushT Gym Simulator (`gym-pusht`).

Supports two intent conditioning modes:
1. `static`: Target solved T-shape goal slots (z_goal)
2. `world_model`: Step-by-step SlotFormer world model sub-goal prediction (z_{t+1})
3. `both`: Side-by-side comparative benchmarking of both modes.

Usage:
    python scripts/eval_pusht_sim.py --num_episodes 50 --max_steps 300 --save_gif
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure repository root and third_party/gym-pusht are on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GYM_PUSHT_PATH = os.path.join(REPO_ROOT, "third_party", "gym-pusht")
if GYM_PUSHT_PATH not in sys.path:
    sys.path.insert(0, GYM_PUSHT_PATH)

import gymnasium as gym
import gym_pusht
from omegaconf import OmegaConf
from src.models.factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate INTACT & SlotFormer in PushT Gym Simulator")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="scratch/checkpoints/ocvp_intact_slotformer_pusht/ocvp_intact_slotformer_best.pt",
        help="Path to trained Stage 2 INTACT SlotFormer model checkpoint",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per mode (default: 50)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum simulation environment steps per episode (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for environment resets (default: 42)",
    )
    parser.add_argument(
        "--goal_mode",
        type=str,
        choices=["static", "world_model", "pidm", "both"],
        default="both",
        help="Intent goal mode: 'static' (z_goal), 'world_model' (z_{t+1}), 'pidm' (goal-conditioned rollout), or 'both'",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=30.0,
        help="Scaling factor to convert INTACT displacement vector to pixel offset (default: 30.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Save animated rollout GIFs in scratch/",
    )
    parser.add_argument(
        "--num_gif_episodes",
        type=int,
        default=3,
        help="Number of rollout GIFs to save (default: 3)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="scratch/eval_pusht_sim_results.json",
        help="Path to output JSON results file",
    )
    return parser.parse_args()


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
                "stage1_ckpt_path": "scratch/checkpoints/savi_pusht/savi_best.pt",
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


def get_static_goal_slots(model, device: torch.device) -> torch.Tensor:
    """
    Generate target goal slots (z_goal) using a dedicated temporary PushT gym environment.
    """
    goal_env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=64, observation_height=64)
    target_obs, _ = goal_env.reset(seed=999)
    for _ in range(5):
        target_obs, _, _, _, _ = goal_env.step(np.array([256.0, 256.0], dtype=np.float32))
    goal_env.close()
    
    goal_tensor = preprocess_obs(target_obs, device)
    with torch.no_grad():
        goal_slots = model.model.extract_slots(goal_tensor)  # [1, 1, K, D]
    return goal_slots[:, 0]  # [1, K, D]


def evaluate_closed_loop(
    env,
    model_wrapper,
    goal_mode: str,
    num_episodes: int,
    max_steps: int,
    base_seed: int,
    action_scale: float,
    device: torch.device,
    save_gif: bool = False,
    num_gif_episodes: int = 3,
):
    """
    Run closed-loop evaluation in PushT Gym simulator.
    """
    inner_model = model_wrapper.model
    results = {
        "mode": goal_mode,
        "episodes": [],
        "coverages": [],
        "returns": [],
        "successes": [],
        "steps_to_solve": [],
    }

    gif_frames_all = []
    print(f"\n" + "=" * 70)
    print(f"STARTING CLOSED-LOOP EVALUATION | Mode: {goal_mode.upper()} | Episodes: {num_episodes}")
    print("=" * 70)

    static_z_goal = None
    if goal_mode in ("static", "pidm"):
        # Both modes plan toward the static goal area; 'pidm' needs the goal
        # too — otherwise plan_action runs with goal_video_or_slots=None and
        # silently degrades to an unconditioned rollout.
        static_z_goal = get_static_goal_slots(model_wrapper, device)

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        obs, info = env.reset(seed=ep_seed)

        pusht_env = env.unwrapped
        agent_pos = np.array([pusht_env.agent.position.x, pusht_env.agent.position.y], dtype=np.float32)

        
        ep_return = 0.0
        final_coverage = info.get("coverage", 0.0)
        solved_step = None
        ep_frames = []

        history_slot_list = []
        prev_action_tensor = None

        history_len = getattr(inner_model, "history_len", 2)

        for step in range(max_steps):
            if save_gif and ep < num_gif_episodes:
                ep_frames.append(obs.copy())

            obs_tensor = preprocess_obs(obs, device)

            with torch.no_grad():
                current_slot = inner_model.extract_slots(obs_tensor)[:, 0]  # [1, K, D]
            history_slot_list.append(current_slot)

            if len(history_slot_list) > history_len:
                history_slot_list.pop(0)

            actor = getattr(inner_model, "idm_actor", getattr(inner_model, "intact_actor", None))
            if actor is None:
                raise RuntimeError(
                    "Checkpoint has no idm_actor/intact_actor — action-conditioned "
                    "goal modes require a PIDM/INTACT model.")

            if goal_mode == "static":
                z_target = static_z_goal
                with torch.no_grad():
                    act_mu, _ = actor(
                        z_curr=current_slot,
                        z_next=z_target,
                        prev_action=prev_action_tensor,
                    )
                    predicted_delta = act_mu[0].cpu().numpy()
            elif goal_mode in ("world_model", "pidm"):
                if len(history_slot_list) < history_len:
                    z_history = current_slot.unsqueeze(1).repeat(1, history_len, 1, 1)
                else:
                    z_history = torch.stack(history_slot_list, dim=1)  # [1, history_len, K, D]

                with torch.no_grad():
                    if hasattr(inner_model, "plan_action"):
                        act_mu = inner_model.plan_action(
                            history_video_or_slots=z_history,
                            goal_video_or_slots=static_z_goal,
                            prev_action=prev_action_tensor,
                        )
                    else:
                        pred_next_slots = inner_model.rollouter(z_history, pred_len=1)  # [1, 1, K, D]
                        z_target = pred_next_slots[:, 0]  # [1, K, D]
                        act_mu, _ = actor(
                            z_curr=current_slot,
                            z_next=z_target,
                            prev_action=prev_action_tensor,
                        )
                    predicted_delta = act_mu[0].cpu().numpy()

            prev_action_tensor = act_mu

            target_x = agent_pos[0] + predicted_delta[0] * action_scale
            target_y = agent_pos[1] + predicted_delta[1] * action_scale
            target_pos = np.array([
                np.clip(target_x, 0.0, 512.0),
                np.clip(target_y, 0.0, 512.0)
            ], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(target_pos)
            agent_pos = np.array([pusht_env.agent.position.x, pusht_env.agent.position.y], dtype=np.float32)

            ep_return += reward
            final_coverage = info.get("coverage", final_coverage)

            if info.get("is_success", False) and solved_step is None:
                solved_step = step + 1

            if terminated or truncated:
                break

        is_success = bool(final_coverage >= 0.95)
        results["coverages"].append(float(final_coverage))
        results["returns"].append(float(ep_return))
        results["successes"].append(is_success)
        if solved_step is not None:
            results["steps_to_solve"].append(solved_step)

        results["episodes"].append({
            "episode_idx": ep,
            "seed": ep_seed,
            "final_coverage": float(final_coverage),
            "total_return": float(ep_return),
            "success": is_success,
            "solved_step": solved_step,
        })

        if save_gif and ep < num_gif_episodes and ep_frames:
            gif_frames_all.append(ep_frames)

        status_str = f"SOLVED at step {solved_step}" if is_success else f"Coverage: {final_coverage*100:.1f}%"
        print(f"  Ep {ep+1:02d}/{num_episodes:02d} (Seed {ep_seed}): Return={ep_return:.2f} | {status_str}")

    avg_coverage = float(np.mean(results["coverages"]))
    max_coverage = float(np.max(results["coverages"]))
    success_rate = float(np.mean(results["successes"])) * 100.0
    avg_return = float(np.mean(results["returns"]))
    avg_steps = float(np.mean(results["steps_to_solve"])) if results["steps_to_solve"] else float(max_steps)

    results["summary"] = {
        "avg_coverage": avg_coverage,
        "max_coverage": max_coverage,
        "success_rate_percent": success_rate,
        "avg_return": avg_return,
        "avg_steps_to_solve": avg_steps,
    }

    print("\n" + "-" * 70)
    print(f"SUMMARY RESULTS ({goal_mode.upper()} MODE):")
    print(f"  Success Rate:        {success_rate:.2f}% ({sum(results['successes'])}/{num_episodes})")
    print(f"  Avg Target Coverage: {avg_coverage * 100:.2f}% (Max: {max_coverage * 100:.2f}%)")
    print(f"  Avg Episode Return:  {avg_return:.2f}")
    print(f"  Avg Steps to Solve:  {avg_steps:.1f}")
    print("-" * 70)

    if save_gif and gif_frames_all:
        os.makedirs("scratch", exist_ok=True)
        gif_filename = f"scratch/pusht_sim_rollout_{goal_mode}.gif"
        
        gif_pil_frames = []
        for frames in gif_frames_all:
            for f in frames:
                gif_pil_frames.append(Image.fromarray(f))

        # Always specify loop=0 per workspace rules for infinite loop play
        gif_pil_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=gif_pil_frames[1:],
            duration=50,
            loop=0,
        )
        print(f"[PushT Eval] Saved rollout GIF: {gif_filename} (loop=0)")
        results["gif_path"] = gif_filename

    return results


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("[PushT Eval] Initializing gym-pusht environment...")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=64, observation_height=64)

    model_wrapper = load_model(args.ckpt_path, device)

    modes_to_run = ["static", "world_model"] if args.goal_mode == "both" else [args.goal_mode]
    all_mode_results = {}

    for mode in modes_to_run:
        res = evaluate_closed_loop(
            env=env,
            model_wrapper=model_wrapper,
            goal_mode=mode,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            base_seed=args.seed,
            action_scale=args.action_scale,
            device=device,
            save_gif=args.save_gif,
            num_gif_episodes=args.num_gif_episodes,
        )
        all_mode_results[mode] = res

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_mode_results, f, indent=2)
    print(f"\n[PushT Eval] Full evaluation results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
