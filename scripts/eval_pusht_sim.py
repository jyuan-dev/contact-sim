#!/usr/bin/env python3
"""
CLOSED-LOOP PUSHT SIMULATOR EVALUATION SCRIPT

Evaluates object-centric visual learning models (SAVi slot perception + INTACT Intent-to-Action Actor + SlotFormer)
in the interactive PushT Gym Simulator (`gym-pusht`).

Supports multiple control modes:
1. `static`: Direct INTACT intent-to-action control towards canonical solved T-shape goal slots (z_goal).
2. `world_model`: Model Predictive Control (CEM shooting) rollout in latent slot space towards z_goal.
3. `both`: Side-by-side comparative benchmarking of both modes.

Usage:
    python scripts/eval_pusht_sim.py --ckpt_path scratch/checkpoints/ocvp_intact_slotformer_pusht_sigreg001/ocvp_intact_slotformer_best.pt --num_episodes 10 --save_gif --device cuda
"""

import os
import sys
import json
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
from src.eval.pusht_sim_utils import load_model, preprocess_obs, get_canonical_goal_slots


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate INTACT & SlotFormer in PushT Gym Simulator")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="scratch/checkpoints/ocvp_intact_slotformer_pusht_sigreg001/ocvp_intact_slotformer_best.pt",
        help="Path to trained Stage 2 INTACT SlotFormer model checkpoint",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per mode (default: 20)",
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
        choices=["static", "subgoal_chaining", "world_model", "cem_mpc", "both", "all"],
        default="both",
        help="Intent goal mode: 'static' (z_goal), 'subgoal_chaining', 'world_model'/'cem_mpc' (CEM latent rollout), 'both', or 'all'",
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


def plan_action_cem(
    model,
    z_history: torch.Tensor,
    z_goal: torch.Tensor,
    horizon: int = 4,
    num_samples: int = 64,
    num_elites: int = 8,
    num_iterations: int = 3,
    action_dim: int = 2,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Model Predictive Control using Cross-Entropy Method (CEM) trajectory optimization
    over candidate actions in latent slot space.
    """
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device) * 0.5

    # Object slots are k >= 1 (excluding robot slot 0)
    target_obj_slots = z_goal[:, 1:]  # [1, K-1, D]

    for _ in range(num_iterations):
        # Sample candidate action sequences [N, H, action_dim]
        actions = torch.randn(num_samples, horizon, action_dim, device=device) * std.unsqueeze(0) + mean.unsqueeze(0)
        actions = actions.clamp(-1.5, 1.5)

        # Expand z_history to [N, history_len, K, D]
        history_expanded = z_history.expand(num_samples, -1, -1, -1)

        # Rollout candidate trajectories in batch
        pred_slots = model.rollouter(history_expanded, pred_len=horizon, actions=actions)  # [N, H, K, D]

        # Evaluate distance of predicted object slots at horizon H against target goal slots
        pred_obj_final = pred_slots[:, -1, 1:]  # [N, K-1, D]
        costs = F.mse_loss(pred_obj_final, target_obj_slots.expand(num_samples, -1, -1), reduction="none").mean(dim=(-2, -1))

        # Select top elite trajectories
        elite_indices = torch.topk(costs, k=num_elites, largest=False).indices
        elite_actions = actions[elite_indices]  # [num_elites, H, action_dim]

        # Update distribution parameters
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.05)

    # Predict the corresponding physical sub-goal slots from the planned action sequence
    best_action_seq = mean.unsqueeze(0)  # [1, H, action_dim]
    best_pred_slots = model.rollouter(z_history, pred_len=horizon, actions=best_action_seq)  # [1, H, K, D]
    z_subgoal = best_pred_slots[:, 0]  # [1, K, D]

    # Return full multi-step action chunk [H, action_dim]
    return mean, z_subgoal


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
    Run closed-loop evaluation in PushT Gym simulator with persistent recurrent SAVi perception.
    """
    inner_model = model_wrapper.model
    inner_savi = (
        inner_model.stage1_model.inner_savi()
        if hasattr(inner_model, "stage1_model")
        else inner_model.inner_savi()
    )
    actor = getattr(inner_model, "idm_actor", getattr(inner_model, "intact_actor", None))

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

    # Extract canonical 100% solved goal slots
    static_z_goal, goal_img = get_canonical_goal_slots(model_wrapper, device)

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        obs, info = env.reset(seed=ep_seed)

        pusht_env = env.unwrapped
        agent_pos = np.array([pusht_env.agent.position.x, pusht_env.agent.position.y], dtype=np.float32)

        ep_return = 0.0
        final_coverage = info.get("coverage", 0.0)
        solved_step = None
        ep_frames = []

        # Reset SAVi recurrent hidden state at start of episode
        if hasattr(inner_savi, "_reset_rnn"):
            inner_savi._reset_rnn()
        prev_slots = None
        history_slot_list = []
        prev_action_tensor = None

        history_len = getattr(inner_model, "history_len", 2)

        for step in range(max_steps):
            if save_gif and ep < num_gif_episodes:
                ep_frames.append(obs.copy())

            obs_tensor = preprocess_obs(obs, device)

            # Persistent sequential SAVi slot extraction
            with torch.no_grad():
                post_slots, _ = inner_savi.encode(obs_tensor, prev_slots=prev_slots)
                current_slot = post_slots[:, 0]  # [1, K, D]
                prev_slots = current_slot

            history_slot_list.append(current_slot)
            if len(history_slot_list) > history_len:
                history_slot_list.pop(0)

            if goal_mode == "static":
                with torch.no_grad():
                    act_mu, _ = actor(
                        z_curr=current_slot,
                        z_next=static_z_goal,
                        prev_action=prev_action_tensor,
                    )
                    predicted_delta = act_mu[0].cpu().numpy()
                    prev_action_tensor = act_mu

            elif goal_mode == "subgoal_chaining":
                # Generate physically consistent intermediate sub-goal from world model rollout
                if len(history_slot_list) < history_len:
                    z_history = current_slot.unsqueeze(1).repeat(1, history_len, 1, 1)
                else:
                    z_history = torch.stack(history_slot_list, dim=1)  # [1, history_len, K, D]

                with torch.no_grad():
                    _, z_subgoal = plan_action_cem(
                        model=inner_model,
                        z_history=z_history,
                        z_goal=static_z_goal,
                        horizon=4,
                        num_samples=64,
                        device=device,
                    )

                    act_mu, _ = actor(
                        z_curr=current_slot,
                        z_next=z_subgoal,
                        prev_action=prev_action_tensor,
                    )
                    predicted_delta = act_mu[0].cpu().numpy()
                    prev_action_tensor = act_mu

            elif goal_mode in ("world_model", "cem_mpc"):
                if len(history_slot_list) < history_len:
                    z_history = current_slot.unsqueeze(1).repeat(1, history_len, 1, 1)
                else:
                    z_history = torch.stack(history_slot_list, dim=1)  # [1, history_len, K, D]

                with torch.no_grad():
                    act_mu, _ = plan_action_cem(
                        model=inner_model,
                        z_history=z_history,
                        z_goal=static_z_goal,
                        horizon=4,
                        num_samples=64,
                        device=device,
                    )
                    predicted_delta = act_mu.cpu().numpy()
                    prev_action_tensor = act_mu[-1:]

            # Unroll multi-step action chunk (K=4)
            actions_to_exec = predicted_delta.reshape(-1, 2)
            for act_sub in actions_to_exec:
                target_x = agent_pos[0] + act_sub[0] * action_scale
                target_y = agent_pos[1] + act_sub[1] * action_scale
                target_pos = np.array([
                    np.clip(target_x, 15.0, 497.0),
                    np.clip(target_y, 15.0, 497.0),
                ], dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(target_pos)
                agent_pos = np.array([pusht_env.agent.position.x, pusht_env.agent.position.y], dtype=np.float32)

                if save_gif and ep < num_gif_episodes:
                    ep_frames.append(obs.copy())

                ep_return += reward
                final_coverage = info.get("coverage", final_coverage)

                if info.get("is_success", False) and solved_step is None:
                    solved_step = step + 1

                if terminated or truncated:
                    break

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
        saved_gif_paths = []
        for ep_i, ep_f in enumerate(gif_frames_all):
            if not ep_f:
                continue
            gif_filename = f"scratch/pusht_sim_rollout_{goal_mode}_ep{ep_i}.gif"
            gif_pil_frames = [Image.fromarray(f) for f in ep_f]
            gif_pil_frames[0].save(
                gif_filename,
                save_all=True,
                append_images=gif_pil_frames[1:],
                duration=50,
                loop=0,
            )
            print(f"[PushT Eval] Saved rollout GIF: {gif_filename} (loop=0)")
            saved_gif_paths.append(gif_filename)
        results["gif_paths"] = saved_gif_paths

    return results


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("[PushT Eval] Initializing gym-pusht environment...")
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=64, observation_height=64)

    model_wrapper = load_model(args.ckpt_path, device)

    if args.goal_mode == "both":
        modes_to_run = ["static", "cem_mpc"]
    elif args.goal_mode == "all":
        modes_to_run = ["static", "subgoal_chaining", "cem_mpc"]
    else:
        modes_to_run = [args.goal_mode]
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
