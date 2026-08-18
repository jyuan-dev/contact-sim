#!/usr/bin/env python3
"""
SUB-GOAL CONDITIONED PUSHT SIMULATION EVALUATION & RENDERING SCRIPT

Demonstrates sub-goal waypoint conditioning for INTACT in the PushT Gym Simulator:
1. Renders visual sub-goal targets (both Ground-Truth Environment Sub-goals and Decoded Slot Sub-goals).
2. Evaluates closed-loop tracking of sequential sub-goals by the INTACT Actor.
3. Saves high-quality composite side-by-side animated GIFs (loop=0) in scratch/.

Usage:
    python scripts/eval_subgoal_pusht.py --ckpt_path scratch/checkpoints/ocvp_intact_slotformer_pusht_sigreg001/ocvp_intact_slotformer_best.pt --num_episodes 5 --num_subgoals 8 --save_gif
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

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
from scripts.eval_pusht_sim import load_model, preprocess_obs, get_canonical_goal_slots


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sub-Goal Navigation & Control in PushT Simulator")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="scratch/checkpoints/ocvp_intact_slotformer_pusht_sigreg001/ocvp_intact_slotformer_best.pt",
        help="Path to trained INTACT checkpoint",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--num_subgoals",
        type=int,
        default=8,
        help="Number of sequential sub-goals interpolated between start and goal (default: 8)",
    )
    parser.add_argument(
        "--steps_per_subgoal",
        type=int,
        default=35,
        help="Maximum simulation steps allocated per sub-goal waypoint (default: 35)",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=30.0,
        help="Scaling factor to convert INTACT delta to pixel offset (default: 30.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for evaluation (default: 42)",
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
        default=True,
        help="Save composite sub-goal visualization GIFs in scratch/",
    )
    parser.add_argument(
        "--out_gif",
        type=str,
        default="scratch/pusht_subgoal_rollout.gif",
        help="Output GIF filepath",
    )
    return parser.parse_args()


def render_subgoal_frame(subgoal_state: np.ndarray) -> np.ndarray:
    """
    Render an RGB frame representing the target sub-goal state [agent_x, agent_y, block_x, block_y, block_angle].
    """
    sub_env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=64, observation_height=64)
    obs, _ = sub_env.reset(seed=42)
    pusht = sub_env.unwrapped

    # Set block angle first, then block position and agent position
    pusht.block.angle = float(subgoal_state[4])
    pusht.block.position = (float(subgoal_state[2]), float(subgoal_state[3]))
    pusht.agent.position = (float(subgoal_state[0]), float(subgoal_state[1]))
    pusht.space.step(1e-4)

    subgoal_img = pusht.get_obs()
    sub_env.close()
    return subgoal_img


def run_subgoal_evaluation(args):
    device = torch.device(args.device)
    model_wrapper = load_model(args.ckpt_path, device)
    inner_model = model_wrapper.model
    inner_savi = inner_model.stage1_model.inner_savi()
    actor = inner_model.intact_actor

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=64, observation_height=64)

    print("\n" + "=" * 75)
    print(f"EVALUATING SUB-GOAL NAVIGATION & CONTROL IN PUSHT SIMULATOR")
    print(f"  Episodes: {args.num_episodes} | Sub-goals per ep: {args.num_subgoals} | Steps per sub-goal: {args.steps_per_subgoal}")
    print("=" * 75)

    all_gif_frames = []
    episode_results = []

    for ep in range(args.num_episodes):
        ep_seed = args.seed + ep
        obs, info = env.reset(seed=ep_seed)
        pusht = env.unwrapped

        agent_start = np.array([pusht.agent.position.x, pusht.agent.position.y], dtype=np.float32)
        block_start = np.array([pusht.block.position.x, pusht.block.position.y, pusht.block.angle], dtype=np.float32)

        # Target goal pose
        goal_pose = pusht.goal_pose  # [256, 256, pi/4]
        block_target = np.array([goal_pose[0], goal_pose[1], goal_pose[2]], dtype=np.float32)

        print(f"\n[Episode {ep+1:02d} (Seed {ep_seed})] Block Start: ({block_start[0]:.1f}, {block_start[1]:.1f}, a={block_start[2]:.2f}) -> Goal: ({block_target[0]:.1f}, {block_target[1]:.1f}, a={block_target[2]:.2f})")

        # Generate sequence of sub-goals
        subgoals = []
        for g_idx in range(1, args.num_subgoals + 1):
            alpha = g_idx / float(args.num_subgoals)
            # Interpolate block position and angle
            b_x = block_start[0] + alpha * (block_target[0] - block_start[0])
            b_y = block_start[1] + alpha * (block_target[1] - block_start[1])
            # Shortest angle interpolation
            diff_angle = (block_target[2] - block_start[2] + np.pi) % (2 * np.pi) - np.pi
            b_a = block_start[2] + alpha * diff_angle

            # Compute pushing direction from block to goal
            push_dir = np.array([block_target[0] - b_x, block_target[1] - b_y], dtype=np.float32)
            norm = np.linalg.norm(push_dir)
            if norm > 1e-4:
                push_dir = push_dir / norm
            else:
                push_dir = np.array([0.0, 1.0], dtype=np.float32)

            # Place agent slightly behind the block along push direction (or at start when approaching)
            if g_idx == 1:
                a_x = (agent_start[0] + b_x) / 2.0
                a_y = (agent_start[1] + b_y) / 2.0
            else:
                # Behind block by ~35 pixels
                a_x = b_x - push_dir[0] * 35.0
                a_y = b_y - push_dir[1] * 35.0

            subgoal_state = np.array([a_x, a_y, b_x, b_y, b_a], dtype=np.float32)
            subgoal_img = render_subgoal_frame(subgoal_state)
            subgoals.append((subgoal_state, subgoal_img))

        # Reset SAVi recurrent perception for the episode
        if hasattr(inner_savi, "_reset_rnn"):
            inner_savi._reset_rnn()
        prev_slots = None
        prev_action_tensor = None

        total_steps = 0
        final_coverage = info.get("coverage", 0.0)
        solved = False

        for g_idx, (sub_state, sub_img) in enumerate(subgoals):
            # Encode sub-goal image to obtain target sub-goal slots z_subgoal
            sub_tensor = preprocess_obs(sub_img, device)
            with torch.no_grad():
                # Extract sub-goal slots cleanly with static init
                sub_savi = model_wrapper.model.stage1_model.inner_savi() if hasattr(model_wrapper.model, "stage1_model") else model_wrapper.model.inner_savi()
                if hasattr(sub_savi, "_reset_rnn"):
                    sub_savi._reset_rnn()
                sub_slots_all, _ = sub_savi.encode(sub_tensor)
                z_subgoal = sub_slots_all[:, 0]  # [1, K, D]

            # Step environment towards this sub-goal
            for s_idx in range(args.steps_per_subgoal):
                agent_pos = np.array([pusht.agent.position.x, pusht.agent.position.y], dtype=np.float32)
                block_pos = np.array([pusht.block.position.x, pusht.block.position.y, pusht.block.angle], dtype=np.float32)

                obs_tensor = preprocess_obs(obs, device)
                with torch.no_grad():
                    post_slots, _ = inner_savi.encode(obs_tensor, prev_slots=prev_slots)
                    current_slot = post_slots[:, 0]
                    prev_slots = current_slot

                    # Query INTACT Actor for action navigating z_curr -> z_subgoal
                    act_mu, act_logstd = actor(
                        z_curr=current_slot,
                        z_next=z_subgoal,
                        prev_action=prev_action_tensor,
                    )

                delta = act_mu[0].cpu().numpy()
                target_pos = np.array([
                    np.clip(agent_pos[0] + delta[0] * args.action_scale, 15.0, 497.0),
                    np.clip(agent_pos[1] + delta[1] * args.action_scale, 15.0, 497.0),
                ], dtype=np.float32)

                prev_action_tensor = act_mu
                obs, reward, terminated, truncated, info = env.step(target_pos)
                final_coverage = info.get("coverage", final_coverage)
                total_steps += 1

                # Construct composite visualization frame: [Live Obs | Target Subgoal | Subgoal Delta Text]
                if args.save_gif and ep < 2:
                    h, w = obs.shape[:2]
                    canvas = np.zeros((h + 24, w * 2 + 10, 3), dtype=np.uint8)
                    canvas[24:24+h, 0:w] = obs
                    canvas[24:24+h, w+10:w*2+10] = sub_img

                    # Overlay labels
                    cv2.putText(canvas, f"Live (Cov: {final_coverage*100:.0f}%)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    cv2.putText(canvas, f"Subgoal {g_idx+1}/{args.num_subgoals}", (w + 14, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1)
                    all_gif_frames.append(canvas)

                if final_coverage >= 0.95:
                    solved = True
                    break

            # Distance to sub-goal block position
            dist_to_sub = np.linalg.norm(block_pos[:2] - sub_state[2:4])
            print(f"  Sub-goal {g_idx+1:02d}/{args.num_subgoals:02d}: Target Block=({sub_state[2]:.1f}, {sub_state[3]:.1f}) | Actual=({block_pos[0]:.1f}, {block_pos[1]:.1f}) | Dist={dist_to_sub:.1f}px | Coverage={final_coverage*100:.1f}%")

            if solved:
                print(f"  🎉 Solved at sub-goal {g_idx+1} (Total Steps: {total_steps})!")
                break

        episode_results.append({
            "episode": ep,
            "seed": ep_seed,
            "final_coverage": float(final_coverage),
            "solved": solved,
            "steps": total_steps,
        })

    env.close()

    if args.save_gif and all_gif_frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_gif)), exist_ok=True)
        pil_frames = [Image.fromarray(f) for f in all_gif_frames]
        pil_frames[0].save(
            args.out_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=50,
            loop=0,
        )
        print(f"\n[PushT Eval] Saved composite Sub-Goal visualization GIF to: {args.out_gif} (loop=0)")

    print("\n" + "=" * 75)
    print("SUB-GOAL EVALUATION SUMMARY:")
    avg_cov = np.mean([r["final_coverage"] for r in episode_results])
    max_cov = np.max([r["final_coverage"] for r in episode_results])
    print(f"  Avg Coverage: {avg_cov*100:.2f}% | Max Coverage: {max_cov*100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    args = parse_args()
    run_subgoal_evaluation(args)
