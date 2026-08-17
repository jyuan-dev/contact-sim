#!/usr/bin/env python3
"""
ONLINE WORLD MODEL SUB-GOAL ROLLOUT & PLANNING IN PUSHT SIMULATOR

Generates reachable physical sub-goals purely online from the Stage 2 SlotFormer World Model:
1. Resets to arbitrary live environment layouts (with full random initial spawn).
2. Uses the SlotFormer World Model (cOCVP) to evaluate candidate action rollouts in latent slot space.
3. Selects the optimal trajectory towards the goal, generating dynamically predicted sub-goals z_{t+1}.
4. Decodes the predicted sub-goals back to RGB images and masks using the SAVi Spatial Decoder.
5. Executes closed-loop control in the live gym simulator and saves a composite GIF (loop=0).

Usage:
    python scripts/eval_world_model_pusht.py --ckpt_path scratch/checkpoints/ocvp_intact_slotformer_pusht_sigreg001/ocvp_intact_slotformer_best.pt --num_episodes 5 --save_gif
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="Evaluate Online World Model Planning in PushT Simulator")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="scratch/checkpoints/ocvp_intact_slotformer_pusht_sigreg001/ocvp_intact_slotformer_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=150,
        help="Maximum simulation steps per episode (default: 150)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="World model rollout planning horizon H (default: 4)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of candidate action trajectories sampled per step (default: 128)",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=30.0,
        help="Scaling factor to convert action to pixel offset (default: 30.0)",
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
        help="Save composite visualization GIFs in scratch/",
    )
    parser.add_argument(
        "--out_gif",
        type=str,
        default="scratch/pusht_world_model_subgoals.gif",
        help="Output GIF filepath",
    )
    return parser.parse_args()


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


def plan_with_world_model(
    model,
    z_history: torch.Tensor,
    z_goal: torch.Tensor,
    agent_pos: np.ndarray,
    block_pos: np.ndarray,
    goal_pos: np.ndarray,
    horizon: int = 4,
    num_samples: int = 128,
    num_elites: int = 16,
    num_iters: int = 3,
    device: torch.device = torch.device("cuda"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Plan optimal action sequence and extract dynamically predicted sub-goal slots z_{t+1}^*.
    Uses physics-guided CEM sampling over the SlotFormer rollouter.
    """
    # 1. Compute directional bias (approach if far, push toward goal if near)
    to_block = block_pos[:2] - agent_pos[:2]
    dist_to_block = np.linalg.norm(to_block)

    if dist_to_block > 45.0:
        # Move towards the block
        bias_dir = to_block / (dist_to_block + 1e-5)
    else:
        # Behind block pushing toward the goal
        to_goal = goal_pos[:2] - block_pos[:2]
        bias_dir = to_goal / (np.linalg.norm(to_goal) + 1e-5)

    prior_mean = torch.tensor(bias_dir, dtype=torch.float32, device=device).unsqueeze(0).repeat(horizon, 1) * 0.4
    mean = prior_mean.clone()
    std = torch.ones(horizon, 2, device=device) * 0.4

    target_obj_slots = z_goal[:, 1:]  # [1, K-1, D] (target block & goal slots)

    for _ in range(num_iters):
        # Sample candidate actions [N, H, 2]
        actions = torch.randn(num_samples, horizon, 2, device=device) * std.unsqueeze(0) + mean.unsqueeze(0)
        actions = actions.clamp(-1.5, 1.5)

        # Expand z_history to [N, history_len, K, D]
        history_expanded = z_history.expand(num_samples, -1, -1, -1)

        # Autoregressive forward physics rollout in latent slot space
        pred_slots = model.rollouter(history_expanded, pred_len=horizon, actions=actions)  # [N, H, K, D]

        # Evaluate distance of predicted object slots at horizon against target goal slots
        pred_obj_final = pred_slots[:, -1, 1:]  # [N, K-1, D]
        costs = F.mse_loss(pred_obj_final, target_obj_slots.expand(num_samples, -1, -1), reduction="none").mean(dim=(-2, -1))

        # Select elite trajectories
        elite_indices = torch.topk(costs, k=num_elites, largest=False).indices
        elite_actions = actions[elite_indices]

        # Update distribution parameters
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.05)

    # Best action trajectory
    best_action_seq = mean.unsqueeze(0)  # [1, H, 2]
    # Predict the corresponding physical sub-goal slots for next step t+1
    best_pred_slots = model.rollouter(z_history, pred_len=1, actions=best_action_seq[:, 0:1])  # [1, 1, K, D]
    z_subgoal = best_pred_slots[:, 0]  # [1, K, D]

    best_action = mean[0:1]  # [1, 2]
    return best_action, z_subgoal


def run_world_model_evaluation(args):
    device = torch.device(args.device)
    model_wrapper = load_model(args.ckpt_path, device)
    inner_model = model_wrapper.model
    inner_savi = inner_model.stage1_model.inner_savi()

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=64, observation_height=64)

    # Canonical target goal slots
    static_z_goal, goal_img = get_canonical_goal_slots(model_wrapper, device)

    print("\n" + "=" * 75)
    print(f"ONLINE WORLD MODEL PLANNING & SUB-GOAL ROLLOUT EVALUATION")
    print(f"  Episodes: {args.num_episodes} | Max Steps: {args.max_steps} | Rollout Horizon: {args.horizon} | Samples: {args.num_samples}")
    print("=" * 75)

    all_gif_frames = []
    episode_results = []

    for ep in range(args.num_episodes):
        ep_seed = args.seed + ep
        obs, info = env.reset(seed=ep_seed)
        pusht = env.unwrapped

        init_agent = np.array([pusht.agent.position.x, pusht.agent.position.y], dtype=np.float32)
        init_block = np.array([pusht.block.position.x, pusht.block.position.y, pusht.block.angle], dtype=np.float32)
        goal_pos = pusht.goal_pose[:2]

        print(f"\n[Episode {ep+1:02d} (Seed {ep_seed})] Initial Agent: ({init_agent[0]:.1f}, {init_agent[1]:.1f}) | Block: ({init_block[0]:.1f}, {init_block[1]:.1f})")

        inner_savi._reset_rnn()
        prev_slots = None
        history_slot_list = []

        history_len = getattr(inner_model, "history_len", 2)
        ep_return = 0.0
        final_coverage = info.get("coverage", 0.0)
        solved_step = None

        for step in range(args.max_steps):
            agent_pos = np.array([pusht.agent.position.x, pusht.agent.position.y], dtype=np.float32)
            block_pos = np.array([pusht.block.position.x, pusht.block.position.y, pusht.block.angle], dtype=np.float32)

            obs_tensor = preprocess_obs(obs, device)
            with torch.no_grad():
                post_slots, _ = inner_savi.encode(obs_tensor, prev_slots=prev_slots)
                current_slot = post_slots[:, 0]  # [1, K, D]
                prev_slots = current_slot

            history_slot_list.append(current_slot)
            if len(history_slot_list) > history_len:
                history_slot_list.pop(0)

            if len(history_slot_list) < history_len:
                z_history = current_slot.unsqueeze(1).repeat(1, history_len, 1, 1)
            else:
                z_history = torch.stack(history_slot_list, dim=1)  # [1, history_len, K, D]

            # 2. Plan action and dynamically roll out the next physical sub-goal with SlotFormer
            with torch.no_grad():
                act_mu, z_subgoal = plan_with_world_model(
                    model=inner_model,
                    z_history=z_history,
                    z_goal=static_z_goal,
                    agent_pos=agent_pos,
                    block_pos=block_pos,
                    goal_pos=goal_pos,
                    horizon=args.horizon,
                    num_samples=args.num_samples,
                    device=device,
                )

            # 3. Decode the dynamically rolled out sub-goal slot state into an image
            subgoal_img = decode_slot_image(inner_model.stage1_model, z_subgoal)

            # 4. Execute the planned action in the live environment
            delta = act_mu[0].cpu().numpy()
            target_pos = np.array([
                np.clip(agent_pos[0] + delta[0] * args.action_scale, 15.0, 497.0),
                np.clip(agent_pos[1] + delta[1] * args.action_scale, 15.0, 497.0),
            ], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(target_pos)
            ep_return += reward
            final_coverage = info.get("coverage", final_coverage)

            if info.get("is_success", False) and solved_step is None:
                solved_step = step + 1

            # Construct 3-panel visualization: [Live Camera Obs | World Model Predicted Sub-Goal | Target Goal]
            if args.save_gif and ep < 2:
                h, w = obs.shape[:2]
                canvas = np.zeros((h + 24, w * 3 + 20, 3), dtype=np.uint8)
                canvas[24:24+h, 0:w] = obs
                canvas[24:24+h, w+10:w*2+10] = subgoal_img
                canvas[24:24+h, w*2+20:w*3+20] = goal_img

                cv2.putText(canvas, f"Live Obs (Cov: {final_coverage*100:.0f}%)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                cv2.putText(canvas, f"WM Sub-goal z_t+1", (w + 14, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 120), 1)
                cv2.putText(canvas, f"Target Goal", (w * 2 + 24, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 255), 1)
                all_gif_frames.append(canvas)

            if step % 25 == 0 or step == args.max_steps - 1 or solved_step:
                print(f"  Step {step:03d}/{args.max_steps}: Agent=({agent_pos[0]:.1f}, {agent_pos[1]:.1f}) | Block=({block_pos[0]:.1f}, {block_pos[1]:.1f}) | Delta=({delta[0]:.3f}, {delta[1]:.3f}) | Cov={final_coverage*100:.1f}%")

            if terminated or truncated:
                break

        is_success = bool(final_coverage >= 0.95)
        print(f"  Result Ep {ep+1:02d}: Return={ep_return:.2f} | Final Coverage={final_coverage*100:.1f}% | {'SOLVED' if is_success else 'Unfinished'}")

        episode_results.append({
            "episode": ep,
            "seed": ep_seed,
            "coverage": float(final_coverage),
            "return": float(ep_return),
            "solved": is_success,
            "steps": solved_step or args.max_steps,
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
        print(f"\n[PushT Eval] Saved World Model Sub-Goal Planning GIF to: {args.out_gif} (loop=0)")

    print("\n" + "=" * 75)
    print("WORLD MODEL ONLINE EVALUATION SUMMARY:")
    avg_cov = np.mean([r["coverage"] for r in episode_results])
    max_cov = np.max([r["coverage"] for r in episode_results])
    success_rate = np.mean([r["solved"] for r in episode_results]) * 100.0
    print(f"  Success Rate: {success_rate:.1f}% | Avg Coverage: {avg_cov*100:.2f}% | Max Coverage: {max_cov*100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    args = parse_args()
    run_world_model_evaluation(args)
