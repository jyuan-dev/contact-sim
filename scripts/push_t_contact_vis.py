"""
scripts/push_t_contact_vis.py
==============================

Extract, replay, and visualize normal/frictional forces and segmentation masks
for PushT episode 0 from the simulator.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
import argparse
import imageio
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Locate repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.replay import PushTReplayer


def main():
    parser = argparse.ArgumentParser(description="PushT Contact and Segmentation Replay Visualizer")
    parser.add_argument("--h5-path", type=str, default="/home/jyuan/.stable-wm/pusht_expert_train.h5",
                        help="Path to PushT HDF5 dataset file")
    parser.add_argument("--unoccluded", action="store_true", default=False,
                        help="Render unoccluded goal mask (default: False)")
    args = parser.parse_args()

    h5_path = args.h5_path
    output_dir = os.path.join(REPO_ROOT, "scratch")
    os.makedirs(output_dir, exist_ok=True)

    gif_output_path = os.path.join(output_dir, "pusht_episode_0_replay.gif")
    seg_gif_path    = os.path.join(output_dir, "pusht_episode_0_segmentation.gif")
    png_output_path = os.path.join(output_dir, "pusht_episode_0_analysis.png")

    # Clean old output files
    for p in (gif_output_path, seg_gif_path, png_output_path):
        if os.path.exists(p):
            os.remove(p)

    print(f"Initializing PushTReplayer for: {h5_path} (unoccluded={args.unoccluded})")
    replayer = PushTReplayer(
        h5_path=h5_path,
        run_physics=True,
        num_workers=1,
        unoccluded_masks=args.unoccluded,
    )

    print("Replaying physics and extracting episode 0...")
    ep_data = next(replayer.iter_episodes())

    T = len(ep_data.states)
    print(f"Episode length: {T} steps")

    scale = 224.0 / 512.0
    arrow_scale = 3.5

    trail_positions = []
    processed_frames = []
    seg_vis_frames = []

    fn_magnitudes = []
    ft_magnitudes = []
    block_pixel_counts = []
    agent_pixel_counts = []
    goal_pixel_counts  = []

    for t in range(T):
        frame = ep_data.frames[t].copy()

        # 1. Agent trajectory trail
        agent_x, agent_y = ep_data.states[t][0], ep_data.states[t][1]
        px_agent = (int(agent_x * scale), int(agent_y * scale))
        trail_positions.append(px_agent)

        for i in range(max(1, t - 15), t + 1):
            pt1 = trail_positions[i - 1]
            pt2 = trail_positions[i]
            alpha = (i - max(0, t - 15)) / 15.0
            color = (0, int(150 * alpha), int(255 * alpha))
            cv2.line(frame, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA)

        # 2. Extract force vectors and contact points
        contact = ep_data.contact_pos[t]
        fn_vec  = ep_data.normal_force[t]
        ft_vec  = ep_data.frictional_force[t]

        fn_mag = float(np.linalg.norm(fn_vec))
        ft_mag = float(np.linalg.norm(ft_vec))
        fn_magnitudes.append(fn_mag)
        ft_magnitudes.append(ft_mag)

        if not np.isnan(contact[0]):
            px_contact = (int(contact[0] * scale), int(contact[1] * scale))
            cv2.circle(frame, px_contact, radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

            if fn_mag > 0.05:
                fn_end_x = int((contact[0] + fn_vec[0] * arrow_scale) * scale)
                fn_end_y = int((contact[1] + fn_vec[1] * arrow_scale) * scale)
                cv2.arrowedLine(frame, px_contact, (fn_end_x, fn_end_y), (255, 255, 0), 2, cv2.LINE_AA, 0, 0.3)

            if ft_mag > 0.05:
                ft_end_x = int((contact[0] + ft_vec[0] * arrow_scale) * scale)
                ft_end_y = int((contact[1] + ft_vec[1] * arrow_scale) * scale)
                cv2.arrowedLine(frame, px_contact, (ft_end_x, ft_end_y), (0, 165, 255), 2, cv2.LINE_AA, 0, 0.3)

        # 3. Extract masks
        block_mask = ep_data.masks["block"][t]
        agent_mask = ep_data.masks["agent"][t]
        goal_mask  = ep_data.masks["goal"][t]

        block_pixel_counts.append(int(np.sum(block_mask > 0)))
        agent_pixel_counts.append(int(np.sum(agent_mask > 0)))
        goal_pixel_counts.append(int(np.sum(goal_mask > 0)))

        # 4. Build 4-panel side-by-side segmentation visualization
        H_f, W_f = frame.shape[:2]
        block_vis = np.zeros((H_f, W_f, 3), dtype=np.uint8); block_vis[block_mask > 0] = (80, 200, 255)
        agent_vis = np.zeros((H_f, W_f, 3), dtype=np.uint8); agent_vis[agent_mask > 0] = (80, 255, 140)
        goal_vis  = np.zeros((H_f, W_f, 3), dtype=np.uint8); goal_vis[goal_mask > 0]   = (255, 100, 80)

        for vis_img, lbl in [(frame.copy(), "Original"), (block_vis, "Block"), (agent_vis, "Agent"), (goal_vis, "Goal")]:
            cv2.putText(vis_img, lbl, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        seg_panel = np.concatenate([frame.copy(), block_vis, agent_vis, goal_vis], axis=1)
        seg_vis_frames.append(seg_panel)

        # 5. Blend colored segmentation masks directly onto source video frame
        masked_frame = frame.copy().astype(np.float32)
        m_b = block_mask > 0
        if np.any(m_b):
            masked_frame[m_b] = masked_frame[m_b] * 0.55 + np.array([80, 200, 255], dtype=np.float32) * 0.45

        m_a = agent_mask > 0
        if np.any(m_a):
            masked_frame[m_a] = masked_frame[m_a] * 0.55 + np.array([80, 255, 140], dtype=np.float32) * 0.45

        m_g = goal_mask > 0
        if np.any(m_g):
            masked_frame[m_g] = masked_frame[m_g] * 0.70 + np.array([255, 100, 80], dtype=np.float32) * 0.30

        frame = np.clip(masked_frame, 0, 255).astype(np.uint8)

        # 6. HUD Overlay
        hud_height = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W_f, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.33
        text_color = (240, 240, 240)
        thickness = 1

        cv2.putText(frame, f"Step: {t}/{T-1}", (8, 12), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Fn (Cyan): {fn_mag:.2f}  Ft (Orange): {ft_mag:.2f}", (8, 25), font, font_scale, text_color, thickness, cv2.LINE_AA)

        processed_frames.append(frame)

    print(f"Saving replay GIF to: {gif_output_path}")
    imageio.mimsave(gif_output_path, processed_frames, fps=10, loop=0)

    print(f"Saving segmentation GIF to: {seg_gif_path}")
    imageio.mimsave(seg_gif_path, seg_vis_frames, fps=10, loop=0)

    # 7. Generate Analysis Plot
    print("Generating forces and segmentation analysis plots...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    timesteps = np.arange(T)
    axes[0].plot(timesteps, fn_magnitudes, color="cyan", linewidth=1.8, label="Normal Force (Fn)")
    axes[0].plot(timesteps, ft_magnitudes, color="orange", linewidth=1.8, label="Frictional Force (Ft)")
    axes[0].set_ylabel("Force Magnitude (N)")
    axes[0].set_title("PushT Episode 0 — Contact Forces Over Time")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    axes[1].plot(timesteps, block_pixel_counts, color="#50c8ff", linewidth=1.8, label="Block Mask Pixels")
    axes[1].plot(timesteps, agent_pixel_counts, color="#50ff8c", linewidth=1.8, label="Agent Mask Pixels")
    axes[1].plot(timesteps, goal_pixel_counts,  color="#ff6450", linewidth=1.8, label="Goal Mask Pixels")
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("Pixel Area (Count)")
    axes[1].set_title("PushT Episode 0 — Object Mask Area Over Time")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(png_output_path, dpi=150)
    plt.close()
    print(f"Saving analysis PNG to: {png_output_path}")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"1. Original Replay GIF (with masks & forces): {gif_output_path}")
    print(f"2. Segmentation GIF (4-panel):                 {seg_gif_path}")
    print(f"3. Forces & Masks Analysis PNG:                {png_output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
