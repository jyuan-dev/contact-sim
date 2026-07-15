import os
import sys
import OpenGL
# Tell PyOpenGL to use EGL platform for headless rendering
OpenGL.PLATFORM = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
SWM_ROOT   = os.path.join(REPO_ROOT, 'third_party', 'stable-worldmodel')

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SWM_ROOT not in sys.path:
    sys.path.insert(0, SWM_ROOT)

import cv2
import numpy as np
import hdf5plugin
import h5py
import imageio
import matplotlib.pyplot as plt
import mujoco

from src.datasets.replay import OGBenchReplayer

def project_3d_to_2d(pos_3d, model, data, camera_name="front_pixels", width=224, height=224):
    """Project a 3D world space coordinate into 2D camera pixel coordinates."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy)
    focal_y = (height / 2.0) / np.tan(fovy_rad / 2.0)
    focal_x = focal_y
    
    rel_pos = pos_3d - cam_pos
    pos_cam = cam_mat.T @ rel_pos
    
    depth = -pos_cam[2]
    if depth < 1e-5:
        return None
        
    px = (pos_cam[0] / depth) * focal_x + (width / 2.0)
    py = (height / 2.0) - (pos_cam[1] / depth) * focal_y
    return (int(px), int(py))

def main():
    dataset_path = "/home/jyuan/.cache/swm/datasets/ogbench/cube_single_expert.lance"
    output_dir = os.path.join(REPO_ROOT, "scratch")
    os.makedirs(output_dir, exist_ok=True)
    
    gif_output_path = os.path.join(output_dir, "ogbench_episode_0_replay.gif")
    seg_gif_path    = os.path.join(output_dir, "ogbench_episode_0_segmentation.gif")
    png_output_path = os.path.join(output_dir, "ogbench_episode_0_analysis.png")
    
    print(f"Initializing OGBenchReplayer for: {dataset_path}")
    replayer = OGBenchReplayer(h5_path=dataset_path, run_physics=True)
    
    print("Loading raw episode 0 data...")
    raw_frames, states, actions = replayer._load_episode_raw(0)
    T = len(states)
    print(f"Episode length: {T} steps")
    
    # Load target block positions and orientations from Lance episode
    episode = replayer._ds.load_episode(0)
    target_pos_seq = np.asarray(episode["privileged/target_block_pos"], dtype=np.float32)
    target_yaw_seq = np.asarray(episode["privileged/target_block_yaw"], dtype=np.float32)
    
    print("Initializing MuJoCo headless environment...")
    env = replayer._get_env()
    model = env._model
    data  = env._data
    nq    = model.nq
    
    num_cubes = env._num_cubes
    cube_geom_id_sets = [set(geom_ids) for geom_ids in env._cube_geom_ids_list]
    target_geom_id_sets = [set(geom_ids) for geom_ids in env._cube_target_geom_ids_list]
    gripper_geom_ids = replayer._find_gripper_geom_ids(model)
    
    # Render dimensions
    H, W = 224, 224
    
    # Buffers to save force values for line plotting
    fn_magnitudes = []
    ft_magnitudes = []
    left_pad_fn_mags = []
    right_pad_fn_mags = []
    left_pad_ft_mags = []
    right_pad_ft_mags = []

    # Find left and right pad geoms dynamically
    left_pad_geoms = set()
    right_pad_geoms = set()
    for gid in range(model.ngeom):
        name = model.geom(gid).name.lower()
        if 'left_pad' in name or 'left_silicone' in name:
            left_pad_geoms.add(gid)
        elif 'right_pad' in name or 'right_silicone' in name:
            right_pad_geoms.add(gid)

    print(f"Dynamically mapped left pad geoms: {sorted(left_pad_geoms)}")
    print(f"Dynamically mapped right pad geoms: {sorted(right_pad_geoms)}")
    
    processed_frames = []
    cube_masks = []
    gripper_masks = []
    target_masks = []
    seg_vis_frames = []  # 4-panel side-by-side segmentation frames
    
    # Scale factor for drawing force vectors (world space meters per Newton)
    arrow_scale = 0.01  # 50 N force -> 0.5 meters in world space

    # Instantiate renderer for RGB visualization
    renderer_rgb = mujoco.Renderer(model, height=H, width=W)
    renderer_seg = mujoco.Renderer(model, height=H, width=W)  # reused for segmentation
    
    print("Replaying physics and drawing overlays...")
    for t in range(T):
        # ── Set state ─────────────────────────────────────────────────
        state = states[t].astype(np.float64)
        if len(state) >= nq:
            data.qpos[:] = state[:nq]
            data.qvel[:] = state[nq: nq + model.nv]
        else:
            data.qpos[:len(state)] = state

        # ── Set target mocap position and orientation ─────────────────
        from ogbench.manipspace import lie
        target_pos = target_pos_seq[t]
        target_yaw = target_yaw_seq[t][0]
        target_quat = lie.SO3.from_z_radians(target_yaw).wxyz.tolist()

        mocap_id = env._cube_target_mocap_ids[env._target_block]
        data.mocap_pos[mocap_id] = target_pos
        data.mocap_quat[mocap_id] = target_quat

        # Make the target block visual indicator visible
        for gid in env._cube_target_geom_ids_list[env._target_block]:
            model.geom(gid).rgba[3] = 0.2

        mujoco.mj_forward(model, data)
        
        # ── Extract contacts and project forces ────────────────────────
        step_normal_mag = 0.0
        step_frict_mag = 0.0
        contact_pts_list = []
        fn_world_accum = np.zeros(3)
        ft_world_accum = np.zeros(3)
        
        left_normal_mag = 0.0
        left_frict_mag = 0.0
        right_normal_mag = 0.0
        right_frict_mag = 0.0
        
        force_buf = np.zeros(6, dtype=np.float64)
        
        for c_idx in range(data.ncon):
            contact = data.contact[c_idx]
            g1, g2  = int(contact.geom1), int(contact.geom2)
            
            # 1. Cube contacts (for drawing force arrows)
            cube_involved = False
            for ci, geom_set in enumerate(cube_geom_id_sets):
                if g1 in geom_set or g2 in geom_set:
                    cube_involved = True
                    break
            if cube_involved:
                mujoco.mj_contactForce(model, data, c_idx, force_buf)
                fn_val = force_buf[0]
                ft_val = np.linalg.norm(force_buf[1:3])
                
                step_normal_mag += fn_val
                step_frict_mag += ft_val
                
                R = contact.frame.reshape(3, 3)
                fn_world_accum += R[0] * fn_val
                ft_world_accum += R[1] * force_buf[1] + R[2] * force_buf[2]
                contact_pts_list.append(contact.pos.copy())
                
            # 2. Gripper left pad contacts
            if g1 in left_pad_geoms or g2 in left_pad_geoms:
                mujoco.mj_contactForce(model, data, c_idx, force_buf)
                left_normal_mag += force_buf[0]
                left_frict_mag += np.linalg.norm(force_buf[1:3])
                
            # 3. Gripper right pad contacts
            if g1 in right_pad_geoms or g2 in right_pad_geoms:
                mujoco.mj_contactForce(model, data, c_idx, force_buf)
                right_normal_mag += force_buf[0]
                right_frict_mag += np.linalg.norm(force_buf[1:3])
                
        fn_magnitudes.append(step_normal_mag)
        ft_magnitudes.append(step_frict_mag)
        left_pad_fn_mags.append(left_normal_mag)
        left_pad_ft_mags.append(left_frict_mag)
        right_pad_fn_mags.append(right_normal_mag)
        right_pad_ft_mags.append(right_frict_mag)
        
        # ── Draw forces on frame ──────────────────────────────────────
        # Render the RGB frame using MuJoCo Renderer so the target block is visible
        renderer_rgb.update_scene(data, camera="front_pixels")
        frame = renderer_rgb.render().copy()
        
        # Draw goal position marker (red diamond)
        goal_px = project_3d_to_2d(target_pos, model, data, "front_pixels", W, H)
        if goal_px is not None:
            gx, gy = goal_px
            cv2.drawMarker(frame, (gx, gy), (0, 0, 255), cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)
            cv2.putText(frame, "Goal", (gx + 9, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1, cv2.LINE_AA)
        
        if len(contact_pts_list) > 0:
            # Average contact position in world space
            c_pos_world = np.mean(contact_pts_list, axis=0)
            
            # Project contact position to 2D
            px_contact = project_3d_to_2d(c_pos_world, model, data, "front_pixels", W, H)
            
            if px_contact is not None:
                # Base circle at contact point (neon green)
                cv2.circle(frame, px_contact, radius=3, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
                
                # Project normal force vector end point
                p_normal_end = c_pos_world + fn_world_accum * arrow_scale
                px_normal_end = project_3d_to_2d(p_normal_end, model, data, "front_pixels", W, H)
                if px_normal_end is not None:
                    # Cyan arrow for Normal Force (Fn)
                    cv2.arrowedLine(frame, px_contact, px_normal_end, (255, 255, 0), 2, cv2.LINE_AA, 0, 0.3)
                    
                # Project friction force vector end point
                p_frict_end = c_pos_world + ft_world_accum * arrow_scale
                px_frict_end = project_3d_to_2d(p_frict_end, model, data, "front_pixels", W, H)
                if px_frict_end is not None:
                    # Orange arrow for Frictional Force (Ft)
                    cv2.arrowedLine(frame, px_contact, px_frict_end, (0, 165, 255), 2, cv2.LINE_AA, 0, 0.3)
                    
        # ── Render and record segmentation masks ──────────────────────
        # Standard segmentation (for gripper and target masks)
        seg = replayer._render_segmentation(env, model, data, H, W, renderer=renderer_seg)
        target_mask  = replayer._seg_to_mask(seg, model, target_geom_id_sets[0])
        gripper_mask = replayer._seg_to_mask(seg, model, gripper_geom_ids)
        
        # Unoccluded cube mask: hide gripper geoms so the grasped cube
        # is fully visible even when the gripper fingers wrap around it.
        cube_mask = replayer._render_unoccluded_mask(
            env, model, data, H, W,
            target_geom_ids=cube_geom_id_sets[0],
            occluder_geom_ids=gripper_geom_ids,
            renderer=renderer_seg,
        )
        
        cube_masks.append(cube_mask)
        target_masks.append(target_mask)
        # Exclude cube and target pixels from gripper mask
        gripper_mask_clean = gripper_mask.copy()
        gripper_mask_clean[cube_mask > 0] = 0
        gripper_mask_clean[target_mask > 0] = 0
        gripper_masks.append(gripper_mask_clean)
        
        # Build 4-panel segmentation visualization (original | cube | gripper | target)
        H_f, W_f = frame.shape[:2]
        cube_vis    = np.zeros((H_f, W_f, 3), dtype=np.uint8); cube_vis[cube_mask > 0] = (80, 200, 255)
        gripper_vis = np.zeros((H_f, W_f, 3), dtype=np.uint8); gripper_vis[gripper_mask_clean > 0] = (80, 255, 140)
        target_vis  = np.zeros((H_f, W_f, 3), dtype=np.uint8); target_vis[target_mask > 0] = (255, 100, 80)
        for vis_img, lbl in [(frame.copy(), "Original"), (cube_vis, "Cube (unoccluded)"), (gripper_vis, "Gripper"), (target_vis, "Goal")]:
            cv2.putText(vis_img, lbl, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
        seg_panel = np.concatenate([frame.copy(), cube_vis, gripper_vis, target_vis], axis=1)
        seg_vis_frames.append(seg_panel)
        
        # ── HUD Overlay ───────────────────────────────────────────────
        hud_height = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.33
        text_color = (255, 255, 255)
        thickness = 1
        
        cv2.putText(frame, f"Step: {t}/{T-1}", (8, 12), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Fn (Cyan): {step_normal_mag:.2f}  Ft (Orange): {step_frict_mag:.2f}", (8, 26), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        processed_frames.append(frame)
        
    # ── Save outputs ──────────────────────────────────────────────────
    print(f"Saving replay GIF to: {gif_output_path}")
    imageio.mimsave(gif_output_path, processed_frames, fps=10)
    
    print(f"Saving segmentation GIF to: {seg_gif_path}")
    imageio.mimsave(seg_gif_path, seg_vis_frames, fps=10)
    
    print("Generating forces and segmentation analysis plots...")
    fig = plt.figure(figsize=(14, 14), dpi=100, facecolor="#0d1117")
    
    # Row 1: Forces over time
    ax_forces = fig.add_subplot(5, 1, 1)
    ax_forces.set_facecolor("#161b22")
    ax_forces.plot(fn_magnitudes, label="Cube Normal Force (Fn)", color="cyan", linewidth=2)
    ax_forces.plot(ft_magnitudes, label="Cube Frictional Force (Ft)", color="orange", linewidth=2)
    ax_forces.plot(left_pad_fn_mags, label="Left Pad Normal Force", color="#22c55e", linestyle="--", linewidth=1.5)
    ax_forces.plot(left_pad_ft_mags, label="Left Pad Frictional Force", color="#22c55e", linestyle=":", linewidth=1.5)
    ax_forces.plot(right_pad_fn_mags, label="Right Pad Normal Force", color="#a855f7", linestyle="--", linewidth=1.5)
    ax_forces.plot(right_pad_ft_mags, label="Right Pad Frictional Force", color="#a855f7", linestyle=":", linewidth=1.5)
    ax_forces.set_title("OGBench Episode 0: Contact Forces (Cube & Gripper Pads) Over Time", color="white", fontsize=13)
    ax_forces.set_xlabel("Time Step", color="#8b949e")
    ax_forces.set_ylabel("Force Magnitude", color="#8b949e")
    ax_forces.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white", ncol=2, fontsize=8)
    ax_forces.grid(True, linestyle="--", alpha=0.4, color="#30363d")
    ax_forces.tick_params(colors="#8b949e")
    for spine in ax_forces.spines.values():
        spine.set_edgecolor("#30363d")
    
    # Rows 2-5: key frames
    key_steps = [10, 35, 60, 85]
    key_steps = [min(s, T-1) for s in key_steps]
    row_info = [
        ("Original + Goal", processed_frames, None),
        ("Cube (grasped)",  cube_masks,       "Blues"),
        ("Gripper",         gripper_masks,    "Greens"),
        ("Target/Goal",     target_masks,     "Reds"),
    ]
    for row_idx, (label, src, cmap) in enumerate(row_info):
        for col_idx, step in enumerate(key_steps):
            ax = fig.add_subplot(5, 4, 4 * (row_idx + 1) + col_idx + 1)
            ax.set_facecolor("#161b22")
            if cmap is None:
                ax.imshow(src[step])
            else:
                ax.imshow(src[step], cmap=cmap, vmin=0, vmax=255)
            ax.set_title(f"Step {step}", color="white", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(label, color="white", fontsize=8)
            ax.axis("off")

    plt.suptitle("OGBench Cube: Replay + Segmentation Analysis", color="white", fontsize=15, y=1.01)
    plt.tight_layout()
    print(f"Saving analysis PNG to: {png_output_path}")
    plt.savefig(png_output_path, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print(f"1. Original Replay GIF (with goal pos): {gif_output_path}")
    print(f"2. Segmentation GIF (4-panel):          {seg_gif_path}")
    print(f"3. Forces & Masks Analysis PNG:         {png_output_path}")
    print("="*60)

if __name__ == '__main__':
    main()
