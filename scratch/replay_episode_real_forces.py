import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import hdf5plugin
import h5py
import imageio
import cv2
import numpy as np
import gymnasium as gym
import gym_pusht
import pymunk

def main():
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    output_dir = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9'
    output_path = os.path.join(output_dir, 'episode_0_real_forces.gif')
    
    print("Initializing environment...")
    env = gym.make('gym_pusht/PushT-v0', obs_type='state')
    env.reset()
    
    print(f"Reading dataset: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        ep_idx = 0
        offset = f['ep_offset'][ep_idx]
        length = f['ep_len'][ep_idx]
        print(f"Episode {ep_idx}: offset={offset}, length={length}")
        
        raw_frames = f['pixels'][offset : offset + length]
        states = f['state'][offset : offset + length]
        actions = f['action'][offset : offset + length]
        
    print("Rendering frames with contact forces...")
    processed_frames = []
    scale = 224.0 / 512.0
    arrow_scale_factor = 1.0  # Scale factor for drawing the force arrow length in pixels
    
    # Store agent positions for the trail
    trail_positions = []
    
    for t in range(length):
        frame = raw_frames[t].copy()
        
        # Current agent position (scaled)
        agent_x, agent_y = states[t][0], states[t][1]
        px_agent = (int(agent_x * scale), int(agent_y * scale))
        trail_positions.append(px_agent)
        
        # 1. Draw the agent trajectory trail
        for i in range(max(1, t - 15), t + 1):
            pt1 = trail_positions[i - 1]
            pt2 = trail_positions[i]
            alpha = (i - max(0, t - 15)) / 15.0
            color = (0, int(150 * alpha), int(255 * alpha))  # Cyan/Blue fading
            cv2.line(frame, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA)
            
        # 2. Simulate the transition to gather exact contact forces and contact points
        # Set states and velocities
        env.unwrapped._set_state(states[t])
        env.unwrapped.agent.velocity = pymunk.Vec2d(states[t][5], states[t][6])
        
        contact_pts = []
        impulses = []
        
        def post_solve_callback(arb, sp, data):
            # Accumulate contact points and impulse magnitude
            contact_pts.extend([(p.point_a + p.point_b) / 2.0 for p in arb.contact_point_set.points])
            impulses.append(arb.total_impulse.length)
            
        handler = env.unwrapped.space.add_collision_handler(0, 0)
        handler.post_solve = post_solve_callback
        
        # Run 10 sub-steps
        if t < length - 1:
            act = actions[t]
            target_pos = states[t][:2] + act * 30.0
            for _ in range(10):
                acc = env.unwrapped.k_p * (target_pos - env.unwrapped.agent.position) + env.unwrapped.k_v * (pymunk.Vec2d(0,0) - env.unwrapped.agent.velocity)
                env.unwrapped.agent.velocity += acc * 0.01
                env.unwrapped.space.step(0.01)
                
        # Average contact force over the step
        has_contact = len(contact_pts) > 0
        force_mag = sum(impulses) / 10.0 if impulses else 0.0
        
        # 3. Draw the force arrow at the contact point if there is contact
        if has_contact and force_mag > 0.05:
            # Average contact points
            cx = sum(p.x for p in contact_pts) / len(contact_pts)
            cy = sum(p.y for p in contact_pts) / len(contact_pts)
            px_contact = (int(cx * scale), int(cy * scale))
            
            # Highlight contact point with a small neon green dot
            cv2.circle(frame, px_contact, radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
            
            # The force normal direction points from the agent's center to the contact point
            dx = cx - agent_x
            dy = cy - agent_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 0.01:
                nx = dx / dist
                ny = dy / dist
                
                # Draw the force arrow starting from the contact point
                # Make the arrow length proportional to the contact force
                arrow_len = force_mag * arrow_scale_factor
                end_x = int((cx + nx * arrow_len) * scale)
                end_y = int((cy + ny * arrow_len) * scale)
                pt_end = (end_x, end_y)
                
                arrow_color = (255, 100, 0)  # Coral/Orange
                cv2.arrowedLine(frame, px_contact, pt_end, arrow_color, 1, cv2.LINE_AA, 0, 0.25)
                
        # 4. Draw HUD overlay with two lines of text to prevent cutoff
        hud_height = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (224, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.33
        text_color = (240, 240, 240)
        thickness = 1
        
        # Line 1: Step info
        cv2.putText(frame, f"Step: {t}/{length-1}", (8, 12), font, font_scale, text_color, thickness, cv2.LINE_AA)
        # Line 2: Contact Force info
        cv2.putText(frame, f"Contact Force: {force_mag:.2f}", (8, 25), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        processed_frames.append(frame)
        
    print(f"Saving video to {output_path}...")
    os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(output_path, processed_frames, fps=10)
    print("Replay with real forces saved successfully!")

if __name__ == '__main__':
    main()
