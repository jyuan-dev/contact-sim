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
    output_path = os.path.join(output_dir, 'episode_0_contact_forces.gif')
    
    # Initialize the gym environment to get contact points
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
    arrow_scale = 80.0
    
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
            
        # 2. Query Pymunk for contacts at this step
        pts = []
        handler = env.unwrapped.space.add_collision_handler(0, 0)
        handler.post_solve = lambda arb, sp, data: pts.extend([(p.point_a + p.point_b) / 2.0 for p in arb.contact_point_set.points])
        
        env.unwrapped._set_state(states[t])
        env.unwrapped.space.step(0.01)
        
        has_contact = len(pts) > 0
        action_x, action_y = actions[t][0], actions[t][1]
        force_mag = np.sqrt(action_x**2 + action_y**2)
        
        # 3. Draw the force arrow at the contact point if there is contact
        if has_contact:
            # Average the contact points if there are multiple
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            
            px_contact = (int(cx * scale), int(cy * scale))
            
            # Highlight the contact point with a small green dot
            cv2.circle(frame, px_contact, radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
            
            # Draw the force arrow starting from the contact point
            end_x = int((cx + action_x * arrow_scale) * scale)
            end_y = int((cy + action_y * arrow_scale) * scale)
            pt_end = (end_x, end_y)
            
            if force_mag > 0.01:
                arrow_color = (255, 100, 0)  # Coral/Orange in RGB
                cv2.arrowedLine(frame, px_contact, pt_end, arrow_color, 1, cv2.LINE_AA, 0, 0.25)
                
        # 4. Add semi-transparent HUD overlay
        hud_height = 25
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (224, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Draw text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        text_color = (240, 240, 240)
        thickness = 1
        
        contact_str = "YES" if has_contact else "NO"
        info_text = f"Step: {t}/{length}  Force: {force_mag:.2f}  Contact: {contact_str}"
        cv2.putText(frame, info_text, (8, 16), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        processed_frames.append(frame)
        
    print(f"Saving video to {output_path}...")
    os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(output_path, processed_frames, fps=10)
    print("Replay with contact forces saved successfully!")

if __name__ == '__main__':
    main()
