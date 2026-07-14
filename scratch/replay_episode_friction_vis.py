"""
Replay episode 0, extract normal and frictional contact forces, and visualize them as distinct overlay arrows.
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import sys
import hdf5plugin
import h5py
import imageio
import cv2
import numpy as np
import gymnasium as gym
import gym_pusht
import pymunk

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GYM_PUSHT  = os.path.join(REPO_ROOT, 'third_party', 'gym-pusht')

if GYM_PUSHT not in sys.path:
    sys.path.insert(0, GYM_PUSHT)

class ContactTracker:
    def __init__(self, agent_body, block_body):
        self.agent_body = agent_body
        self.block_body = block_body
        self.reset()
        
    def reset(self):
        self.contact_exists = False
        self.total_normal_impulse = pymunk.Vec2d(0, 0)
        self.total_tangential_impulse = pymunk.Vec2d(0, 0)
        self.contact_pts = []
        
    def post_solve(self, arb, space, data):
        body_a = arb.shapes[0].body
        body_b = arb.shapes[1].body
        is_agent_block = (body_a == self.agent_body and body_b == self.block_body) or \
                         (body_a == self.block_body and body_b == self.agent_body)
        if is_agent_block:
            self.contact_exists = True
            normal = arb.normal
            impulse_vec = arb.total_impulse
            
            normal_magnitude = impulse_vec.dot(normal)
            normal_imp_vec = normal * normal_magnitude
            tangential_imp_vec = impulse_vec - normal_imp_vec
            
            self.total_normal_impulse += normal_imp_vec
            self.total_tangential_impulse += tangential_imp_vec
            
            for p in arb.contact_point_set.points:
                self.contact_pts.append((p.point_a + p.point_b) / 2.0)

def step_physics_with_tracker(env, state, action, tracker):
    raw_env = env.unwrapped
    tracker.reset()
    
    # 1. Set state and initial velocity
    raw_env._set_state(state[:5])
    raw_env.agent.velocity = pymunk.Vec2d(state[5], state[6])
    
    agent_body = raw_env.agent
    target_pos = state[:2] + action * 30.0
    
    # Run 10 sub-steps
    for _ in range(10):
        acc = raw_env.k_p * (target_pos - agent_body.position) + raw_env.k_v * (pymunk.Vec2d(0, 0) - agent_body.velocity)
        agent_body.velocity += acc * 0.01
        raw_env.space.step(0.01)

def main():
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    output_dir = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9'
    output_path = os.path.join(output_dir, 'episode_0_friction_forces.gif')
    
    print("Initializing environment...")
    env = gym.make('gym_pusht/PushT-v0', obs_type='state')
    env.reset()
    raw_env = env.unwrapped
    
    # Enable shape friction dynamically
    for shape in raw_env.agent.shapes:
        shape.friction = 1.0
    for shape in raw_env.block.shapes:
        shape.friction = 1.0
        
    # Register collision tracker
    tracker = ContactTracker(raw_env.agent, raw_env.block)
    handler = raw_env.space.add_collision_handler(0, 0)
    handler.post_solve = tracker.post_solve
        
    print(f"Reading dataset: {h5_path}")
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    with h5py.File(h5_path, 'r') as f:
        ep_idx = 0
        offset = f['ep_offset'][ep_idx]
        length = f['ep_len'][ep_idx]
        print(f"Episode {ep_idx}: offset={offset}, length={length}")
        
        raw_frames = f['pixels'][offset : offset + length]
        states = f['state'][offset : offset + length]
        actions = f['action'][offset : offset + length]
        
    print("Rendering frames with normal and frictional forces...")
    processed_frames = []
    scale = 224.0 / 512.0
    arrow_scale = 3.5  # Scale factor to display force vectors in pixels
    
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
            
        # 2. Simulate step with action and collect impulses
        if t < length - 1:
            step_physics_with_tracker(env, states[t], actions[t], tracker)
            
        contact_exists = tracker.contact_exists
        normal_imp = tracker.total_normal_impulse
        friction_imp = tracker.total_tangential_impulse
        contact_pts = tracker.contact_pts
            
        # 3. Draw force arrows at contact point
        fn_mag = normal_imp.length / 10.0  # Average per physics sub-step
        ft_mag = friction_imp.length / 10.0
        
        if contact_exists and len(contact_pts) > 0:
            # Average contact point coordinates
            cx = sum(p.x for p in contact_pts) / len(contact_pts)
            cy = sum(p.y for p in contact_pts) / len(contact_pts)
            px_contact = (int(cx * scale), int(cy * scale))
            
            # Highlight contact point with small neon green circle
            cv2.circle(frame, px_contact, radius=2, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
            
            # Net normal force vector (reaction force on the block)
            # Contact impulse is applied to block. Force is opposite to solver impulse.
            normal_vec = -normal_imp / 10.0
            friction_vec = -friction_imp / 10.0
            
            # Cyan arrow for Normal Force (Fn)
            if fn_mag > 0.05:
                fn_end_x = int((cx + normal_vec.x * arrow_scale) * scale)
                fn_end_y = int((cy + normal_vec.y * arrow_scale) * scale)
                cv2.arrowedLine(frame, px_contact, (fn_end_x, fn_end_y), (255, 255, 0), 2, cv2.LINE_AA, 0, 0.3) # Cyan/Teal in BGR: (255, 255, 0)
                
            # Orange/Magenta arrow for Frictional Force (Ft)
            if ft_mag > 0.05:
                ft_end_x = int((cx + friction_vec.x * arrow_scale) * scale)
                ft_end_y = int((cy + friction_vec.y * arrow_scale) * scale)
                cv2.arrowedLine(frame, px_contact, (ft_end_x, ft_end_y), (0, 165, 255), 2, cv2.LINE_AA, 0, 0.3) # Orange in BGR: (0, 165, 255)
                
        # 4. HUD overlay
        hud_height = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (224, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.33
        text_color = (240, 240, 240)
        thickness = 1
        
        cv2.putText(frame, f"Step: {t}/{length-1}", (8, 12), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Fn (Cyan): {fn_mag:.2f}  Ft (Orange): {ft_mag:.2f}", (8, 25), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        processed_frames.append(frame)
        
    print(f"Saving video to {output_path}...")
    os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(output_path, processed_frames, fps=10)
    print("Replay with frictional forces saved successfully!")

if __name__ == '__main__':
    main()
