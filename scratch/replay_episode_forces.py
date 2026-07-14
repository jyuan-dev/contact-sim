"""
Replay episode 0, extract contact forces, and draw them as overlay vectors.
"""
import sys
import os
import h5py
import numpy as np
import cv2
import pygame
import pymunk
from PIL import Image

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GYM_PUSHT  = os.path.join(REPO_ROOT, 'third_party', 'gym-pusht')

if GYM_PUSHT not in sys.path:
    sys.path.insert(0, GYM_PUSHT)

import gymnasium as gym
import gym_pusht
def step_and_collect_forces(env, action):
    raw_env = env.unwrapped
    n_steps = int(1 / (raw_env.dt * raw_env.control_hz))  # 10 steps
    
    agent_body = raw_env.agent
    block_body = raw_env.block
    
    total_impulse = pymunk.Vec2d(0, 0)
    contact_points = []
    contact_exists = False
    
    for _ in range(n_steps):
        # Step PD control
        acceleration = raw_env.k_p * (action - agent_body.position) + raw_env.k_v * (
            pymunk.Vec2d(0, 0) - agent_body.velocity
        )
        agent_body.velocity += acceleration * raw_env.dt
        
        # Step physics
        raw_env.space.step(raw_env.dt)
        
        # Collect impulses
        arbiters = []
        agent_body.each_arbiter(lambda arb: arbiters.append(arb))
        for arb in arbiters:
            body_a = arb.shapes[0].body
            body_b = arb.shapes[1].body
            is_agent_block = (body_a == agent_body and body_b == block_body) or \
                             (body_a == block_body and body_b == agent_body)
            if is_agent_block:
                contact_exists = True
                total_impulse += arb.total_impulse
                for p in arb.contact_point_set.points:
                    contact_points.append(tuple(p.point_a))
                    
    return contact_exists, total_impulse, contact_points

def main():
    print("Initializing PushT Env...")
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    env.reset()
    raw_env = env.unwrapped
    
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    print(f"Loading episode 0 from {h5_path}...")
    
    import hdf5plugin
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    
    with h5py.File(h5_path, 'r') as f:
        ep_len = f['ep_len'][0]
        offset = f['ep_offset'][0]
        states = f['state'][offset : offset + ep_len]
        actions = f['action'][offset : offset + ep_len]
        pixels = f['pixels'][offset : offset + ep_len]
        
    print(f"Episode 0 length: {ep_len} steps. Processing...")
    frames = []
    
    for i in range(ep_len - 1):
        state_curr = states[i]
        action_t = actions[i]
        orig_img = pixels[i].copy()  # (224, 224, 3)
        
        # Inject state and run physics step to collect forces
        raw_env._set_state(state_curr[:5])
        contact_exists, impulse, points = step_and_collect_forces(env, action_t)
        
        # Draw on top of the original RGB image (224x224)
        vis_img = orig_img.copy()
        
        if contact_exists and impulse.length > 1.0:
            # We want to draw the contact force vector (direction and magnitude)
            # Normal pymunk coordinate space is 512x512, dataset pixel space is 224x224.
            # Scale coordinates from 512 to 224
            scale = 224.0 / 512.0
            
            # The contact points are in local or world space. Let's get the pusher's position as center.
            agent_pos = np.array(raw_env.agent.position) * scale
            
            # Contact force is opposite to pymunk resolving impulse
            force_vec = -np.array([impulse.x, impulse.y])
            force_mag = np.linalg.norm(force_vec)
            
            if force_mag > 1e-3:
                # Normalize and cap the force arrow length for visualization
                force_dir = force_vec / force_mag
                arrow_len = min(40, int(force_mag / 400.0) + 10)
                arrow_end = agent_pos + force_dir * arrow_len
                
                # Convert to integer pixel coordinates
                pt_start = (int(agent_pos[0]), int(agent_pos[1]))
                pt_end = (int(arrow_end[0]), int(arrow_end[1]))
                
                # Draw red arrow on the frame
                cv2.arrowedLine(vis_img, pt_start, pt_end, (255, 0, 0), 2, tipLength=0.3)
                
                # Draw a small yellow circle at contact point(s)
                if points:
                    for pt in points:
                        pt_scaled = (int(pt[0] * scale), int(pt[1] * scale))
                        cv2.circle(vis_img, pt_scaled, 2, (0, 255, 255), -1)
                        
        frames.append(Image.fromarray(vis_img))
        
    out_gif = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9/episode_0_forces.gif'
    print(f"Compiling and saving GIF to {out_gif}...")
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print("Done!")

if __name__ == '__main__':
    main()
