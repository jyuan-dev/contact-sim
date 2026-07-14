"""
Test groundtruth contact force extraction from the PushT Gym simulator.
"""
import sys
import os
import h5py
import numpy as np
import pygame
import pymunk

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GYM_PUSHT  = os.path.join(REPO_ROOT, 'third_party', 'gym-pusht')

if GYM_PUSHT not in sys.path:
    sys.path.insert(0, GYM_PUSHT)

import gymnasium as gym
import gym_pusht

def main():
    # Load env
    print("Initializing PushT Env...")
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    env.reset()
    raw_env = env.unwrapped
    
    # Load states from dataset
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    print(f"Loading states from {h5_path}...")
    
    import hdf5plugin
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    
    with h5py.File(h5_path, 'r') as f:
        states = f['state'][0:300]  # first 300 steps
        
    print(f"Processing 300 steps to find contact instances...")
    contact_count = 0
    
    for step_idx in range(len(states)):
        state = states[step_idx]
        
        # Inject state and run physics step to resolve contact/forces
        raw_env._set_state(state[:5])
        
        # Look for contact between agent and block
        agent_body = raw_env.agent
        block_body = raw_env.block
        
        contact_found = False
        total_impulse = pymunk.Vec2d(0, 0)
        contact_points = []
        
        # Collect active arbiters associated with the agent body
        arbiters = []
        agent_body.each_arbiter(lambda arb: arbiters.append(arb))
        
        for arbiter in arbiters:
            body_a = arbiter.shapes[0].body
            body_b = arbiter.shapes[1].body
            
            # Check if this arbiter is between agent and block
            is_agent_block = (body_a == agent_body and body_b == block_body) or \
                             (body_a == block_body and body_b == agent_body)
                             
            if is_agent_block:
                contact_found = True
                total_impulse += arbiter.total_impulse
                # Get contact points
                for p in arbiter.contact_point_set.points:
                    contact_points.append(tuple(p.point_a)) # point in body coordinate space
                    
        if contact_found:
            contact_count += 1
            impulse_magnitude = total_impulse.length
            if contact_count <= 10:
                print(f"Step {step_idx:3d} | CONTACT FOUND! "
                      f"Impulse={total_impulse} (Mag={impulse_magnitude:.4f}) | "
                      f"Points={contact_points}")
                
    print(f"\nFinished processing. Found {contact_count} steps with contacts out of 300 steps.")

if __name__ == '__main__':
    main()
