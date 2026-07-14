"""
Test groundtruth impulse extraction by stepping physics manually with dataset actions.
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
    print(f"Loading from {h5_path}...")
    
    import hdf5plugin
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    
    with h5py.File(h5_path, 'r') as f:
        states = f['state'][0:100]
        actions = f['action'][0:100]
        
    print("Processing 100 steps...")
    contact_count = 0
    for i in range(len(states) - 1):
        state_t = states[i]
        action_t = actions[i]
        
        # Set state to t
        raw_env._set_state(state_t[:5])
        
        # Step env with action and collect forces
        contact_exists, impulse, points = step_and_collect_forces(env, action_t)
        
        if contact_exists:
            contact_count += 1
            if contact_count <= 15:
                print(f"Step {i:2d} | CONTACT! Impulse={impulse} (Mag={impulse.length:.4f}) | Points={len(points)}")
                
    print(f"\nTotal steps with contact: {contact_count} / 99")

if __name__ == '__main__':
    main()
