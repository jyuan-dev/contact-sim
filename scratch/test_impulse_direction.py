"""
Check alignment between agent movement, block movement, and contact impulse.
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
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    env.reset()
    raw_env = env.unwrapped
    
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    
    import hdf5plugin
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    
    with h5py.File(h5_path, 'r') as f:
        states = f['state'][60:75]
        actions = f['action'][60:75]
        
    for i in range(len(states) - 1):
        state_curr = states[i]
        state_next = states[i+1]
        action_t = actions[i]
        
        raw_env._set_state(state_curr[:5])
        contact_exists, impulse, points = step_and_collect_forces(env, action_t)
        
        if contact_exists and impulse.length > 1.0:
            agent_disp = state_next[:2] - state_curr[:2]
            block_disp = state_next[2:4] - state_curr[2:4]
            print(f"Step {60+i:2d}:")
            print(f"  Agent displacement: {agent_disp}")
            print(f"  Block displacement: {block_disp}")
            print(f"  Contact Impulse:    {impulse}")
            # cosine similarity
            if np.linalg.norm(block_disp) > 1e-3:
                cos_sim = np.dot(block_disp, (impulse.x, impulse.y)) / (np.linalg.norm(block_disp) * impulse.length)
                print(f"  Cos Sim (Block, Impulse): {cos_sim:.4f}")
                
if __name__ == '__main__':
    main()
