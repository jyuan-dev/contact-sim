"""
Test extraction of normal and tangential (frictional) contact forces from PushT.
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

def step_and_collect_friction(env, state, action):
    raw_env = env.unwrapped
    
    # 1. Set state and initial velocity
    raw_env._set_state(state[:5])
    raw_env.agent.velocity = pymunk.Vec2d(state[5], state[6])
    
    agent_body = raw_env.agent
    block_body = raw_env.block
    
    # Target position calculation
    target_pos = state[:2] + action * 30.0
    
    total_normal_impulse = pymunk.Vec2d(0, 0)
    total_tangential_impulse = pymunk.Vec2d(0, 0)
    contact_exists = False
    
    def post_solve_callback(arb, sp, data):
        nonlocal contact_exists, total_normal_impulse, total_tangential_impulse
        body_a = arb.shapes[0].body
        body_b = arb.shapes[1].body
        is_agent_block = (body_a == agent_body and body_b == block_body) or \
                         (body_a == block_body and body_b == agent_body)
        if is_agent_block:
            contact_exists = True
            normal = arb.normal
            impulse_vec = arb.total_impulse
            
            normal_magnitude = impulse_vec.dot(normal)
            normal_imp_vec = normal * normal_magnitude
            tangential_imp_vec = impulse_vec - normal_imp_vec
            
            total_normal_impulse += normal_imp_vec
            total_tangential_impulse += tangential_imp_vec
            
    handler = raw_env.space.add_collision_handler(0, 0)
    handler.post_solve = post_solve_callback
    
    # Run 10 sub-steps
    for _ in range(10):
        acc = raw_env.k_p * (target_pos - agent_body.position) + raw_env.k_v * (pymunk.Vec2d(0, 0) - agent_body.velocity)
        agent_body.velocity += acc * 0.01
        raw_env.space.step(0.01)
        
    handler.post_solve = None
    return contact_exists, total_normal_impulse, total_tangential_impulse

def main():
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    env.reset()
    raw_env = env.unwrapped
    
    # Enable shape friction (normally 0.0 by default in the simulator)
    for shape in raw_env.agent.shapes:
        shape.friction = 1.0
    for shape in raw_env.block.shapes:
        shape.friction = 1.0
    
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    
    import hdf5plugin
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    
    with h5py.File(h5_path, 'r') as f:
        states = f['state'][60:75]
        actions = f['action'][60:75]
        
    print("Replaying steps 60 to 75 to analyze normal & frictional forces...")
    for i in range(len(states) - 1):
        state_curr = states[i]
        action_t = actions[i]
        
        contact_exists, normal_imp, friction_imp = step_and_collect_friction(env, state_curr, action_t)
        
        if contact_exists:
            print(f"Step {60+i:2d}:")
            print(f"  Normal Impulse:     {normal_imp} (Mag={normal_imp.length:.4f})")
            print(f"  Frictional Impulse: {friction_imp} (Mag={friction_imp.length:.4f})")
            if normal_imp.length > 0:
                print(f"  Friction Ratio (F_t / F_n): {friction_imp.length / normal_imp.length:.4f}")

if __name__ == '__main__':
    main()
