"""
Proof of concept: Generate groundtruth segmentation masks directly from the PushT Gym simulator.
"""
import sys
import os
import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
import h5py
import numpy as np
import torch
import cv2
import pygame
import pymunk
from PIL import Image

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOTFORMER = os.path.join(REPO_ROOT, 'third_party', 'cjepa', 'src',
                          'third_party', 'slotformer')
HDF5_DS    = os.path.join(SLOTFORMER, 'base_slots')
GYM_PUSHT  = os.path.join(REPO_ROOT, 'third_party', 'gym-pusht')

for p in [REPO_ROOT, SLOTFORMER, HDF5_DS, GYM_PUSHT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import gymnasium as gym
import gym_pusht
from gym_pusht.envs.pymunk_override import DrawOptions

def render_mask(env, state, target='block', width=224, height=224):
    """
    Render a groundtruth segmentation mask for a target object ('block', 'agent', or 'goal').
    """
    raw_env = env.unwrapped

    # 1. Set state
    raw_env._set_state(state[:5])
    
    # 2. Save original colors of all shapes in the space
    orig_colors = {}
    for shape in raw_env.space.shapes:
        orig_colors[shape] = shape.color
        
    # 3. Create a black surface
    screen = pygame.Surface((512, 512))
    screen.fill((0, 0, 0))  # black background
    
    draw_options = DrawOptions(screen)
    
    if target == 'goal':
        # Draw ONLY the goal pose in white
        goal_body = raw_env.get_goal_pose_body(raw_env.goal_pose)
        for shape in raw_env.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("white"), goal_points)
    else:
        # Configure colors for target shapes
        for shape in raw_env.space.shapes:
            if target == 'agent' and shape.body == raw_env.agent:
                shape.color = pygame.Color("white")
            elif target == 'block' and shape.body == raw_env.block:
                shape.color = pygame.Color("white")
            else:
                shape.color = pygame.Color("black")
                
        # Draw shapes
        raw_env.space.debug_draw(draw_options)
        
    # 4. Restore original colors
    for shape, color in orig_colors.items():
        shape.color = color
        
    # 5. Extract image, convert to binary mask
    img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    img = cv2.resize(img, (width, height))
    mask = (img[:, :, 0] > 127).astype(np.uint8) * 255
    return mask

def main():
    # Load env
    print("Initializing PushT Env...")
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    env.reset()
    
    # Load a state from dataset
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    print(f"Loading state from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Get step 0 values
        state = f['state'][0]
        orig_img = f['pixels'][0]  # (224, 224, 3)
        
    print("State loaded:", state)
    
    # Generate masks
    print("Generating masks...")
    block_mask = render_mask(env, state, target='block', width=224, height=224)
    agent_mask = render_mask(env, state, target='agent', width=224, height=224)
    goal_mask  = render_mask(env, state, target='goal', width=224, height=224)
    
    # Save the output visualization images
    out_dir = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9'
    
    # Save original image
    Image.fromarray(orig_img).save(os.path.join(out_dir, 'gt_orig.png'))
    # Save mask images
    Image.fromarray(block_mask).save(os.path.join(out_dir, 'gt_mask_block.png'))
    Image.fromarray(agent_mask).save(os.path.join(out_dir, 'gt_mask_agent.png'))
    Image.fromarray(goal_mask).save(os.path.join(out_dir, 'gt_mask_goal.png'))
    
    print("Groundtruth masks saved successfully to:", out_dir)

if __name__ == '__main__':
    main()
