import os
import sys
import gymnasium as gym
import h5py
import hdf5plugin
import numpy as np
import cv2
import pygame
import pymunk
from PIL import Image

REPO_ROOT = '/home/jyuan/jyuan-ws/contact-sim'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'third_party', 'gym-pusht'))

import gym_pusht
from gym_pusht.envs.pymunk_override import DrawOptions

def render_mask_fixed(env, state, target='block', width=224, height=224):
    raw_env = env.unwrapped
    orig_cog = raw_env.block.center_of_gravity
    orig_colors = {}
    for shape in raw_env.space.shapes:
        orig_colors[shape] = shape.color

    raw_env.block.center_of_gravity = (0, 0)
    
    raw_env.agent.position = list(state[:2])
    raw_env.block.position = list(state[2:4])
    raw_env.block.angle = state[4]
    raw_env.space.step(raw_env.dt)

    screen = pygame.Surface((512, 512))
    screen.fill((0, 0, 0))
    draw_options = DrawOptions(screen)

    if target == 'goal':
        goal_body = raw_env.get_goal_pose_body(raw_env.goal_pose)
        for shape in raw_env.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [pymunk.pygame_util.to_pygame(point, draw_options.surface) for point in goal_points]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(screen, pygame.Color("white"), goal_points)
    else:
        for shape in raw_env.space.shapes:
            if target == 'agent' and shape.body == raw_env.agent:
                shape.color = pygame.Color("white")
            elif target == 'block' and shape.body == raw_env.block:
                shape.color = pygame.Color("white")
            else:
                shape.color = pygame.Color("black")
        raw_env.space.debug_draw(draw_options)

    raw_env.block.center_of_gravity = orig_cog
    for shape, color in orig_colors.items():
        shape.color = color

    img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    img = cv2.resize(img, (width, height))
    mask = (img[:, :, 0] > 127).astype(np.uint8) * 255
    return mask

def main():
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    env.reset()
    
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    
    out_dir = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9'
    
    steps_to_test = [1000, 2500, 6000]
    
    with h5py.File(h5_path, 'r') as f:
        for step in steps_to_test:
            state = f['state'][step]
            orig_img = f['pixels'][step]
            
            block_mask = render_mask_fixed(env, state, target='block', width=224, height=224)
            agent_mask = render_mask_fixed(env, state, target='agent', width=224, height=224)
            
            # Create overlay
            overlay = orig_img.copy()
            # Red channel highlight for block
            overlay[block_mask > 127] = [0, 0, 255]
            # Green channel highlight for agent
            overlay[agent_mask > 127] = [0, 255, 0]
            
            save_path = os.path.join(out_dir, f'sample_step_{step}.png')
            Image.fromarray(overlay).save(save_path)
            print(f"Saved step {step} overlay to: {save_path}")

if __name__ == '__main__':
    main()
