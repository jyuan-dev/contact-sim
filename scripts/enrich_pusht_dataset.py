"""
Enrich the PushT HDF5 dataset with groundtruth masks, contact state, and normal/frictional forces.
Parallelized with multiprocessing for fast execution.
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import sys
import argparse
import hdf5plugin
import h5py
import numpy as np
import cv2
import pygame
import pymunk
from tqdm import tqdm
from multiprocessing import Pool

# Setup paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
GYM_PUSHT  = os.path.join(REPO_ROOT, 'third_party', 'gym-pusht')

if GYM_PUSHT not in sys.path:
    sys.path.insert(0, GYM_PUSHT)

import gymnasium as gym
import gym_pusht
from gym_pusht.envs.pymunk_override import DrawOptions

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
    
    # Position agent/block
    # CRITICAL: Always set block COG to (0, 0) before setting position to avoid shifts
    raw_env.block.center_of_gravity = (0, 0)
    raw_env.agent.position = list(state[:2])
    raw_env.block.position = list(state[2:4])
    raw_env.block.angle = state[4]
    
    raw_env.agent.velocity = pymunk.Vec2d(state[5], state[6])
    
    agent_body = raw_env.agent
    target_pos = state[:2] + action * 30.0
    
    for _ in range(10):
        acc = raw_env.k_p * (target_pos - agent_body.position) + raw_env.k_v * (pymunk.Vec2d(0, 0) - agent_body.velocity)
        agent_body.velocity += acc * 0.01
        raw_env.space.step(0.01)

def render_mask(env, target='block', width=224, height=224):
    """
    Render a groundtruth segmentation mask for a target object.
    Targets: 'block', 'agent', or 'goal'.
    """
    raw_env = env.unwrapped
    
    # Save original colors
    orig_colors = {}
    for shape in raw_env.space.shapes:
        orig_colors[shape] = shape.color
        
    # CRITICAL FIX: Ensure COG is (0, 0) so setting angle does not shift position
    raw_env.block.center_of_gravity = (0, 0)
    
    screen = pygame.Surface((512, 512))
    screen.fill((0, 0, 0))  # black background
    draw_options = DrawOptions(screen)
    
    if target == 'goal':
        goal_body = raw_env.get_goal_pose_body(raw_env.goal_pose)
        for shape in raw_env.block.shapes:
            goal_points = [goal_body.local_to_world(v) for v in shape.get_vertices()]
            goal_points = [pymunk.pygame_util.to_pygame(pt, screen) for pt in goal_points]
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
                
        # Draw shapes
        raw_env.space.debug_draw(draw_options)
        
    # Restore original colors
    for shape, color in orig_colors.items():
        shape.color = color
        
    img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    img = cv2.resize(img, (width, height))
    mask = (img[:, :, 0] > 127).astype(np.uint8) * 255
    return mask

def process_episode_chunk(chunk_args):
    """
    Worker function to process a chunk of episodes in parallel.
    """
    chunk_idx, episodes_to_run, unoccluded = chunk_args
    
    # Initialize pygame for this process (SDL dummy driver must be set)
    pygame.init()
    
    # Initialize PushT env with block_cog override to prevent alignment shifts
    env = gym.make('gym_pusht/PushT-v0', obs_type='state', block_cog=(0, 0))
    env.reset()
    raw_env = env.unwrapped
    
    # Enable shape friction
    for shape in raw_env.agent.shapes:
        shape.friction = 1.0
    for shape in raw_env.block.shapes:
        shape.friction = 1.0
        
    # Setup contact tracker
    tracker = ContactTracker(raw_env.agent, raw_env.block)
    handler = raw_env.space.add_collision_handler(0, 0)
    handler.post_solve = tracker.post_solve
    
    # Lists to store results
    chunk_b_masks = []
    chunk_a_masks = []
    chunk_g_masks = []
    chunk_contact_pos = []
    chunk_normal_forces = []
    chunk_frictional_forces = []
    
    # Process episodes sequentially inside this worker
    for global_step_idx, ep_states, ep_actions in episodes_to_run:
        length = len(ep_states)
        for t in range(length):
            # 1. Step physics & resolve forces
            if t < length - 1:
                step_physics_with_tracker(env, ep_states[t], ep_actions[t], tracker)
                if tracker.contact_exists and len(tracker.contact_pts) > 0:
                    cx = sum(p.x for p in tracker.contact_pts) / len(tracker.contact_pts)
                    cy = sum(p.y for p in tracker.contact_pts) / len(tracker.contact_pts)
                    contact_pos_val = [cx, cy]
                else:
                    contact_pos_val = [float('nan'), float('nan')]
                fn_vec = -tracker.total_normal_impulse / 10.0
                ft_vec = -tracker.total_tangential_impulse / 10.0
            else:
                contact_pos_val = [float('nan'), float('nan')]
                fn_vec = pymunk.Vec2d(0, 0)
                ft_vec = pymunk.Vec2d(0, 0)
                
            # 2. Position block and render masks
            raw_env.block.center_of_gravity = (0, 0)
            raw_env.agent.position = list(ep_states[t][:2])
            raw_env.block.position = list(ep_states[t][2:4])
            raw_env.block.angle = ep_states[t][4]
            raw_env.space.step(raw_env.dt)
            
            b_mask = render_mask(env, target='block', width=224, height=224)
            a_mask = render_mask(env, target='agent', width=224, height=224)
            g_mask = render_mask(env, target='goal', width=224, height=224)
            
            if not unoccluded:
                # Subtract agent and block masks from the goal mask to show only camera-visible parts
                g_mask_clean = g_mask.copy()
                g_mask_clean[b_mask > 0] = 0
                g_mask_clean[a_mask > 0] = 0
                g_mask = g_mask_clean

            chunk_b_masks.append(b_mask)
            chunk_a_masks.append(a_mask)
            chunk_g_masks.append(g_mask)
            chunk_contact_pos.append(contact_pos_val)
            chunk_normal_forces.append([fn_vec.x, fn_vec.y])
            chunk_frictional_forces.append([ft_vec.x, ft_vec.y])
            
    pygame.quit()
    return (
        chunk_idx,
        chunk_b_masks,
        chunk_a_masks,
        chunk_g_masks,
        chunk_contact_pos,
        chunk_normal_forces,
        chunk_frictional_forces
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/jyuan/.stable-wm/pusht_expert_train.h5',
                        help='Path to the input expert h5 dataset')
    parser.add_argument('--output', type=str, default='/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5',
                        help='Path to save the enriched h5 dataset')
    parser.add_argument('--test', action='store_true',
                        help='If set, runs on a test subset of 1 episode and writes to scratch')
    parser.add_argument('--num-workers', type=int, default=24,
                        help='Number of workers for parallel mask generation')
    parser.add_argument('--unoccluded', action='store_true', default=False,
                        help='If set, target T-shape masks will not be occluded by block or agent')
    args = parser.parse_args()

    input_h5 = args.input
    output_h5 = args.output
    if args.test:
        output_h5 = os.path.join(REPO_ROOT, 'scratch', 'pusht_expert_train_test_enriched.h5')
        print(f"Running in TEST mode. Output will be saved to: {output_h5}")

    if not os.path.exists(input_h5):
        print(f"Input file not found at {input_h5}")
        sys.exit(1)

    print("Opening input dataset...")
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    
    with h5py.File(input_h5, 'r') as f_in:
        # Determine number of episodes to process
        num_episodes = f_in['ep_len'].shape[0]
        if args.test:
            num_episodes = min(num_episodes, 2)  # process only 2 episodes for test
            
        print(f"Dataset has {f_in['ep_len'].shape[0]} episodes. Processing {num_episodes} episodes...")
        
        # Calculate total steps to process
        total_steps = sum(f_in['ep_len'][:num_episodes])
        print(f"Total steps to process: {total_steps}")
        
        # 1. Partition episodes into chunks of size e.g. 50 episodes
        episodes_per_chunk = 50
        chunks = []
        chunk_idx = 0
        current_chunk_episodes = []
        
        global_step_idx = 0
        for ep_idx in range(num_episodes):
            offset = f_in['ep_offset'][ep_idx]
            length = f_in['ep_len'][ep_idx]
            
            ep_states = f_in['state'][offset : offset + length]
            ep_actions = f_in['action'][offset : offset + length]
            
            current_chunk_episodes.append((global_step_idx, ep_states, ep_actions))
            global_step_idx += length
            
            if len(current_chunk_episodes) >= episodes_per_chunk or ep_idx == num_episodes - 1:
                chunks.append((chunk_idx, current_chunk_episodes, args.unoccluded))
                current_chunk_episodes = []
                chunk_idx += 1
                
        print(f"Divided into {len(chunks)} chunks for parallel rendering.")
        
        # Create output file
        print(f"Creating output dataset: {output_h5}...")
        os.makedirs(os.path.dirname(output_h5), exist_ok=True)
        
        with h5py.File(output_h5, 'w') as f_out:
            # Copy original keys first (resizing them if in test mode)
            for key in f_in.keys():
                if args.test:
                    if key in ['ep_len', 'ep_offset']:
                        data = f_in[key][:num_episodes]
                    else:
                        data = f_in[key][:total_steps]
                    f_out.create_dataset(key, data=data, **hdf5plugin.Zstd())
                else:
                    f_in.copy(key, f_out)
                print(f"Copied dataset key: {key}")

            # Create new datasets for enriched attributes
            block_masks = f_out.create_dataset('block_masks', shape=(total_steps, 224, 224), dtype=np.uint8,
                                               chunks=(1, 224, 224), **hdf5plugin.Zstd())
            agent_masks = f_out.create_dataset('agent_masks', shape=(total_steps, 224, 224), dtype=np.uint8,
                                               chunks=(1, 224, 224), **hdf5plugin.Zstd())
            goal_masks = f_out.create_dataset('goal_masks', shape=(total_steps, 224, 224), dtype=np.uint8,
                                              chunks=(1, 224, 224), **hdf5plugin.Zstd())
            chunk_size = min(1000, total_steps)
            contact_pos = f_out.create_dataset('contact_pos', shape=(total_steps, 2), dtype=np.float32,
                                               chunks=(chunk_size, 2), **hdf5plugin.Zstd())
            normal_forces = f_out.create_dataset('normal_force', shape=(total_steps, 2), dtype=np.float32,
                                                 chunks=(chunk_size, 2), **hdf5plugin.Zstd())
            frictional_forces = f_out.create_dataset('frictional_force', shape=(total_steps, 2), dtype=np.float32,
                                                     chunks=(chunk_size, 2), **hdf5plugin.Zstd())
            
            print(f"Starting parallel processing with {args.num_workers} workers...")
            with Pool(args.num_workers) as pool:
                for (
                    c_idx,
                    b_masks,
                    a_masks,
                    g_masks,
                    c_pos,
                    fn_f,
                    ft_f
                ) in tqdm(pool.imap_unordered(process_episode_chunk, chunks), total=len(chunks)):
                    # Get start step index of this chunk
                    start_step_idx = chunks[c_idx][1][0][0]
                    n_steps = len(b_masks)
                    
                    block_masks[start_step_idx : start_step_idx + n_steps] = b_masks
                    agent_masks[start_step_idx : start_step_idx + n_steps] = a_masks
                    goal_masks[start_step_idx : start_step_idx + n_steps] = g_masks
                    contact_pos[start_step_idx : start_step_idx + n_steps] = c_pos
                    normal_forces[start_step_idx : start_step_idx + n_steps] = fn_f
                    frictional_forces[start_step_idx : start_step_idx + n_steps] = ft_f

    print(f"\nSuccessfully enriched dataset. File saved to: {output_h5}")

if __name__ == '__main__':
    main()
