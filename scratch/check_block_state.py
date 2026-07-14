import sys
import os
import h5py
import hdf5plugin
import gymnasium as gym
import numpy as np

# Path setup
REPO_ROOT = '/home/jyuan/jyuan-ws/contact-sim'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'third_party', 'gym-pusht'))

import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
env.reset()
raw_env = env.unwrapped

h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

with h5py.File(h5_path, 'r') as f:
    state = f['state'][0]

print("Original block state in dataset:", state[2:5])

# Manually set the positions step-by-step
raw_env.agent.position = list(state[:2])
print("After setting agent position: block position =", raw_env.block.position)

raw_env.block.position = list(state[2:4])
print("After setting block position: block position =", raw_env.block.position)

raw_env.block.angle = state[4]
print("After setting block angle: block position =", raw_env.block.position)

raw_env.space.step(raw_env.dt)
print("After space.step(): block position =", raw_env.block.position)

