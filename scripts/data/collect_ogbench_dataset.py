#!/usr/bin/env python3
"""
Script to collect and save OGBench expert dataset episodes using stable-worldmodel.
"""

import argparse
import os
from pathlib import Path
import sys

# Ensure EGL backend for headless rendering
import OpenGL
OpenGL.PLATFORM = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

# Add third_party path if needed
sys.path.insert(0, os.path.abspath('third_party/stable-worldmodel'))

try:
    import stable_worldmodel as swm
    from stable_worldmodel.envs.ogbench import ExpertPolicy
except ImportError:
    print("Error: stable_worldmodel is required. Ensure third_party/stable-worldmodel is available.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect OGBench expert dataset episodes.")
    parser.add_argument(
        "--env",
        type=str,
        default="swm/OGBCube-v0",
        help="OGBench environment ID (default: swm/OGBCube-v0)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/jyuan/.cache/swm/datasets/ogbench/cube_single_expert.lance",
        help="Target output path for the collected Lance dataset"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to collect (default: 2)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image resolution [width, height] (default: 224 224)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Initializing World '{args.env}'...")
    world = swm.World(
        args.env,
        num_envs=1,
        image_shape=tuple(args.resolution),
        env_type='single',
        multiview=False,
        width=args.resolution[0],
        height=args.resolution[1],
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
    )

    print("Setting expert policy...")
    world.set_policy(ExpertPolicy())

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.episodes} episodes → {output_path} ...")
    world.collect(
        output_path,
        episodes=args.episodes,
        seed=args.seed,
    )
    print("🎉 Done collecting OGBench dataset!")


if __name__ == '__main__':
    main()
