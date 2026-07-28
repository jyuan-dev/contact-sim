#!/usr/bin/env python3
"""
Script to download PushT dataset from Hugging Face / LeRobot repository.
"""

import argparse
import os
import sys
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is required. Install via `pip install huggingface_hub`.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Download PushT dataset.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/pusht",
        help="HuggingFace dataset repo ID for PushT (default: lerobot/pusht)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/home/jyuan/.stable-wm/pusht",
        help="Local target directory for PushT dataset (default: /home/jyuan/.stable-wm/pusht)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of download threads (default: 8)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    print(f"Starting PushT dataset download from '{args.repo_id}' to '{args.target_dir}'...")
    start_time = time.time()

    try:
        download_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.target_dir,
            max_workers=args.max_workers,
            resume_download=True,
        )
        elapsed = time.time() - start_time
        print(f"✅ Success! PushT dataset downloaded to {download_path} in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ Error downloading PushT dataset: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
