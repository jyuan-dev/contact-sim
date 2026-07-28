#!/usr/bin/env python3
"""
Script to download Hugging Face datasets (such as Pokuang/ContactWorld) using
high-performance multi-threaded transfers.
"""

import argparse
import os
import sys
import time

# Enable high-performance Rust multi-thread transfer if installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is required. Install via `pip install huggingface_hub`.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets efficiently.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Pokuang/ContactWorld",
        help="HuggingFace dataset repository ID (default: Pokuang/ContactWorld)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/data/ContactWorld",
        help="Local directory to download the dataset into (default: /data/ContactWorld)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of concurrent download worker threads (default: 8)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    print(f"Starting download of '{args.repo_id}' to '{args.target_dir}'...")
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
        print(f"✅ Success! Dataset downloaded to {download_path} in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
