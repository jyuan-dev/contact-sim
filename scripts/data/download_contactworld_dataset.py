#!/usr/bin/env python3
"""
Script to download Hugging Face datasets (such as Pokuang/ContactWorld) with
guaranteed rate limiting across all download backends (httpx, requests, urllib3).
"""

import argparse
import os
import sys
import threading
import time

# Disable fast C/Rust extensions (xet, hf_transfer) that bypass Python network rate limiting
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

try:
    import httpx
    import requests
    import urllib3.response
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub, httpx, and requests are required. Install via `pip install huggingface_hub httpx requests`.")
    sys.exit(1)


class RateLimiter:
    """Thread-safe global rate limiter for controlling download bandwidth."""
    def __init__(self, max_bytes_per_sec):
        self.max_bytes_per_sec = max_bytes_per_sec
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.total_bytes = 0

    def consume(self, num_bytes):
        if num_bytes <= 0 or self.max_bytes_per_sec is None:
            return
        with self.lock:
            self.total_bytes += num_bytes
            now = time.time()
            elapsed = now - self.start_time
            expected_time = self.total_bytes / self.max_bytes_per_sec
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)


def enable_rate_limiting(max_mb_per_sec):
    if max_mb_per_sec is None or max_mb_per_sec <= 0:
        return
    max_bytes_per_sec = float(max_mb_per_sec) * 1024 * 1024
    limiter = RateLimiter(max_bytes_per_sec)

    # 1. Patch httpx (used by modern huggingface_hub http_get)
    orig_httpx_bytes = httpx.Response.iter_bytes
    def rate_limited_httpx_bytes(self, chunk_size=None):
        for chunk in orig_httpx_bytes(self, chunk_size=chunk_size):
            if chunk:
                limiter.consume(len(chunk))
            yield chunk
    httpx.Response.iter_bytes = rate_limited_httpx_bytes

    orig_httpx_raw = httpx.Response.iter_raw
    def rate_limited_httpx_raw(self, chunk_size=None):
        for chunk in orig_httpx_raw(self, chunk_size=chunk_size):
            if chunk:
                limiter.consume(len(chunk))
            yield chunk
    httpx.Response.iter_raw = rate_limited_httpx_raw

    # 2. Patch requests & urllib3 (fallback HTTP backends)
    orig_req_iter = requests.Response.iter_content
    def rate_limited_req_iter(self, chunk_size=1024 * 1024, decode_unicode=False):
        for chunk in orig_req_iter(self, chunk_size=chunk_size, decode_unicode=decode_unicode):
            if chunk:
                limiter.consume(len(chunk))
            yield chunk
    requests.Response.iter_content = rate_limited_req_iter

    orig_u3_read = urllib3.response.HTTPResponse.read
    def rate_limited_u3_read(self, amt=None, *args, **kwargs):
        data = orig_u3_read(self, amt, *args, **kwargs)
        if data:
            limiter.consume(len(data))
        return data
    urllib3.response.HTTPResponse.read = rate_limited_u3_read

    print(f"🔒 Rate limit enabled: strictly capped at {max_mb_per_sec:.2f} MB/s ({max_bytes_per_sec / 1024:.1f} KB/s) across all backends.")


def parse_args():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets with speed limiting.")
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
        "--max-mb-per-sec",
        type=float,
        default=0.5,
        help="Maximum download speed limit in MB/s (default: 0.5 MB/s = 500 KB/s)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of concurrent download worker threads (default: 2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    if args.max_mb_per_sec:
        enable_rate_limiting(args.max_mb_per_sec)

    print(f"Starting download of '{args.repo_id}' to '{args.target_dir}'...")
    start_time = time.time()

    try:
        download_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.target_dir,
            max_workers=args.max_workers,
        )
        elapsed = time.time() - start_time
        print(f"✅ Success! Dataset downloaded to {download_path} in {elapsed:.2f}s")
    except Exception as e:
        print(f"❌ Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
