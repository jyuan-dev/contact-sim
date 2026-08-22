#!/usr/bin/env python3
"""
Direct HTTP stream downloader for Hugging Face datasets (such as Pokuang/ContactWorld)
with resume capability (.part files via HTTP Range), rate limiting, and progress tracking.
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
    from huggingface_hub import HfApi, hf_hub_url
except ImportError:
    print("Error: requests, tqdm, and huggingface_hub are required. Install via `pip install requests tqdm huggingface_hub`.")
    sys.exit(1)


# ============================================================
# Helpers
# ============================================================

def format_time(seconds):
    if seconds is None or seconds == float("inf"):
        return "unknown"

    seconds = int(seconds)

    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    if minutes:
        return f"{minutes}m {seconds:02d}s"

    return f"{seconds}s"


def get_headers():
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_file_size(url):
    """
    Get remote file size after redirects.
    """
    headers = get_headers()
    try:
        r = requests.head(
            url,
            headers=headers,
            allow_redirects=True,
            timeout=30,
        )
        r.raise_for_status()

        size = r.headers.get("Content-Length")
        if size is not None:
            return int(size)
    except Exception as e:
        print(f"Warning: could not get size for {url}: {e}")

    return None


# ============================================================
# Rate limiter
# ============================================================

class RateLimiter:
    """
    Limits aggregate bandwidth across the entire Python process.
    If rate_bytes_per_sec <= 0, throttling is disabled.
    """

    def __init__(self, rate_bytes_per_sec):
        self.rate = rate_bytes_per_sec
        self.start_time = time.monotonic()
        self.bytes_downloaded = 0

    def throttle(self, num_bytes):
        if self.rate is None or self.rate <= 0 or num_bytes <= 0:
            return

        self.bytes_downloaded += num_bytes

        expected_time = self.bytes_downloaded / self.rate
        actual_time = time.monotonic() - self.start_time

        sleep_time = expected_time - actual_time

        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================
# Download
# ============================================================

def download_file_with_retry(
    filename,
    url,
    output_path,
    file_size,
    overall_bar,
    limiter,
    chunk_size,
    max_retries=20,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part_path = Path(str(output_path) + ".part")

    retries = 0
    while retries < max_retries:
        try:
            existing = 0
            if part_path.exists():
                existing = part_path.stat().st_size

            # If the part file already equals or exceeds the full size, check or finalize
            if file_size is not None and existing >= file_size:
                part_path.replace(output_path)
                return

            headers = get_headers()
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"

            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 300),
            ) as response:

                # Handle 416 Range Not Satisfiable (file already fully downloaded in .part)
                if response.status_code == 416:
                    if part_path.exists():
                        part_path.replace(output_path)
                        return

                response.raise_for_status()

                # Resume was accepted
                if existing > 0 and response.status_code == 206:
                    mode = "ab"
                # Server ignored Range request -> restart file
                elif existing > 0:
                    print(
                        f"\nServer did not support resume for {filename}; restarting file."
                    )
                    existing = 0
                    mode = "wb"
                else:
                    mode = "wb"

                with open(part_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue

                        f.write(chunk)
                        n = len(chunk)

                        # Rate limiting
                        limiter.throttle(n)

                        # Overall progress
                        overall_bar.update(n)

            part_path.replace(output_path)
            return

        except (requests.RequestException, IOError) as e:
            retries += 1
            wait_sec = min(30, 2 ** min(retries, 5))
            print(f"\n⚠️ Error downloading {filename} ({e}). Retrying {retries}/{max_retries} in {wait_sec}s...")
            time.sleep(wait_sec)

    raise RuntimeError(f"Failed to download {filename} after {max_retries} retries.")


# ============================================================
# Argument Parsing & Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face datasets with resume and bandwidth control."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Pokuang/ContactWorld",
        help="HuggingFace dataset repository ID (default: Pokuang/ContactWorld)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/data/ContactWorld",
        help="Local target directory for dataset (default: /data/ContactWorld)",
    )
    parser.add_argument(
        "--rate-mb-s",
        "--max-mb-per-sec",
        type=float,
        default=0.0,
        dest="rate_mb_s",
        help="Bandwidth limit in MB/s (default: 0.0 = unlimited)",
    )
    parser.add_argument(
        "--chunk-size-kb",
        type=int,
        default=256,
        help="Chunk size in KiB (default: 256)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=20,
        help="Max retries per file (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_id = args.repo_id
    local_dir = Path(args.target_dir)
    rate_mb_s = args.rate_mb_s
    chunk_size = args.chunk_size_kb * 1024
    rate_bytes_s = rate_mb_s * 1024 * 1024 if rate_mb_s > 0 else 0

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Hugging Face Stream Downloader")
    print("=" * 70)
    print(f"Dataset      : {repo_id}")
    print(f"Target folder: {local_dir}")
    print(f"Rate limit   : {'Unlimited' if rate_mb_s <= 0 else f'{rate_mb_s:.2f} MB/s'}")
    print("=" * 70)
    print()

    api = HfApi()

    # --------------------------------------------------------
    # List repository files
    # --------------------------------------------------------
    print("Getting dataset file list...")
    filenames = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Found {len(filenames)} files.\n")

    # --------------------------------------------------------
    # Determine sizes & existing progress
    # --------------------------------------------------------
    print("Checking download sizes & existing files...")
    files = []
    total_size = 0
    already_downloaded = 0

    for i, filename in enumerate(filenames, 1):
        output_path = local_dir / filename
        part_path = Path(str(output_path) + ".part")

        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )

        # Already completely downloaded
        if output_path.exists():
            size = output_path.stat().st_size
            files.append(
                {
                    "name": filename,
                    "url": url,
                    "path": output_path,
                    "size": size,
                    "done": True,
                }
            )
            total_size += size
            already_downloaded += size
            print(f"[{i:3d}/{len(filenames)}] [DONE] {filename} ({size / 1024**2:.1f} MB)")
            continue

        size = get_file_size(url)
        part_size = part_path.stat().st_size if part_path.exists() else 0

        files.append(
            {
                "name": filename,
                "url": url,
                "path": output_path,
                "size": size,
                "done": False,
            }
        )

        if size is not None:
            total_size += size
            already_downloaded += part_size
            part_str = f" [partially resumed: {part_size / 1024**2:.1f} MB]" if part_size > 0 else ""
            print(f"[{i:3d}/{len(filenames)}] [TODO] {filename} ({size / 1024**2:.1f} MB){part_str}")
        else:
            print(f"[{i:3d}/{len(filenames)}] [TODO] {filename} (size unknown)")

    print()

    # --------------------------------------------------------
    # Estimate total time
    # --------------------------------------------------------
    remaining = total_size - already_downloaded
    expected_seconds = (remaining / rate_bytes_s) if rate_bytes_s > 0 else 0

    print("=" * 70)
    print(f"Total dataset size : {total_size / 1024**3:.2f} GB")
    print(f"Already on disk    : {already_downloaded / 1024**3:.2f} GB")
    print(f"Remaining to fetch : {remaining / 1024**3:.2f} GB")
    if rate_bytes_s > 0:
        print(f"Bandwidth limit    : {rate_mb_s:.2f} MB/s")
        print(f"Expected time      : {format_time(expected_seconds)}")
    else:
        print(f"Bandwidth limit    : Unlimited (Max Speed)")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Global rate limiter
    # --------------------------------------------------------
    limiter = RateLimiter(rate_bytes_s)

    # --------------------------------------------------------
    # Overall progress bar
    # --------------------------------------------------------
    with tqdm(
        total=total_size,
        initial=already_downloaded,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Overall",
        dynamic_ncols=True,
    ) as overall_bar:

        for index, file in enumerate(files, 1):
            if file["done"]:
                continue

            overall_bar.write(f"[{index}/{len(files)}] Downloading: {file['name']}")
            overall_bar.set_description(f"[{index}/{len(files)}] {file['name'][:25]}")
            download_file_with_retry(
                filename=file["name"],
                url=file["url"],
                output_path=file["path"],
                file_size=file["size"],
                overall_bar=overall_bar,
                limiter=limiter,
                chunk_size=chunk_size,
                max_retries=args.max_retries,
            )

    print()
    print("=" * 70)
    print("✅ Download complete.")
    print(f"Saved to: {local_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
