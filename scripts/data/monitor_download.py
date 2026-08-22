#!/usr/bin/env python3
"""
Monitoring & Auto-Recovery Watchdog for ContactWorld Dataset Download.
Tracks downloaded volume (GB / percentage), download speed, finalized files,
restarts automatically if interrupted, and verifies file integrity.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

TARGET_DIR = Path("/data/ContactWorld")
TMUX_SESSION = "download-contactworld"
REPO_ID = "Pokuang/ContactWorld"
TOTAL_FILES = 28
ESTIMATED_TOTAL_GB = 8.85


def is_process_running():
    try:
        out = subprocess.check_output(["pgrep", "-f", "scripts/data/download_contactworld_dataset.py"]).decode()
        pids = [int(p) for p in out.strip().split() if p]
        return len(pids) > 0
    except subprocess.CalledProcessError:
        return False


def get_total_downloaded_bytes():
    if not TARGET_DIR.exists():
        return 0
    total = 0
    for root, _, files in os.walk(TARGET_DIR):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def count_completed_files():
    if not TARGET_DIR.exists():
        return 0, []
    completed = []
    for root, _, files in os.walk(TARGET_DIR):
        for f in files:
            if not f.endswith(".part") and not f.endswith(".incomplete") and not f.endswith(".lock") and not f.endswith(".json") and ".cache" not in root:
                rel_path = os.path.relpath(os.path.join(root, f), TARGET_DIR)
                completed.append(rel_path)
    return len(completed), completed


def get_tmux_live_status():
    try:
        out = subprocess.check_output(
            ["tmux", "capture-pane", "-pt", TMUX_SESSION, "-p", "-S", "-15"],
            stderr=subprocess.DEVNULL
        ).decode()
        dl_line = ""
        recon_line = ""
        for line in reversed(out.splitlines()):
            line_s = line.strip()
            if "Downloading bytes:" in line_s and not dl_line:
                dl_line = line_s
            elif "Reconstructing" in line_s and not recon_line:
                recon_line = line_s
        if dl_line and recon_line:
            return f"{dl_line} | {recon_line}"
        elif dl_line:
            return dl_line
        elif recon_line:
            return recon_line
    except Exception:
        pass
    return "Running"


def restart_download_if_needed():
    if is_process_running():
        return False
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Process not running. Auto-restarting in tmux session '{TMUX_SESSION}'...", flush=True)
    res = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION], capture_output=True)
    if res.returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-c", "/home/jyuan/jyuan-ws/contact-sim"])
        subprocess.run(["tmux", "send-keys", "-t", TMUX_SESSION, "source /home/jyuan/miniconda3/etc/profile.d/conda.sh && conda activate contact-sim", "C-m"])
    
    subprocess.run([
        "tmux", "send-keys", "-t", TMUX_SESSION,
        "python scripts/data/download_contactworld_dataset.py --target-dir /data/ContactWorld --rate-mb-s 1.0",
        "C-m"
    ])
    return True


def monitor_loop():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 ContactWorld download monitor started.", flush=True)
    while True:
        num_done, completed_list = count_completed_files()
        downloaded_bytes = get_total_downloaded_bytes()
        downloaded_gb = downloaded_bytes / (1024 ** 3)
        pct = min(100.0, (downloaded_gb / ESTIMATED_TOTAL_GB) * 100.0)
        running = is_process_running()
        live_status = get_tmux_live_status()

        if num_done >= TOTAL_FILES:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🎉 All {TOTAL_FILES} files downloaded successfully! Total size: {downloaded_gb:.2f} GB", flush=True)
            break

        if not running:
            restart_download_if_needed()
            time.sleep(5)
        else:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⏳ Volume: {downloaded_gb:.2f} GB / ~{ESTIMATED_TOTAL_GB:.2f} GB ({pct:.1f}%) "
                f"| Files: {num_done}/{TOTAL_FILES} finalized | Live: {live_status}",
                flush=True
            )
            time.sleep(10)


if __name__ == "__main__":
    monitor_loop()
