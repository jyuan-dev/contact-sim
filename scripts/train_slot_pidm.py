"""
Legacy Shim: Train Slot-PIDM (Stage 2) on PushT dataset.

Delegates execution to the unified slot training entrypoint `scripts/train_slot.py`.
"""

import sys
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    train_slot_script = os.path.join(REPO_ROOT, "scripts", "train_slot.py")
    
    cmd = [sys.executable, train_slot_script, "--mode", "slot_pidm"]
    
    # Forward provided CLI arguments if config not specified
    if not any(arg.startswith("--config") for arg in sys.argv[1:]):
        cmd.extend(["--config", "configs/savi/slot_pidm_pusht.yaml"])
        
    cmd.extend(sys.argv[1:])
    print(f"[Legacy Shim train_slot_pidm.py] Launching: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
