"""
Legacy Shim: Evaluation Script for Slot-PIDM.

Delegates execution to `eval/eval_slot.py`.
"""

import sys
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    eval_slot_script = os.path.join(REPO_ROOT, "eval", "eval_slot.py")
    cmd = [sys.executable, eval_slot_script, "--model", "slot_pidm"] + sys.argv[1:]
    print(f"[Legacy Shim eval_slot_pidm.py] Launching: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
