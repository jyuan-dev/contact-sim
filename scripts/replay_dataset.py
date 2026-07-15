"""
scripts/replay_dataset.py
=========================

CLI entry-point for the dataset replay engine.

Replays episodes from an HDF5 dataset — either re-running the simulator to
derive contacts / forces / masks (--run-physics), or reading pre-computed
enriched data from an already-enriched file — and writes the results back
to a new HDF5 file with the same schema as enrich_pusht_dataset.py.

Examples
--------
Re-run physics (full pipeline):

    python scripts/replay_dataset.py \\
        --dataset pusht \\
        --input  /path/to/pusht_expert_train.h5 \\
        --output /path/to/pusht_replayed.h5 \\
        --num-workers 16 \\
        --run-physics

Read pre-computed data (no simulation):

    python scripts/replay_dataset.py \\
        --dataset pusht \\
        --input  /path/to/pusht_expert_train_enriched.h5 \\
        --output /path/to/pusht_replayed.h5

Smoke-test (2 episodes → scratch/):

    python scripts/replay_dataset.py \\
        --dataset pusht \\
        --input  /path/to/pusht_expert_train.h5 \\
        --run-physics \\
        --test
"""

import os
import sys
import argparse

# ── Ensure repo root is on sys.path ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Set SDL dummy driver before any pygame-importing code ────────────────────
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

from src.datasets.replay import (
    EpisodeData,
    PushTReplayer,
    OGBenchReplayer,
    LiberoReplayer,
)


# ════════════════════════════════════════════════════════════════════════════
# Replayer dispatch
# ════════════════════════════════════════════════════════════════════════════

_REPLAYER_MAP = {
    "pusht":   PushTReplayer,
    "ogbench": OGBenchReplayer,
    "libero":  LiberoReplayer,
}


def build_replayer(args) -> "BaseReplayer":
    cls      = _REPLAYER_MAP[args.dataset]
    episodes = args.episodes if args.episodes else None
    return cls(
        h5_path     = args.input,
        run_physics  = args.run_physics,
        num_workers  = args.num_workers,
        episodes     = episodes,
    )


# ════════════════════════════════════════════════════════════════════════════
# HDF5 output writer
# ════════════════════════════════════════════════════════════════════════════

def write_output(
    replayer,
    output_h5: str,
    input_h5: str,
    run_physics: bool,
    test_mode: bool,
) -> None:
    """Iterate episodes and write EpisodeData fields to an output HDF5 file.

    The output schema:
    - Writes raw keys (pixels, state, action, ep_len, ep_offset).
    - Writes enriched keys (contact_pos, normal_force, frictional_force, and individual object masks).
    """
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)

    is_h5 = os.path.isfile(input_h5) and (input_h5.endswith(".h5") or input_h5.endswith(".hdf5"))

    if is_h5:
        with h5py.File(input_h5, "r") as f_in:
            ep_lens_all = f_in["ep_len"][:]
            ep_offs_all = f_in["ep_offset"][:]
    else:
        # For OGBench or other non-HDF5 formats, read from replayer._ds (LanceDataset)
        if hasattr(replayer, "_ds"):
            ep_lens_all = replayer._ds.lengths
            ep_offs_all = replayer._ds.offsets
        else:
            raise ValueError("Unsupported dataset replayer for non-HDF5 format.")

    # Determine which episodes we're writing (subset in test / episodes mode)
    ep_indices = replayer.episode_indices
    if test_mode:
        ep_indices = ep_indices[:2]
        replayer.episode_indices = ep_indices

    # Total steps across the selected episodes
    total_steps = int(sum(ep_lens_all[i] for i in ep_indices))
    print(
        f"Processing {len(ep_indices)} episodes "
        f"({total_steps} total steps) → {output_h5}"
    )

    # ── Inspect first episode to get shapes and dimensions ───────────────
    print("Inspecting first episode to determine output dimensions...")
    sample_ep = replayer._process_one(ep_indices[0])
    state_dim = sample_ep.states.shape[1]
    action_dim = sample_ep.actions.shape[1]
    frame_h, frame_w = sample_ep.frames.shape[1], sample_ep.frames.shape[2]
    contact_dim = sample_ep.contact_pos.shape[1]
    force_dim = sample_ep.normal_force.shape[1]

    if sample_ep.masks:
        first_mask = next(iter(sample_ep.masks.values()))
        mask_h, mask_w = first_mask.shape[1], first_mask.shape[2]
    else:
        mask_h, mask_w = 224, 224

    # ── Open output file and write ───────────────────────────────────────
    with h5py.File(output_h5, "w") as f_out:
        # Pre-allocate raw datasets
        pixels_ds = f_out.create_dataset(
            "pixels", shape=(total_steps, frame_h, frame_w, 3), dtype=np.uint8,
            chunks=(1, frame_h, frame_w, 3), **hdf5plugin.Zstd()
        )
        state_ds = f_out.create_dataset(
            "state", shape=(total_steps, state_dim), dtype=np.float32,
            chunks=(min(1000, total_steps), state_dim), **hdf5plugin.Zstd()
        )
        action_ds = f_out.create_dataset(
            "action", shape=(total_steps, action_dim), dtype=np.float32,
            chunks=(min(1000, total_steps), action_dim), **hdf5plugin.Zstd()
        )

        # Pre-allocate enriched datasets
        contact_pos_ds = f_out.create_dataset(
            "contact_pos", shape=(total_steps, contact_dim), dtype=np.float32,
            chunks=(min(1000, total_steps), contact_dim), **hdf5plugin.Zstd()
        )
        normal_force_ds = f_out.create_dataset(
            "normal_force", shape=(total_steps, force_dim), dtype=np.float32,
            chunks=(min(1000, total_steps), force_dim), **hdf5plugin.Zstd()
        )
        frictional_force_ds = f_out.create_dataset(
            "frictional_force", shape=(total_steps, force_dim), dtype=np.float32,
            chunks=(min(1000, total_steps), force_dim), **hdf5plugin.Zstd()
        )

        # Pre-allocate masks dynamically
        mask_datasets = {}
        for mask_name in sample_ep.masks.keys():
            # In HDF5, write as e.g. "block_masks" or "cube_0_masks"
            ds_name = f"{mask_name}_masks"
            mask_datasets[mask_name] = f_out.create_dataset(
                ds_name, shape=(total_steps, mask_h, mask_w), dtype=np.uint8,
                chunks=(1, mask_h, mask_w), **hdf5plugin.Zstd()
            )

        # Recomputed lists
        new_ep_lens = []
        new_ep_offs = []

        # ── Replay and write ──────────────────────────────────────────────
        write_ptr = 0  # current position in the flat output arrays

        for ep_data in tqdm(
            replayer.iter_episodes(),
            total=len(ep_indices),
            desc="Replaying episodes",
            unit="ep",
        ):
            T = len(ep_data.frames)
            new_ep_lens.append(T)
            new_ep_offs.append(write_ptr)

            # Write raw
            pixels_ds[write_ptr : write_ptr + T] = ep_data.frames
            state_ds[write_ptr : write_ptr + T]  = ep_data.states
            action_ds[write_ptr : write_ptr + T] = ep_data.actions

            # Write enriched
            contact_pos_ds[write_ptr : write_ptr + T]      = ep_data.contact_pos
            normal_force_ds[write_ptr : write_ptr + T]     = ep_data.normal_force
            frictional_force_ds[write_ptr : write_ptr + T] = ep_data.frictional_force

            # Write masks
            for mask_name, mask_ds in mask_datasets.items():
                mask_ds[write_ptr : write_ptr + T] = ep_data.masks.get(
                    mask_name, np.zeros((T, mask_h, mask_w), np.uint8)
                )

            write_ptr += T

        # Write metadata indices
        f_out.create_dataset("ep_len", data=np.array(new_ep_lens, dtype=np.int32))
        f_out.create_dataset("ep_offset", data=np.array(new_ep_offs, dtype=np.int32))

    print(f"\nDone. Output saved to: {output_h5}")


# ════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a contact-sim dataset (re-run physics or read pre-computed data).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(_REPLAYER_MAP.keys()),
        help="Which dataset / replayer to use.",
    )
    parser.add_argument(
        "--input", type=str,
        default="/home/jyuan/.stable-wm/pusht_expert_train.h5",
        help="Path to the input dataset (HDF5 file or Lance directory).",
    )
    parser.add_argument(
        "--output", type=str,
        default="/home/jyuan/.stable-wm/pusht_replayed.h5",
        help="Path to write the output HDF5 file.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel worker processes (1 = serial).",
    )
    parser.add_argument(
        "--run-physics", action="store_true", default=False,
        help="Re-run the simulator to derive contacts/forces/masks. "
             "If omitted, reads pre-computed enriched data from --input.",
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None,
        help="Explicit episode indices to replay (default: all).",
    )
    parser.add_argument(
        "--test", action="store_true", default=False,
        help="Process only 2 episodes and write output to scratch/ for quick validation.",
    )
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input dataset path not found: {args.input}")
        sys.exit(1)

    output_h5 = args.output
    if args.test:
        scratch_dir = os.path.join(REPO_ROOT, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        output_h5 = os.path.join(scratch_dir, f"{args.dataset}_replay_test.h5")
        print(f"[TEST MODE] Output → {output_h5}")

    print(
        f"Dataset : {args.dataset}\n"
        f"Input   : {args.input}\n"
        f"Output  : {output_h5}\n"
        f"Physics : {args.run_physics}\n"
        f"Workers : {args.num_workers}\n"
        f"Episodes: {args.episodes if args.episodes else 'all'}"
    )

    replayer = build_replayer(args)
    write_output(
        replayer    = replayer,
        output_h5   = output_h5,
        input_h5    = args.input,
        run_physics  = args.run_physics,
        test_mode   = args.test,
    )


if __name__ == "__main__":
    main()
