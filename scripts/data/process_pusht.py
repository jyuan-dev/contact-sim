"""
PushT Dataset Processing Tool: Enrich with physics ground-truth & resize resolutions.

Usage:
    # Enrich dataset:
    python scripts/data/process_pusht.py enrich --in-h5 path/to/input.h5 --out-h5 path/to/enriched.h5

    # Resize dataset:
    python scripts/data/process_pusht.py resize --in-h5 path/to/input.h5 --out-h5 path/to/resized.h5 --size 64
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="PushT Data Processing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Enrich subcommand
    enrich_parser = subparsers.add_parser("enrich", help="Enrich dataset with GT masks & contact forces")
    enrich_parser.add_argument("--in-h5", type=str, required=True, help="Input PushT HDF5 dataset")
    enrich_parser.add_argument("--out-h5", type=str, required=True, help="Output enriched HDF5 dataset")
    enrich_parser.add_argument("--num-workers", type=int, default=8, help="Multiprocessing workers")

    # Resize subcommand
    resize_parser = subparsers.add_parser("resize", help="Resize dataset image and mask resolution")
    resize_parser.add_argument("--in-h5", type=str, required=True, help="Input HDF5 dataset")
    resize_parser.add_argument("--out-h5", type=str, required=True, help="Output resized HDF5 dataset")
    resize_parser.add_argument("--size", type=int, default=64, help="Target resolution (e.g. 64 for 64x64)")

    args, unknown = parser.parse_known_args()

    if args.command == "enrich":
        print(f"[PushT Process] Running enrich pipeline: {args.in_h5} -> {args.out_h5}")
        # Execute enrich logic
    elif args.command == "resize":
        print(f"[PushT Process] Running resize pipeline ({args.size}x{args.size}): {args.in_h5} -> {args.out_h5}")
        # Execute resize logic
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
