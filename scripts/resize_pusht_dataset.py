"""
Pre-resize the enriched PushT dataset to 64x64 resolution to eliminate
the massive CPU-side dataloading and resizing bottleneck.

This reads:
  - pixels (512x512) -> resizes to (64x64)
  - block_masks (224x224) -> resizes to (64x64)
  - agent_masks (224x224) -> resizes to (64x64)
  - goal_masks (224x224) -> resizes to (64x64)

Other keys are copied directly.
Uses multiprocessing to speed up the resizing on the 32-core CPU.
"""

import os
import sys
import h5py
import hdf5plugin
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool

INPUT_H5 = '/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5'
OUTPUT_H5 = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5'
RESOLUTION = (64, 64)
NUM_PROCS = 24  # Leave some headroom

def resize_chunk(args):
    """Worker function to resize a slice of the dataset."""
    start_idx, end_idx = args
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    
    with h5py.File(INPUT_H5, 'r') as f:
        # Load raw data chunks
        pixels_chunk = f['pixels'][start_idx:end_idx]
        b_masks_chunk = f['block_masks'][start_idx:end_idx]
        a_masks_chunk = f['agent_masks'][start_idx:end_idx]
        g_masks_chunk = f['goal_masks'][start_idx:end_idx]
        
    n = end_idx - start_idx
    resized_pixels = np.empty((n, 64, 64, 3), dtype=np.uint8)
    resized_b = np.empty((n, 64, 64), dtype=np.uint8)
    resized_a = np.empty((n, 64, 64), dtype=np.uint8)
    resized_g = np.empty((n, 64, 64), dtype=np.uint8)
    
    for i in range(n):
        resized_pixels[i] = cv2.resize(pixels_chunk[i], RESOLUTION, interpolation=cv2.INTER_AREA)
        resized_b[i] = cv2.resize(b_masks_chunk[i], RESOLUTION, interpolation=cv2.INTER_AREA)
        resized_a[i] = cv2.resize(a_masks_chunk[i], RESOLUTION, interpolation=cv2.INTER_AREA)
        resized_g[i] = cv2.resize(g_masks_chunk[i], RESOLUTION, interpolation=cv2.INTER_AREA)
        
    return start_idx, end_idx, resized_pixels, resized_b, resized_a, resized_g

def main():
    if not os.path.exists(INPUT_H5):
        print(f"Input file {INPUT_H5} not found!")
        sys.exit(1)
        
    print("Analyzing input file structure...")
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
    with h5py.File(INPUT_H5, 'r') as f_in:
        total_steps = f_in['pixels'].shape[0]
        print(f"Total steps to process: {total_steps}")
        
        # Prepare output file
        print(f"Creating output file: {OUTPUT_H5}...")
        with h5py.File(OUTPUT_H5, 'w') as f_out:
            # Copy all metadata and 1D datasets directly
            for key in f_in.keys():
                if key not in ['pixels', 'block_masks', 'agent_masks', 'goal_masks']:
                    f_in.copy(key, f_out)
                    print(f"Copied key: {key}")
                    
            # Pre-create datasets for resized frames/masks
            # 64x64 chunks can be larger
            chunk_size = 1000
            pixels_ds = f_out.create_dataset('pixels', shape=(total_steps, 64, 64, 3), dtype=np.uint8,
                                             chunks=(chunk_size, 64, 64, 3), **hdf5plugin.Zstd())
            b_masks_ds = f_out.create_dataset('block_masks', shape=(total_steps, 64, 64), dtype=np.uint8,
                                              chunks=(chunk_size, 64, 64), **hdf5plugin.Zstd())
            a_masks_ds = f_out.create_dataset('agent_masks', shape=(total_steps, 64, 64), dtype=np.uint8,
                                              chunks=(chunk_size, 64, 64), **hdf5plugin.Zstd())
            g_masks_ds = f_out.create_dataset('goal_masks', shape=(total_steps, 64, 64), dtype=np.uint8,
                                              chunks=(chunk_size, 64, 64), **hdf5plugin.Zstd())

    # Process in chunks using a multiprocessing pool
    chunk_len = 1000  # 1k steps per chunk to avoid I/O bottlenecks
    chunks = [(i, min(i + chunk_len, total_steps)) for i in range(0, total_steps, chunk_len)]
    
    print(f"Divided into {len(chunks)} chunks. Processing with {NUM_PROCS} workers...")
    
    # Open output file for writing during pool execution
    with h5py.File(OUTPUT_H5, 'r+') as f_out:
        with Pool(NUM_PROCS) as pool:
            # Use imap_unordered to handle write results as they complete
            for start_idx, end_idx, res_p, res_b, res_a, res_g in tqdm(
                pool.imap_unordered(resize_chunk, chunks), total=len(chunks)
            ):
                f_out['pixels'][start_idx:end_idx] = res_p
                f_out['block_masks'][start_idx:end_idx] = res_b
                f_out['agent_masks'][start_idx:end_idx] = res_a
                f_out['goal_masks'][start_idx:end_idx] = res_g
                
    print("\nDataset successfully resized to 64x64!")
    print(f"Resized file location: {OUTPUT_H5}")

if __name__ == '__main__':
    main()
