"""
Analyze slot attention center-of-mass trajectories from the trained SAVi model
and correlate them with ground-truth agent/block physical states.

Usage:
    python eval/analyze_slots.py --ckpt-path /path/to/SAVi_BlockPush.pth
"""
import os
import sys
import argparse
import json
import cv2
import numpy as np
import h5py
import hdf5plugin
import torch

# Append PlaySlot to path to import official SAVi modules
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYSLOT  = os.path.join(REPO_ROOT, 'third_party', 'PlaySlot', 'src')
if PLAYSLOT not in sys.path:
    sys.path.insert(0, PLAYSLOT)

from models.SAVi import SAVi

PLAYSLOT_CFG = os.path.join(PLAYSLOT, 'configs', 'models', 'SAVi.json')
H5_DEFAULT   = '/home/jyuan/.stable-wm/pusht_expert_train.h5'


def main():
    parser = argparse.ArgumentParser(description='Analyze SAVi slot attention trajectories')
    parser.add_argument('--ckpt-path', required=True, help='Path to SAVi checkpoint (.pth file)')
    parser.add_argument('--h5-path', default=H5_DEFAULT, help='Path to PushT HDF5 dataset')
    parser.add_argument('--num-slots', type=int, default=3, help='Number of slots (default: 3)')
    parser.add_argument('--num-frames', type=int, default=109, help='Number of frames to analyze (default: 109)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load SAVi config and model
    with open(PLAYSLOT_CFG) as f:
        savi_config = json.load(f)
    savi_config['initializer'] = 'Learned'
    savi_config['num_slots']   = args.num_slots

    model = SAVi(**savi_config).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        if k == 'initializer.slots' and v.shape[1] > args.num_slots:
            v = v[:, :args.num_slots, :]
        model_state_dict[k] = v
    model.load_state_dict(model_state_dict)
    model.eval()
    print(f'SAVi loaded with {args.num_slots} slots.')

    # Load episode 0 data
    with h5py.File(args.h5_path, 'r') as f:
        raw_pixels = f['pixels'][0:args.num_frames]
        raw_states = f['state'][0:args.num_frames]

    processed_images = []
    for img in raw_pixels:
        resized = cv2.resize(img, (64, 64))
        t = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        processed_images.append(t)
    images_tensor = torch.stack(processed_images).to(device)

    # Run model
    with torch.no_grad():
        model_out = model(images_tensor.unsqueeze(0), num_imgs=args.num_frames, decode=True)
        masks = model_out['masks'][0].cpu()  # (T, num_slots, 1, H, W)

    # Analyze center of mass
    print(f'\nAnalyzing {args.num_slots} slots tracking behavior over {args.num_frames} frames...')
    slot_coms = {s: [] for s in range(args.num_slots)}

    for t in range(args.num_frames):
        mask_t = masks[t].squeeze(1).numpy()  # (num_slots, H, W)
        mask_t = mask_t / (mask_t.sum(axis=0, keepdims=True) + 1e-8)
        y_indices, x_indices = np.indices((64, 64))
        for s in range(args.num_slots):
            m = mask_t[s]
            total_mass = m.sum()
            if total_mass > 0.01:
                com_y = (m * y_indices).sum() / total_mass
                com_x = (m * x_indices).sum() / total_mass
                slot_coms[s].append([com_x, com_y])
            else:
                slot_coms[s].append([0.0, 0.0])

    # Compare with physical states
    pusher_traj = raw_states[:, 0:2] * (64.0 / 512.0)  # [T, 2]
    block_traj  = raw_states[:, 2:4] * (64.0 / 512.0)  # [T, 2]

    for s in range(args.num_slots):
        traj = np.array(slot_coms[s])
        corr_pusher = np.corrcoef(traj[:, 0], pusher_traj[:, 0])[0, 1]
        corr_block  = np.corrcoef(traj[:, 0], block_traj[:, 0])[0, 1]
        std_x = traj[:, 0].std()
        std_y = traj[:, 1].std()
        print(f'\nSlot {s}:')
        print(f'  --> Std of Center of Mass: X={std_x:.2f}, Y={std_y:.2f}')
        print(f'  --> Correlation with Pusher Agent: {corr_pusher:.4f}')
        print(f'  --> Correlation with T-Block:      {corr_block:.4f}')


if __name__ == '__main__':
    main()
