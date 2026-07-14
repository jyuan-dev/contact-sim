"""
Visualize slot attention masks from the trained SAVi (PlaySlot) model on PushT expert frames.

Usage:
    python eval/visualize_savi.py --ckpt-path /path/to/savi_pusht.pt [--save-gif] [--output-dir ./eval_results]
"""
import sys
import os
import argparse
import json
import h5py
import hdf5plugin
import numpy as np
import torch
import cv2
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYSLOT  = os.path.join(REPO_ROOT, 'third_party', 'PlaySlot', 'src')
if PLAYSLOT not in sys.path:
    sys.path.insert(0, PLAYSLOT)

from models.SAVi import SAVi

PLAYSLOT_CFG = os.path.join(PLAYSLOT, 'configs', 'models', 'SAVi.json')
H5_DEFAULT   = '/home/jyuan/.stable-wm/pusht_expert_train.h5'

# Colors (RGB 0–255) for each slot
SLOT_COLORS = [
    (76,  114, 176),
    (221, 132,  82),
    (85,  168, 104),
]


def main():
    parser = argparse.ArgumentParser(description='Visualize SAVi slot masks on PushT frames')
    parser.add_argument('--ckpt-path', required=True, help='Path to SAVi checkpoint (.pt file)')
    parser.add_argument('--h5-path', default=H5_DEFAULT, help='Path to PushT HDF5 dataset')
    parser.add_argument('--num-slots',  type=int, default=3,  help='Number of slots (default: 3)')
    parser.add_argument('--num-frames', type=int, default=20, help='Number of frames to visualize (default: 20)')
    parser.add_argument('--output-dir', default='.', help='Directory to save outputs (default: current dir)')
    parser.add_argument('--save-gif', action='store_true', help='Save visualization GIF to output-dir')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build SAVi model
    with open(PLAYSLOT_CFG) as f:
        cfg = json.load(f)

    model = SAVi(
        num_slots               = args.num_slots,
        slot_dim                = cfg['slot_dim'],
        num_iterations          = cfg['num_iterations'],
        num_iterations_first    = cfg['num_iterations_first'],
        in_channels             = cfg['in_channels'],
        mlp_hidden              = cfg['mlp_hidden'],
        mlp_encoder_dim         = cfg['mlp_encoder_dim'],
        initializer             = cfg['initializer'],
        encoder                 = cfg['encoder'],
        decoder                 = cfg['decoder'],
        transition_module_params= cfg['transition_module_params'],
    ).to(device).eval()

    print(f'Loading SAVi checkpoint from {args.ckpt_path}...')
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    state = {}
    for k, v in ckpt['model_state_dict'].items():
        k = k.replace('module.', '')
        if k == 'initializer.slots' and v.shape[1] > args.num_slots:
            v = v[:, :args.num_slots, :]
        state[k] = v
    model.load_state_dict(state)
    print('SAVi loaded successfully.')

    # Load frames
    print(f'Loading {args.num_frames} frames from {args.h5_path}...')
    with h5py.File(args.h5_path, 'r') as f:
        raw_imgs = f['pixels'][0:args.num_frames]  # (T, H, W, 3) uint8

    video_np = np.stack([
        cv2.resize(img, (64, 64)) for img in raw_imgs
    ], axis=0).astype(np.float32) / 255.0

    video_t = torch.tensor(video_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, 3, H, W)

    # Forward pass
    print('Running SAVi forward pass...')
    with torch.no_grad():
        out = model(video_t, num_imgs=args.num_frames, decode=True)

    masks = out['masks_history'][0].squeeze(2).cpu().numpy()  # (T, num_slots, H, W)

    # Render
    print('Rendering overlay frames...')
    gif_frames = []
    for t in range(args.num_frames):
        frame_big = cv2.resize((video_np[t] * 255).astype(np.uint8), (196, 196), interpolation=cv2.INTER_NEAREST)
        hard_mask = np.argmax(masks[t], axis=0)
        hard_big  = cv2.resize(hard_mask.astype(np.uint8), (196, 196), interpolation=cv2.INTER_NEAREST)

        color_overlay = np.zeros((196, 196, 3), dtype=np.float32)
        for s, color in enumerate(SLOT_COLORS[:args.num_slots]):
            color_overlay[hard_big == s] = color

        blended = (0.55 * frame_big.astype(np.float32) + 0.45 * color_overlay).clip(0, 255).astype(np.uint8)
        gif_frames.append(blended)

    if args.save_gif:
        os.makedirs(args.output_dir, exist_ok=True)
        out_gif = os.path.join(args.output_dir, 'savi_slot_visualization.gif')
        pil_frames = [Image.fromarray(f) for f in gif_frames]
        pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:], duration=150, loop=0)
        print(f'Saved GIF to: {out_gif}')
    else:
        print('Done. Pass --save-gif and --output-dir to save visualization.')


if __name__ == '__main__':
    main()
