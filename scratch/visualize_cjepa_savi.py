"""
Apply the trained C-JEPA StoSAVi model to PushT expert samples and visualize slot attention masks.
"""
import sys
import os
import h5py
import hdf5plugin
import numpy as np
import torch
import cv2
from PIL import Image

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOTFORMER = os.path.join(REPO_ROOT, 'third_party', 'cjepa', 'src',
                          'third_party', 'slotformer')
HDF5_DS    = os.path.join(SLOTFORMER, 'base_slots')

for p in [REPO_ROOT, SLOTFORMER, HDF5_DS,
          os.path.join(SLOTFORMER, 'base_slots', 'models')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from base_slots.models.savi import StoSAVi

# -------------------------------------------------------
# Config
# -------------------------------------------------------
SAVI_CKPT = '/home/jyuan/.stable-wm/savi_cjepa_pusht/stosavi_pusht_epoch_5.pt'
H5_PATH   = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
OUT_GIF   = '/home/jyuan/.gemini/antigravity-ide/brain/7e779a15-4a5e-4e10-b75a-2cf8ce71c057/cjepa_savi_segmentation.gif'

DEVICE   = 'cuda'
IMG_SIZE = 64
N_FRAMES = 20
N_SLOTS  = 4

# Colors (RGB 0-255) for each slot
COLORS = [
    (76,  114, 176),   # blue
    (221, 132,  82),   # orange
    (85,  168, 104),   # green
    (196, 78,  82),    # red
]

# -------------------------------------------------------
# Locate and load checkpoint first to get config
# -------------------------------------------------------
if not os.path.exists(SAVI_CKPT):
    print(f"Checkpoint not found at {SAVI_CKPT}. Searching for other checkpoints...")
    # Find latest epoch checkpoint
    ckpts = [f for f in os.listdir('/home/jyuan/.stable-wm/savi_cjepa_pusht') if f.endswith('.pt')]
    if ckpts:
        ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join('/home/jyuan/.stable-wm/savi_cjepa_pusht', x)))
        SAVI_CKPT = os.path.join('/home/jyuan/.stable-wm/savi_cjepa_pusht', ckpts[-1])
        print(f"Using latest checkpoint: {SAVI_CKPT}")
    else:
        print("No checkpoints found. Exiting.")
        sys.exit(0)

print(f"Loading SAVi checkpoint from {SAVI_CKPT}...")
ckpt = torch.load(SAVI_CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']

# -------------------------------------------------------
# Build SAVi model using config from checkpoint
# -------------------------------------------------------
print(f"Building StoSAVi model with config from checkpoint...")
model = StoSAVi(
    resolution = (IMG_SIZE, IMG_SIZE),
    clip_len   = N_FRAMES,
    eps        = 1e-6,
    slot_dict  = cfg['slot_dict'],
    enc_dict   = cfg['enc_dict'],
    dec_dict   = cfg['dec_dict'],
    pred_dict  = cfg['pred_dict'],
    loss_dict  = cfg['loss_dict'],
).to(DEVICE).eval()

model.load_state_dict(ckpt['model'])
print("SAVi loaded successfully!")

# -------------------------------------------------------
# Load PushT expert frames
# -------------------------------------------------------
print("Loading PushT frames from dataset...")
with h5py.File(H5_PATH, 'r') as f:
    raw_imgs = f['pixels'][0:N_FRAMES]   # (N_FRAMES, H, W, 3) uint8

# Preprocess: resize to IMG_SIZE, normalize to [-1,1] (matching dataloader transform)
video_np = np.stack([
    cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in raw_imgs
], axis=0).astype(np.float32) / 255.0

video_t = torch.tensor(video_np).permute(0, 3, 1, 2)
# Normalize to [-1, 1]
video_t = (video_t - 0.5) / 0.5
video_t = video_t.unsqueeze(0).to(DEVICE)                   # (1, T, 3, H, W)

# -------------------------------------------------------
# Run SAVi
# -------------------------------------------------------
print("Running SAVi forward pass...")
with torch.no_grad():
    out = model({'img': video_t})

# post_masks: (1, T, num_slots, 1, H, W)
masks = out['post_masks'][0].squeeze(2).cpu().numpy()   # (T, N_slots, H, W) in [0,1]

# -------------------------------------------------------
# Render overlay GIF
# -------------------------------------------------------
print("Rendering overlay frames...")
gif_frames = []
for t in range(N_FRAMES):
    # Upscale to 196x196 for clarity
    frame_big = cv2.resize((video_np[t] * 255).astype(np.uint8), (196, 196), interpolation=cv2.INTER_NEAREST)

    # Argmax to color the most active slot
    hard_mask = np.argmax(masks[t], axis=0)   # (H, W)
    hard_big  = cv2.resize(hard_mask.astype(np.uint8), (196, 196), interpolation=cv2.INTER_NEAREST)

    color_overlay = np.zeros((196, 196, 3), dtype=np.float32)
    for s, color in enumerate(COLORS):
        color_overlay[hard_big == s] = color

    blended = (0.55 * frame_big.astype(np.float32) + 0.45 * color_overlay).clip(0, 255).astype(np.uint8)
    gif_frames.append(blended)

print(f"Saving GIF to {OUT_GIF}...")
pil_frames = [Image.fromarray(f) for f in gif_frames]
pil_frames[0].save(OUT_GIF, save_all=True, append_images=pil_frames[1:], duration=150, loop=0)
print("Done!")
