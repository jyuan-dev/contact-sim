"""
Apply the trained SAVi model to PushT expert samples and visualize slot attention masks.
"""
import sys
import os
import h5py
import hdf5plugin
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.append('/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src')

from models.SAVi import SAVi
import json

# -------------------------------------------------------
# Config
# -------------------------------------------------------
SAVI_CKPT     = '/home/jyuan/.stable-wm/savi_pusht.pt'
SAVi_CFG_PATH = '/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src/configs/models/SAVi.json'
H5_PATH       = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
OUT_GIF       = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9/savi_pusht_segmentation.gif'

DEVICE   = 'cuda'
IMG_SIZE = 64
N_FRAMES = 20
N_SLOTS  = 3

# Colors (RGB 0-255) for each slot
COLORS = [
    (76,  114, 176),
    (221, 132,  82),
    (85,  168, 104),
]

# -------------------------------------------------------
# Build SAVi model — use exact keys from SAVi.json
# -------------------------------------------------------
print(f"Loading SAVi config...")
with open(SAVi_CFG_PATH) as f:
    cfg = json.load(f)

model = SAVi(
    num_slots               = N_SLOTS,
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
).to(DEVICE).eval()

# -------------------------------------------------------
# Load weights
# -------------------------------------------------------
print(f"Loading SAVi checkpoint from {SAVI_CKPT}...")
ckpt = torch.load(SAVI_CKPT, map_location=DEVICE, weights_only=False)
state = {}
for k, v in ckpt['model_state_dict'].items():
    k = k.replace('module.', '')
    if k == 'initializer.slots' and v.shape[1] > N_SLOTS:
        v = v[:, :N_SLOTS, :]
    state[k] = v
model.load_state_dict(state)
print("SAVi loaded successfully!")

# -------------------------------------------------------
# Load PushT expert frames
# -------------------------------------------------------
print("Loading PushT frames from dataset...")
with h5py.File(H5_PATH, 'r') as f:
    raw_imgs = f['pixels'][0:N_FRAMES]   # (N_FRAMES, H, W, 3) uint8

video_np = np.stack([
    cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in raw_imgs
], axis=0).astype(np.float32) / 255.0

video_t = torch.tensor(video_np).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)  # (1, T, 3, H, W)

# -------------------------------------------------------
# Run SAVi
# -------------------------------------------------------
print("Running SAVi forward pass...")
with torch.no_grad():
    out = model(video_t, num_imgs=N_FRAMES, decode=True)

masks = out['masks_history'][0].squeeze(2).cpu().numpy()   # (T, N_slots, H, W) in [0,1]

# -------------------------------------------------------
# Render overlay GIF
# -------------------------------------------------------
print("Rendering overlay frames...")
gif_frames = []
for t in range(N_FRAMES):
    frame_big = cv2.resize((video_np[t] * 255).astype(np.uint8), (196, 196), interpolation=cv2.INTER_NEAREST)

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
