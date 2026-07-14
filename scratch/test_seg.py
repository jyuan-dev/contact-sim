import sys
import os
import h5py
import hdf5plugin
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.extend([
    '/home/jyuan/jyuan-ws/contact-sim',
    '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa',
    '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src',
    '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src/third_party',
    '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src/third_party/videosaur'
])

import custom_models
from src.third_party.videosaur.videosaur import configuration, models
import torchvision.transforms as T
import seaborn as sns
from torchvision.utils import draw_segmentation_masks

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()

def main():
    print("Loading VideoSAUR model...")
    cfg_path = '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src/third_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml'
    weight_path = '/home/jyuan/.stable-wm/pusht_videosaur_model.ckpt'
    
    conf = configuration.load_config(cfg_path)
    model = models.build(conf.model, conf.optimizer).cuda().eval()
    ckpt = torch.load(weight_path, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'])
    print("Model loaded successfully!")
    
    # Load 15 frames from the dataset
    print("Loading expert sequence...")
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    with h5py.File(h5_path, 'r') as f:
        # Load the first 15 frames of the first episode
        raw_imgs = f['pixels'][0:15]  # [15, 64, 64, 3] uint8
        
    # Resize and normalize
    video_list = []
    video_vis_list = []
    for img in raw_imgs:
        img_rgb = img  # dataset pixels are RGB
        # Visual original frame resized to 196x196
        img_resized = cv2.resize(img_rgb, (196, 196))
        video_vis_list.append(torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0)
        
        # Normalized frame
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).cuda() / 255.0
        img_norm = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
        video_list.append(img_norm)
        
    video_b = torch.stack(video_list, dim=0).unsqueeze(0)  # [1, 15, 3, 196, 196]
    video_vis = torch.stack(video_vis_list, dim=1).unsqueeze(0)  # [1, 3, 15, 196, 196]
    
    inputs = {
        'video': video_b,
        'video_visualization': video_vis
    }
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(inputs)
        aux = model.aux_forward(inputs, outputs)
        
    print("Aux outputs keys:", aux.keys())
    
    # We want grouping_masks or decoder_masks
    mask_key = "grouping_masks" if "grouping_masks" in aux else "decoder_masks"
    if mask_key not in aux:
        # Try raw corrector masks
        # grouping_masks = outputs["processor"]["corrector"]["masks"]
        print("No masks found in aux. Keys in outputs['processor']: ", outputs["processor"].keys())
        return
        
    masks = aux[mask_key][0]  # [T, n_slots, H, W]
    print(f"Using {mask_key} with shape:", masks.shape)
    
    # Generate visualization frames
    n_slots = masks.shape[1]
    palette = sns.color_palette("deep", n_slots)
    colors = [tuple(int(c * 255) for c in color) for color in palette]
    
    visual_frames = []
    for t in range(masks.shape[0]):
        frame_vis = (video_vis[0, :, t] * 255).to(torch.uint8)
        mask_t = masks[t] > 0.5  # bool tensor [n_slots, H, W]
        # Overlay masks on the original frame
        overlay = draw_segmentation_masks(frame_vis.cpu(), mask_t.cpu(), colors=colors, alpha=0.4)
        # Reshape to HWC for cv2/PIL
        visual_frames.append(overlay.permute(1, 2, 0).numpy())
        
    # Save as GIF
    save_path = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9/cjepa_segmentation.gif'
    pil_frames = [Image.fromarray(f) for f in visual_frames]
    pil_frames[0].save(save_path, save_all=True, append_images=pil_frames[1:], duration=150, loop=0)
    print(f"Segmentation visualization saved successfully to {save_path}!")

if __name__ == '__main__':
    main()
