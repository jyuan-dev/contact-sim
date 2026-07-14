import os
import sys
import json
import cv2
import numpy as np
import h5py
import hdf5plugin
import torch
from PIL import Image

# Append PlaySlot to path to import official SAVi modules
sys.path.append('/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src')
from models.SAVi import SAVi

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load config and change num_slots to 3
    savi_config_path = '/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src/configs/models/SAVi.json'
    with open(savi_config_path, 'r') as f:
        savi_config = json.load(f)
    savi_config["initializer"] = "Learned"
    savi_config["num_slots"] = 3  # Change to 3 slots
    
    savi_model = SAVi(**savi_config).to(device)
    
    # 2. Load pre-trained weights and slice initializer slots parameter to length 3
    savi_checkpoint_path = '/home/jyuan/.stable-wm/SAVi_BlockPush.pth'
    checkpoint = torch.load(savi_checkpoint_path, map_location=device, weights_only=False)
    
    model_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith("module."):
            k = k[7:]
        # Slice initializer slots tensor
        if k == "initializer.slots":
            v = v[:, :3, :]  # Take the first 3 slots
        model_state_dict[k] = v
        
    savi_model.load_state_dict(model_state_dict)
    savi_model.eval()
    
    # 3. Load Episode 0
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    with h5py.File(h5_path, 'r') as f:
        raw_pixels = f['pixels'][0 : 109]
        
    # Preprocess
    H_w, W_w = 64, 64
    processed_images = []
    for img in raw_pixels:
        resized = cv2.resize(img, (H_w, W_w))
        tensor_img = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        processed_images.append(tensor_img)
        
    images_tensor = torch.stack(processed_images).to(device)
    
    # 4. Run model
    with torch.no_grad():
        x = images_tensor.unsqueeze(0)
        model_out = savi_model(x, num_imgs=109, decode=True)
        recons = model_out["recons_imgs"][0].cpu()
        masks = model_out["masks"][0].cpu()
        
    # 5. Generate validation GIF
    colors = [
        [255, 0, 0],    # Red (Slot 0)
        [0, 255, 0],    # Green (Slot 1)
        [0, 0, 255]     # Blue (Slot 2)
    ]
    
    visual_frames = []
    for t in range(109):
        orig_img = images_tensor[t].permute(1, 2, 0).cpu().numpy()
        recon_img = recons[t].permute(1, 2, 0).clamp(0, 1).numpy()
        
        mask_t = masks[t].squeeze(1).permute(1, 2, 0).numpy()
        mask_t = mask_t / (mask_t.sum(axis=-1, keepdims=True) + 1e-8)
        
        mask_rgb = np.zeros((H_w, W_w, 3), dtype=np.float32)
        for s in range(3):
            color = np.array(colors[s], dtype=np.float32) / 255.0
            mask_rgb += mask_t[:, :, s:s+1] * color
            
        overlay_img = 0.5 * orig_img + 0.5 * mask_rgb
        combined = np.hstack([orig_img, recon_img, overlay_img])
        combined_uint8 = (combined * 255.0).astype(np.uint8)
        upscaled = cv2.resize(combined_uint8, (384, 128), interpolation=cv2.INTER_NEAREST)
        visual_frames.append(upscaled)
        
    gif_path = '/home/jyuan/.stable-wm/test_3_slots.gif'
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in visual_frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
    print("Success: Generated test_3_slots.gif")

if __name__ == '__main__':
    main()
