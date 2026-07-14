import os
import sys
import json
import cv2
import numpy as np
import h5py
import hdf5plugin
import torch

# Append PlaySlot to path to import official SAVi modules
sys.path.append('/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src')
from models.SAVi import SAVi

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load config and model
    savi_config_path = '/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src/configs/models/SAVi.json'
    with open(savi_config_path, 'r') as f:
        savi_config = json.load(f)
    savi_config["initializer"] = "Learned"
    savi_config["num_slots"] = 3
    
    savi_model = SAVi(**savi_config).to(device)
    
    # Load weights
    savi_checkpoint_path = '/home/jyuan/.stable-wm/SAVi_BlockPush.pth'
    checkpoint = torch.load(savi_checkpoint_path, map_location=device, weights_only=False)
    model_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith("module."):
            k = k[7:]
        if k == "initializer.slots" and v.shape[1] > 3:
            v = v[:, :3, :]
        model_state_dict[k] = v
    savi_model.load_state_dict(model_state_dict)
    savi_model.eval()
    
    # 2. Load Episode 0 data
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    with h5py.File(h5_path, 'r') as f:
        raw_pixels = f['pixels'][0 : 109]
        # State contains agent pos [0:2] and block pos [2:5]
        raw_states = f['state'][0 : 109]
        
    H_w, W_w = 64, 64
    processed_images = []
    for img in raw_pixels:
        resized = cv2.resize(img, (H_w, W_w))
        tensor_img = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        processed_images.append(tensor_img)
    images_tensor = torch.stack(processed_images).to(device)
    
    # 3. Run model with decode=True to extract attention masks
    with torch.no_grad():
        model_out = savi_model(images_tensor.unsqueeze(0), num_imgs=109, decode=True)
        masks = model_out["masks"][0].cpu()  # (109, 3, 1, 64, 64)
        
    # 4. Analyze Center of Mass of each slot attention mask
    print("Analyzing 3 slots tracking behavior...")
    
    slot_coms = {0: [], 1: [], 2: []}
    
    for t in range(109):
        # Normalize masks per pixel
        mask_t = masks[t].squeeze(1).numpy()  # (3, 64, 64)
        mask_t = mask_t / (mask_t.sum(axis=0, keepdims=True) + 1e-8)
        
        # Calculate CoM for each slot
        y_indices, x_indices = np.indices((64, 64))
        for s in range(3):
            m = mask_t[s]
            total_mass = m.sum()
            if total_mass > 0.01:
                com_y = (m * y_indices).sum() / total_mass
                com_x = (m * x_indices).sum() / total_mass
                slot_coms[s].append([com_x, com_y])
            else:
                slot_coms[s].append([0.0, 0.0])
                
    # Compare with physical states (pusher agent is state[:, 0:2], block is state[:, 2:4])
    pusher_traj = raw_states[:, 0:2]  # Trajectory scale: [0, 512] -> scale to [0, 64]
    pusher_traj_scaled = pusher_traj * (64.0 / 512.0)
    
    block_traj = raw_states[:, 2:4]
    block_traj_scaled = block_traj * (64.0 / 512.0)
    
    for s in range(3):
        traj = np.array(slot_coms[s])
        # Compute correlation with pusher trajectory
        corr_pusher = np.corrcoef(traj[:, 0], pusher_traj_scaled[:, 0])[0, 1]
        corr_block = np.corrcoef(traj[:, 0], block_traj_scaled[:, 0])[0, 1]
        
        # Standard deviation of Com (to detect background)
        std_x = traj[:, 0].std()
        std_y = traj[:, 1].std()
        
        print(f"\nSlot {s}:")
        print(f"  --> Standard Deviation of Center of Mass: X={std_x:.2f}, Y={std_y:.2f}")
        print(f"  --> Correlation with Pusher Agent: {corr_pusher:.4f}")
        print(f"  --> Correlation with T-Block: {corr_block:.4f}")

if __name__ == '__main__':
    main()
