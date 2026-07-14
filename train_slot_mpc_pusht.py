import os
import sys
sys.path.append('/home/jyuan/jyuan-ws/contact-sim/third_party/gym-pusht')
sys.path.append('/home/jyuan/jyuan-ws/contact-sim')
import argparse
import math
import json
import cv2
import numpy as np
import h5py
import hdf5plugin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import gym_pusht
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# Append PlaySlot to path to import official SAVi modules
sys.path.append('/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src')
from models.SAVi import SAVi

# Headless video setup
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# ----------------------------------------------------
# 1. Custom SAVi Checkpoint Loader (bypasses PyTorch 2.6 weights_only error)
# ----------------------------------------------------
def load_savi_weights(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith("module."):
            k = k[7:]
        # Slice learned initializer slots parameter from 8 down to model's num_slots (3)
        if k == "initializer.slots" and v.shape[1] > model.num_slots:
            v = v[:, :model.num_slots, :]
        model_state_dict[k] = v
    model.load_state_dict(model_state_dict)
    return model

# ----------------------------------------------------
# 2. SAVi Learning Rate Scheduler (Linear Warmup + Cosine Annealing)
# ----------------------------------------------------
class SAViScheduler:
    def __init__(self, optimizer, base_lr=1e-4, warmup_steps=4000, max_steps=100000):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear Warmup
            lr = self.base_lr * float(self.current_step) / float(self.warmup_steps)
        else:
            # Cosine Annealing
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# ----------------------------------------------------
# 3. Dynamics Learning Rate Scheduler (Cosine Annealing)
# ----------------------------------------------------
class DynamicsScheduler:
    def __init__(self, optimizer, base_lr=2e-4, max_steps=100000):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        progress = float(self.current_step) / float(max(1, self.max_steps))
        progress = min(max(progress, 0.0), 1.0)
        lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# ----------------------------------------------------
# 4. Relative Action Wrapper
# ----------------------------------------------------
class RelativeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_delta=30.0):
        super().__init__(env)
        self.max_delta = max_delta
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        delta = action * self.max_delta
        current_pos = np.array(self.env.unwrapped.agent.position)
        target_pos = current_pos + delta
        target_pos = np.clip(target_pos, 0, 512)
        return target_pos

# ----------------------------------------------------
# 5. PushT Dataset Loader (Supporting Train/Val splits)
# ----------------------------------------------------
from datasets.pusht import PushTDataset, normalize_img, denormalize_img, augment_background

# ----------------------------------------------------
# 6. cOCVP Dynamics Model (Action-Conditioned Transformer)
# ----------------------------------------------------
class cOCVPDynamics(nn.Module):
    def __init__(self, slot_dim=128, action_dim=2, num_layers=4, num_heads=8, ff_dim=1024):
        super().__init__()
        self.slot_dim = slot_dim
        self.action_proj = nn.Linear(action_dim, slot_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(slot_dim, slot_dim)

    def get_sinusoidal_pos_embedding(self, num_timesteps, d_model):
        pe = torch.zeros(num_timesteps, d_model)
        position = torch.arange(0, num_timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, slots_seq, actions_seq):
        B, T, N, D = slots_seq.shape
        act_emb = self.action_proj(actions_seq).unsqueeze(2)
        conditioned_slots = slots_seq + act_emb
        
        flat_tokens = conditioned_slots.view(B, T * N, D)
        
        pos_emb = self.get_sinusoidal_pos_embedding(T, D).to(slots_seq.device)
        pos_emb = pos_emb.unsqueeze(1).expand(-1, N, -1)
        pos_emb = pos_emb.reshape(1, T * N, D).expand(B, -1, -1)
        
        tokens_with_pos = flat_tokens + pos_emb
        
        mask = torch.ones(T * N, T * N, dtype=torch.bool, device=slots_seq.device)
        for i in range(T * N):
            for j in range(T * N):
                if (j // N) <= (i // N):
                    mask[i, j] = False
                    
        out_tokens = self.transformer(tokens_with_pos, mask=mask)
        out_slots = out_tokens.view(B, T, N, D)
        
        return self.output_head(out_slots)

    def rollout(self, seed_slots, action_seq):
        B, H, _ = action_seq.shape
        N = seed_slots.shape[2]
        slots_history = seed_slots
        
        for t in range(H):
            actions_in = action_seq[:, :t+1, :]
            pred = self.forward(slots_history, actions_in)
            next_slots = pred[:, -1, :, :].unsqueeze(1)
            slots_history = torch.cat([slots_history, next_slots], dim=1)
            
        return slots_history[:, 1:, :, :]

# ----------------------------------------------------
# 7. Periodic Validation GIF Generator
# ----------------------------------------------------
def save_validation_gif(savi_model, val_loader, device, step, save_dir):
    savi_model.eval()
    
    # Extract the first batch's first sequence (1, T, C, H, W)
    for val_images, _ in val_loader:
        images = val_images[0:1].to(device)
        break
        
    with torch.no_grad():
        model_out = savi_model(images, num_imgs=images.shape[1], decode=True)
        recons = model_out["recons_imgs"][0].cpu()  # (T, C, H, W)
        masks = model_out["masks"][0].cpu()  # (T, num_slots, 1, H, W)
        
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 128, 0],  # Orange
        [128, 0, 255]   # Purple
    ]
    
    H_w, W_w = 64, 64
    visual_frames = []
    
    # Helper to draw text with drop shadow
    def draw_text_with_shadow(img, text, pos):
        cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    
    for t in range(images.shape[1]):
        # Denormalize from ImageNet stats back to [0,1] for visualization
        orig_img = denormalize_img(images[0, t]).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        recon_img = denormalize_img(recons[t]).permute(1, 2, 0).clamp(0, 1).numpy()
        
        # Build colored slot attention overlay
        mask_t = masks[t].squeeze(1).permute(1, 2, 0).numpy()
        mask_t = mask_t / (mask_t.sum(axis=-1, keepdims=True) + 1e-8)
        
        mask_rgb = np.zeros((H_w, W_w, 3), dtype=np.float32)
        for s in range(savi_model.num_slots):
            color = np.array(colors[s], dtype=np.float32) / 255.0
            mask_rgb += mask_t[:, :, s:s+1] * color
            
        overlay_img = 0.5 * orig_img + 0.5 * mask_rgb
        
        # Horizontal stack: Original | Reconstructed | Overlay
        combined = np.hstack([orig_img, recon_img, overlay_img])
        combined_uint8 = (combined * 255.0).astype(np.uint8)
        upscaled = cv2.resize(combined_uint8, (384, 128), interpolation=cv2.INTER_NEAREST)
        
        # Draw labels at the top of each panel
        draw_text_with_shadow(upscaled, "Input", (5, 12))
        draw_text_with_shadow(upscaled, "Recon", (133, 12))
        draw_text_with_shadow(upscaled, "Segmentation", (261, 12))
        
        visual_frames.append(upscaled)
        
    vis_dir = os.path.join(save_dir, "tb_logs", "val_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    gif_path = os.path.join(vis_dir, f"val_recon_step_{step}.gif")
    
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in visual_frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
    print(f"Validation GIF saved to {gif_path}")

# ----------------------------------------------------
# 8. MPC Planner
# ----------------------------------------------------
class GradientMPC:
    def __init__(self, dynamics_model, horizon=5, lr=0.1, num_iters=20):
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.lr = lr
        self.num_iters = num_iters

    def plan(self, seed_slots, goal_slots, action_dim=2):
        B = seed_slots.shape[0]
        actions_seq = torch.randn(B, self.horizon, action_dim, requires_grad=True, device=seed_slots.device)
        optimizer = optim.Adam([actions_seq], lr=self.lr)
        
        for _ in range(self.num_iters):
            optimizer.zero_grad()
            pred_slots_seq = self.dynamics.rollout(seed_slots, actions_seq)
            pred_final_slots = pred_slots_seq[:, -1, :, :]
            loss = F.mse_loss(pred_final_slots, goal_slots)
            loss.backward()
            optimizer.step()
            
        return actions_seq.detach()

class CjepaMPC:
    def __init__(self, dynamics_model, horizon=5, lr=0.1, num_iters=15):
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.lr = lr
        self.num_iters = num_iters

    def plan(self, info_dict, hist_actions, action_dim=6):
        B = hist_actions.shape[0]
        N = hist_actions.shape[1]
        
        # Optimize future action sequence
        actions_seq = torch.zeros(B, N, self.horizon, action_dim, requires_grad=True, device=hist_actions.device)
        with torch.no_grad():
            actions_seq.copy_(torch.randn_like(actions_seq) * 0.1)
            
        optimizer = optim.Adam([actions_seq], lr=self.lr)
        
        for _ in range(self.num_iters):
            optimizer.zero_grad()
            action_candidates = torch.cat([hist_actions, actions_seq], dim=2)
            # Create shallow copy of info_dict to avoid in-place rollout side-effects
            info_dict_iter = {k: v for k, v in info_dict.items()}
            cost = self.dynamics.get_cost(info_dict_iter, action_candidates)
            loss = cost.sum()
            loss.backward()
            optimizer.step()
            
        return actions_seq.detach()

# ----------------------------------------------------
# 9. Execution Modes (Train SAVi, Train Dynamics, Evaluate)
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Slot-MPC PushT Training and Planning Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train_savi", "train_dynamics", "evaluate"],
                        help="Execution mode")
    parser.add_argument("--h5_path", type=str, default="/home/jyuan/.stable-wm/pusht_expert_train.h5",
                        help="Path to PushT HDF5 dataset")
    parser.add_argument("--save_dir", type=str, default="/home/jyuan/.stable-wm/",
                        help="Directory to save/load checkpoints")
    parser.add_argument("--log_dir", type=str, default="tb_logs",
                        help="TensorBoard log directory path")
    parser.add_argument("--savi_ckpt", type=str, default="SAVi_BlockPush.pth",
                        help="Pre-trained SAVi weights file name")
    parser.add_argument("--batch_size", type=int, default=16, help="Physical batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon for MPC")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    savi_checkpoint_path = os.path.join(args.save_dir, args.savi_ckpt)
    dynamics_checkpoint_path = os.path.join(args.save_dir, "dynamics_pusht.pt")

    # Load official SAVi configuration from PlaySlot
    savi_config_path = '/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src/configs/models/SAVi.json'
    print(f"Loading official SAVi configuration from {savi_config_path}...")
    with open(savi_config_path, 'r') as f:
        savi_config = json.load(f)
    
    # Force Learned initializer to match pre-trained checkpoint weights keys
    savi_config["initializer"] = "Learned"
    # Reduce slots to exactly 3 (background, block, pusher agent)
    savi_config["num_slots"] = 3
    
    # Initialize SAVi Model
    savi_model = SAVi(**savi_config).to(device)
    
    N_slots = savi_config["num_slots"]
    D_slot = savi_config["slot_dim"]
    D_act = 2
    C, H, W = 3, 64, 64

    # Initialize cOCVP dynamics
    dynamics = cOCVPDynamics(slot_dim=D_slot, action_dim=D_act).to(device)

    # ----------------------------------------------------
    # MODE 1: Train SAVi (Pre-training)
    # ----------------------------------------------------
    if args.mode == "train_savi":
        print("Starting Phase 1: Training SAVi on PushT dataset...")
        # Train with seq_len=8 containing shuffled individual frames to break temporal correlation
        train_dataset = PushTDataset(args.h5_path, seq_len=8, image_size=(H, W), split="train", shuffle_sequence=True, augment_bg=True)
        # Keep validation sequence sequential to visualize temporal tracking performance
        val_dataset = PushTDataset(args.h5_path, seq_len=8, image_size=(H, W), split="val", shuffle_sequence=False)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Setup TensorBoard SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, args.log_dir, 'savi'))
        
        optimizer = optim.Adam(savi_model.parameters(), lr=1e-4)
        
        # Calculate steps and configure gradient accumulation to simulate effective batch size 64
        accum_steps = max(1, 64 // args.batch_size)
        total_batches = len(train_loader)
        effective_steps = (args.epochs * total_batches) // accum_steps
        
        # Setup scheduler using warmup of 4000 effective optimization steps
        scheduler = SAViScheduler(
            optimizer=optimizer, 
            base_lr=1e-4, 
            warmup_steps=4000, 
            max_steps=effective_steps
        )
        
        savi_model.train()
        optimizer.zero_grad()
        
        steps = 0
        optim_steps = 0
        
        for epoch in range(args.epochs):
            total_train_loss = 0.0
            savi_model.train()
            for images, _ in train_loader:
                images = images.to(device)
                
                # Official SAVi forward processes (B, T, C, H, W)
                model_out = savi_model(images, num_imgs=images.shape[1], decode=True)
                recons = model_out["recons_imgs"]
                
                # Divide loss by accumulation steps
                loss = F.mse_loss(recons, images) / accum_steps
                loss.backward()
                
                total_train_loss += loss.item() * accum_steps
                steps += 1
                
                if steps % accum_steps == 0:
                    # Clip gradients to a max norm of 0.05
                    torch.nn.utils.clip_grad_norm_(savi_model.parameters(), max_norm=0.05)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update learning rate schedule
                    current_lr = scheduler.step()
                    optim_steps += 1
                    
                    # Log to TensorBoard
                    tb_writer.add_scalar("Loss/train_savi", loss.item() * accum_steps, optim_steps)
                    tb_writer.add_scalar("LR/train_savi", current_lr, optim_steps)
                    
                    # Periodic validation every 100 steps
                    if optim_steps % 100 == 0:
                        savi_model.eval()
                        total_val_loss = 0.0
                        val_steps = 0
                        with torch.no_grad():
                            for val_images, _ in val_loader:
                                val_images = val_images.to(device)
                                val_out = savi_model(val_images, num_imgs=val_images.shape[1], decode=True)
                                val_recons = val_out["recons_imgs"]
                                val_loss = F.mse_loss(val_recons, val_images)
                                
                                total_val_loss += val_loss.item()
                                val_steps += 1
                                if val_steps >= 10:  # quick validation check
                                    break
                        avg_val_loss = total_val_loss / val_steps
                        tb_writer.add_scalar("Loss/val_savi", avg_val_loss, optim_steps)
                        print(f"Step {optim_steps} | Periodic Val Loss: {avg_val_loss:.4f}")
                        
                        # Generate and save validation visualization GIF
                        save_validation_gif(
                            savi_model=savi_model, 
                            val_loader=val_loader, 
                            device=device, 
                            step=optim_steps, 
                            save_dir=args.save_dir
                        )
                        savi_model.train()
                    
                    if optim_steps % 10 == 0:
                        print(f"Epoch {epoch+1}/{args.epochs} | Step {optim_steps} | Train Loss: {loss.item() * accum_steps:.4f} | LR: {current_lr:.2e}")
            
            # Validation Step at the end of Epoch
            print("Running validation at end of epoch...")
            savi_model.eval()
            total_val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_images, _ in val_loader:
                    val_images = val_images.to(device)
                    val_out = savi_model(val_images, num_imgs=val_images.shape[1], decode=True)
                    val_recons = val_out["recons_imgs"]
                    val_loss = F.mse_loss(val_recons, val_images)
                    
                    total_val_loss += val_loss.item()
                    val_steps += 1
                    if val_steps >= 20:
                        break
                        
            avg_val_loss = total_val_loss / val_steps
            tb_writer.add_scalar("Loss/val_savi", avg_val_loss, optim_steps)
            print(f"Epoch {epoch+1} Completed. Avg Train Loss: {total_train_loss/steps:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
            
        tb_writer.close()
        torch.save({"model_state_dict": savi_model.state_dict()}, savi_checkpoint_path)
        print(f"SAVi weights successfully saved to {savi_checkpoint_path}!")

    # ----------------------------------------------------
    # MODE 2: Train cOCVP (Dynamics)
    # ----------------------------------------------------
    elif args.mode == "train_dynamics":
        print("Starting Phase 2: Training cOCVP Latent Dynamics...")
        
        # Load official pre-trained SAVi weights
        if not os.path.exists(savi_checkpoint_path):
            raise FileNotFoundError(f"Missing pre-trained SAVi weights at {savi_checkpoint_path}. Please download or train first.")
        
        print(f"Loading pre-trained SAVi weights from {savi_checkpoint_path}...")
        savi_model = load_savi_weights(
            checkpoint_path=savi_checkpoint_path, 
            model=savi_model, 
            device=device
        )
        savi_model.eval()
        
        # Split 95% training, 5% validation
        train_dataset = PushTDataset(args.h5_path, seq_len=8, image_size=(H, W), split="train")
        val_dataset = PushTDataset(args.h5_path, seq_len=8, image_size=(H, W), split="val")
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Setup TensorBoard SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, args.log_dir, 'dynamics'))
        
        optimizer = optim.Adam(dynamics.parameters(), lr=2e-4)
        
        # Gradient accumulation for Dynamics training to simulate effective batch size of 64
        accum_steps = max(1, 64 // args.batch_size)
        total_batches = len(train_loader)
        effective_steps = (args.epochs * total_batches) // accum_steps
        
        # Setup Cosine Annealing scheduler (decreases from 2e-4 base lr)
        scheduler = DynamicsScheduler(
            optimizer=optimizer, 
            base_lr=2e-4, 
            max_steps=effective_steps
        )
        
        dynamics.train()
        optimizer.zero_grad()
        
        steps = 0
        optim_steps = 0
        
        for epoch in range(args.epochs):
            total_train_loss = 0.0
            total_train_slot_loss = 0.0
            total_train_img_loss = 0.0
            dynamics.train()
            
            for images, actions in train_loader:
                B_curr, T_curr = images.shape[0], images.shape[1]
                images = images.to(device)
                actions = actions.to(device)
                
                # Extract ground-truth slots using frozen pre-trained SAVi
                with torch.no_grad():
                    model_out = savi_model(images, num_imgs=T_curr, decode=False)
                    gt_slots = model_out["slot_history"]  # (B, T, N_slots, D_slot)
                
                predicted_slots = dynamics(gt_slots, actions)
                
                # Compute Slot Alignment Loss
                slot_loss = F.mse_loss(predicted_slots[:, 1:, :, :], gt_slots[:, 1:, :, :])
                
                # Compute Future Image Prediction Loss using frozen SAVi decoder
                # Reshape predicted slots to batch-decode: (B * T, N_slots, D_slot)
                flat_pred_slots = predicted_slots.reshape(-1, N_slots, D_slot)
                with torch.no_grad():
                    flat_recon, _ = savi_model.decode(flat_pred_slots)
                # Reshape decoded frames back: (B, T, C, H, W)
                predicted_images = flat_recon.reshape(B_curr, T_curr, C, H, W)
                img_loss = F.mse_loss(predicted_images[:, 1:, :, :, :], images[:, 1:, :, :, :])
                
                # Combined objective loss (with loss weights lambda_slot = 1, lambda_img = 1)
                loss = (slot_loss + img_loss) / accum_steps
                loss.backward()
                
                total_train_loss += loss.item() * accum_steps
                total_train_slot_loss += slot_loss.item()
                total_train_img_loss += img_loss.item()
                steps += 1
                
                if steps % accum_steps == 0:
                    # Clip gradients to a max norm of 0.05
                    torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=0.05)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update learning rate schedule
                    current_lr = scheduler.step()
                    optim_steps += 1
                    
                    # Log to TensorBoard
                    tb_writer.add_scalar("Loss/train_dynamics", loss.item() * accum_steps, optim_steps)
                    tb_writer.add_scalar("Loss/train_dynamics_slot", slot_loss.item(), optim_steps)
                    tb_writer.add_scalar("Loss/train_dynamics_img", img_loss.item(), optim_steps)
                    tb_writer.add_scalar("LR/train_dynamics", current_lr, optim_steps)
                    
                    # Periodic validation every 100 steps
                    if optim_steps % 100 == 0:
                        dynamics.eval()
                        total_val_loss = 0.0
                        val_steps = 0
                        with torch.no_grad():
                            for val_images, val_actions in val_loader:
                                val_B, val_T = val_images.shape[0], val_images.shape[1]
                                val_images = val_images.to(device)
                                val_actions = val_actions.to(device)
                                
                                val_out = savi_model(val_images, num_imgs=val_T, decode=False)
                                val_gt_slots = val_out["slot_history"]
                                
                                val_pred_slots = dynamics(val_gt_slots, val_actions)
                                val_slot_loss = F.mse_loss(val_pred_slots[:, 1:, :, :], val_gt_slots[:, 1:, :, :])
                                
                                val_flat_pred = val_pred_slots.reshape(-1, N_slots, D_slot)
                                val_flat_recon, _ = savi_model.decode(val_flat_pred)
                                val_pred_imgs = val_flat_recon.reshape(val_B, val_T, C, H, W)
                                val_img_loss = F.mse_loss(val_pred_imgs[:, 1:, :, :, :], val_images[:, 1:, :, :, :])
                                
                                val_loss = val_slot_loss + val_img_loss
                                total_val_loss += val_loss.item()
                                val_steps += 1
                                if val_steps >= 10:
                                    break
                        avg_val_loss = total_val_loss / val_steps
                        tb_writer.add_scalar("Loss/val_dynamics", avg_val_loss, optim_steps)
                        print(f"Step {optim_steps} | Periodic Val Dynamics Loss: {avg_val_loss:.4f}")
                        dynamics.train()
                    
                    if optim_steps % 10 == 0:
                        print(f"Epoch {epoch+1}/{args.epochs} | Step {optim_steps} | Loss: {loss.item() * accum_steps:.4f} (Slot: {slot_loss.item():.4f}, Img: {img_loss.item():.4f}) | LR: {current_lr:.2e}")
            
            # Validation step at the end of epoch
            print("Running validation at end of epoch...")
            dynamics.eval()
            total_val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_images, val_actions in val_loader:
                    val_B, val_T = val_images.shape[0], val_images.shape[1]
                    val_images = val_images.to(device)
                    val_actions = val_actions.to(device)
                    
                    val_out = savi_model(val_images, num_imgs=val_T, decode=False)
                    val_gt_slots = val_out["slot_history"]
                    
                    val_pred_slots = dynamics(val_gt_slots, val_actions)
                    val_slot_loss = F.mse_loss(val_pred_slots[:, 1:, :, :], val_gt_slots[:, 1:, :, :])
                    
                    val_flat_pred = val_pred_slots.reshape(-1, N_slots, D_slot)
                    val_flat_recon, _ = savi_model.decode(val_flat_pred)
                    val_pred_imgs = val_flat_recon.reshape(val_B, val_T, C, H, W)
                    val_img_loss = F.mse_loss(val_pred_imgs[:, 1:, :, :, :], val_images[:, 1:, :, :, :])
                    
                    val_loss = val_slot_loss + val_img_loss
                    total_val_loss += val_loss.item()
                    val_steps += 1
                    if val_steps >= 20:
                        break
                        
            avg_val_loss = total_val_loss / val_steps
            tb_writer.add_scalar("Loss/val_dynamics", avg_val_loss, optim_steps)
            print(f"Epoch {epoch+1} Completed. Avg Train Loss: {total_train_loss/steps:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
            
        tb_writer.close()
        torch.save(dynamics.state_dict(), dynamics_checkpoint_path)
        print(f"cOCVP weights successfully saved to {dynamics_checkpoint_path}!")

    # ----------------------------------------------------
    # MODE 3: Closed-Loop Planning Evaluation in gym_pusht
    # ----------------------------------------------------
    elif args.mode == "evaluate":
        print("Starting Inference: Evaluating Slot-MPC in gym_pusht...")
        
        # Setup paths and import bridge
        import sys
        sys.path.extend([
            '/home/jyuan/jyuan-ws/contact-sim',
            '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa',
            '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src',
            '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src/third_party'
        ])
        import custom_models
        import src.world_models.dinowm_causal as dinowm_causal
        sys.modules['stable_worldmodel.wm.dinowm'] = dinowm_causal
        
        checkpoint_path = '/home/jyuan/.stable-wm/pusht_videosaur_0_epoch_30_object.ckpt'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"C-JEPA World Model checkpoint missing at {checkpoint_path}")
            
        print(f"Loading pre-trained C-JEPA World Model from {checkpoint_path}...")
        pl_module = torch.load(checkpoint_path, map_location=device, weights_only=False)
        world_model = pl_module.model.to(device).eval()
        print("World model loaded successfully!")
        
        # Load VideoSAUR model for slot attention visualization
        from src.third_party.videosaur.videosaur import configuration, models
        import seaborn as sns
        from torchvision.utils import draw_segmentation_masks
        
        print("Loading VideoSAUR model for visualizations...")
        vs_cfg_path = '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src/third_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml'
        vs_weight_path = '/home/jyuan/.stable-wm/pusht_videosaur_model.ckpt'
        vs_conf = configuration.load_config(vs_cfg_path)
        videosaur_model = models.build(vs_conf.model, vs_conf.optimizer).to(device).eval()
        vs_ckpt = torch.load(vs_weight_path, map_location=device)
        videosaur_model.load_state_dict(vs_ckpt['state_dict'])
        print("VideoSAUR model loaded successfully!")
        
        # Load expert dataset to fit standard scalers
        import h5py
        from sklearn.preprocessing import StandardScaler
        
        print("Fitting StandardScaler on dataset action and proprioception...")
        with h5py.File(args.h5_path, 'r') as f:
            actions_all = f['action'][:]
            proprio_all = f['proprio'][:]
            
        action_scaler = StandardScaler()
        action_scaler.fit(actions_all)
        
        proprio_scaler = StandardScaler()
        proprio_scaler.fit(proprio_all)
        print("Scalers fitted successfully!")
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        
        def preprocess_image(img_np):
            # img_np is HWC RGB image in [0, 255]
            img_tensor = torch.tensor(img_np.copy(), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
            # resize to 196x196
            img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(196, 196), mode='bilinear', align_corners=False).squeeze(0)
            # normalize
            img_norm = (img_resized - mean) / std
            return img_norm
            
        raw_env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        env = RelativeActionWrapper(raw_env, max_delta=30.0)
        
        planner = CjepaMPC(world_model, horizon=args.horizon, lr=0.1, num_iters=15)
        
        print("Calculating goal configurations...")
        env.reset()
        goal_pose = np.array([256.0, 256.0, 256.0, 256.0, np.pi/4])
        env.unwrapped._set_state(goal_pose)
        
        goal_img = env.render()
        goal_img_norm = preprocess_image(goal_img)
        goal_img_tensor = goal_img_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 3, 196, 196]
        
        goal_pos = np.array(env.unwrapped.agent.position)
        goal_vel = np.array(env.unwrapped.agent.velocity)
        goal_proprio = np.concatenate((goal_pos, goal_vel))
        goal_proprio_norm = torch.tensor(proprio_scaler.transform(goal_proprio[None, :])[0], dtype=torch.float32, device=device)
        goal_proprio_tensor = goal_proprio_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 4]
        
        print("Running evaluation episode...")
        obs, info = env.reset()
        done = False
        step = 0
        max_steps = 150
        cumulative_reward = 0.0
        
        # History parameters
        T_history = 5
        pixels_history = []
        proprio_history = []
        actions_history = []
        
        # Get initial state
        curr_img = env.render()
        curr_img_norm = preprocess_image(curr_img)
        
        curr_pos = np.array(env.unwrapped.agent.position)
        curr_vel = np.array(env.unwrapped.agent.velocity)
        curr_proprio = np.concatenate((curr_pos, curr_vel))
        curr_proprio_norm = torch.tensor(proprio_scaler.transform(curr_proprio[None, :])[0], dtype=torch.float32, device=device)
        
        zero_action = torch.zeros(6, dtype=torch.float32, device=device)
        
        # Pre-populate history
        for _ in range(T_history):
            pixels_history.append(curr_img_norm)
            proprio_history.append(curr_proprio_norm)
            actions_history.append(zero_action)
            
        planning_step = 0
        gif_frames = []
        
        # Set colors for masks
        palette = sns.color_palette("deep", 4)
        colors = [tuple(int(c * 255) for c in color) for color in palette]
        
        while not done and step < max_steps:
            # Build inputs
            pixels_tensor = torch.stack(pixels_history, dim=0).unsqueeze(0).unsqueeze(0)       # [1, 1, 5, 3, 196, 196]
            proprio_tensor = torch.stack(proprio_history, dim=0).unsqueeze(0).unsqueeze(0)     # [1, 1, 5, 4]
            hist_actions_tensor = torch.stack(actions_history, dim=0).unsqueeze(0).unsqueeze(0) # [1, 1, 5, 6]
            
            info_dict = {
                "pixels": pixels_tensor,
                "proprio": proprio_tensor,
                "action": hist_actions_tensor,
                "goal": goal_img_tensor,
                "goal_proprio": goal_proprio_tensor,
                "id": torch.zeros(1, T_history, dtype=torch.long, device=device),
                "step_idx": torch.full((1, T_history), planning_step, dtype=torch.long, device=device)
            }
            
            planned_actions = planner.plan(info_dict, hist_actions_tensor, action_dim=6)  # [1, 1, horizon, 6]
            action_norm_6d = planned_actions[0, 0, 0].cpu().numpy()  # [6]
            
            # Denormalize 6D action to three 2D actions
            action_raw_3steps = action_scaler.inverse_transform(action_norm_6d.reshape(3, 2))  # [3, 2]
            
            # Run VideoSAUR aux_forward on the current history to get grouping masks for visualization
            video_b = pixels_tensor.squeeze(1)  # [1, 5, 3, 196, 196]
            video_vis = video_b * std + mean
            video_vis = torch.clamp(video_vis, 0.0, 1.0).permute(0, 2, 1, 3, 4)  # [1, 3, 5, 196, 196]
            vs_inputs = {
                'video': video_b,
                'video_visualization': video_vis
            }
            with torch.no_grad():
                vs_outputs = videosaur_model(vs_inputs)
                vs_aux = videosaur_model.aux_forward(vs_inputs, vs_outputs)
                
            mask_key = "grouping_masks" if "grouping_masks" in vs_aux else "decoder_masks"
            if mask_key in vs_aux:
                masks = vs_aux[mask_key][0]  # [5, 4, 196, 196]
                curr_mask = masks[-1] > 0.5  # [4, 196, 196] bool
                frame_vis = (video_vis[0, :, -1] * 255).to(torch.uint8)
                overlay = draw_segmentation_masks(frame_vis.cpu(), curr_mask.cpu(), colors=colors, alpha=0.4)
                gif_frames.append(overlay.permute(1, 2, 0).numpy())
            
            # Execute 3-step action block
            reward_block = 0.0
            for sub_step in range(3):
                action_cmd = action_raw_3steps[sub_step]
                obs, reward, terminated, truncated, info = env.step(action_cmd)
                reward_block += reward
                step += 1
                done = terminated or truncated
                if done:
                    break
                    
            cumulative_reward += reward_block
            
            # Get observation after the block to update history
            curr_img = env.render()
            curr_img_norm = preprocess_image(curr_img)
            
            curr_pos = np.array(env.unwrapped.agent.position)
            curr_vel = np.array(env.unwrapped.agent.velocity)
            curr_proprio = np.concatenate((curr_pos, curr_vel))
            curr_proprio_norm = torch.tensor(proprio_scaler.transform(curr_proprio[None, :])[0], dtype=torch.float32, device=device)
            
            block_action_norm = torch.tensor(action_norm_6d, dtype=torch.float32, device=device)
            
            # Update history sliding window
            pixels_history.append(curr_img_norm)
            pixels_history.pop(0)
            
            proprio_history.append(curr_proprio_norm)
            proprio_history.pop(0)
            
            actions_history.append(block_action_norm)
            actions_history.pop(0)
            
            planning_step += 1
            print(f"Step {step} | Block Reward: {reward_block:.4f} | Coverage: {info.get('coverage', 0.0):.4f}")
            
        print(f"\nEpisode Finished after {step} steps.")
        print(f"Cumulative Reward: {cumulative_reward:.4f}")
        print(f"Final Goal Coverage: {info.get('coverage', 0.0)*100:.2f}%")
        print(f"Success Status: {'SUCCESS' if info.get('is_success', False) else 'FAILED'}")
        
        # Save evaluation GIF
        if gif_frames:
            gif_path = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9/cjepa_evaluate.gif'
            pil_frames = [Image.fromarray(f) for f in gif_frames]
            pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=150, loop=0)
            print(f"Evaluation visualization saved successfully to {gif_path}!")
            
        env.close()

if __name__ == '__main__':
    main()
