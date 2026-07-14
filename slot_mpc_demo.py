import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import h5py
import hdf5plugin
import cv2

# ----------------------------------------------------
# 1. Real PushT Dataset Loader
# ----------------------------------------------------
class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, seq_len=4, image_size=(64, 64)):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.image_size = image_size
        self.f = None
        
        # Read episode offsets and lengths
        with h5py.File(h5_path, 'r') as f:
            self.ep_offsets = f['ep_offset'][:]
            self.ep_lens = f['ep_len'][:]
            self.num_episodes = len(self.ep_lens)

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        # Open HDF5 file lazily per worker process
        if self.f is None:
            self.f = h5py.File(self.h5_path, 'r')
            
        offset = self.ep_offsets[idx]
        length = self.ep_lens[idx]
        
        # Randomly sample a sequence window of length seq_len from the episode
        start_idx = np.random.randint(0, length - self.seq_len + 1)
        abs_start = offset + start_idx
        
        # Load pixels and actions for the window
        raw_pixels = self.f['pixels'][abs_start : abs_start + self.seq_len]
        actions = self.f['action'][abs_start : abs_start + self.seq_len]
        
        # Process frames (resize to self.image_size and permute/normalize)
        processed_images = []
        for img in raw_pixels:
            resized = cv2.resize(img, self.image_size)
            tensor_img = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            processed_images.append(tensor_img)
            
        images_tensor = torch.stack(processed_images)  # (seq_len, C, H, W)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)  # (seq_len, D_act)
        
        return images_tensor, actions_tensor

# ----------------------------------------------------
# 2. Slot Attention Module (inspired by SAVi)
# ----------------------------------------------------
class SlotAttention(nn.Module):
    def __init__(self, num_slots, input_dim, slot_dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.scale = slot_dim ** -0.5
        self.eps = eps

        # Slot initialization parameters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        
        # Projections
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)
        
        # Update gate (GRU)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # Residual MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(),
            nn.Linear(slot_dim * 4, slot_dim)
        )
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs, num_slots=None):
        B, N_feat, _ = inputs.shape
        N_slots = num_slots if num_slots is not None else self.num_slots
        
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        
        # Sample initial slots
        mu = self.slot_mu.expand(B, N_slots, -1)
        sigma = self.slot_logsigma.exp().expand(B, N_slots, -1)
        slots = mu + sigma * torch.randn_like(sigma)
        
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=1)
            
            attn = attn + self.eps
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.matmul(attn_norm, v)
            
            slots = self.gru(
                updates.view(-1, self.slot_dim), 
                slots_prev.view(-1, self.slot_dim)
            )
            slots = slots.view(B, N_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        return slots

# ----------------------------------------------------
# 3. cOCVP Dynamics Model (Action-Conditioned Transformer)
# ----------------------------------------------------
class cOCVPDynamics(nn.Module):
    def __init__(self, slot_dim=256, action_dim=2, num_layers=4, num_heads=8, ff_dim=1024):
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
        act_emb = self.action_proj(actions_seq).unsqueeze(2)  # (B, T, 1, D)
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
# 4. Slot Decoder (Spatial Broadcast Decoder)
# ----------------------------------------------------
class SlotDecoder(nn.Module):
    def __init__(self, slot_dim=256, out_channels=3, resolution=(64, 64)):
        super().__init__()
        self.resolution = resolution
        self.slot_dim = slot_dim
        self.proj = nn.Linear(slot_dim + 2, slot_dim)
        
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(slot_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels + 1, kernel_size=3, padding=1)
        )

    def forward(self, slots):
        B, N, D = slots.shape
        H, W = self.resolution
        
        slots_broadcast = slots.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W, -1)
        
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).to(slots.device)
        grid = grid.unsqueeze(0).unsqueeze(1).expand(B, N, -1, -1, -1)
        
        slots_with_grid = torch.cat([slots_broadcast, grid], dim=-1)
        slots_with_grid = slots_with_grid.reshape(B * N, H, W, D + 2)
        
        feat = self.proj(slots_with_grid).permute(0, 3, 1, 2)
        dec_out = self.decoder_cnn(feat)
        dec_out = dec_out.reshape(B, N, -1, H, W)
        
        recons = dec_out[:, :, :-1, :, :]
        masks = dec_out[:, :, -1:, :, :]
        
        masks = F.softmax(masks, dim=1)
        recon_img = torch.sum(recons * masks, dim=1)
        return recon_img

# ----------------------------------------------------
# 5. Gradient-based Model Predictive Control (MPC) Planner
# ----------------------------------------------------
class GradientMPC:
    def __init__(self, dynamics_model, horizon=10, lr=0.1, num_iters=20):
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

# ----------------------------------------------------
# 6. Main Training and Planning Loop
# ----------------------------------------------------
def run_demo():
    print("--- Running Slot-MPC PushT Demo ---")
    
    # Configuration
    B = 2
    T = 4
    N_slots = 3
    D_slot = 256
    D_act = 2
    C, H, W = 3, 64, 64
    
    h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'
    print(f"Loading real PushT expert dataset from {h5_path}...")
    dataset = PushTDataset(h5_path, seq_len=T, image_size=(H, W))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)
    
    # Load a real batch of sequences
    images, actions = next(iter(dataloader))
    print(f"Loaded PushT batch: images shape={images.shape}, actions shape={actions.shape}")
    
    # Initialize Modules
    slot_attention = SlotAttention(num_slots=N_slots, input_dim=64, slot_dim=D_slot)
    dynamics = cOCVPDynamics(slot_dim=D_slot, action_dim=D_act)
    decoder = SlotDecoder(slot_dim=D_slot, out_channels=C, resolution=(H, W))
    
    cnn_backbone = nn.Sequential(
        nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),  # 32x32
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
        nn.ReLU(),
    )
    
    # Setup Optimizers
    savi_optimizer = optim.Adam(list(cnn_backbone.parameters()) + list(slot_attention.parameters()) + list(decoder.parameters()), lr=1e-4)
    dynamics_optimizer = optim.Adam(dynamics.parameters(), lr=2e-4)
    
    print("\n--- PHASE 1: Slot Representation Pre-training (SAVi Reconstructions) ---")
    savi_optimizer.zero_grad()
    
    # 1. Extract visual features from actual PushT frames
    flat_images = images.view(-1, C, H, W)
    features = cnn_backbone(flat_images)  # (B*T, 64, 16, 16)
    features = features.flatten(2).transpose(1, 2)  # (B*T, 256, 64)
    
    # 2. Extract slots
    slots = slot_attention(features)  # (B*T, N_slots, D_slot)
    
    # 3. Reconstruct image
    recons = decoder(slots)  # (B*T, C, H, W)
    
    # 4. Reconstruction Loss
    loss_recon = F.mse_loss(recons, flat_images)
    loss_recon.backward()
    savi_optimizer.step()
    print(f"Reconstruction Loss: {loss_recon.item():.4f}")
    
    print("\n--- PHASE 2: cOCVP Latent Dynamics Training ---")
    cnn_backbone.eval()
    slot_attention.eval()
    decoder.eval()
    
    dynamics_optimizer.zero_grad()
    
    # Extract slots from ground-truth frames to align predictions
    with torch.no_grad():
        gt_features = cnn_backbone(images.view(-1, C, H, W)).flatten(2).transpose(1, 2)
        gt_slots = slot_attention(gt_features).view(B, T, N_slots, D_slot)
        
    # Predict future slots given initial slot and actions
    predicted_slots = dynamics(gt_slots, actions)
    
    # Compute combined cOCVP loss (latent slot alignment + image prediction)
    loss_slot = F.mse_loss(predicted_slots[:, 1:, :, :], gt_slots[:, 1:, :, :])
    
    flat_predicted_slots = predicted_slots[:, 1:, :, :].reshape(-1, N_slots, D_slot)
    predicted_recons = decoder(flat_predicted_slots)
    flat_target_images = images[:, 1:, :, :, :].reshape(-1, C, H, W)
    loss_img = F.mse_loss(predicted_recons, flat_target_images)
    
    loss_dynamics = loss_slot + loss_img
    loss_dynamics.backward()
    dynamics_optimizer.step()
    print(f"Dynamics Joint Loss: {loss_dynamics.item():.4f} (Slot Alignment: {loss_slot.item():.4f}, Image Prediction: {loss_img.item():.4f})")
    
    print("\n--- INFERENCE TIME: Gradient-based MPC Planning ---")
    horizon = 5
    planner = GradientMPC(dynamics, horizon=horizon, lr=0.1, num_iters=10)
    
    # Set seed slots (from starting frame) and target goal slots (from target final frame)
    seed_slots = gt_slots[:, 0:1, :, :]  # (B, 1, N_slots, D_slot)
    goal_slots = gt_slots[:, -1, :, :]   # (B, N_slots, D_slot)
    
    planned_actions = planner.plan(seed_slots, goal_slots, action_dim=D_act)
    print(f"Planned action sequence shape: {planned_actions.shape}")
    print("Planned actions sequence (Batch 0):\n", planned_actions[0])

if __name__ == '__main__':
    run_demo()
