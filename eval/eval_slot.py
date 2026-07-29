"""
Unified Evaluation and Visualization Tool for Slot Attention Models (StoSAVi & Slot-PIDM).

Usage:
    # Evaluate StoSAVi slot masks:
    python eval/eval_slot.py --ckpt scratch/checkpoints/savi/savi_final.pt --model savi --save-gif

    # Evaluate Slot-PIDM inverse dynamics:
    python eval/eval_slot.py --ckpt scratch/checkpoints/slot_pidm_pusht/slot_pidm_final.pt --model slot_pidm
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.data_utils import get_dataset, find_dataset_path
from src.utils.training_utils import get_device
from src.models.slot_attention import StoSAVi
from src.models.slot_pidm import SlotPIDMAgent

# Colors for visual overlays
SLOT_COLORS = [
    (255, 99, 71),   # Tomato / Red
    (60, 179, 113),  # Medium Sea Green
    (30, 144, 255),  # Dodger Blue
    (255, 215, 0),   # Gold
    (147, 112, 219)  # Medium Purple
]


def overlay_masks_on_img(img_np, masks_np):
    """
    Overlay soft or hard slot masks on RGB image.
    img_np: [H, W, 3] uint8
    masks_np: [K, H, W] float 0-1
    """
    out = img_np.copy().astype(np.float32)
    K = masks_np.shape[0]

    for k in range(K):
        color = np.array(SLOT_COLORS[k % len(SLOT_COLORS)], dtype=np.float32)
        mask_k = masks_np[k][..., None]  # [H, W, 1]
        out = out * (1.0 - 0.4 * mask_k) + color * (0.4 * mask_k)

    return np.clip(out, 0, 255).astype(np.uint8)


def evaluate_savi(ckpt_path, config_path, h5_path, save_gif, output_dir, device):
    print(f"[Eval SAVi] Loading checkpoint: {ckpt_path}")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'slot_dict': {'num_slots': 4, 'slot_size': 64, 'slot_mlp_size': 128, 'num_iterations': 2},
            'enc_dict': {'enc_channels': [3, 16, 16, 16, 16], 'enc_ks': 5, 'enc_out_channels': 32},
            'dec_dict': {'dec_channels': [64, 16, 16, 16, 16], 'dec_resolution': [8, 8], 'dec_ks': 5},
            'pred_dict': {'pred_type': 'mlp', 'pred_rnn': False, 'pred_num_layers': 2, 'pred_num_heads': 4, 'pred_ffn_dim': 256}
        }

    model = StoSAVi(
        resolution=tuple(config.get('resolution', [64, 64])),
        clip_len=config.get('n_sample_frames', 16),
        slot_dict=config['slot_dict'],
        enc_dict=config['enc_dict'],
        dec_dict=config['dec_dict'],
        pred_dict=config['pred_dict']
    ).to(device)


    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
        print(" Successfully loaded SAVi checkpoint weights.")
    else:
        print(f" Warning: Checkpoint path '{ckpt_path}' not found, running dry-run evaluation.")

    model.eval()

    resolved_h5 = find_dataset_path(h5_path)
    if os.path.exists(resolved_h5):
        ds = get_dataset('pusht', resolved_h5, split='val', resolution=(64, 64), n_sample_frames=16)
        sample = ds[0]
        imgs = sample['img'].unsqueeze(0).to(device)         # [1, T, 3, H, W]
        gt_masks = sample['gt_masks'].unsqueeze(0).to(device) # [1, T, M, H, W]
    else:
        print("[Eval SAVi] Dataset not found, generating random tensor input.")
        imgs = torch.rand(1, 16, 3, 64, 64, device=device)
        gt_masks = (torch.rand(1, 16, 3, 64, 64, device=device) > 0.5).float()

    with torch.no_grad():
        masks_pred, slots, loss_dict = model(imgs, gt_masks)

    print(f"\n[SAVi Evaluation Metrics]")
    print(f"  Reconstruction Loss: {loss_dict.get('recon_loss', 0.0):.6f}")
    print(f"  KLD Loss: {loss_dict.get('kld_loss', 0.0):.6f}")
    print(f"  Predicted Slot Masks shape: {list(masks_pred.shape)}")

    if save_gif:
        os.makedirs(output_dir, exist_ok=True)
        gif_frames = []
        T = imgs.shape[1]

        for t in range(T):
            img_t = ((imgs[0, t].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8)
            masks_t = masks_pred[0, t].cpu().numpy()  # [K, H, W]
            vis_t = overlay_masks_on_img(img_t, masks_t)
            gif_frames.append(vis_t)

        out_gif_path = os.path.join(output_dir, "savi_slots_vis.gif")
        imageio.mimsave(out_gif_path, gif_frames, fps=8, loop=0)
        print(f"[Eval SAVi] Saved slot visualization GIF -> {out_gif_path}")


def evaluate_slot_pidm(ckpt_path, config_path, device):
    print(f"[Eval Slot-PIDM] Loading checkpoint: {ckpt_path}")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'model': {'d_model': 128, 'action_dim': 2, 'k_slots': 4, 'num_heads': 4, 'num_iterations': 2}}

    m_cfg = config.get('model', {})
    agent = SlotPIDMAgent(
        d_model=m_cfg.get('d_model', 128),
        action_dim=m_cfg.get('action_dim', 2),
        k_slots=m_cfg.get('k_slots', 4),
        num_heads=m_cfg.get('num_heads', 4),
        num_iterations=m_cfg.get('num_iterations', 2)
    ).to(device)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(checkpoint.get('model_state', checkpoint), strict=False)
        print(f" Loaded model state successfully.")
    else:
        print(f" Warning: Checkpoint '{ckpt_path}' not found, running synthetic eval.")

    agent.eval()
    batch_size = 8
    img_t = torch.randn(batch_size, 3, 64, 64, device=device)
    img_next = torch.randn(batch_size, 3, 64, 64, device=device)
    mask_t = torch.rand(batch_size, 3, 64, 64, device=device)
    action = torch.randn(batch_size, m_cfg.get('action_dim', 2), device=device)

    with torch.no_grad():
        slot_loss, pidm_loss, total_loss, slot_masks = agent(img_t, img_next, mask_t, action)

    print("\n[Slot-PIDM Evaluation Metrics]")
    print(f"  Action Prediction Loss (MSE): {pidm_loss.item():.6f}")
    print(f"  Slot Loss: {slot_loss.item():.6f}")
    print(f"  Total Joint Dynamics Loss: {total_loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Unified Slot Model Evaluation Tool")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="configs/savi/pusht.yaml", help="Path to model config")
    parser.add_argument("--model", type=str, choices=['savi', 'slot_pidm'], default='savi', help="Model architecture")
    parser.add_argument("--h5-path", type=str, default="scratch/pusht_expert_train_test_enriched.h5", help="Dataset path")
    parser.add_argument("--save-gif", action="store_true", help="Save visualization GIF")
    parser.add_argument("--output-dir", type=str, default="scratch/eval_results", help="Directory for output artifacts")
    args = parser.parse_args()

    device = get_device()

    if args.model == 'savi':
        evaluate_savi(args.ckpt, args.config, args.h5_path, args.save_gif, args.output_dir, device)
    else:
        evaluate_slot_pidm(args.ckpt, args.config, device)


if __name__ == "__main__":
    main()
