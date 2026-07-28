"""
Inference & evaluation for the StoSAVi model trained with DETR-style bipartite matching.

Loads the final checkpoint, runs on a few validation episodes, saves slot-decomposition
images and a GIF, and reports mean Dice / IoU for matched slots.
"""

import sys
import os
import math
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torchvision.utils as vutils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.slot_attention import StoSAVi, DETRHungarianMatcher

try:
    from PIL import Image
    import imageio
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow / imageio not found - images will not be saved.")


def to_rgb_np(t):
    return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def compute_miou(pred_bin, gt_bin, eps=1e-6):
    """Hungarian-matched mean IoU. pred_bin/gt_bin: [S,H,W] / [M,H,W]."""
    pred_flat = pred_bin.flatten(1).float()
    gt_flat   = gt_bin.flatten(1).float()
    inter = torch.einsum('sh,mh->sm', pred_flat, gt_flat)
    sum_p = pred_flat.sum(-1, keepdim=True)
    sum_g = gt_flat.sum(-1, keepdim=True).T
    union = sum_p + sum_g - inter + eps
    iou_mat = (inter + eps) / union
    cost = -iou_mat.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return iou_mat[row_ind, col_ind].mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='/home/jyuan/.stable-wm/savi_mask_detr/savi_final.pt')
    parser.add_argument('--dataset', type=str, default='pusht',
                        choices=['pusht', 'ogbench', 'libero'])
    parser.add_argument('--n_episodes', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--out_dir', type=str,
                        default='/home/jyuan/.stable-wm/savi_mask_detr/infer_results')
    parser.add_argument('--num_gt', type=int, default=3, choices=[2, 3])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config_path = os.path.join(REPO_ROOT, 'configs', 'savi', f'{args.dataset}.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    model = StoSAVi(
        resolution=tuple(cfg['resolution']),
        clip_len=cfg['n_sample_frames'],
        slot_dict=cfg['slot_dict'],
        enc_dict=cfg['enc_dict'],
        dec_dict=cfg['dec_dict'],
        pred_dict=cfg['pred_dict'],
        loss_dict=cfg['loss_dict'],
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params/1e6:.2f}M")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}, global_step: {ckpt.get('global_step', '?')}")

    if args.dataset == 'pusht':
        from src.datasets.pusht import PushTMaskHDF5Dataset
        val_ds = PushTMaskHDF5Dataset(
            h5_path=cfg['h5_path'], split='val',
            resolution=tuple(cfg['resolution']),
            n_sample_frames=cfg['n_sample_frames'],
            frame_offset=cfg['frame_offset'],
            train_frac=cfg['train_frac'])
    elif args.dataset == 'ogbench':
        from src.datasets.ogbench import OGBenchCubeDataset
        val_ds = OGBenchCubeDataset(
            data_path=cfg['h5_path'], split='val',
            resolution=tuple(cfg['resolution']),
            n_sample_frames=cfg['n_sample_frames'])
    elif args.dataset == 'libero':
        from src.datasets.libero import LiberoDataset
        val_ds = LiberoDataset(
            data_path=cfg['h5_path'], split='val',
            resolution=tuple(cfg['resolution']),
            n_sample_frames=cfg['n_sample_frames'])

    print(f"Val episodes: {len(val_ds._episode_indices)}")
    matcher = DETRHungarianMatcher(cost_bce=1.0, cost_dice=1.0)

    all_ious, all_dice = [], []
    gif_frames = []
    n_episodes = min(args.n_episodes, len(val_ds._episode_indices))

    for ep_i in range(n_episodes):
        ep_idx = val_ds._episode_indices[ep_i]
        data   = val_ds.get_video(ep_idx)
        video  = data['video'].float().to(device)
        gt_m   = data['gt_masks'].float().to(device)

        T       = min(video.shape[0], cfg['n_sample_frames'])
        clip    = video[:T].unsqueeze(0)
        gt_clip = gt_m[:T, :args.num_gt]

        with torch.no_grad():
            out = model({'img': clip})

        recon       = out['post_recon_combined'][0]   # [T, 3, H, W]
        masks       = out['post_masks'][0]             # [T, S, 1, H, W]
        recon_slots = out['post_recons'][0]            # [T, S, 3, H, W]
        S = masks.shape[1]

        ep_ious, ep_dices = [], []
        row_imgs = []

        for t in range(T):
            pred_m_t = masks[t, :, 0, :, :]   # [S, H, W]
            gt_m_t   = gt_clip[t]              # [M, H, W]

            # IoU
            pred_bin = (pred_m_t > args.threshold).float()
            gt_bin   = (gt_m_t   > args.threshold).float()
            ep_ious.append(compute_miou(pred_bin, gt_bin))

            # Dice (soft)
            indices = matcher(pred_m_t.unsqueeze(0), gt_m_t.unsqueeze(0))
            src_i, tgt_i = indices[0]
            p_flat = pred_m_t[src_i].flatten(1)
            g_flat = gt_m_t[tgt_i].flatten(1)
            inter  = (p_flat * g_flat).sum(-1)
            denom  = p_flat.sum(-1) + g_flat.sum(-1)
            ep_dices.append((2 * inter / (denom + 1e-6)).mean().item())

            # Visualisation row
            orig_rgb = (clip[0, t] * 0.5 + 0.5).clamp(0, 1)
            rec_rgb  = (recon[t]   * 0.5 + 0.5).clamp(0, 1)
            row = [orig_rgb, rec_rgb]

            for s in range(S):
                slot_img = (recon_slots[t, s] * 0.5 + 0.5).clamp(0, 1)
                m_s      = masks[t, s, 0]
                row.append((slot_img * m_s + (1 - m_s)).clamp(0, 1))

            for g in range(gt_clip.shape[1]):
                m = gt_m_t[g]
                c = torch.zeros(3, m.shape[0], m.shape[1], device=device)
                c[g % 3] = m
                row.append(c)

            row_imgs.append(torch.stack(row, 0))

        all_ious.extend(ep_ious)
        all_dice.extend(ep_dices)
        print(f"  Episode {ep_i:3d} (idx={ep_idx}) | mIoU={np.mean(ep_ious):.4f}  Dice={np.mean(ep_dices):.4f}")

        if not HAS_PIL:
            continue

        n_cols = 2 + S + args.num_gt
        grid_t = torch.stack(row_imgs, 0).flatten(0, 1)
        grid   = vutils.make_grid(grid_t, nrow=n_cols, pad_value=1.0, padding=2)
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(os.path.join(args.out_dir, f'ep{ep_i:03d}.png'))
        gif_frames.append(grid_np)

    print("\n" + "="*60)
    print(f"  Overall mean IoU  : {np.mean(all_ious):.4f}")
    print(f"  Overall mean Dice : {np.mean(all_dice):.4f}")
    print(f"  Episodes evaluated: {n_episodes}")
    print("="*60)

    if HAS_PIL and gif_frames:
        gif_path = os.path.join(args.out_dir, 'overview.gif')
        imageio.mimsave(gif_path, gif_frames, fps=2)
        print(f"  GIF saved: {gif_path}")

    print(f"All images saved in: {args.out_dir}")


if __name__ == '__main__':
    main()
