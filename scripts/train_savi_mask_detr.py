"""
Train StoSAVi on PushT with groundtruth mask supervision using DETR-style bipartite matching.

Uses the enriched dataset (pusht_expert_train_enriched.h5) which contains:
  - pixels: RGB frames
  - block_masks, agent_masks, goal_masks: binary GT segmentation masks

Bipartite matching is computed efficiently in a vectorized manner using BCE and Dice cost.

Usage:
    python scripts/train_savi_mask_detr.py
    python scripts/train_savi_mask_detr.py --resume /path/to/checkpoint.pt
"""

import sys
import os
import argparse
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import h5py
import hdf5plugin
from scipy.optimize import linear_sum_assignment

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.slot_attention import StoSAVi

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    h5_path        = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5',
    resolution     = (64, 64),
    n_sample_frames= 6,
    frame_offset   = 1,
    train_frac     = 0.8,

    # Model (same architecture as before)
    slot_dict = dict(
        num_slots      = 4,
        slot_size      = 64,
        slot_mlp_size  = 128,
        num_iterations = 2,
        kernel_mlp     = False,
    ),
    enc_dict = dict(
        enc_channels   = (3, 16, 16, 16, 16),
        enc_ks         = 5,
        enc_out_channels = 32,
        enc_norm       = '',
    ),
    dec_dict = dict(
        dec_channels   = (64, 16, 16, 16, 16),
        dec_resolution = (8, 8),
        dec_ks         = 5,
        dec_norm       = '',
    ),
    pred_dict = dict(
        pred_type      = 'mlp',
        pred_rnn       = False,
        pred_norm_first= True,
        pred_num_layers= 2,
        pred_num_heads = 4,
        pred_ffn_dim   = 256,
        pred_sg_every  = None,
    ),
    loss_dict = dict(
        use_post_recon_loss = True,
        kld_method          = 'var-0.01',
    ),

    # Training
    max_epochs      = 8,
    batch_size      = 256,
    num_workers     = 12,
    lr              = 2e-4,
    clip_grad       = 0.05,
    warmup_pct      = 0.025,

    # Loss weights
    recon_loss_w    = 1.0,
    kld_loss_w      = 1e-4,
    mask_loss_w     = 0.5,

    # I/O
    ckpt_dir        = '/home/jyuan/.stable-wm/savi_mask_detr',
    tb_dir          = '/home/jyuan/.stable-wm/savi_mask_detr/tb_logs',
    save_every_n_epochs = 1,
    vis_every_n_epochs  = 1,
    n_vis_samples       = 4,
)


# ── Dataset & Model Components ────────────────────────────────────────────────
from datasets.pusht import PushTMaskHDF5Dataset
from models.slot_attention import DETRHungarianMatcher, DETRMaskLoss



# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_anneal_with_warmup(step, total_steps, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def to_rgb(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def visualize_slots(model, val_ds, n_samples, device, writer, epoch):
    model.eval()
    grids = []
    for i in range(min(n_samples, len(val_ds._episode_indices))):
        ep_idx = val_ds._episode_indices[i]
        data   = val_ds.get_video(ep_idx)
        video  = data['video'].float().to(device)
        gt_m   = data['gt_masks'].float().to(device)
        T      = min(video.shape[0], CFG['n_sample_frames'])
        clip   = video[:T].unsqueeze(0)

        out = model({'img': clip})
        recon      = out['post_recon_combined'][0]
        masks      = out['post_masks'][0]
        recon_slots= out['post_recons'][0]

        rows = []
        for t in range(T):
            row = [to_rgb(clip[0, t]), to_rgb(recon[t])]
            # Slot masks
            for s in range(masks.shape[1]):
                slot_img = to_rgb(recon_slots[t, s] * masks[t, s] +
                                  (1 - masks[t, s]))
                row.append(slot_img)
            # GT masks (block=red, agent=green, goal=blue)
            for g in range(gt_m.shape[1]):
                m = gt_m[t, g]
                c = torch.zeros(3, m.shape[0], m.shape[1], device=device)
                c[g % 3] = m
                row.append(c)
            rows.append(torch.stack(row, dim=0))

        grid = torch.stack(rows, dim=0).flatten(0, 1)
        n_cols = 2 + masks.shape[1] + gt_m.shape[1]
        grids.append(vutils.make_grid(grid, nrow=n_cols, pad_value=1.0))

    writer.add_image('val/slot_decomposition',
                     torch.stack(grids, dim=0).mean(0),
                     global_step=epoch)
    model.train()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--weight_bce', type=float, default=1.0,
                        help='Weight of BCE loss in matcher and loss computation')
    parser.add_argument('--weight_dice', type=float, default=1.0,
                        help='Weight of Dice loss in matcher and loss computation')
    parser.add_argument('--num_gt', type=int, default=3, choices=[2, 3],
                        help='Number of GT masks to supervise: 2 (block, agent) or 3 (block, agent, goal)')
    parser.add_argument('--limit_train_batches', type=int, default=None,
                        help='Limit train batches per epoch (for verification)')
    parser.add_argument('--limit_val_batches', type=int, default=None,
                        help='Limit validation batches per epoch (for verification)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override maximum training epochs')
    args = parser.parse_args()

    if args.max_epochs is not None:
        CFG['max_epochs'] = args.max_epochs

    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    os.makedirs(CFG['tb_dir'],   exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building datasets …")
    train_ds = PushTMaskHDF5Dataset(
        CFG['h5_path'], split='train',
        resolution=CFG['resolution'],
        n_sample_frames=CFG['n_sample_frames'],
        frame_offset=CFG['frame_offset'],
        train_frac=CFG['train_frac'],
    )
    val_ds = PushTMaskHDF5Dataset(
        CFG['h5_path'], split='val',
        resolution=CFG['resolution'],
        n_sample_frames=CFG['n_sample_frames'],
        frame_offset=CFG['frame_offset'],
        train_frac=CFG['train_frac'],
    )

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=CFG['num_workers'],
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=CFG['num_workers'],
                              pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building StoSAVi model …")
    model = StoSAVi(
        resolution = CFG['resolution'],
        clip_len   = CFG['n_sample_frames'],
        eps        = 1e-6,
        slot_dict  = CFG['slot_dict'],
        enc_dict   = CFG['enc_dict'],
        dec_dict   = CFG['dec_dict'],
        pred_dict  = CFG['pred_dict'],
        loss_dict  = CFG['loss_dict'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ── Matcher & Loss ────────────────────────────────────────────────────────
    matcher = DETRHungarianMatcher(cost_bce=args.weight_bce, cost_dice=args.weight_dice)
    mask_loss_fn = DETRMaskLoss(weight_bce=args.weight_bce, weight_dice=args.weight_dice)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    steps_per_epoch = len(train_loader) if args.limit_train_batches is None else args.limit_train_batches
    total_steps  = CFG['max_epochs'] * steps_per_epoch
    warmup_steps = int(CFG['warmup_pct'] * total_steps)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=CFG['tb_dir'])
    writer.add_text('config', str(CFG), global_step=0)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Starting training: {CFG['max_epochs']} epochs, {steps_per_epoch} steps/epoch")

    for epoch in range(start_epoch, CFG['max_epochs']):
        model.train()
        epoch_losses = {'total': 0., 'recon': 0., 'kld': 0., 'mask': 0., 'bce': 0., 'dice': 0.}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_idx >= args.limit_train_batches:
                break

            img      = batch['img'].to(device, non_blocking=True)       # (B, T, 3, H, W)
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)  # (B, T, 3, H, W)

            # LR schedule
            lr = cosine_anneal_with_warmup(global_step, total_steps,
                                           warmup_steps, CFG['lr'],
                                           CFG['lr'] / 100.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward
            out = model({'img': img})
            loss_dict = model.calc_train_loss({'img': img}, out)

            recon_loss = loss_dict.get('post_recon_loss', torch.tensor(0.))
            kld_loss   = loss_dict.get('kld_loss', torch.tensor(0.))

            # Bipartite Mask Matching & Loss
            pred_masks = out['post_masks'].flatten(0, 1)  # (B*T, S, 1, H, W)
            gt_flat    = gt_masks.flatten(0, 1)          # (B*T, 3, H, W)

            # Select the number of GT channels to match (2 or 3)
            gt_selected = gt_flat[:, :args.num_gt]

            # Matching (run on pred_masks[:, :, 0] to remove channels)
            indices = matcher(pred_masks[:, :, 0], gt_selected)

            # Loss computation
            mask_loss, bce_loss, dice_loss = mask_loss_fn(pred_masks, gt_selected, indices)

            loss = (CFG['recon_loss_w'] * recon_loss +
                    CFG['kld_loss_w']   * kld_loss +
                    CFG['mask_loss_w']  * mask_loss)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if CFG['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
            optimizer.step()

            # Accumulate
            epoch_losses['total'] += loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kld']   += kld_loss.item()
            epoch_losses['mask']  += mask_loss.item()
            epoch_losses['bce']   += bce_loss.item()
            epoch_losses['dice']  += dice_loss.item()

            writer.add_scalar('train/lr',         lr,                global_step)
            writer.add_scalar('train/loss',       loss.item(),       global_step)
            writer.add_scalar('train/recon_loss', recon_loss.item(), global_step)
            writer.add_scalar('train/kld_loss',   kld_loss.item(),   global_step)
            writer.add_scalar('train/mask_loss',  mask_loss.item(),  global_step)
            writer.add_scalar('train/bce_loss',   bce_loss.item(),   global_step)
            writer.add_scalar('train/dice_loss',  dice_loss.item(),  global_step)

            if global_step % 100 == 0:
                print(f"  Step {global_step:6d}/{total_steps:6d} | "
                      f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                      f"mask={mask_loss.item():.4f} bce={bce_loss.item():.4f} "
                      f"dice={dice_loss.item():.4f} kld={kld_loss.item():.6f} "
                      f"lr={lr:.2e}", flush=True)

            global_step += 1

        # ── End of epoch ──────────────────────────────────────────────────────
        n = steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{CFG['max_epochs']} | "
              f"loss={epoch_losses['total']/n:.4f}  "
              f"recon={epoch_losses['recon']/n:.4f}  "
              f"mask={epoch_losses['mask']/n:.4f}  "
              f"bce={epoch_losses['bce']/n:.4f}  "
              f"dice={epoch_losses['dice']/n:.4f}  "
              f"kld={epoch_losses['kld']/n:.6f}  "
              f"lr={lr:.2e}  [{elapsed:.0f}s]", flush=True)

        writer.add_scalar('epoch/loss',       epoch_losses['total'] / n, epoch)
        writer.add_scalar('epoch/recon_loss', epoch_losses['recon'] / n, epoch)
        writer.add_scalar('epoch/mask_loss',  epoch_losses['mask']  / n, epoch)
        writer.add_scalar('epoch/bce_loss',   epoch_losses['bce']   / n, epoch)
        writer.add_scalar('epoch/dice_loss',  epoch_losses['dice']  / n, epoch)
        writer.add_scalar('epoch/kld_loss',   epoch_losses['kld']   / n, epoch)

        # Validation
        model.eval()
        val_losses = {'total': 0., 'recon': 0., 'mask': 0., 'bce': 0., 'dice': 0.}
        val_steps = len(val_loader) if args.limit_val_batches is None else args.limit_val_batches

        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_loader):
                if args.limit_val_batches is not None and val_idx >= args.limit_val_batches:
                    break

                img      = val_batch['img'].to(device, non_blocking=True)
                gt_masks = val_batch['gt_masks'].to(device, non_blocking=True)
                out = model({'img': img})
                ld  = model.calc_train_loss({'img': img}, out)

                recon = ld.get('post_recon_loss', torch.tensor(0.))
                kld   = ld.get('kld_loss',        torch.tensor(0.))
                pred_masks_val = out['post_masks'].flatten(0, 1)  # (B*T, S, 1, H, W)
                gt_fl = gt_masks.flatten(0, 1)                    # (B*T, 3, H, W)
                gt_sel_val = gt_fl[:, :args.num_gt]

                indices_val = matcher(pred_masks_val[:, :, 0], gt_sel_val)
                mask_l, bce_l, dice_l = mask_loss_fn(pred_masks_val, gt_sel_val, indices_val)

                val_losses['recon'] += recon.item()
                val_losses['mask']  += mask_l.item()
                val_losses['bce']   += bce_l.item()
                val_losses['dice']  += dice_l.item()
                val_losses['total'] += (CFG['recon_loss_w'] * recon +
                                        CFG['kld_loss_w']   * kld +
                                        CFG['mask_loss_w']  * mask_l).item()

        vn = val_steps
        writer.add_scalar('val/loss',       val_losses['total'] / vn, epoch)
        writer.add_scalar('val/recon_loss', val_losses['recon'] / vn, epoch)
        writer.add_scalar('val/mask_loss',  val_losses['mask']  / vn, epoch)
        writer.add_scalar('val/bce_loss',   val_losses['bce']   / vn, epoch)
        writer.add_scalar('val/dice_loss',  val_losses['dice']  / vn, epoch)
        print(f"          Val  | loss={val_losses['total']/vn:.4f}  "
              f"recon={val_losses['recon']/vn:.4f}  "
              f"mask={val_losses['mask']/vn:.4f}  "
              f"bce={val_losses['bce']/vn:.4f}  "
              f"dice={val_losses['dice']/vn:.4f}", flush=True)

        if (epoch + 1) % CFG['vis_every_n_epochs'] == 0:
            visualize_slots(model, val_ds, CFG['n_vis_samples'], device, writer, epoch)

        if (epoch + 1) % CFG['save_every_n_epochs'] == 0:
            ckpt_path = os.path.join(CFG['ckpt_dir'], f'stosavi_mask_epoch_{epoch+1}.pt')
            torch.save({
                'epoch':       epoch,
                'global_step': global_step,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'config':      CFG,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    final_path = os.path.join(CFG['ckpt_dir'], 'stosavi_mask_final.pt')
    torch.save({'epoch': CFG['max_epochs'] - 1, 'global_step': global_step,
                'model': model.state_dict(), 'config': CFG}, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()


if __name__ == '__main__':
    main()
