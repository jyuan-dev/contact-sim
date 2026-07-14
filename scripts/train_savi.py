"""
Train StoSAVi on PushT or other datasets with groundtruth mask supervision using DETR-style bipartite matching.
"""

import sys
import os
import argparse
import time
import math
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# Add workspace root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.slot_attention import StoSAVi, DETRHungarianMatcher, DETRMaskLoss

# ── Dynamic Dataset Loader ───────────────────────────────────────────────────
def get_dataset(dataset_name, h5_path, split, resolution, n_sample_frames, frame_offset, train_frac):
    if dataset_name == 'pusht':
        from src.datasets.pusht import PushTMaskHDF5Dataset
        return PushTMaskHDF5Dataset(
            h5_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            train_frac=train_frac
        )
    elif dataset_name == 'ogbench':
        from src.datasets.ogbench import OGBenchCubeDataset
        return OGBenchCubeDataset(
            data_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames
        )
    elif dataset_name == 'libero':
        from src.datasets.libero import LiberoDataset
        return LiberoDataset(
            data_path=h5_path,
            split=split,
            resolution=resolution,
            n_sample_frames=n_sample_frames
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


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
def visualize_slots(model, val_ds, n_samples, device, writer, epoch, cfg):
    model.eval()
    grids = []
    
    # We check if dataset is a placeholder stub
    if not hasattr(val_ds, '_episode_indices'):
        print("[visualize_slots] Skipping visualization: dataset does not support episode indexing (placeholder stub).")
        return

    for i in range(min(n_samples, len(val_ds._episode_indices))):
        ep_idx = val_ds._episode_indices[i]
        data   = val_ds.get_video(ep_idx)
        video  = data['video'].float().to(device)
        gt_m   = data['gt_masks'].float().to(device)
        T      = min(video.shape[0], cfg['n_sample_frames'])
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
    parser.add_argument('--dataset', type=str, default='pusht', choices=['pusht', 'ogbench', 'libero'],
                        help='Dataset name to load config and run training')
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

    # Load Config from YAML file
    config_path = os.path.join(REPO_ROOT, 'configs', 'savi', f'{args.dataset}.yaml')
    print(f"Loading SAVi configuration from {config_path}...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.max_epochs is not None:
        cfg['max_epochs'] = args.max_epochs

    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    os.makedirs(cfg['tb_dir'],   exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Building datasets for {args.dataset} …")
    train_ds = get_dataset(
        dataset_name=args.dataset,
        h5_path=cfg['h5_path'],
        split='train',
        resolution=tuple(cfg['resolution']),
        n_sample_frames=cfg['n_sample_frames'],
        frame_offset=cfg['frame_offset'],
        train_frac=cfg['train_frac']
    )
    val_ds = get_dataset(
        dataset_name=args.dataset,
        h5_path=cfg['h5_path'],
        split='val',
        resolution=tuple(cfg['resolution']),
        n_sample_frames=cfg['n_sample_frames'],
        frame_offset=cfg['frame_offset'],
        train_frac=cfg['train_frac']
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'],
                              pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building StoSAVi model …")
    model = StoSAVi(
        resolution=tuple(cfg['resolution']),
        clip_len=cfg['n_sample_frames'],
        slot_dict=cfg['slot_dict'],
        enc_dict=cfg['enc_dict'],
        dec_dict=cfg['dec_dict'],
        pred_dict=cfg['pred_dict'],
        loss_dict=cfg['loss_dict']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ── Matcher & Loss ────────────────────────────────────────────────────────
    matcher = DETRHungarianMatcher(cost_bce=args.weight_bce, cost_dice=args.weight_dice)
    criterion = DETRMaskLoss(weight_bce=args.weight_bce, weight_dice=args.weight_dice)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    steps_per_epoch = len(train_loader) if args.limit_train_batches is None else args.limit_train_batches
    total_steps  = cfg['max_epochs'] * steps_per_epoch
    warmup_steps = int(cfg['warmup_pct'] * total_steps)

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
    writer = SummaryWriter(log_dir=cfg['tb_dir'])

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"Starting training: {cfg['max_epochs']} epochs, {steps_per_epoch} steps/epoch")
    model.train()

    for epoch in range(start_epoch, cfg['max_epochs']):
        t0 = time.time()
        epoch_losses = {'total': 0., 'recon': 0., 'mask': 0.}
        
        for step, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and step >= args.limit_train_batches:
                break

            # Cosine LR warmup/decay
            lr = cosine_anneal_with_warmup(global_step, total_steps, warmup_steps, float(cfg['lr']), 1e-6)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            img      = batch['img'].to(device, non_blocking=True)
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)

            # Limit target masks to num_gt classes if specified (block, agent, goal)
            gt_masks = gt_masks[:, :, :args.num_gt, :, :]

            # Forward
            out = model({'img': img})
            recon = out['post_recon_combined']
            loss_recon = F.mse_loss(recon, img)

            # Extract predicted slot masks: shape [B, T, S, 1, H, W]
            pred_masks = out['post_masks']
            B, T, S, _, H, W = pred_masks.shape
            M = gt_masks.shape[2] # target masks classes

            # Bipartite matching and mask loss computed per timestep
            loss_mask_total = 0.
            bce_loss_total = 0.
            dice_loss_total = 0.

            for t in range(T):
                indices = matcher(pred_masks[:, t, :, 0, :, :], gt_masks[:, t, :, :, :])
                loss_m, l_bce, l_dice = criterion(pred_masks[:, t, :, :, :, :], gt_masks[:, t, :, :, :], indices)
                loss_mask_total += loss_m
                bce_loss_total  += l_bce
                dice_loss_total += l_dice

            loss_mask_total /= T
            bce_loss_total  /= T
            dice_loss_total /= T

            # Compute KLD loss from model's training loss function
            loss_dict_savi = model.calc_train_loss({'img': img}, out)
            loss_kld = loss_dict_savi.get('kld_loss', torch.tensor(0., device=device))

            # Total Loss
            loss = (cfg['recon_loss_w'] * loss_recon +
                    cfg['mask_loss_w'] * loss_mask_total +
                    cfg['kld_loss_w'] * loss_kld)

            # Backward & Optimize
            optimizer.zero_grad()
            loss.backward()
            if cfg['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
            optimizer.step()

            # Record
            epoch_losses['total'] += loss.item()
            epoch_losses['recon'] += loss_recon.item()
            epoch_losses['mask']  += loss_mask_total.item()

            writer.add_scalar('train/loss',       loss.item(),            global_step)
            writer.add_scalar('train/loss_recon', loss_recon.item(),      global_step)
            writer.add_scalar('train/loss_mask',  loss_mask_total.item(), global_step)
            writer.add_scalar('train/loss_bce',   bce_loss_total.item(),  global_step)
            writer.add_scalar('train/loss_dice',  dice_loss_total.item(), global_step)
            writer.add_scalar('train/loss_kld',   loss_kld.item(),        global_step)
            writer.add_scalar('train/lr',         lr,                     global_step)

            if global_step % 100 == 0:
                print(f"  Step {global_step:6d}/{total_steps} | "
                      f"loss={loss.item():.4f} recon={loss_recon.item():.4f} mask={loss_mask_total.item():.4f} "
                      f"bce={bce_loss_total.item():.4f} dice={dice_loss_total.item():.4f} "
                      f"kld={loss_kld.item():.6f} lr={lr:.2e}", flush=True)

            global_step += 1

        # ── End of epoch ──────────────────────────────────────────────────────
        n = steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{cfg['max_epochs']} | "
              f"loss={epoch_losses['total']/n:.4f}  "
              f"recon={epoch_losses['recon']/n:.4f}  "
              f"mask={epoch_losses['mask']/n:.4f}  "
              f"lr={lr:.2e}  [{elapsed:.0f}s]", flush=True)

        writer.add_scalar('epoch/loss',       epoch_losses['total'] / n, epoch)
        writer.add_scalar('epoch/loss_recon', epoch_losses['recon'] / n, epoch)
        writer.add_scalar('epoch/loss_mask',  epoch_losses['mask'] / n,  epoch)

        # Validation
        model.eval()
        val_losses = {'total': 0., 'recon': 0., 'mask': 0.}
        val_steps = len(val_loader) if args.limit_val_batches is None else args.limit_val_batches

        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_loader):
                if args.limit_val_batches is not None and val_idx >= args.limit_val_batches:
                    break

                img      = val_batch['img'].to(device, non_blocking=True)
                gt_masks = val_batch['gt_masks'].to(device, non_blocking=True)
                gt_masks = gt_masks[:, :, :args.num_gt, :, :]

                out = model({'img': img})
                recon = out['post_recon_combined']
                loss_recon = F.mse_loss(recon, img)

                pred_masks = out['post_masks']
                B, T, S, _, H, W = pred_masks.shape

                loss_mask_total = 0.
                for t in range(T):
                    indices = matcher(pred_masks[:, t, :, 0, :, :], gt_masks[:, t, :, :, :])
                    loss_m, _, _ = criterion(pred_masks[:, t, :, :, :, :], gt_masks[:, t, :, :, :], indices)
                    loss_mask_total += loss_m
                loss_mask_total /= T

                loss_dict_savi = model.calc_train_loss({'img': img}, out)
                loss_kld = loss_dict_savi.get('kld_loss', torch.tensor(0., device=device))
                loss = (cfg['recon_loss_w'] * loss_recon +
                        cfg['mask_loss_w'] * loss_mask_total +
                        cfg['kld_loss_w'] * loss_kld)

                val_losses['recon'] += loss_recon.item()
                val_losses['mask']  += loss_mask_total.item()
                val_losses['total'] += loss.item()

        vn = val_steps
        writer.add_scalar('val/loss',       val_losses['total'] / vn, epoch)
        writer.add_scalar('val/loss_recon', val_losses['recon'] / vn, epoch)
        writer.add_scalar('val/loss_mask',  val_losses['mask'] / vn,  epoch)
        print(f"          Val  | loss={val_losses['total']/vn:.4f}  "
              f"recon={val_losses['recon']/vn:.4f}  "
              f"mask={val_losses['mask']/vn:.4f}", flush=True)

        if (epoch + 1) % cfg['vis_every_n_epochs'] == 0:
            visualize_slots(model, val_ds, cfg['n_vis_samples'], device, writer, epoch, cfg)

        if (epoch + 1) % cfg['save_every_n_epochs'] == 0:
            ckpt_path = os.path.join(cfg['ckpt_dir'], f'savi_epoch_{epoch+1}.pt')
            torch.save({
                'epoch':       epoch,
                'global_step': global_step,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'config':      cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    final_path = os.path.join(cfg['ckpt_dir'], 'savi_final.pt')
    torch.save({'epoch': cfg['max_epochs'] - 1, 'global_step': global_step,
                'model': model.state_dict(), 'config': cfg}, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()


if __name__ == '__main__':
    main()
