"""
Evaluation CLI entrypoint for evaluating a single StoSAVi model checkpoint on validation data.
Outputs JSON formatted metrics for shell script aggregation.
"""

import os
import sys
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.data_utils import get_dataset
from src.models.slot_attention import build_savi_model
from src.metrics.eval_metrics import (
    compute_psnr, compute_ssim, compute_fg_ari, compute_latent_std, compute_sigreg_stat
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a StoSAVi checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--name", type=str, default="Variant", help="Display name for variant")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--max_batches", type=int, default=15, help="Number of validation batches to evaluate")
    parser.add_argument("--out_json", type=str, default=None, help="Output path for JSON evaluation result")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resolution = tuple(config['resolution'])
    n_sample_frames = config['n_sample_frames']
    h5_path = config['h5_path']

    val_ds = get_dataset('pusht', h5_path, 'val', resolution, n_sample_frames, config.get('frame_offset', 1), config.get('train_frac', 0.8))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)


    model = build_savi_model(
        resolution=resolution,
        clip_len=n_sample_frames,
        slot_dict=config['slot_dict'],
        enc_dict=config['enc_dict'],
        dec_dict=config['dec_dict'],
        pred_dict=config['pred_dict'],
        loss_dict=config.get('loss_dict', None)
    ).to(device)

    if os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded weights from '{args.ckpt_path}' for {args.name}", file=sys.stderr)
    else:
        print(f"Warning: Checkpoint '{args.ckpt_path}' not found! Evaluating uninitialized model.", file=sys.stderr)

    model.eval()
    val_loss_sum, psnr_sum, ssim_sum, std_sum, sigreg_sum, ari_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n_batches, ari_batches = 0, 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break
            imgs = batch['img'].to(device)
            out_dict = model({'img': imgs})
            loss_dict = model.calc_train_loss({'img': imgs}, out_dict)
            recon_loss = loss_dict.get('post_recon_loss', torch.tensor(0.0, device=device))

            recon_img = out_dict['post_recon_combined']
            psnr = compute_psnr(recon_img, imgs)
            ssim = compute_ssim(recon_img, imgs)
            std = compute_latent_std(out_dict['post_slots'])
            sigreg = compute_sigreg_stat(out_dict['post_slots'])

            if 'gt_masks' in batch:
                ari = compute_fg_ari(out_dict['post_masks'], batch['gt_masks'])
                ari_sum += ari
                ari_batches += 1

            val_loss_sum += recon_loss.item()
            psnr_sum += psnr
            ssim_sum += ssim
            std_sum += std
            sigreg_sum += sigreg
            n_batches += 1

    res = {
        'name': args.name,
        'recon_loss': float(val_loss_sum / max(1, n_batches)),
        'psnr': float(psnr_sum / max(1, n_batches)),
        'ssim': float(ssim_sum / max(1, n_batches)),
        'fg_ari': float((ari_sum / max(1, ari_batches)) * 100),
        'latent_std': float(std_sum / max(1, n_batches)),
        'sigreg_stat': float(sigreg_sum / max(1, n_batches))
    }

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(res, f, indent=2)

    print(json.dumps(res))


if __name__ == '__main__':
    main()
