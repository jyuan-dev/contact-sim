"""
infer_savi_pusht.py
====================
Dedicated inference script for the SAVi checkpoint:
    /home/jyuan/.stable-wm/savi_mask_detr/savi_epoch_8.pt

Loads a StoSAVi model (via src.models.savi.SAVi), runs it on T=6 frame clips
from the 64×64 PushT dataset, then visualises the per-slot attention masks and
reconstructed image using cv2.

Usage:
    python scripts/infer_savi_pusht.py [--ckpt PATH] [--dataset PATH]
                                       [--n 4] [--clip-len 6] [--device cuda]

Outputs (saved to scratch/):
    savi_masks_grid.png   — grid: input | slot masks | reconstruction per clip
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import numpy as np

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.savi import SAVi

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH    = "/home/jyuan/.stable-wm/savi_mask_detr/savi_epoch_8.pt"
DATASET_PATH = "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5"
OUT_DIR      = ROOT / "scratch"
OUT_DIR.mkdir(exist_ok=True)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.38
THICKNESS  = 1

# Distinct BGR colours per slot
SLOT_COLORS_BGR = [
    ( 60, 100, 231),   # slot 0 — red
    (219, 152,  52),   # slot 1 — blue
    (113, 204,  46),   # slot 2 — green
    (  0, 200, 200),   # slot 3 — yellow
    (200,   0, 200),   # slot 4 — magenta
]


# ── Build & load ──────────────────────────────────────────────────────────────
def _infer_savi_cfg(sd: dict) -> dict:
    """Reverse-engineer SAVi hyper-params from checkpoint weight shapes."""
    # num_slots / slot_dim from init_latents: (1, K, D)
    num_slots = sd["init_latents"].shape[1]
    slot_dim  = sd["init_latents"].shape[2]

    # enc_channels: ConvTranspose2d weight shape is (in_ch, out_ch, kH, kW)
    in_ch  = sd["encoder.0.0.weight"].shape[1]   # RGB = 3
    enc_c1 = sd["encoder.0.0.weight"].shape[0]
    n_enc  = sum(1 for k in sd if k.startswith("encoder.") and k.endswith(".0.weight"))
    enc_channels = tuple([in_ch] + [enc_c1] * n_enc)

    # enc_out_channels
    enc_out = sd["encoder_out_layer.1.weight"].shape[0]

    # dec_channels: ConvTranspose2d weight is (in_ch, out_ch/groups, kH, kW)
    # dec_channels[0] must equal slot_dim (enforced by StoSAVi)
    # decoder.0.0.weight.shape = (slot_dim, dec_c1, kH, kW)
    dec_c1 = sd["decoder.0.0.weight"].shape[1]   # out-channels of first ConvTranspose2d
    # count transposed-conv blocks (have ".0.weight" sub-key)
    n_dec_blocks = sum(1 for k in sd
                       if k.startswith("decoder.") and k.endswith(".0.weight"))
    # dec_channels = [slot_dim, dec_c1, dec_c1, …]  (n_dec_blocks intermediate channels)
    dec_channels = tuple([slot_dim] + [dec_c1] * n_dec_blocks)

    # kernel_mlp: True if kernel_dist_layer has >1 linear layer
    kernel_linears = sum(1 for k in sd
                         if k.startswith("kernel_dist_layer.") and k.endswith(".weight"))
    kernel_mlp = kernel_linears > 1

    # predictor type
    has_transformer = any("transformer_encoder" in k for k in sd)
    pred_type = "transformer" if has_transformer else "mlp"
    pred_rnn  = "predictor.rnn.weight_ih_l0" in sd

    return dict(
        num_slots=num_slots,
        slot_dim=slot_dim,
        enc_channels=enc_channels,
        enc_out_channels=enc_out,
        dec_channels=dec_channels,
        kernel_mlp=kernel_mlp,
        pred_type=pred_type,
        pred_rnn=pred_rnn,
    )


def build_savi(ckpt_path: str, device: torch.device) -> SAVi:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"] if "model" in ckpt else ckpt.get("model_state")
    cfg  = ckpt.get("config", {})

    # Broad hyper-params from stored config
    model_cfg  = cfg.get("model", {})
    clip_len   = model_cfg.get("n_sample_frames", model_cfg.get("clip_len", 6))
    resolution = tuple(model_cfg.get("resolution", [64, 64]))

    # Fine-grained architecture inferred from weight shapes
    arch = _infer_savi_cfg(sd)
    num_slots = arch["num_slots"]
    slot_dim  = arch["slot_dim"]

    print(f"SAVi arch (inferred): {arch}")
    print(f"  clip_len={clip_len}, resolution={resolution}")
    print(f"  epoch={ckpt.get('epoch','?')}, global_step={ckpt.get('global_step','?')}")
    print(f"  state dict: {len(sd)} tensors")

    model = SAVi(
        resolution=resolution,
        clip_len=clip_len,
        num_slots=num_slots,
        slot_dim=slot_dim,
        slot_dict=dict(
            num_slots=num_slots,
            slot_size=slot_dim,
            slot_mlp_size=slot_dim * 2,
            num_iterations=3,
            kernel_mlp=arch["kernel_mlp"],
        ),
        enc_dict=dict(
            enc_channels=arch["enc_channels"],
            enc_ks=5,
            enc_out_channels=arch["enc_out_channels"],
            enc_norm='',
        ),
        dec_dict=dict(
            dec_channels=arch["dec_channels"],
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm='',
        ),
        pred_dict=dict(
            pred_type=arch["pred_type"],
            pred_rnn=arch["pred_rnn"],
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=256,
            pred_sg_every=None,
        ),
        loss_dict=dict(
            use_post_recon_loss=True,
            kld_method='none',
        ),
    )

    missing, unexpected = model.model.load_state_dict(sd, strict=True)
    assert not missing,    f"Missing: {missing}"
    assert not unexpected, f"Unexpected: {unexpected}"
    print("✓ Checkpoint loaded (strict=True)")

    return model.eval().to(device)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_clips(dataset_path: str, n_clips: int, clip_len: int,
               device: torch.device):
    """
    Return clips tensor [n_clips, clip_len, 3, 64, 64] float32.

    Reads consecutive clip_len frames starting from a random episode offset
    so the clips represent genuine temporal sequences.
    """
    import h5py

    clips = []
    with h5py.File(dataset_path, "r") as f:
        pixels    = f["pixels"]          # (N, 64, 64, 3) uint8
        ep_offset = f["ep_offset"][:]    # (E,) int64
        ep_len    = f["ep_len"][:]       # (E,) int32
        total_ep  = len(ep_offset)

        rng       = np.random.default_rng(42)
        # Pick episodes that are long enough
        valid_eps = np.where(ep_len >= clip_len)[0]
        chosen    = rng.choice(valid_eps, size=n_clips, replace=(n_clips > len(valid_eps)))

        for ep_idx in chosen:
            start  = ep_offset[ep_idx]
            length = ep_len[ep_idx]
            t0     = rng.integers(0, length - clip_len + 1)
            frames = pixels[int(start + t0): int(start + t0 + clip_len)]  # (T, 64, 64, 3)
            clip   = torch.from_numpy(frames.astype(np.float32) / 255.0)
            clip   = clip.permute(0, 3, 1, 2)   # (T, 3, 64, 64)
            clips.append(clip)

    return torch.stack(clips).to(device)   # (n_clips, T, 3, 64, 64)


# ── cv2 visualisation ─────────────────────────────────────────────────────────
UPSCALE = 3   # upscale 64px → 192px for readability


def to_uint8(tensor_chw: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] → HWC uint8 BGR."""
    img = tensor_chw.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)


def mask_to_overlay(img_bgr: np.ndarray, mask_hw: np.ndarray,
                    color_bgr: tuple, alpha: float = 0.55) -> np.ndarray:
    """Blend a soft mask (HW, float [0,1]) onto a BGR image."""
    m = cv2.resize(mask_hw, (img_bgr.shape[1], img_bgr.shape[0]),
                   interpolation=cv2.INTER_LINEAR)
    overlay = np.zeros_like(img_bgr)
    overlay[:] = color_bgr
    weight = (m * alpha)[..., None]
    return (img_bgr * (1 - weight) + overlay * weight).astype(np.uint8)


def label_img(img: np.ndarray, text: str, color=(220, 220, 220)) -> np.ndarray:
    """Stamp a label on the top-left corner of an image (in-place copy)."""
    out = img.copy()
    cv2.putText(out, text, (3, 13), FONT, FONT_SCALE, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (3, 13), FONT, FONT_SCALE, color,     1, cv2.LINE_AA)
    return out


def build_clip_row(input_clip, masks, recon, clip_idx: int,
                   num_slots: int, t_show: int) -> np.ndarray:
    """
    Build one row of visualisation for a single clip at a chosen timestep.

    Layout: [input | slot-0 overlay | slot-1 overlay | ... | recon]
    """
    # input frame
    inp_bgr  = to_uint8(input_clip[t_show])
    inp_bgr  = label_img(inp_bgr, f"C{clip_idx} t={t_show}")

    cells = [inp_bgr]

    # per-slot masks  — masks: (T, K, H, W) or (T, K, 1, H, W)
    m = masks[t_show]                   # (K, H, W) or (K, 1, H, W)
    if m.dim() == 4:
        m = m.squeeze(1)                # (K, H, W)
    m_np = m.cpu().numpy()              # (K, H, W)

    # build base grayscale image for mask overlay
    base = to_uint8(input_clip[t_show])
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for k in range(num_slots):
        color = SLOT_COLORS_BGR[k % len(SLOT_COLORS_BGR)]
        cell  = mask_to_overlay(gray_bgr, m_np[k], color)
        cell  = label_img(cell, f"S{k}", color)
        cells.append(cell)

    # reconstruction
    if recon is not None:
        rec_bgr = to_uint8(recon[t_show])
        rec_bgr = label_img(rec_bgr, "recon")
        cells.append(rec_bgr)

    # add thin vertical separator
    sep = np.full((cells[0].shape[0], 2, 3), 40, dtype=np.uint8)
    row = []
    for i, c in enumerate(cells):
        if i > 0:
            row.append(sep)
        row.append(c)
    return np.concatenate(row, axis=1)


def build_grid(rows: list, pad: int = 4) -> np.ndarray:
    h_sep = np.full((pad, rows[0].shape[1], 3), 25, dtype=np.uint8)
    out   = []
    for i, r in enumerate(rows):
        if i > 0:
            out.append(h_sep)
        out.append(r)
    return np.concatenate(out, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAVi PushT inference (cv2)")
    parser.add_argument("--ckpt",      default=CKPT_PATH)
    parser.add_argument("--dataset",   default=DATASET_PATH)
    parser.add_argument("--n",         type=int,   default=4,    help="Number of clips")
    parser.add_argument("--clip-len",  type=int,   default=6,    help="Frames per clip")
    parser.add_argument("--t-show",    type=int,   default=-1,   help="Timestep to display (-1=last)")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device   = torch.device(args.device)
    clip_len = args.clip_len
    t_show   = args.t_show if args.t_show >= 0 else clip_len - 1
    print(f"Device: {device}  |  clip_len={clip_len}  |  t_show={t_show}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.ckpt}")
    model = build_savi(args.ckpt, device)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {args.n} clips (len={clip_len}) from: {args.dataset}")
    clips = load_clips(args.dataset, args.n, clip_len, device)
    print(f"Clips tensor: {clips.shape}  range=[{clips.min():.3f}, {clips.max():.3f}]")

    # ── Forward pass ──────────────────────────────────────────────────────────
    print("Running SAVi forward pass …")
    with torch.no_grad():
        raw_out = model({'img': clips})

    # Extract masks and reconstruction — use explicit None checks (tensors don't support 'or')
    def _first(*keys):
        for k in keys:
            v = raw_out.get(k)
            if v is not None:
                return v
        return None

    masks = _first("post_masks", "masks", "prior_masks")
    recon = _first("post_recon_combined", "recon_combined")

    print(f"Output keys: {list(raw_out.keys())}")
    if masks is not None:
        print(f"  masks:  {tuple(masks.shape)}")
    if recon is not None:
        print(f"  recon:  {tuple(recon.shape)}")

    # Squeeze singleton dim if present: (B, T, K, 1, H, W) → (B, T, K, H, W)
    if masks is not None and masks.dim() == 6 and masks.shape[3] == 1:
        masks = masks.squeeze(3)

    num_slots = masks.shape[2] if masks is not None else 4

    # ── Visualise ─────────────────────────────────────────────────────────────
    rows = []
    for ci in range(args.n):
        clip_masks = masks[ci] if masks is not None else None   # (T, K, H, W)
        clip_recon = recon[ci] if recon is not None else None   # (T, 3, H, W)
        row = build_clip_row(clips[ci], clip_masks, clip_recon,
                             ci, num_slots, t_show)
        rows.append(row)

    grid = build_grid(rows)

    # Title bar
    title_bar = np.full((22, grid.shape[1], 3), (18, 18, 18), dtype=np.uint8)
    title_txt = (f"SAVi PushT  |  savi_epoch_8.pt  |  "
                 f"slots={num_slots}  t_show={t_show}/{clip_len-1}")
    cv2.putText(title_bar, title_txt, (6, 15), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    final = np.concatenate([title_bar, grid], axis=0)

    out_path = str(OUT_DIR / "savi_masks_grid.png")
    cv2.imwrite(out_path, final)
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
