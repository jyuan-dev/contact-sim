"""
infer_detr_pusht.py
====================
Dedicated inference script for the trained DETR checkpoint saved at
    /home/jyuan/.stable-wm/detr_pusht/detr_final.pt

The checkpoint was saved from a bare DETR instance whose Transformer used
PyTorch's built-in nn.TransformerDecoder (key prefix: `transformer.decoder.layers.*`).
The current Transformer class uses a manual ModuleList (`transformer.decoder_layers.*`),
so we define a LegacyTransformer here that exactly mirrors the saved architecture
and load the checkpoint directly without any key remapping.

Usage:
    python scripts/infer_detr_pusht.py [--ckpt PATH] [--dataset PATH] [--n 8] [--device cuda]

Outputs (saved to scratch/):
    detr_pred_grid.png   — grid of input images with predicted boxes overlaid
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.detr import ResNetBackbone, MLP, PositionEmbeddingSine

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH    = "/home/jyuan/.stable-wm/detr_pusht/detr_final.pt"
DATASET_PATH = "/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5"
OUT_DIR      = ROOT / "scratch"
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["block", "agent", "goal"]   # index len(CLASS_NAMES) = no-object

# BGR colours for cv2 (one per class)
COLORS_BGR = [
    (  60, 100, 231),  # block  — warm red   #e74c3c
    ( 219, 152,  52),  # agent  — blue        #3498db
    ( 113, 204,  46),  # goal   — green       #2ecc71
]


# ── Legacy Transformer (matches checkpoint architecture) ──────────────────────
class LegacyTransformer(nn.Module):
    """
    Uses PyTorch's built-in nn.TransformerDecoder so that parameter keys match
    the `transformer.decoder.layers.*` layout saved in detr_final.pt.
    """

    def __init__(self, d_model=128, nhead=4, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, src, pos_embed, query_embed):
        HW, B, C = src.shape
        tgt = torch.zeros(query_embed.shape[0], B, self.d_model, device=src.device)
        query_embed_exp = query_embed.unsqueeze(1).expand(-1, B, -1)

        memory      = self.encoder(src + pos_embed)
        mem_with_pos = memory + pos_embed

        output = tgt + query_embed_exp
        all_outputs = []
        for layer in self.decoder.layers:
            output = layer(output, mem_with_pos)
            all_outputs.append(output)

        return torch.stack(all_outputs, dim=0)   # [L, Q, B, D]


# ── DETR (same as src/models/detr.py but uses LegacyTransformer) ──────────────
class LegacyDETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries):
        super().__init__()
        self.backbone    = backbone
        self.transformer = transformer
        hidden_dim       = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj  = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, x):
        src     = self.input_proj(self.backbone(x))
        pos     = PositionEmbeddingSine(self.transformer.d_model // 2)(src)
        src_f   = src.flatten(2).permute(2, 0, 1)
        pos_f   = pos.flatten(2).permute(2, 0, 1)

        hs = self.transformer(src_f, pos_f, self.query_embed.weight)
        hs = hs.permute(2, 0, 1, 3)          # [B, L, Q, D]

        return {
            "pred_logits": self.class_embed(hs),
            "pred_boxes":  self.bbox_embed(hs).sigmoid(),
        }


# ── Build & load ──────────────────────────────────────────────────────────────
def build_legacy_detr(ckpt_path: str, device: torch.device) -> LegacyDETR:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]

    num_queries, d_model = sd["query_embed.weight"].shape
    num_classes          = sd["class_embed.weight"].shape[0] - 1
    num_enc = sum(1 for k in sd
                  if k.startswith("transformer.encoder.layers.")
                  and k.endswith(".self_attn.in_proj_weight"))
    num_dec = sum(1 for k in sd
                  if k.startswith("transformer.decoder.layers.")
                  and k.endswith(".self_attn.in_proj_weight"))
    dim_ff  = sd["transformer.encoder.layers.0.linear1.weight"].shape[0]
    nhead   = max(1, d_model // 32)

    print(f"Checkpoint config: num_queries={num_queries}, num_classes={num_classes}, "
          f"d_model={d_model}, nhead={nhead}, enc={num_enc}, dec={num_dec}, ff={dim_ff}")
    print(f"  epoch={ckpt.get('epoch','?')}, global_step={ckpt.get('global_step','?')}")

    backbone    = ResNetBackbone(train_backbone=False)
    transformer = LegacyTransformer(d_model=d_model, nhead=nhead,
                                    num_encoder_layers=num_enc,
                                    num_decoder_layers=num_dec,
                                    dim_feedforward=dim_ff)
    model = LegacyDETR(backbone, transformer,
                       num_classes=num_classes, num_queries=num_queries)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    print("✓ Checkpoint loaded (strict=True)")

    return model.eval().to(device)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_sample_frames(dataset_path: str, n: int, device: torch.device):
    """Return (imgs_tensor [N,3,64,64] float32 on device, list of raw uint8 HWC numpy).

    Supports flat layout (pusht_expert_train_64x64.h5):
        top-level datasets: pixels (N,64,64,3) uint8, ep_offset, ep_len, …
    """
    import h5py

    imgs_raw, imgs_tensor = [], []
    with h5py.File(dataset_path, "r") as f:
        total = f["pixels"].shape[0]
        rng   = np.random.default_rng(42)
        idxs  = np.sort(rng.choice(total, size=n, replace=(n > total)))
        for i in idxs:
            img = f["pixels"][int(i)]           # (64, 64, 3) uint8
            imgs_raw.append(img.copy())
            imgs_tensor.append(
                torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            )

    return torch.stack(imgs_tensor).to(device), imgs_raw


# ── cv2 drawing ───────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1


def draw_predictions(img_hwc_uint8: np.ndarray,
                     pred_logits: torch.Tensor,
                     pred_boxes: torch.Tensor,
                     score_thresh: float = 0.3) -> np.ndarray:
    """
    Draw bounding boxes on a uint8 HWC image (RGB).

    Args:
        img_hwc_uint8: (H, W, 3) uint8 RGB image
        pred_logits:   (Q, C+1) logits from the last decoder layer
        pred_boxes:    (Q, 4)   cx,cy,w,h in [0,1]
        score_thresh:  minimum confidence to draw

    Returns:
        Annotated (H, W, 3) uint8 BGR image ready for cv2.imwrite / hstack.
    """
    H, W = img_hwc_uint8.shape[:2]
    canvas = cv2.cvtColor(img_hwc_uint8, cv2.COLOR_RGB2BGR).copy()

    probs          = pred_logits.softmax(-1).cpu()           # [Q, C+1]
    scores, labels = probs[:, :-1].max(-1)

    for q in range(scores.shape[0]):
        s = scores[q].item()
        if s < score_thresh:
            continue
        cls = labels[q].item()
        cx, cy, w, h = pred_boxes[q].cpu().tolist()

        x0 = int((cx - w / 2) * W)
        y0 = int((cy - h / 2) * H)
        x1 = int((cx + w / 2) * W)
        y1 = int((cy + h / 2) * H)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W - 1, x1), min(H - 1, y1)

        color = COLORS_BGR[cls % len(COLORS_BGR)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)

        name  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        label = f"{name} {s:.2f}"
        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)

        ty = max(y0 - 4, th + 2)
        cv2.rectangle(canvas, (x0, ty - th - 2), (x0 + tw + 2, ty + bl), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x0 + 1, ty - 1),
                    FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

    return canvas


def make_legend(class_names, colors_bgr, cell_h=20, width=300) -> np.ndarray:
    """Build a small legend strip."""
    h = cell_h * len(class_names) + 4
    legend = np.zeros((h, width, 3), dtype=np.uint8)
    for i, (name, color) in enumerate(zip(class_names, colors_bgr)):
        y = i * cell_h + 2
        cv2.rectangle(legend, (4, y + 2), (20, y + cell_h - 2), color, -1)
        cv2.putText(legend, name, (26, y + cell_h - 6),
                    FONT, FONT_SCALE, (220, 220, 220), THICKNESS, cv2.LINE_AA)
    return legend


def build_grid(annotated: list, ncols: int = 4, pad: int = 4,
               pad_color=(30, 30, 30)) -> np.ndarray:
    """Tile a list of equal-sized BGR images into a grid."""
    H, W = annotated[0].shape[:2]
    n    = len(annotated)
    nrows = math.ceil(n / ncols)
    blank = np.full((H, W, 3), pad_color, dtype=np.uint8)

    rows = []
    for r in range(nrows):
        row_imgs = []
        for c in range(ncols):
            idx = r * ncols + c
            img = annotated[idx] if idx < n else blank
            if c > 0:
                row_imgs.append(np.full((H, pad, 3), pad_color, dtype=np.uint8))
            row_imgs.append(img)
        rows.append(np.concatenate(row_imgs, axis=1))
        if r < nrows - 1:
            rows.append(np.full((pad, rows[-1].shape[1], 3), pad_color, dtype=np.uint8))

    return np.concatenate(rows, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DETR PushT inference (cv2)")
    parser.add_argument("--ckpt",    default=CKPT_PATH)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--n",       type=int,   default=8)
    parser.add_argument("--thresh",  type=float, default=0.3)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.ckpt}")
    model = build_legacy_detr(args.ckpt, device)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {args.n} frames from: {args.dataset}")
    imgs_t, imgs_raw = load_sample_frames(args.dataset, args.n, device)
    print(f"Input: {imgs_t.shape}  range=[{imgs_t.min():.3f}, {imgs_t.max():.3f}]")

    # ── Forward pass ──────────────────────────────────────────────────────────
    print("Running inference …")
    with torch.no_grad():
        out = model(imgs_t)

    pred_logits = out["pred_logits"][:, -1]   # [N, Q, C+1]  last decoder layer
    pred_boxes  = out["pred_boxes"][:, -1]    # [N, Q, 4]

    probs  = pred_logits.softmax(-1)
    scores = probs[:, :, :-1].max(-1).values
    print(f"pred_logits: {pred_logits.shape}")
    print(f"pred_boxes:  {pred_boxes.shape}")
    print(f"Max scores per image: {scores.max(-1).values.cpu().numpy().round(3)}")

    # ── Annotate with cv2 ─────────────────────────────────────────────────────
    annotated = []
    for i in range(len(imgs_raw)):
        img_u8 = (imgs_raw[i] if imgs_raw[i].dtype == np.uint8
                  else (imgs_raw[i] * 255).clip(0, 255).astype(np.uint8))
        frame_num_text = f"#{i}"
        ann = draw_predictions(img_u8, pred_logits[i], pred_boxes[i], args.thresh)
        # Frame index label (top-left corner)
        cv2.putText(ann, frame_num_text, (3, 12),
                    FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        annotated.append(ann)

    grid   = build_grid(annotated, ncols=4, pad=4)
    legend = make_legend(CLASS_NAMES, COLORS_BGR, width=grid.shape[1])

    # Title bar
    title_bar = np.full((24, grid.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    title_txt = f"DETR PushT  |  detr_final.pt  |  thresh={args.thresh}"
    cv2.putText(title_bar, title_txt, (6, 16),
                FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    final = np.concatenate([title_bar, grid, legend], axis=0)

    out_path = str(OUT_DIR / "detr_pred_grid.png")
    cv2.imwrite(out_path, final)
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
