"""
Train an authentic DETR model on PushT data for object detection.

Predicts coordinates [cx, cy, w, h] and labels for:
  - Class 0: block
  - Class 1: agent
  - Class 2: goal
  - Class 3: background (no-object)

Optimized via Hungarian bipartite matching using L1 box loss, GIoU loss, and Cross Entropy classification.
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
import cv2
from scipy.optimize import linear_sum_assignment

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOTFORMER = os.path.join(REPO_ROOT, 'third_party', 'cjepa', 'src',
                          'third_party', 'slotformer')
HDF5_DS    = os.path.join(SLOTFORMER, 'base_slots')

for p in [REPO_ROOT, SLOTFORMER, HDF5_DS]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    h5_path        = '/home/jyuan/.stable-wm/pusht_expert_train_64x64.h5',
    resolution     = (64, 64),
    n_sample_frames= 6,
    frame_offset   = 1,
    train_frac     = 0.8,

    # Model parameters
    num_queries    = 5,      # 3 target objects + 2 background slots
    num_classes    = 3,      # 0: block, 1: agent, 2: goal
    d_model        = 128,    # Keep it lightweight for 64x64 inputs
    nhead          = 4,
    num_encoder_layers = 3,
    num_decoder_layers = 3,
    dim_feedforward = 512,

    # Training
    max_epochs      = 15,
    batch_size      = 256,
    num_workers     = 8,
    lr              = 2e-4,
    clip_grad       = 0.1,
    warmup_pct      = 0.05,

    # Loss weights
    weight_class   = 1.0,
    weight_bbox    = 5.0,
    weight_giou    = 2.0,
    eos_coef       = 0.1,    # Scale class loss for background queries

    # I/O
    ckpt_dir        = '/home/jyuan/.stable-wm/detr_pusht',
    tb_dir          = '/home/jyuan/.stable-wm/detr_pusht/tb_logs',
    save_every_n_epochs = 2,
    vis_every_n_epochs  = 1,
    n_vis_samples       = 4,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PushTMaskHDF5Dataset(Dataset):
    MASK_KEYS = ['block_masks', 'agent_masks', 'goal_masks']

    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        resolution=(64, 64),
        n_sample_frames: int = 6,
        frame_offset: int = 1,
        train_frac: float = 0.9,
        seed: int = 42,
    ):
        assert split in ('train', 'val')
        self.h5_path = h5_path
        self.split = split
        self.resolution = resolution
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset

        with h5py.File(h5_path, 'r') as f:
            ep_lens = np.array(f['ep_len'])
            ep_offs = np.array(f['ep_offset'])

        self._ep_lens = ep_lens.tolist()
        self._ep_offs = ep_offs.tolist()
        n_episodes = len(ep_lens)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * train_frac)

        if split == 'train':
            self._episode_indices = sorted(idx[:n_train].tolist())
        else:
            self._episode_indices = sorted(idx[n_train:].tolist())

        self._index = self._build_index()
        print(f"[PushTMaskHDF5Dataset] {split}: {len(self._episode_indices)} episodes, "
              f"{len(self._index)} clips")

    def _build_index(self):
        clip_len = (self.n_sample_frames - 1) * self.frame_offset + 1
        index = []
        for ep in self._episode_indices:
            ep_len = self._ep_lens[ep]
            for start in range(0, ep_len - clip_len + 1):
                index.append((ep, start))
        return index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        episode_idx, start_frame = self._index[idx]
        frame_idxs = [start_frame + t * self.frame_offset
                      for t in range(self.n_sample_frames)]

        offset = int(self._ep_offs[episode_idx])
        abs_idxs = [offset + i for i in frame_idxs]

        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, 'r') as f:
            frames = f['pixels'][abs_idxs]
            masks  = {k: f[k][abs_idxs] for k in self.MASK_KEYS}

        video = (frames.astype(np.float32) / 127.5) - 1.0
        img = torch.from_numpy(video.transpose(0, 3, 1, 2))

        gt_masks = np.stack([masks[k] for k in self.MASK_KEYS], axis=1)
        gt_masks = torch.from_numpy(gt_masks).float() / 255.0

        return {
            'data_idx': idx,
            'img': img,
            'gt_masks': gt_masks,
        }

    def get_video(self, episode_idx):
        offset = int(self._ep_offs[episode_idx])
        ep_len = self._ep_lens[episode_idx]

        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
        with h5py.File(self.h5_path, 'r') as f:
            frames = f['pixels'][offset:offset + ep_len]
            masks  = {k: f[k][offset:offset + ep_len] for k in self.MASK_KEYS}

        video = (frames.astype(np.float32) / 127.5) - 1.0
        video = torch.from_numpy(video.transpose(0, 3, 1, 2))

        gt_masks = np.stack([masks[k] for k in self.MASK_KEYS], axis=1)
        gt_masks = torch.from_numpy(gt_masks).float() / 255.0

        return {'video': video, 'gt_masks': gt_masks, 'data_idx': episode_idx}


# ── Model Components ──────────────────────────────────────────────────────────

class ResNetBackbone(torch.nn.Module):
    def __init__(self, name='resnet18', train_backbone=True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        # ResNet18 down to layer3 (output channels: 256).
        # Input 64x64 -> conv1 (32x32) -> maxpool (16x16) -> layer1 (16x16) -> layer2 (8x8) -> layer3 (4x4)
        self.body = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.num_channels = 256
        if not train_backbone:
            for p in self.body.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.body(x)


class PositionEmbeddingSine(torch.nn.Module):
    """Sine positional embedding (similar to DETR paper)."""
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, tensor):
        # tensor: [B, C, H, W]
        x_embed = tensor.sum(dim=1, keepdim=True)
        mask = torch.zeros_like(x_embed, dtype=torch.bool)[:, 0]
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Transformer(torch.nn.Module):
    """Transformer Encoder-Decoder."""
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, src, pos_embed, query_embed):
        # src: [HW, B, C]
        # pos_embed: [HW, B, C]
        # query_embed: [num_queries, C]
        HW, B, C = src.shape
        num_queries = query_embed.shape[0]
        
        tgt = torch.zeros(num_queries, B, self.d_model, device=src.device)
        query_embed_expanded = query_embed.unsqueeze(1).expand(-1, B, -1)

        memory = self.encoder(src + pos_embed)
        out = self.decoder(tgt + query_embed_expanded, memory + pos_embed)
        return out


class MLP(torch.nn.Module):
    """Simple multi-layer perceptron (used for box regression)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(torch.nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = torch.nn.Embedding(num_queries, hidden_dim)
        self.input_proj = torch.nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 3, H, W]
        features = self.backbone(x)
        src = self.input_proj(features)
        
        pos = PositionEmbeddingSine(self.transformer.d_model // 2)(src)
        
        src_flat = src.flatten(2).permute(2, 0, 1)
        pos_flat = pos.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight
        
        hs = self.transformer(src_flat, pos_flat, query_embed)
        hs = hs.permute(1, 0, 2)
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}


# ── Box Helpers & Hungarian Matcher ───────────────────────────────────────────

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    # boxes1: [N, 4], boxes2: [M, 4] in xyxy format
    # clamping coordinates to ensure non-negative areas
    boxes1 = torch.cat([boxes1[:, :2], torch.max(boxes1[:, 2:], boxes1[:, :2])], dim=1)
    boxes2 = torch.cat([boxes2[:, :2], torch.max(boxes2[:, 2:], boxes2[:, :2])], dim=1)
    
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / area.clamp(min=1e-6)
    return giou


class HungarianMatcher(torch.nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing 'pred_logits' [B, Q, num_classes+1] and 'pred_boxes' [B, Q, 4]
            targets: list of dicts: {'labels': [M], 'boxes': [M, 4]}
        """
        B, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten predictions across batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B * Q, num_classes + 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)              # [B * Q, 4]

        # Concat target classes and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Cost matrix: classification cost
        cost_class = -out_prob[:, tgt_ids]

        # Cost matrix: L1 box cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Cost matrix: GIoU box cost
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

        # Combine costs
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(B, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c in enumerate(cost.split(sizes, -1)):
            row_ind, col_ind = linear_sum_assignment(c[i].numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices


# ── Set Criterion Loss ────────────────────────────────────────────────────────

class SetCriterion(torch.nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox_mean = loss_bbox.sum() / num_boxes

        # GIoU Loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - torch.diagonal(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        loss_giou_mean = loss_giou.sum() / num_boxes
        
        return {'loss_bbox': loss_bbox_mean, 'loss_giou': loss_giou_mean}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses


# ── Box Processing & Visualization Helpers ────────────────────────────────────

def masks_to_boxes_and_labels(gt_masks):
    """
    Args:
        gt_masks: Tensor of shape [N, 3, H, W] where channels are: block, agent, goal
    Returns:
        targets: List of dicts, each with 'boxes' [M, 4] and 'labels' [M]
    """
    N, M, H, W = gt_masks.shape
    targets = []
    
    for i in range(N):
        boxes = []
        labels = []
        for m_idx in range(M):
            mask = gt_masks[i, m_idx]
            coords = torch.nonzero(mask)
            if len(coords) == 0:
                continue
            ymin, xmin = coords.min(dim=0).values
            ymax, xmax = coords.max(dim=0).values
            
            # cx, cy, w, h normalized to [0, 1]
            h = (ymax - ymin + 1) / H
            w = (xmax - xmin + 1) / W
            cy = (ymin + ymax) / 2.0 / H
            cx = (xmin + xmax) / 2.0 / W
            
            boxes.append(torch.stack([cx, cy, w, h]))
            labels.append(torch.tensor(m_idx, dtype=torch.long, device=gt_masks.device))
            
        if len(boxes) > 0:
            boxes = torch.stack(boxes)
            labels = torch.stack(labels)
        else:
            boxes = torch.zeros((0, 4), device=gt_masks.device)
            labels = torch.zeros((0,), dtype=torch.long, device=gt_masks.device)
            
        targets.append({"boxes": boxes, "labels": labels})
        
    return targets


def draw_box(img, box_xyxy, color, thickness=1):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = min(max(int(x1 * W), 0), W - 1)
    x2 = min(max(int(x2 * W), 0), W - 1)
    y1 = min(max(int(y1 * H), 0), H - 1)
    y2 = min(max(int(y2 * H), 0), H - 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


@torch.no_grad()
def visualize_detr(model, val_ds, n_samples, device, writer, epoch):
    model.eval()
    grids = []
    
    for i in range(min(n_samples, len(val_ds._episode_indices))):
        ep_idx = val_ds._episode_indices[i]
        data   = val_ds.get_video(ep_idx)
        video  = data['video'].float().to(device)
        gt_m   = data['gt_masks'].float().to(device)
        T      = min(video.shape[0], CFG['n_sample_frames'])
        
        clip   = video[:T]
        gt_clip= gt_m[:T]
        
        outputs = model(clip)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        gt_targets = masks_to_boxes_and_labels(gt_clip)
        
        rows = []
        for t in range(T):
            img_tensor = clip[t]
            img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            
            # 1. Ground Truth image
            gt_img = img_np.copy()
            gt_t = gt_targets[t]
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_t['boxes'])
            for box, label in zip(gt_boxes_xyxy, gt_t['labels']):
                color = (255, 0, 0) if label == 0 else ((0, 255, 0) if label == 1 else (0, 0, 255))
                gt_img = draw_box(gt_img, box.cpu().numpy(), color, thickness=1)
                
            # 2. Predicted image (confidence threshold score > 0.3)
            pred_img = img_np.copy()
            probs = pred_logits[t].softmax(-1)
            scores, labels = probs[:, :-1].max(-1)
            
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[t])
            for q_idx in range(len(scores)):
                score = scores[q_idx].item()
                if score > 0.3:
                    label = labels[q_idx].item()
                    color = (255, 0, 0) if label == 0 else ((0, 255, 0) if label == 1 else (0, 0, 255))
                    pred_img = draw_box(pred_img, pred_boxes_xyxy[q_idx].cpu().numpy(), color, thickness=1)
                    
            gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0
            pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).float() / 255.0
            orig_tensor = (img_tensor * 0.5 + 0.5).clamp(0, 1).cpu()
            
            # Row contents: Original Frame | GT Bounding Boxes | Pred Bounding Boxes
            rows.append(torch.stack([orig_tensor, gt_tensor, pred_tensor], dim=0))
            
        grid = torch.stack(rows, dim=0).flatten(0, 1)
        grids.append(vutils.make_grid(grid, nrow=3, pad_value=1.0))
        
    writer.add_image('val/detections',
                     torch.stack(grids, dim=0).mean(0),
                     global_step=epoch)
    model.train()


# ── Training Helpers ──────────────────────────────────────────────────────────

def cosine_anneal_with_warmup(step, total_steps, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
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
    print("Building DETR model …")
    backbone = ResNetBackbone(train_backbone=True)
    transformer = Transformer(
        d_model=CFG['d_model'],
        nhead=CFG['nhead'],
        num_encoder_layers=CFG['num_encoder_layers'],
        num_decoder_layers=CFG['num_decoder_layers'],
        dim_feedforward=CFG['dim_feedforward']
    )
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=CFG['num_classes'],
        num_queries=CFG['num_queries']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ── Matcher & Loss ────────────────────────────────────────────────────────
    matcher = HungarianMatcher(
        cost_class=CFG['weight_class'],
        cost_bbox=CFG['weight_bbox'],
        cost_giou=CFG['weight_giou']
    )
    
    weight_dict = {
        'loss_ce': CFG['weight_class'],
        'loss_bbox': CFG['weight_bbox'],
        'loss_giou': CFG['weight_giou']
    }
    
    criterion = SetCriterion(
        num_classes=CFG['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=CFG['eos_coef'],
        losses=['labels', 'boxes']
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
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
        epoch_losses = {'total': 0., 'ce': 0., 'bbox': 0., 'giou': 0.}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_idx >= args.limit_train_batches:
                break

            img      = batch['img'].to(device, non_blocking=True)       # [B, T, 3, H, W]
            gt_masks = batch['gt_masks'].to(device, non_blocking=True)  # [B, T, 3, H, W]

            # Flatten temporal dim: treats each frame as a standalone object detection image
            # B, T, C, H, W -> B * T, C, H, W
            B, T = img.shape[:2]
            img_flat = img.flatten(0, 1)
            gt_flat = gt_masks.flatten(0, 1)

            # Convert GT masks to bounding box targets
            targets = masks_to_boxes_and_labels(gt_flat)

            # LR schedule
            lr = cosine_anneal_with_warmup(global_step, total_steps,
                                           warmup_steps, CFG['lr'],
                                           CFG['lr'] / 100.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward
            outputs = model(img_flat)
            
            # Loss calculation
            loss_dict = criterion(outputs, targets)
            
            loss_ce = loss_dict['loss_ce']
            loss_bbox = loss_dict['loss_bbox']
            loss_giou = loss_dict['loss_giou']

            loss = (CFG['weight_class'] * loss_ce +
                    CFG['weight_bbox']  * loss_bbox +
                    CFG['weight_giou']  * loss_giou)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if CFG['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
            optimizer.step()

            # Accumulate
            epoch_losses['total'] += loss.item()
            epoch_losses['ce']    += loss_ce.item()
            epoch_losses['bbox']  += loss_bbox.item()
            epoch_losses['giou']  += loss_giou.item()

            writer.add_scalar('train/lr',        lr,               global_step)
            writer.add_scalar('train/loss',      loss.item(),      global_step)
            writer.add_scalar('train/loss_ce',   loss_ce.item(),   global_step)
            writer.add_scalar('train/loss_bbox', loss_bbox.item(), global_step)
            writer.add_scalar('train/loss_giou', loss_giou.item(), global_step)

            if global_step % 100 == 0:
                print(f"  Step {global_step:6d}/{total_steps:6d} | "
                      f"loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                      f"bbox={loss_bbox.item():.4f} giou={loss_giou.item():.4f} "
                      f"lr={lr:.2e}", flush=True)

            global_step += 1

        # ── End of epoch ──────────────────────────────────────────────────────
        n = steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{CFG['max_epochs']} | "
              f"loss={epoch_losses['total']/n:.4f}  "
              f"ce={epoch_losses['ce']/n:.4f}  "
              f"bbox={epoch_losses['bbox']/n:.4f}  "
              f"giou={epoch_losses['giou']/n:.4f}  "
              f"lr={lr:.2e}  [{elapsed:.0f}s]", flush=True)

        writer.add_scalar('epoch/loss',      epoch_losses['total'] / n, epoch)
        writer.add_scalar('epoch/loss_ce',   epoch_losses['ce'] / n, epoch)
        writer.add_scalar('epoch/loss_bbox', epoch_losses['bbox'] / n, epoch)
        writer.add_scalar('epoch/loss_giou', epoch_losses['giou'] / n, epoch)

        # Validation
        model.eval()
        val_losses = {'total': 0., 'ce': 0., 'bbox': 0., 'giou': 0.}
        val_steps = len(val_loader) if args.limit_val_batches is None else args.limit_val_batches

        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_loader):
                if args.limit_val_batches is not None and val_idx >= args.limit_val_batches:
                    break

                img      = val_batch['img'].to(device, non_blocking=True)
                gt_masks = val_batch['gt_masks'].to(device, non_blocking=True)
                
                B_v, T_v = img.shape[:2]
                img_v_flat = img.flatten(0, 1)
                gt_v_flat = gt_masks.flatten(0, 1)
                
                targets_v = masks_to_boxes_and_labels(gt_v_flat)
                
                outputs_v = model(img_v_flat)
                loss_dict_v = criterion(outputs_v, targets_v)
                
                l_ce = loss_dict_v['loss_ce']
                l_bbox = loss_dict_v['loss_bbox']
                l_giou = loss_dict_v['loss_giou']
                
                val_losses['ce']    += l_ce.item()
                val_losses['bbox']  += l_bbox.item()
                val_losses['giou']  += l_giou.item()
                val_losses['total'] += (CFG['weight_class'] * l_ce +
                                        CFG['weight_bbox']  * l_bbox +
                                        CFG['weight_giou']  * l_giou).item()

        vn = val_steps
        writer.add_scalar('val/loss',      val_losses['total'] / vn, epoch)
        writer.add_scalar('val/loss_ce',   val_losses['ce'] / vn, epoch)
        writer.add_scalar('val/loss_bbox', val_losses['bbox'] / vn, epoch)
        writer.add_scalar('val/loss_giou', val_losses['giou'] / vn, epoch)
        print(f"          Val  | loss={val_losses['total']/vn:.4f}  "
              f"ce={val_losses['ce']/vn:.4f}  "
              f"bbox={val_losses['bbox']/vn:.4f}  "
              f"giou={val_losses['giou']/vn:.4f}", flush=True)

        if (epoch + 1) % CFG['vis_every_n_epochs'] == 0:
            visualize_detr(model, val_ds, CFG['n_vis_samples'], device, writer, epoch)

        if (epoch + 1) % CFG['save_every_n_epochs'] == 0:
            ckpt_path = os.path.join(CFG['ckpt_dir'], f'detr_epoch_{epoch+1}.pt')
            torch.save({
                'epoch':       epoch,
                'global_step': global_step,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'config':      CFG,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    final_path = os.path.join(CFG['ckpt_dir'], 'detr_final.pt')
    torch.save({'epoch': CFG['max_epochs'] - 1, 'global_step': global_step,
                'model': model.state_dict(), 'config': CFG}, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()


if __name__ == '__main__':
    main()
