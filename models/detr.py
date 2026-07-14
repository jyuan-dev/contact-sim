import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ── 1. Backbone Components ────────────────────────────────────────────────────
class ResNetBackbone(nn.Module):
    def __init__(self, name='resnet18', train_backbone=True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        # ResNet18 down to layer3 (output channels: 256).
        # Input 64x64 -> conv1 (32x32) -> maxpool (16x16) -> layer1 (16x16) -> layer2 (8x8) -> layer3 (4x4)
        self.body = nn.Sequential(
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


class PositionEmbeddingSine(nn.Module):
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


# ── 2. Transformer Components ─────────────────────────────────────────────────
class Transformer(nn.Module):
    """Transformer Encoder-Decoder."""
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

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


class MLP(nn.Module):
    """Simple multi-layer perceptron (used for box regression)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ── 3. DETR Model ─────────────────────────────────────────────────────────────
class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

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


# ── 4. Box Helper Functions ────────────────────────────────────────────────────
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
    boxes1 = torch.cat([boxes1[:, :2], torch.max(boxes1[:, 2:], boxes1[:, :2])], dim=1)
    boxes2 = torch.cat([boxes2[:, :2], torch.max(boxes2[:, 2:], boxes2[:, :2])], dim=1)
    
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / area.clamp(min=1e-6)
    return giou


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
        
        if len(boxes) == 0:
            targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=gt_masks.device),
                'labels': torch.zeros((0,), dtype=torch.long, device=gt_masks.device)
            })
        else:
            targets.append({
                'boxes': torch.stack(boxes),
                'labels': torch.stack(labels)
            })
    return targets


# ── 5. Loss Criterion & Matcher ───────────────────────────────────────────────
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, num_queries = outputs["pred_logits"].shape[:2]
        
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B * Q, num_classes + 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)              # [B * Q, 4]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(B, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c in enumerate(cost.split(sizes, -1)):
            row_ind, col_ind = linear_sum_assignment(c[i].numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices


class SetCriterion(nn.Module):
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

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox_mean = loss_bbox.sum() / num_boxes

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
