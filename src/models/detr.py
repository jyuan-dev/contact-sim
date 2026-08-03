import math
import copy
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
        B, C, H, W = tensor.shape
        device = tensor.device

        y_embed = torch.arange(1, H + 1, device=device, dtype=torch.float32).view(1, H, 1).expand(B, H, W)
        x_embed = torch.arange(1, W + 1, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)

        if self.normalize:
            y_embed = y_embed / y_embed[:, -1:, :] * self.scale
            x_embed = x_embed / x_embed[:, :, -1:] * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ── 2. Transformer Components ─────────────────────────────────────────────────
class Transformer(nn.Module):
    """Transformer Encoder-Decoder with intermediate decoder layer outputs.

    Returns stacked outputs from all decoder layers — required for DETR's
    auxiliary decoding losses (Carion et al., 2020).
    """

    def __init__(self, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1):
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
        self.decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer)
                                              for _ in range(num_decoder_layers)])

    def forward(self, src, pos_embed, query_embed):
        # src: [HW, B, C]; pos_embed: [HW, B, C]; query_embed: [num_queries, C]
        HW, B, C = src.shape
        num_queries = query_embed.shape[0]

        tgt = torch.zeros(num_queries, B, self.d_model, device=src.device)
        query_embed_expanded = query_embed.unsqueeze(1).expand(-1, B, -1)

        memory = self.encoder(src + pos_embed)
        mem_with_pos = memory + pos_embed

        # Manually iterate decoder layers to collect intermediate outputs
        output = tgt + query_embed_expanded
        all_outputs = []
        for layer in self.decoder_layers:
            output = layer(output, mem_with_pos)
            all_outputs.append(output)

        # [num_decoder_layers, num_queries, B, d_model]
        return torch.stack(all_outputs, dim=0)


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

        # hs: [num_decoder_layers, num_queries, B, d_model]
        hs = self.transformer(src_flat, pos_flat, query_embed)

        # Apply shared FFN heads to every decoder layer output
        # (same class_embed/bbox_embed for all layers, per DETR paper)
        hs_flat = hs.permute(2, 0, 1, 3)  # [B, num_layers, num_queries, d_model]
        B, L, Q, D = hs_flat.shape

        outputs_class = self.class_embed(hs_flat)      # [B, L, Q, num_classes+1]
        outputs_coord = self.bbox_embed(hs_flat).sigmoid()  # [B, L, Q, 4]

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


def batched_generalized_box_iou(boxes1, boxes2):
    """
    Computes GIoU for batched boxes: boxes1 [B, Q, 4], boxes2 [B, M, 4] in cxcywh format.
    Returns tensor of shape [B, Q, M].
    """
    b1 = box_cxcywh_to_xyxy(boxes1)  # [B, Q, 4]
    b2 = box_cxcywh_to_xyxy(boxes2)  # [B, M, 4]

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])  # [B, Q]
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])  # [B, M]

    lt = torch.maximum(b1[..., None, :2], b2[..., None, :, :2])  # [B, Q, M, 2]
    rb = torch.minimum(b1[..., None, 2:], b2[..., None, :, 2:])  # [B, Q, M, 2]

    wh = (rb - lt).clamp(min=0)  # [B, Q, M, 2]
    inter = wh[..., 0] * wh[..., 1]  # [B, Q, M]

    union = area1[..., None] + area2[..., None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.minimum(b1[..., None, :2], b2[..., None, :, :2])
    rb_c = torch.maximum(b1[..., None, 2:], b2[..., None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


def masks_to_boxes_and_labels(gt_masks):
    """
    Args:
        gt_masks: Tensor of shape [N, 3, H, W] where channels are: block, agent, goal
    Returns:
        targets: List of dicts, each with 'boxes' [M, 4] and 'labels' [M]
    """
    N, M, H, W = gt_masks.shape
    device = gt_masks.device
    mask_bool = gt_masks > 0.5
    has_obj = mask_bool.flatten(2).any(dim=-1)  # [N, M]

    y_grid = torch.arange(H, device=device).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device).view(1, 1, 1, W)

    y_min_val = torch.where(mask_bool, y_grid, torch.tensor(1e5, device=device)).flatten(2).min(dim=-1).values
    y_max_val = torch.where(mask_bool, y_grid, torch.tensor(-1e5, device=device)).flatten(2).max(dim=-1).values
    x_min_val = torch.where(mask_bool, x_grid, torch.tensor(1e5, device=device)).flatten(2).min(dim=-1).values
    x_max_val = torch.where(mask_bool, x_grid, torch.tensor(-1e5, device=device)).flatten(2).max(dim=-1).values

    h_norm = (y_max_val - y_min_val + 1.0) / H
    w_norm = (x_max_val - x_min_val + 1.0) / W
    cy_norm = (y_min_val + y_max_val) / 2.0 / H
    cx_norm = (x_min_val + x_max_val) / 2.0 / W

    boxes_all = torch.stack([cx_norm, cy_norm, w_norm, h_norm], dim=-1)  # [N, M, 4]

    targets = []
    for i in range(N):
        valid = has_obj[i]
        if not valid.any():
            targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device)
            })
        else:
            boxes = boxes_all[i, valid]
            labels = torch.arange(M, device=device)[valid]
            targets.append({'boxes': boxes, 'labels': labels})
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
        """
        Batched GPU vectorized cost computation.
        outputs['pred_logits']: [B, Q, num_classes + 1]
        outputs['pred_boxes']:  [B, Q, 4]
        targets: list of N dicts, each with 'boxes' [M_i, 4] and 'labels' [M_i]
        """
        out_logits = outputs["pred_logits"]
        out_boxes = outputs["pred_boxes"]
        B, Q, C = out_logits.shape
        device = out_logits.device

        sizes = [len(v["boxes"]) for v in targets]
        max_m = max(sizes) if sizes else 0

        if max_m == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(B)]

        tgt_labels_padded = torch.zeros(B, max_m, dtype=torch.long, device=device)
        tgt_boxes_padded = torch.zeros(B, max_m, 4, dtype=torch.float32, device=device)

        for i, t in enumerate(targets):
            m = sizes[i]
            if m > 0:
                tgt_labels_padded[i, :m] = t["labels"].to(device)
                tgt_boxes_padded[i, :m] = t["boxes"].to(device)

        out_prob = out_logits.softmax(-1)  # [B, Q, C]

        tgt_idx_expanded = tgt_labels_padded.unsqueeze(1).expand(-1, Q, -1)  # [B, Q, max_m]
        c_class = -torch.gather(out_prob, 2, tgt_idx_expanded)  # [B, Q, max_m]

        c_bbox = torch.cdist(out_boxes, tgt_boxes_padded, p=1)
        c_giou = -batched_generalized_box_iou(out_boxes, tgt_boxes_padded)

        cost = self.cost_bbox * c_bbox + self.cost_class * c_class + self.cost_giou * c_giou
        cost_np = cost.detach().cpu().numpy()

        indices = []
        for i in range(B):
            m = sizes[i]
            if m == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
            else:
                c_i = cost_np[i, :, :m]
                row_ind, col_ind = linear_sum_assignment(c_i)
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
        if sum(len(J) for _, J in indices) == 0:
            target_classes_o = torch.empty((0,), dtype=torch.int64, device=src_logits.device)
        else:
            target_classes_o = torch.cat([t["labels"][J].to(src_logits.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        if sum(len(i) for _, i in indices) == 0:
            target_boxes = torch.empty((0, 4), dtype=torch.float32, device=src_boxes.device)
        else:
            target_boxes = torch.cat([t['boxes'][i].to(src_boxes.device) for t, (_, i) in zip(targets, indices)], dim=0)

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
        """Compute losses for all decoder layers (auxiliary losses per DETR paper).

        outputs['pred_logits'] is [B, L, Q, C+1] (L = num decoder layers).
        The last layer is treated as the primary output; earlier layers are aux.
        """
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        if pred_logits.ndim == 4:
            B, L, Q, C = pred_logits.shape
        else:
            B, Q, C = pred_logits.shape
            L = 1

        if L > 1:
            out_all = {
                'pred_logits': pred_logits.view(B * L, Q, C),
                'pred_boxes': pred_boxes.view(B * L, Q, 4)
            }
            all_indices = self.matcher(out_all, targets * L)
            indices_per_layer = [all_indices[l * B : (l + 1) * B] for l in range(L)]
        else:
            indices_per_layer = [self.matcher(outputs, targets)]

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)
        num_boxes_val = torch.clamp(num_boxes_t, min=1).item()

        losses = {}
        for layer in range(L):
            if L > 1:
                layer_out = {'pred_logits': pred_logits[:, layer], 'pred_boxes': pred_boxes[:, layer]}
            else:
                layer_out = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}

            indices = indices_per_layer[layer]
            suffix = f'_aux_{layer}' if layer < L - 1 else ''

            for loss in self.losses:
                if loss == 'labels':
                    l = self.loss_labels(layer_out, targets, indices, num_boxes_val)
                elif loss == 'boxes':
                    l = self.loss_boxes(layer_out, targets, indices, num_boxes_val)
                else:
                    continue
                for k, v in l.items():
                    losses[f'{k}{suffix}'] = v

        return losses
