"""
Model Factory for Contact-Sim / Slot-Worldmodel Baselines.

Provides a unified build_model(cfg) function that instantiates:
  - DETR Bounding-Box / Mask model
  - StoSAVi / SAVi Slot Attention model

Enforces a Standardized Model Output Dictionary Contract:
  {
      'pred_boxes': Tensor or None,   # [B, Q, 4] normalized (cx, cy, w, h)
      'pred_masks': Tensor or None,   # [B, (T,) K, H, W] soft attention masks
      'pred_logits': Tensor or None,  # [B, Q, num_classes + 1] class logits
      'recon_img': Tensor or None,    # [B, T, C, H, W] image reconstructions
      'input_img': Tensor,            # [B, T, C, H, W] input image (passthrough)
  }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detr import DETR, ResNetBackbone, Transformer


def _resolve_cfg(cfg):
    """Resolve Hydra DictConfig to plain dict if needed."""
    try:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return cfg


def _flatten_video(x):
    """Flatten [B, T, C, H, W] → [B*T, C, H, W] if 5D."""
    if x.ndim == 5:
        B, T, C, H, W = x.shape
        return x.reshape(B * T, C, H, W)
    return x


class StandardizedDETRWrapper(nn.Module):
    """Wraps DETR model to adhere to standardized output dictionary contract."""

    def __init__(self, model, criterion=None, weight_dict=None):
        super().__init__()
        self.model = model
        self._criterion = criterion
        self._weight_dict = weight_dict

    def forward(self, x):
        """Forward pass. Input: [B, C, H, W] or [B, T, C, H, W]."""
        x_in = _flatten_video(x)

        raw_out = self.model(x_in)

        # Extract last decoder layer for the standardized contract (eval/metrics).
        # raw_out['pred_logits'] is [B, L, Q, C] (multi-layer) or [B, Q, C] (single).
        pred_logits = raw_out['pred_logits']
        pred_boxes = raw_out['pred_boxes']
        if pred_logits.ndim == 4:
            final_logits = pred_logits[:, -1]
            final_boxes = pred_boxes[:, -1]
        else:
            final_logits = pred_logits
            final_boxes = pred_boxes

        return {
            'pred_boxes': final_boxes,
            'pred_masks': None,
            'pred_logits': final_logits,
            'recon_img': None,
            'input_img': x,
            # Pass full layer-stacked output to compute_loss for aux losses
            'pred_logits_all': pred_logits,
            'pred_boxes_all': pred_boxes,
        }

    def compute_loss(self, out, batch):
        """Compute DETR loss using the pre-built criterion (with aux losses)."""
        from src.losses.model_losses import compute_detr_loss
        gt_masks = batch.get('gt_masks') if isinstance(batch, dict) else None
        # Use full layer-stacked outputs for aux losses, fall back to contract keys
        loss_out = {
            'pred_logits': out.get('pred_logits_all', out['pred_logits']),
            'pred_boxes': out.get('pred_boxes_all', out['pred_boxes']),
        }
        return compute_detr_loss(loss_out, gt_masks, self._criterion, self._weight_dict)


class StandardizedSAViWrapper(nn.Module):
    """Wraps StoSAVi / SAVi model to adhere to standardized output dictionary contract."""

    def __init__(self, model, weight_dict=None):
        super().__init__()
        self.model = model
        self._weight_dict = weight_dict

    def forward(self, x):
        """Forward pass. Input: {'img': [B, T, C, H, W]} or Tensor [B, T, C, H, W]."""
        if isinstance(x, torch.Tensor):
            x = {'img': x}

        img_tensor = x['img']
        if img_tensor.shape[-2:] != (64, 64):
            B, T, C, H, W = img_tensor.shape
            img_tensor = F.interpolate(
                img_tensor.view(B * T, C, H, W), size=(64, 64),
                mode='bilinear', align_corners=False,
            ).view(B, T, C, 64, 64)
            x = dict(x, img=img_tensor)

        raw_out = self.model(x)

        post_masks = raw_out.get('post_masks', raw_out.get('masks', raw_out.get('prior_masks')))
        if post_masks is not None and post_masks.ndim == 6 and post_masks.shape[3] == 1:
            post_masks = post_masks.squeeze(3)

        recon_img = raw_out.get('post_recon_combined', raw_out.get('recon_combined'))

        return {
            'pred_boxes': None,
            'pred_masks': post_masks,
            'pred_logits': None,
            'recon_img': recon_img,
            'input_img': img_tensor,
        }

    def compute_loss(self, out, batch):
        """Compute SAVi slot attention loss."""
        from src.losses.model_losses import compute_savi_loss
        gt_masks = batch.get('gt_masks') if isinstance(batch, dict) else None
        return compute_savi_loss(out, gt_masks, weight_dict=self._weight_dict)


def build_model(cfg):
    """
    Constructs a baseline model wrapped in standardized output contract.

    Args:
        cfg (dict or DictConfig): Configuration loaded via Hydra / OmegaConf or YAML.

    Returns:
        nn.Module: Standardized PyTorch model wrapper.
    """
    cfg = _resolve_cfg(cfg)
    model_cfg = cfg.get('model', cfg)

    model_type = model_cfg.get('type', model_cfg.get('name', 'detr')).lower()

    if model_type in ('detr',):
        backbone = ResNetBackbone(train_backbone=model_cfg.get('train_backbone', False))
        transformer = Transformer(
            d_model=model_cfg.get('d_model', 128),
            nhead=model_cfg.get('nhead', 4),
            num_encoder_layers=model_cfg.get('num_encoder_layers', 3),
            num_decoder_layers=model_cfg.get('num_decoder_layers', 3),
            dim_feedforward=model_cfg.get('dim_feedforward', 512),
        )
        base_model = DETR(
            backbone=backbone,
            transformer=transformer,
            num_classes=model_cfg.get('num_classes', 3),
            num_queries=model_cfg.get('num_queries', 5),
        )
        # Build criterion inside the wrapper so train.py doesn't need to reach in
        from src.models.detr import HungarianMatcher, SetCriterion
        w_dict = dict(model_cfg.get('weight_dict', {'class': 1.0, 'bbox': 5.0, 'giou': 2.0}))
        # Ensure aux weight entries exist (use same weights as main by default)
        for prefix in ('class', 'bbox', 'giou'):
            w_dict.setdefault(f'{prefix}_aux', w_dict.get(prefix, 1.0))
        matcher = HungarianMatcher(
            cost_class=w_dict.get('class', 1.0),
            cost_bbox=w_dict.get('bbox', 5.0),
            cost_giou=w_dict.get('giou', 2.0),
        )
        criterion = SetCriterion(
            num_classes=model_cfg.get('num_classes', 3),
            matcher=matcher,
            weight_dict=w_dict,
            eos_coef=0.1,
            losses=['labels', 'boxes'],
        )
        wrapper = StandardizedDETRWrapper(base_model, criterion=criterion, weight_dict=w_dict)
        return wrapper

    elif model_type in ('savi', 'stosavi'):
        from src.models.savi import SAVi
        res = tuple(model_cfg.get('resolution', [64, 64]))
        clip_len = model_cfg.get('n_sample_frames', model_cfg.get('clip_len', 6))
        base_model = SAVi(
            resolution=res,
            clip_len=clip_len,
            num_slots=model_cfg.get('num_slots', 4),
            slot_dim=model_cfg.get('slot_dim', 64),
            num_iterations=model_cfg.get('num_iterations', 3),
            in_channels=model_cfg.get('in_channels', 3),
            slot_dict=model_cfg.get('slot_dict'),
            enc_dict=model_cfg.get('enc_dict'),
            dec_dict=model_cfg.get('dec_dict'),
            pred_dict=model_cfg.get('pred_dict'),
            loss_dict=model_cfg.get('loss_dict'),
        )
        weight_dict = cfg.get('weight_dict', model_cfg.get('weight_dict'))
        return StandardizedSAViWrapper(base_model, weight_dict=weight_dict)

    elif model_type in ('deformable_savi', 'deformable-savi'):
        from src.models.deformable_savi import DeformableSAVi
        res = tuple(model_cfg.get('resolution', [64, 64]))
        clip_len = model_cfg.get('n_sample_frames', model_cfg.get('clip_len', 6))
        base_model = DeformableSAVi(
            resolution=res,
            clip_len=clip_len,
            num_slots=model_cfg.get('num_slots', 4),
            slot_dim=model_cfg.get('slot_dim', 64),
            num_iterations=model_cfg.get('num_iterations', 3),
            n_heads=model_cfg.get('n_heads', 4),
            n_points=model_cfg.get('n_points', 4),
            in_channels=model_cfg.get('in_channels', 3),
            slot_dict=model_cfg.get('slot_dict'),
            enc_dict=model_cfg.get('enc_dict'),
            dec_dict=model_cfg.get('dec_dict'),
            pred_dict=model_cfg.get('pred_dict'),
            loss_dict=model_cfg.get('loss_dict'),
        )
        weight_dict = cfg.get('weight_dict', model_cfg.get('weight_dict'))
        return StandardizedSAViWrapper(base_model, weight_dict=weight_dict)


    elif model_type in ('deformable_detr', 'deformable-detr'):
        from src.models.deformable_detr import DeformableDETR
        from src.models.detr import HungarianMatcher, SetCriterion
        base_model = DeformableDETR(
            num_classes=model_cfg.get('num_classes', 3),
            num_queries=model_cfg.get('num_queries', 10),
            d_model=model_cfg.get('d_model', 128),
            nhead=model_cfg.get('nhead', 4),
            num_encoder_layers=model_cfg.get('num_encoder_layers', 3),
            num_decoder_layers=model_cfg.get('num_decoder_layers', 3),
            dim_feedforward=model_cfg.get('dim_feedforward', 512),
            dropout=model_cfg.get('dropout', 0.1),
            backbone_name=model_cfg.get('backbone_name', 'resnet18'),
            train_backbone=model_cfg.get('train_backbone', True),
        )
        w_dict = dict(model_cfg.get('weight_dict', {'class': 1.0, 'bbox': 5.0, 'giou': 2.0}))
        for prefix in ('class', 'bbox', 'giou'):
            w_dict.setdefault(f'{prefix}_aux', w_dict.get(prefix, 1.0))
        matcher = HungarianMatcher(
            cost_class=w_dict.get('class', 1.0),
            cost_bbox=w_dict.get('bbox', 5.0),
            cost_giou=w_dict.get('giou', 2.0),
        )
        criterion = SetCriterion(
            num_classes=model_cfg.get('num_classes', 3),
            matcher=matcher,
            weight_dict=w_dict,
            eos_coef=0.1,
            losses=['labels', 'boxes'],
        )
        return StandardizedDETRWrapper(base_model, criterion=criterion, weight_dict=w_dict)

    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Supported: 'detr', 'deformable_detr', 'savi'.")

