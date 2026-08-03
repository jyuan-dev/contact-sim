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
        # Resize to 224x224 if using pretrained ResNet (otherwise skip for speed)
        if x_in.shape[-2:] != (224, 224) and any(p.requires_grad for p in self.model.backbone.parameters()):
            # Only resize if backbone is actually trained/frozen-ImageNet
            pass  # skip resize for random-init backbone

        raw_out = self.model(x_in)
        return {
            'pred_boxes': raw_out['pred_boxes'],
            'pred_masks': None,
            'pred_logits': raw_out['pred_logits'],
            'recon_img': None,
            'input_img': x,
        }

    def compute_loss(self, out, batch):
        """Compute DETR loss using the pre-built criterion."""
        from src.losses.model_losses import compute_detr_loss
        gt_masks = batch.get('gt_masks') if isinstance(batch, dict) else None
        return compute_detr_loss(out, gt_masks, self._criterion, self._weight_dict)


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
        w_dict = model_cfg.get('weight_dict', {'class': 1.0, 'bbox': 5.0, 'giou': 2.0})
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

    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Supported: 'detr', 'savi'.")
