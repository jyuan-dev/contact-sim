"""
Model Factory for Contact-Sim / Slot-Worldmodel Baselines.

Provides a unified build_model(cfg) function that instantiates:
  - DETR Bounding-Box / Mask model
  - StoSAVi / SAVi Slot Attention model
  - Slot-PIDM Model

Enforces a Standardized Model Output Dictionary Contract:
  {
      'pred_boxes': Tensor or None,   # [B, Q, 4] normalized (cx, cy, w, h)
      'pred_masks': Tensor or None,   # [B, (T,) K, H, W] soft attention masks
      'pred_logits': Tensor or None,  # [B, Q, num_classes + 1] class logits
      'recon_img': Tensor or None     # [B, T, C, H, W] image reconstructions
  }
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.detr import DETR, ResNetBackbone, Transformer
from src.models.slot_attention import build_savi_model


class StandardizedDETRWrapper(nn.Module):
    """Wraps DETR model to adhere to standardized output dictionary contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass.
        Input x: [B, C, H, W] or [B, T, C, H, W]
        """
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x_in = x.reshape(B * T, C, H, W)
        else:
            x_in = x

        # Interpolate to 224x224 if required by ResNet backbone
        if x_in.shape[-2:] != (224, 224):
            x_in = F.interpolate(x_in, size=(224, 224), mode='bilinear', align_corners=False)

        raw_out = self.model(x_in)
        pred_logits = raw_out['pred_logits'] # [B_total, Q, num_classes + 1]
        pred_boxes = raw_out['pred_boxes']   # [B_total, Q, 4]

        return {
            'pred_boxes': pred_boxes,
            'pred_masks': None,
            'pred_logits': pred_logits,
            'recon_img': None
        }


class StandardizedSAViWrapper(nn.Module):
    """Wraps StoSAVi / SAVi model to adhere to standardized output dictionary contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass.
        Input x: {'img': [B, T, C, H, W]} or Tensor [B, T, C, H, W]
        """
        if isinstance(x, torch.Tensor):
            x_dict = {'img': x}
        else:
            x_dict = x

        img_tensor = x_dict['img']
        if img_tensor.shape[-2:] != (64, 64):
            B, T, C, H, W = img_tensor.shape
            img_tensor = F.interpolate(img_tensor.view(B * T, C, H, W), size=(64, 64), mode='bilinear', align_corners=False).view(B, T, C, 64, 64)
            x_dict['img'] = img_tensor

        raw_out = self.model(x_dict)
        
        post_masks = raw_out.get('post_masks', None)
        if post_masks is not None and post_masks.ndim == 6:
            post_masks = post_masks.squeeze(3) # [B, T, K, H, W]

        recon_img = raw_out.get('recon_combined', None)

        return {
            'pred_boxes': None,
            'pred_masks': post_masks,
            'pred_logits': None,
            'recon_img': recon_img,
            'raw_out': raw_out
        }


def build_model(cfg):
    """
    Constructs a baseline model wrapped in standardized output contract.

    Args:
        cfg (dict or DictConfig): Configuration loaded via Hydra / OmegaConf or YAML.

    Returns:
        nn.Module: Standardized PyTorch model wrapper.
    """
    try:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    model_cfg = cfg.get('model', cfg)
    if isinstance(cfg, dict):
        merged_cfg = {**cfg, **model_cfg}
    else:
        merged_cfg = model_cfg

    model_type = merged_cfg.get('type', merged_cfg.get('name', 'detr')).lower()

    if 'detr' in model_type:
        backbone = ResNetBackbone(train_backbone=model_cfg.get('train_backbone', False))
        transformer = Transformer(
            d_model=model_cfg.get('d_model', 128),
            nhead=model_cfg.get('nhead', 4),
            num_encoder_layers=model_cfg.get('num_encoder_layers', 3),
            num_decoder_layers=model_cfg.get('num_decoder_layers', 3),
            dim_feedforward=model_cfg.get('dim_feedforward', 512)
        )
        base_model = DETR(
            backbone=backbone,
            transformer=transformer,
            num_classes=model_cfg.get('num_classes', 3),
            num_queries=model_cfg.get('num_queries', 5)
        )
        return StandardizedDETRWrapper(base_model)

    elif 'savi' in model_type or 'slot' in model_type:
        res = tuple(merged_cfg.get('resolution', [64, 64]))
        clip_len = merged_cfg.get('n_sample_frames', merged_cfg.get('clip_len', 16))
        
        base_model = build_savi_model(
            resolution=res,
            clip_len=clip_len,
            slot_dict=merged_cfg.get('slot_dict', {}),
            enc_dict=merged_cfg.get('enc_dict', {}),
            dec_dict=merged_cfg.get('dec_dict', {}),
            pred_dict=merged_cfg.get('pred_dict', {}),
            loss_dict=merged_cfg.get('loss_dict', None)
        )
        return StandardizedSAViWrapper(base_model)

    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Supported: 'detr', 'savi', 'stosavi'.")
