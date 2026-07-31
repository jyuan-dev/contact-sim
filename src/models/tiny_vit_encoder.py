"""
TinyViT Encoder module for StoSAVi slot attention models.
Uses timm TinyViT backbones (tiny_vit_5m_224 / tiny_vit_11m_224) with optional ImageNet pretraining.
"""

import torch
import torch.nn as nn
import timm

class TinyViTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = 'tiny_vit_5m_224',
        pretrained: bool = True,
        out_channels: int = 32,
        target_size: tuple = (224, 224)
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_channels = out_channels
        self.target_size = target_size

        # Create timm backbone without classification head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Infer feature channels from dummy forward pass
        dummy_x = torch.randn(1, 3, target_size[0], target_size[1])
        with torch.no_grad():
            dummy_out = self.backbone.forward_features(dummy_x)

        if dummy_out.ndim == 4:
            self.feat_dim = dummy_out.shape[-1] if dummy_out.shape[1] in [7, 14] else dummy_out.shape[1]
        else:
            self.feat_dim = dummy_out.shape[-1]

        # Token projection layer
        self.proj = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B*T, 3, H, W]
        Returns:
            tokens: Projected feature tokens [B*T, N_tokens, out_channels]
        """
        if x.shape[-2:] != self.target_size:
            x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        feats = self.backbone.forward_features(x)

        if feats.ndim == 4:
            if feats.shape[1] in [7, 14]: # NHWC format from timm TinyVit: [B*T, H, W, C]
                tokens = feats.flatten(1, 2) # [B*T, H*W, C]
            else: # NCHW format: [B*T, C, H, W]
                tokens = feats.flatten(2).transpose(1, 2)
        else:
            tokens = feats

        out_tokens = self.proj(tokens)
        return out_tokens
