import os
import sys
import types
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLOTFORMER_DIR = os.path.join(REPO_ROOT, 'third_party', 'slotformer')
BASE_SLOTS_DIR = os.path.join(SLOTFORMER_DIR, 'slotformer', 'base_slots')

# ── One-time nerv shim setup ──────────────────────────────────────────────────
_SAVI_SETUP_DONE = False


def _setup_savi_imports():
    global _SAVI_SETUP_DONE
    if _SAVI_SETUP_DONE:
        return
    _SAVI_SETUP_DONE = True

    # Register fake nerv modules (needed by third_party/slotformer imports).
    if 'nerv' not in sys.modules:
        nerv_mod = types.ModuleType('nerv')

        nerv_train = types.ModuleType('nerv.training')

        class BaseModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

        nerv_train.BaseModel = BaseModel

        nerv_models = types.ModuleType('nerv.models')

        def deconv_out_shape(in_size, stride=1, padding=0, kernel_size=1, output_padding=0):
            if isinstance(in_size, (list, tuple)):
                in_size = in_size[0]
            return (in_size - 1) * stride - 2 * padding + kernel_size + output_padding

        def _make_conv_norm_act(conv_cls, in_channels, out_channels, kernel_size,
                                 stride=1, padding=0, output_padding=0, norm='', act='relu'):
            layers = [conv_cls(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding,
                               **({'output_padding': output_padding} if conv_cls is nn.ConvTranspose2d else {}))]
            if act == 'relu':
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        def conv_norm_act(in_channels, out_channels, kernel_size=5, stride=1,
                          padding=None, norm='', act='relu'):
            if padding is None:
                padding = kernel_size // 2
            return _make_conv_norm_act(nn.Conv2d, in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, norm=norm, act=act)

        def deconv_norm_act(in_channels, out_channels, kernel_size=5, stride=1,
                            padding=None, output_padding=None, norm='', act='relu'):
            if padding is None:
                padding = kernel_size // 2
            if output_padding is None:
                output_padding = stride - 1
            return _make_conv_norm_act(nn.ConvTranspose2d, in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding,
                                       output_padding=output_padding, norm=norm, act=act)

        nerv_models.deconv_out_shape = deconv_out_shape
        nerv_models.conv_norm_act = conv_norm_act
        nerv_models.deconv_norm_act = deconv_norm_act

        sys.modules['nerv'] = nerv_mod
        sys.modules['nerv.training'] = nerv_train
        sys.modules['nerv.models'] = nerv_models

    # Ensure the slotformer base_slots directory is importable.
    if BASE_SLOTS_DIR not in sys.path:
        sys.path.insert(0, BASE_SLOTS_DIR)


_setup_savi_imports()

# Clear stale 'models' package so `from models.savi import StoSAVi` resolves
# to third_party/slotformer/slotformer/base_slots/models/savi.py.
for k in list(sys.modules.keys()):
    if k == 'models' or k.startswith('models.'):
        del sys.modules[k]

from models.savi import StoSAVi


class SAVi(nn.Module):
    """
    Standard SAVi (Slot Attention for Video) model wrapper.
    Instantiates StoSAVi with deterministic or stochastic slot attention defaults.

    Accepts either flat kwargs or nested third-party dicts for slot/enc/dec/pred/loss
    config. Flat kwargs serve as defaults; nested dicts take precedence when provided.
    """

    def __init__(
        self,
        resolution=(64, 64),
        clip_len=6,
        num_slots=4,
        slot_dim=64,
        num_iterations=3,
        in_channels=3,
        slot_dict=None,
        enc_dict=None,
        dec_dict=None,
        pred_dict=None,
        loss_dict=None,
        **kwargs,
    ):
        super().__init__()
        self.resolution = tuple(resolution)

        slot_dict = dict(
            num_slots=num_slots,
            slot_size=slot_dim,
            slot_mlp_size=slot_dim * 2,
            num_iterations=num_iterations,
            kernel_mlp=True,
        ) | (slot_dict or {})

        enc_dict = dict(
            enc_channels=(in_channels, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=slot_dim,
            enc_norm='',
        ) | (enc_dict or {})

        dec_dict = dict(
            dec_channels=(slot_dim, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm='',
        ) | (dec_dict or {})

        pred_dict = dict(
            pred_type='transformer',
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=256,
            pred_sg_every=None,
        ) | (pred_dict or {})

        loss_dict = dict(
            use_post_recon_loss=True,
            kld_method='none',
        ) | (loss_dict or {})

        self.model = StoSAVi(
            resolution=self.resolution,
            clip_len=clip_len,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            pred_dict=pred_dict,
            loss_dict=loss_dict,
        )

    def forward(self, x, **kwargs):
        """Forward pass. Accepts tensor [B, T, C, H, W] or dict {'img': ...}."""
        if isinstance(x, torch.Tensor):
            x = {'img': x}
        return self.model(x)
