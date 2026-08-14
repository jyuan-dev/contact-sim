import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLOTFORMER_DIR = os.path.join(REPO_ROOT, 'third_party', 'slotformer')
BASE_SLOTS_DIR = os.path.join(SLOTFORMER_DIR, 'slotformer', 'base_slots')

# ── One-time nerv shim setup ──────────────────────────────────────────────────
_SAVI_SETUP_DONE = False



def _setup_savi_imports():
    """
    Set up synthetic 'nerv' module stubs and python path bindings.

    This function creates in-memory mock modules for 'nerv' (nerv.training, nerv.models)
    and adds third_party/slotformer to sys.path so that third-party StoSAVi can be
    imported without editing third-party source files or installing external dependencies.
    """
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


class SlotAttentionWithBN(nn.Module):
    """
    Slot attention module with optional BatchNorm at the end of the residual update.
    """

    def __init__(
        self,
        in_features: int,
        num_iterations: int,
        num_slots: int,
        slot_size: int,
        mlp_hidden_size: int,
        eps: float = 1e-6,
        use_residual_bn: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size**-0.5
        self.use_residual_bn = use_residual_bn

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )
        if self.use_residual_bn:
            self.residual_bn = nn.BatchNorm1d(self.slot_size)
        else:
            self.residual_bn = None

    def forward(self, inputs: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.project_q(slots)
            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)

            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)

            res = self.mlp(slots)
            slots = slots + res
            if self.residual_bn is not None:
                slots = self.residual_bn(slots.view(bs * self.num_slots, self.slot_size)).view(
                    bs, self.num_slots, self.slot_size
                )

        return slots

    @property
    def dtype(self):
        return self.project_k.weight.dtype

    @property
    def device(self):
        return self.project_k.weight.device


class SAVi(nn.Module):
    """
    Standard SAVi (Slot Attention for Video) model wrapper.
    Instantiates StoSAVi with deterministic or stochastic slot attention defaults,
    with optional BatchNorm on the encoder output and slot residual updates.

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
        use_encoder_bn: bool = False,
        use_residual_bn: bool = False,
        slot_dict=None,
        enc_dict=None,
        dec_dict=None,
        pred_dict=None,
        loss_dict=None,
        **kwargs,
    ):
        super().__init__()
        self.resolution = tuple(resolution)
        self.use_encoder_bn = kwargs.pop("use_encoder_bn", use_encoder_bn) or kwargs.pop("use_bn", False)
        self.use_residual_bn = kwargs.pop("use_residual_bn", use_residual_bn) or kwargs.pop("use_bn", False)

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
        # Bind dynamic dtype property to third-party StoSAVi instance
        type(self.model).dtype = property(lambda self: next(self.parameters()).dtype)

        # ── Optional BatchNorm at the end of slot residual ───────────────────
        if self.use_residual_bn:
            self.model.slot_attention = SlotAttentionWithBN(
                in_features=slot_dim,
                num_iterations=num_iterations,
                num_slots=num_slots,
                slot_size=slot_dim,
                mlp_hidden_size=slot_dict.get("slot_mlp_size", slot_dim * 2),
                use_residual_bn=True,
            )

        # ── Optional BatchNorm at the end of encoder ─────────────────────────
        if self.use_encoder_bn:
            self.encoder_bn = nn.BatchNorm1d(slot_dim)
            self.model.encoder_bn = self.encoder_bn
            orig_get_encoder_out = self.model._get_encoder_out

            def _bn_get_encoder_out(img):
                enc_out = orig_get_encoder_out(img)  # [B, HW, enc_out_channels]
                B, HW, C = enc_out.shape
                enc_out = self.encoder_bn(enc_out.view(B * HW, C)).view(B, HW, C)
                return enc_out

            self.model._get_encoder_out = _bn_get_encoder_out
        else:
            self.encoder_bn = None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def load_state_dict(self, state_dict, strict=True):
        """Handle state dict loading for both direct StoSAVi state dicts and wrapper state dicts."""
        if any(k.startswith('model.') for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict=strict)
        return self.model.load_state_dict(state_dict, strict=strict)

    def forward(self, x, **kwargs):
        """Forward pass. Accepts tensor [B, T, C, H, W] or dict {'img': ...}."""
        if isinstance(x, torch.Tensor):
            x = {'img': x}
        return self.model(x)

