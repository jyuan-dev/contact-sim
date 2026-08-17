"""
SAVi (Slot Attention for Video) — native, self-contained implementation.

Deterministic Slot Attention for Video, ported natively from the SlotFormer
codebase so this project no longer depends on ``third_party/slotformer``.
Supports optional BatchNorm regularization on the encoder output and on the
slot residual update.

Module layout (state-dict keys are pinned by existing checkpoints)::

    SAVi (public wrapper)
    └── model: StoSAVi
        ├── encoder / encoder_pos_embedding / encoder_out_layer / [encoder_bn]
        ├── init_latents / kernel_dist_layer / prior_slot_layer
        ├── slot_attention [.residual_bn]
        ├── decoder / decoder_pos_embedding
        └── predictor (.base_predictor / .rnn / .out_projector)

The core :class:`StoSAVi` exposes the API that rollout / slotformer / pidm /
extract_slots rely on: ``encode``, ``decode``, ``_get_encoder_out``,
``_sample_dist``, ``_reset_rnn``, ``kernel_dist_layer``, ``slot_attention``,
``predictor``, ``init_latents``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Union

from jaxtyping import Float

from src.utils.tensor_checks import check_tensor_shape, typechecked


# ── CNN / shape helpers ───────────────────────────────────────────────────────

def deconv_out_shape(in_size: Union[int, tuple, list], stride: int = 1,
                     padding: int = 0, kernel_size: int = 1,
                     output_padding: int = 0) -> int:
    size = in_size[0] if isinstance(in_size, (list, tuple)) else in_size
    return (size - 1) * stride - 2 * padding + kernel_size + output_padding


def conv_norm_act(in_channels: int, out_channels: int, kernel_size: int = 5,
                  stride: int = 1, padding: Optional[int] = None,
                  act: str = 'relu') -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding)]
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def deconv_norm_act(in_channels: int, out_channels: int, kernel_size: int = 5,
                    stride: int = 1, padding: Optional[int] = None,
                    output_padding: Optional[int] = None,
                    act: str = 'relu') -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    if output_padding is None:
        output_padding = stride - 1
    layers: list[nn.Module] = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)]
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def assert_shape(actual: Union[tuple, list], expected: Union[tuple, list],
                 message: str = "") -> None:
    assert list(actual) == list(expected), \
        f"Expected shape: {expected} but passed shape: {actual}. {message}"


# ── Position embedding ────────────────────────────────────────────────────────

def build_grid(resolution: tuple) -> torch.Tensor:
    """Return a coordinate grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed(nn.Module):
    """Add a learned embedding of normalized coordinates to a feature map."""

    def __init__(self, hidden_size: int, resolution: tuple) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer('grid', build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        return inputs + emb_proj


# ── Slot attention ────────────────────────────────────────────────────────────

class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention,
    with optional BatchNorm on the end of each residual update."""

    def __init__(
        self,
        in_features: int,
        num_iterations: int,
        num_slots: int,
        slot_size: int,
        mlp_hidden_size: int,
        eps: float = 1e-6,
        use_residual_bn: bool = False,
    ) -> None:
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
        self.residual_bn = nn.BatchNorm1d(self.slot_size) if use_residual_bn else None

    @typechecked
    def forward(self, inputs: Float[torch.Tensor, "B N C"],
                slots: Float[torch.Tensor, "B K D"]) -> Float[torch.Tensor, "B K D"]:
        """`inputs`: [B, N, C] flattened per-pixel features,
        `slots`: [B, num_slots, C] slot inits.

        The last dims use distinct names (C vs D) so no accidental
        cross-argument equality is implied.
        """
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B, num_inputs, slot_size]
        v = self.project_v(inputs)  # [B, num_inputs, slot_size]

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
            slots = slots + self.mlp(slots)
            if self.residual_bn is not None:
                slots = self.residual_bn(slots.view(bs * self.num_slots, self.slot_size)).view(
                    bs, self.num_slots, self.slot_size
                )

        return slots

    @property
    def dtype(self) -> torch.dtype:
        return self.project_k.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.project_k.weight.device


# Backward-compatible alias pinned by tests/modular/test_savi_bn.py.
SlotAttentionWithBN = SlotAttention


# ── Slot transition predictors ────────────────────────────────────────────────

class TransformerPredictor(nn.Module):
    """Transformer encoder modeling interaction between slots."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        ffn_dim: int = 256,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x)


class RNNPredictorWrapper(nn.Module):
    """Predictor wrapped in an LSTM for sequential scene dynamics."""

    def __init__(
        self,
        base_predictor: nn.Module,
        input_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.base_predictor = base_predictor
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers)
        self.out_projector = nn.Linear(hidden_size, input_size)
        self.hidden_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_predictor(x)
        out_shape = out.shape
        # Re-pack cuDNN weights after optimizer steps (cheap no-op when fresh).
        self.rnn.flatten_parameters()
        out, self.hidden_state = self.rnn(
            out.view(1, -1, out_shape[-1]), self.hidden_state)
        return self.out_projector(out[0]).view(out_shape)

    def reset(self) -> None:
        """Clear the LSTM hidden state."""
        self.hidden_state = None


# ── Core SAVi model ───────────────────────────────────────────────────────────

class StoSAVi(nn.Module):
    """Deterministic Slot Attention for Video (native port of SlotFormer's StoSAVi)."""

    def __init__(
        self,
        resolution: tuple,
        clip_len: int,
        slot_dict: Optional[dict[str, Any]] = None,
        enc_dict: Optional[dict[str, Any]] = None,
        dec_dict: Optional[dict[str, Any]] = None,
        pred_dict: Optional[dict[str, Any]] = None,
        # accepted for config compatibility; unused (deterministic SAVi)
        loss_dict: Optional[dict[str, Any]] = None,
        use_encoder_bn: bool = False,
        use_residual_bn: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        slot_dict = dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
        ) | (slot_dict or {})
        enc_dict = dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
        ) | (enc_dict or {})
        dec_dict = dict(
            dec_channels=(128, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
        ) | (dec_dict or {})
        pred_dict = dict(
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
        ) | (pred_dict or {})

        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps
        self.use_encoder_bn = use_encoder_bn
        self.use_residual_bn = use_residual_bn

        self._build_slot_attention(slot_dict, enc_dict)
        self._build_encoder(enc_dict)
        self._build_decoder(dec_dict)
        self._build_predictor(pred_dict)

    def _build_slot_attention(self, slot_dict: dict[str, Any],
                              enc_dict: dict[str, Any]) -> None:
        self.enc_out_channels = enc_dict['enc_out_channels']
        self.num_slots = slot_dict['num_slots']
        self.slot_size = slot_dict['slot_size']
        self.slot_mlp_size = slot_dict['slot_mlp_size']
        self.num_iterations = slot_dict['num_iterations']

        # Learnable per-slot initialization.
        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))

        # Predicts the (mu, log-var) of the SA input "kernels"; the
        # deterministic model samples mu only (see `_sample_dist`).
        self.kernel_dist_layer = nn.Sequential(
            nn.Linear(self.slot_size, self.slot_size * 2),
            nn.LayerNorm(self.slot_size * 2),
            nn.ReLU(),
            nn.Linear(self.slot_size * 2, self.slot_size * 2),
        )

        # Unused; kept for state-dict compatibility with pre-trained weights.
        self.prior_slot_layer = nn.Sequential(
            nn.Linear(self.slot_size, self.slot_size),
            nn.LayerNorm(self.slot_size),
            nn.ReLU(),
            nn.Linear(self.slot_size, self.slot_size),
        )

        self.slot_attention = SlotAttention(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
            use_residual_bn=self.use_residual_bn,
        )

    def _build_encoder(self, enc_dict: dict[str, Any]) -> None:
        self.enc_channels = list(enc_dict['enc_channels'])  # CNN channels
        self.enc_ks = enc_dict['enc_ks']  # kernel size in CNN
        self.visual_resolution = (64, 64)  # CNN out visual resolution
        self.visual_channels = self.enc_channels[-1]  # CNN out visual channels

        enc_layers = len(self.enc_channels) - 1
        self.encoder = nn.Sequential(*[
            conv_norm_act(
                self.enc_channels[i],
                self.enc_channels[i + 1],
                kernel_size=self.enc_ks,
                # 2x downsampling when training on 128x128 images
                stride=2 if (i == 0 and self.resolution[0] == 128) else 1,
                act='relu' if i != (enc_layers - 1) else '',
            )
            for i in range(enc_layers)
        ])

        self.encoder_pos_embedding = SoftPositionEmbed(self.visual_channels,
                                                       self.visual_resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.enc_out_channels),
            nn.ReLU(),
            nn.Linear(self.enc_out_channels, self.enc_out_channels),
        )
        self.encoder_bn = nn.BatchNorm1d(self.enc_out_channels) if self.use_encoder_bn else None

    def _build_decoder(self, dec_dict: dict[str, Any]) -> None:
        self.dec_channels = dec_dict['dec_channels']  # CNN channels
        self.dec_resolution = dec_dict['dec_resolution']  # broadcast size
        self.dec_ks = dec_dict['dec_ks']  # kernel size
        assert self.dec_channels[0] == self.slot_size, \
            'wrong in_channels for Decoder'

        modules = []
        out_size = self.dec_resolution[0]
        stride = 2
        for i in range(len(self.dec_channels) - 1):
            if out_size == self.resolution[0]:
                stride = 1
            modules.append(
                deconv_norm_act(
                    self.dec_channels[i],
                    self.dec_channels[i + 1],
                    kernel_size=self.dec_ks,
                    stride=stride,
                    act='relu'))
            out_size = deconv_out_shape(out_size, stride, self.dec_ks // 2,
                                        self.dec_ks, stride - 1)

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )

        # Output conv for RGB and segmentation mask.
        modules.append(
            nn.Conv2d(
                self.dec_channels[-1], 4, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.slot_size,
                                                       self.dec_resolution)

    def _build_predictor(self, pred_dict: dict[str, Any]) -> None:
        """Predictor transitioning slots from time t to t+1:
        Transformer (object interaction) wrapped in LSTM (scene dynamics)."""
        self.predictor = TransformerPredictor(
            self.slot_size,
            pred_dict['pred_num_layers'],
            pred_dict['pred_num_heads'],
            pred_dict['pred_ffn_dim'],
            norm_first=pred_dict['pred_norm_first'],
        )
        if pred_dict['pred_rnn']:
            self.predictor = RNNPredictorWrapper(
                self.predictor,
                self.slot_size,
                self.slot_mlp_size,
                num_layers=1,
            )

    def _sample_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """Deterministic sampling: return the mean half of (mu, log-var)."""
        assert dist.shape[-1] == self.slot_size * 2
        return dist[..., :self.slot_size]

    def _get_encoder_out(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image, add pos embed, project, optionally BatchNorm.

        `img`: [B, C, H, W] -> [B, H*W, enc_out_channels].
        """
        encoder_out = self.encoder(img).type(self.dtype)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        encoder_out = encoder_out.permute(0, 2, 1).contiguous()
        encoder_out = self.encoder_out_layer(encoder_out)
        if self.encoder_bn is not None:
            B, HW, C = encoder_out.shape
            encoder_out = self.encoder_bn(encoder_out.view(B * HW, C)).view(B, HW, C)
        return encoder_out

    @typechecked
    def encode(self, img: Float[torch.Tensor, "B T C H W"],
               prev_slots: Optional[Float[torch.Tensor, "B K D"]] = None) -> tuple[
                   Float[torch.Tensor, "B T K D"],
                   Float[torch.Tensor, "B T HW EncOutC"],
               ]:
        """Encode a clip to post-slots.

        Returns (post_slots [B, T, num_slots, slot_size],
                 encoder_out [B, T, H*W, enc_out_channels]).
        """
        B, T, C, H, W = img.shape
        encoder_out = self._get_encoder_out(img.flatten(0, 1))
        encoder_out = encoder_out.unflatten(0, (B, T))

        # Apply slot attention per frame, reusing slots across time. The
        # predictor is called exactly once per slot transition — no trailing
        # call — so chained encodes (rollout.py's two-phase pattern) evolve
        # the RNN identically to a single full-clip encode.
        if prev_slots is None:
            latents = self.init_latents.expand(B, -1, -1)
        all_post_slots = []
        for idx in range(T):
            if prev_slots is not None:
                latents = self.predictor(prev_slots)
            kernels = self._sample_dist(self.kernel_dist_layer(latents))
            post_slots = self.slot_attention(encoder_out[:, idx], kernels)
            all_post_slots.append(post_slots)
            prev_slots = post_slots

        return torch.stack(all_post_slots, dim=1), encoder_out

    def _reset_rnn(self) -> None:
        self.predictor.reset()

    def forward(self, data_dict: dict) -> dict:
        """Forward pass. `data_dict['img']`: [B, T, C, H, W]."""
        if not isinstance(data_dict, dict):
            raise TypeError(f"data_dict must be a dict, got {type(data_dict).__name__}")
        if "img" not in data_dict:
            raise ValueError("data_dict must contain the 'img' key")
        return self._forward(data_dict['img'], None)

    def _forward(self, img: torch.Tensor,
                 prev_slots: Optional[torch.Tensor] = None) -> dict:
        if prev_slots is None:
            self._reset_rnn()

        B, T = img.shape[:2]
        post_slots, encoder_out = self.encode(img, prev_slots=prev_slots)

        out_dict = {
            'post_slots': post_slots,  # [B, T, num_slots, C]
            'img': img,  # [B, T, 3, H, W]
        }
        post_recon_img, post_recons, post_masks, _ = \
            self.decode(post_slots.flatten(0, 1))
        out_dict.update({
            'post_recon_combined': post_recon_img.unflatten(0, (B, T)),
            'post_recons': post_recons.unflatten(0, (B, T)),
            'post_masks': post_masks.unflatten(0, (B, T)),
        })
        return out_dict

    @typechecked
    def decode(self, slots: Float[torch.Tensor, "B K D"]) -> tuple[
        Float[torch.Tensor, "B 3 H W"],
        Float[torch.Tensor, "B K 3 H W"],
        Float[torch.Tensor, "B K 1 H W"],
        Float[torch.Tensor, "B K D"],
    ]:
        """Decode slots to reconstructed images and masks.

        `slots`: [B, num_slots, slot_size].
        """
        bs, num_slots, slot_size = slots.shape
        height, width = self.resolution
        num_channels = 3

        # Spatial broadcast.
        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                       self.dec_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [B*num_slots, 4, H, W].

        out = out.view(bs, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, 3, H, W]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, 3, H, W]
        return recon_combined, recons, masks, slots

    def encode_slots(self, video: torch.Tensor) -> torch.Tensor:
        """Extract per-frame slots [B, T, K, D] for video [B, T, C, H, W]."""
        if hasattr(self, "_reset_rnn"):
            self._reset_rnn()
        post_slots, _ = self.encode(video)
        return post_slots

    def decode_slots(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode slots [B, T, K, D] or [B, K, D] to (recon_img, pred_masks).
        """
        is_5d = (slots.ndim == 4)
        if is_5d:
            B, T, K, D = slots.shape
            slots_flat = slots.flatten(0, 1)
        else:
            B, K, D = slots.shape
            slots_flat = slots

        recon_combined, _, masks, _ = self.decode(slots_flat)

        if is_5d:
            recon_img = recon_combined.unflatten(0, (B, T))
            pred_masks = masks.squeeze(2).unflatten(0, (B, T))
        else:
            recon_img = recon_combined
            pred_masks = masks.squeeze(2)

        return recon_img, pred_masks

    @property
    def dtype(self) -> torch.dtype:
        return self.init_latents.dtype

    @property
    def device(self) -> torch.device:
        return self.init_latents.device


# ── Public wrapper ────────────────────────────────────────────────────────────

class SAVi(nn.Module):
    """
    Standard deterministic SAVi model wrapper.

    Thin wrapper around the native :class:`StoSAVi` core. Accepts flat kwargs
    or nested dicts for the slot/enc/dec/pred configs — flat kwargs act as
    defaults, nested dicts take precedence when provided.
    """

    def __init__(
        self,
        resolution: tuple = (64, 64),
        clip_len: int = 6,
        num_slots: int = 4,
        slot_dim: int = 64,
        num_iterations: int = 3,
        in_channels: int = 3,
        use_encoder_bn: bool = False,
        use_residual_bn: bool = False,
        slot_dict: Optional[dict[str, Any]] = None,
        enc_dict: Optional[dict[str, Any]] = None,
        dec_dict: Optional[dict[str, Any]] = None,
        pred_dict: Optional[dict[str, Any]] = None,
        loss_dict: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.resolution = tuple(resolution)
        self.use_encoder_bn = use_encoder_bn
        self.use_residual_bn = use_residual_bn

        slot_dict = dict(
            num_slots=num_slots,
            slot_size=slot_dim,
            slot_mlp_size=slot_dim * 2,
            num_iterations=num_iterations,
        ) | (slot_dict or {})

        enc_dict = dict(
            enc_channels=(in_channels, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=slot_dim,
        ) | (enc_dict or {})

        dec_dict = dict(
            dec_channels=(slot_dim, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
        ) | (dec_dict or {})

        pred_dict = dict(
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=256,
        ) | (pred_dict or {})

        self.model = StoSAVi(
            resolution=self.resolution,
            clip_len=clip_len,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            pred_dict=pred_dict,
            loss_dict=loss_dict,
            use_encoder_bn=use_encoder_bn,
            use_residual_bn=use_residual_bn,
        )

    @property
    def encoder_bn(self) -> Optional[nn.BatchNorm1d]:
        return self.model.encoder_bn

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def load_state_dict(self, state_dict: dict[str, torch.Tensor],
                        strict: bool = True) -> Any:
        """Handle state dicts for both the wrapper ('model.'-prefixed keys)
        and the bare core model."""
        if any(k.startswith('model.') for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict=strict)
        return self.model.load_state_dict(state_dict, strict=strict)

    def encode_slots(self, video: torch.Tensor) -> torch.Tensor:
        """Extract per-frame slots [B, T, K, D] for video [B, T, C, H, W]."""
        return self.model.encode_slots(video)

    def decode_slots(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode slots to (recon_img, pred_masks)."""
        return self.model.decode_slots(slots)

    def forward(self, x: Union[torch.Tensor, dict], **kwargs: Any) -> dict:
        """Forward pass. Accepts tensor [B, T, C, H, W] or dict {'img': ...}."""
        if isinstance(x, torch.Tensor):
            check_tensor_shape(x, "x", ndim=5)
            x = {'img': x}
        return self.model(x)
