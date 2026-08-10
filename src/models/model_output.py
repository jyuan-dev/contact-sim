"""
Standardized Model Output Contract.

All model wrappers must return a ``ModelOutput`` (or compatible subtype) from
their ``forward()`` method.  Using ``TypedDict`` rather than a plain ``dict``
makes the contract explicit, enables IDE type-checking, and serves as
self-documenting code.

Key layout
----------
pred_boxes   : [B, Q, 4]  normalized (cx, cy, w, h) — detection models
pred_masks   : [B, T, K, H, W] soft slot attention masks — SAVi models
pred_logits  : [B, Q, num_classes + 1] class logits — detection models
recon_img    : [B, T, C, H, W] reconstructed images — SAVi models
input_img    : [B, T, C, H, W] passthrough input (always populated)
post_slots   : [B, T, K, D] slot embeddings — SAVi models

DETR-specific passthrough keys (for aux-loss computation in compute_loss):
pred_logits_all  : [B, L, Q, C] multi-layer logits stack
pred_boxes_all   : [B, L, Q, 4] multi-layer box stack
"""

from __future__ import annotations

from typing import Optional
try:
    from typing import Required, NotRequired
except ImportError:          # Python < 3.11
    from typing_extensions import Required, NotRequired

from torch import Tensor

# ---------------------------------------------------------------------------
# Base contract shared by all wrappers
# ---------------------------------------------------------------------------

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class ModelOutput(TypedDict, total=False):
    """
    Standardized forward-pass output dictionary.

    Keys marked as Required must always be present.
    All other keys are model-family specific and may be ``None``.
    """

    # Always present — even if the model doesn't produce detection outputs.
    input_img: Required[Tensor]         # [B, (T,) C, H, W]

    # Detection outputs (DETR family)
    pred_boxes: Optional[Tensor]        # [B, Q, 4]
    pred_logits: Optional[Tensor]       # [B, Q, num_classes + 1]

    # Segmentation / reconstruction outputs (SAVi family)
    pred_masks: Optional[Tensor]        # [B, T, K, H, W]
    recon_img: Optional[Tensor]         # [B, T, C, H, W]
    post_slots: Optional[Tensor]        # [B, T, K, D]

    # DETR-internal: full layer stack for auxiliary loss computation.
    # Not consumed by metrics or eval code — only by compute_loss().
    pred_logits_all: NotRequired[Optional[Tensor]]   # [B, L, Q, C]
    pred_boxes_all: NotRequired[Optional[Tensor]]    # [B, L, Q, 4]


