"""
BaseModelWrapper + ModelOutput contract — foundation for all Contact-Sim wrappers.

Every model family must:
1. Subclass ``BaseModelWrapper``.
2. Implement ``forward(x) -> ModelOutput``.
3. Implement ``compute_loss(out, batch) -> tuple[Tensor, dict]``.
4. Implement the ``build(model_cfg)`` classmethod.

Usage::

    @register_model("my_new_model")
    class MyNewModelWrapper(BaseModelWrapper):
        @classmethod
        def build(cls, model_cfg: dict) -> "MyNewModelWrapper":
            base = MyNewModel(...)
            return cls(base)

        def forward(self, x) -> ModelOutput: ...
        def compute_loss(self, out: ModelOutput, batch: dict): ...
"""

from __future__ import annotations

import abc
from typing import Any, Optional

from typing_extensions import NotRequired, Required

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from jaxtyping import Float
import torch.nn as nn
from torch import Tensor


# ── Output contract ────────────────────────────────────────────────────────────

class ModelOutput(TypedDict, total=False):
    """Standardized forward-pass output dictionary.

    Keys marked Required must always be present.  Other keys are model-family
    specific and may be ``None``.
    """

    # Always present
    input_img: Required[Float[Tensor, "B ..."]]         # [B, (T,) C, H, W]

    # Detection outputs (DETR family)
    pred_boxes: Optional[Float[Tensor, "B Q 4"]]        # [B, Q, 4]
    pred_logits: Optional[Float[Tensor, "B Q NumClassesPlus1"]]  # [B, Q, num_classes + 1]

    # Segmentation / reconstruction outputs (SAVi family) — one shape per key;
    # the wrapper normalizes, consumers take these keys strictly.
    pred_masks: Optional[Float[Tensor, "B T K H W"]]    # [B, T, K, H, W] (always 5-D, squeezed)
    recon_img: Optional[Float[Tensor, "B T C H W"]]     # [B, T, C, H, W]
    post_slots: Optional[Float[Tensor, "B T K D"]]      # [B, T, K, D]

    # DETR-internal: full layer stack for auxiliary loss computation
    pred_logits_all: NotRequired[Optional[Float[Tensor, "L B Q NumClassesPlus1"]]]
    pred_boxes_all: NotRequired[Optional[Float[Tensor, "L B Q 4"]]]

    # Rollout metadata
    is_rollout_mask: NotRequired[Optional[Tensor]]      # [T] boolean tensor

    # Extra model-specific tensors / diagnostics
    extra: NotRequired[Optional[dict[str, Any]]]


# ── Abstract base class ────────────────────────────────────────────────────────

class BaseModelWrapper(abc.ABC, nn.Module):
    """
    Abstract base class for all standardized model wrappers.

    Subclasses must implement:
      - ``build(cls, model_cfg)``  — classmethod factory
      - ``forward(self, x)``       — returns a ``ModelOutput``-compatible dict
      - ``compute_loss(self, out, batch)`` — returns (total_loss, loss_dict)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def build(cls, model_cfg: dict) -> "BaseModelWrapper":
        """
        Construct a fully initialized wrapper instance from a model config dict.

        ``model_cfg`` is the ``model`` sub-dict of the resolved Hydra config
        (plain Python dict, not DictConfig).  All hyper-parameters should be
        read from here — no hardcoded magic numbers in implementations.

        Args:
            model_cfg (dict): Model-specific config section, e.g.::

                {
                    "type": "savi",
                    "num_slots": 4,
                    "slot_dim": 64,
                    ...
                }

        Returns:
            An initialized ``BaseModelWrapper`` instance ready to be moved to
            a device with ``.to(device)``.
        """
        ...

    @abc.abstractmethod
    def forward(self, x: Any) -> ModelOutput:
        """
        Run the model forward pass.

        Args:
            x: Input tensor(s).  Concrete type depends on the model family:
               - Detection models: ``Tensor [B, C, H, W]`` or ``[B, T, C, H, W]``
               - Slot-Attention models: ``Tensor [B, T, C, H, W]`` or
                 ``dict {"img": Tensor, ...}``

        Returns:
            A ``ModelOutput``-compatible dict.  The ``"input_img"`` key is
            always required; other keys may be ``None`` when not applicable.
        """
        ...

    @abc.abstractmethod
    def compute_loss(
        self,
        out: ModelOutput,
        batch: dict,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the training loss from a forward-pass output and the batch.

        Args:
            out (ModelOutput): The dict returned by ``forward()``.
            batch (dict):      Raw data batch from the DataLoader, containing
                               ``"img"``, ``"gt_masks"`` (if available), etc.

        Returns:
            (total_loss, loss_dict) where:
              - ``total_loss`` is a scalar Tensor (already weighted sum)
              - ``loss_dict``  is a plain ``dict[str, float]`` with individual
                loss terms (e.g. ``{"recon_loss": 0.42, "mask_bce": 0.11}``)
        """
        ...

    def rollout(
        self,
        video: Float[Tensor, "B T C H W"],
        n_cond_frames: int = 2,
        actions: Float[Tensor, "B Tact ActDim"] | None = None,
        goal_slots: Float[Tensor, "B K D"] | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """
        Autoregressively rollout future states given initial context frames.

        Args:
            video: Video tensor [B, T, C, H, W].
            n_cond_frames: Number of initial context frames to condition on.
            actions: Action sequence for action-conditioned models [B, T_act, ActDim].
            goal_slots: Target goal slots for goal-conditioned models [B, K, D].

        Returns:
            A ModelOutput dict containing rollout predictions.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement rollout()")
