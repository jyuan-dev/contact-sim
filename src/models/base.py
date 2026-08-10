"""
BaseModelWrapper — Abstract Base Class for all Contact-Sim model wrappers.

Every model family (DETR, SAVi, DeformableSAVi, …) must:

1. Subclass ``BaseModelWrapper``.
2. Implement ``forward(x) -> ModelOutput``.
3. Implement ``compute_loss(out, batch) -> tuple[Tensor, dict]``.
4. Implement the ``build(model_cfg)`` classmethod that self-constructs from a
   plain-dict model config section (already resolved from Hydra / OmegaConf).

This makes adding new model families trivial:

    @register_model("my_new_model")
    class MyNewModelWrapper(BaseModelWrapper):

        @classmethod
        def build(cls, model_cfg: dict) -> "MyNewModelWrapper":
            base = MyNewModel(...)
            return cls(base)

        def forward(self, x) -> ModelOutput:
            ...

        def compute_loss(self, out: ModelOutput, batch: dict):
            ...

The ``@register_model`` decorator (from ``src.models.factory``) wires the
class into the global registry so that ``build_model(cfg)`` can find it.
"""

from __future__ import annotations

import abc
from typing import Any

import torch.nn as nn
from torch import Tensor

from src.models.model_output import ModelOutput


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
                    "weight_dict": {"recon": 1.0, "mask": 1.0},
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
