"""
src.models.wrappers — Standardized model wrapper implementations.

Importing this package triggers all ``@register_model`` decorators, populating
the global registry in ``src.models.factory``.

Available wrappers (registered names):
  - "savi", "stosavi"        → StandardizedSAViWrapper
  - "slotformer", "slot_former" → StandardizedSlotFormerWrapper
"""

from src.models.wrappers.savi_wrapper import (
    StandardizedSAViWrapper,
)
from src.models.wrappers.slotformer_wrapper import (
    StandardizedSlotFormerWrapper,
)
from src.models.wrappers.lewm_wrapper import (
    StandardizedLeWMWrapper,
)

__all__ = [
    "StandardizedSAViWrapper",
    "StandardizedSlotFormerWrapper",
    "StandardizedLeWMWrapper",
]
