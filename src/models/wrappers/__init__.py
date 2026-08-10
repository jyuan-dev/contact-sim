"""
src.models.wrappers — Standardized model wrapper implementations.

Importing this package triggers all ``@register_model`` decorators, populating
the global registry in ``src.models.factory``.

Available wrappers (registered names):
  - "savi", "stosavi"  → StandardizedSAViWrapper
  - "deformable_savi"  → StandardizedDeformableSAViWrapper
"""

from src.models.wrappers.savi_wrapper import StandardizedSAViWrapper
from src.models.wrappers.deformable_savi_wrapper import StandardizedDeformableSAViWrapper

__all__ = [
    "StandardizedSAViWrapper",
    "StandardizedDeformableSAViWrapper",
]

