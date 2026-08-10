"""
src.models — Model families for Contact-Sim / Slot-Worldmodel Baselines.

Public API
----------
build_model(cfg)        Build a registered model wrapper from a config dict.
register_model(name)    Decorator to register a new BaseModelWrapper subclass.
list_models()           List all registered model names.
BaseModelWrapper        ABC that all wrappers must inherit from.
ModelOutput             TypedDict defining the standardized forward output contract.

Wrapper classes (importable directly after build_model or list_models triggers
registration via _ensure_wrappers_imported):
  StandardizedDETRWrapper
  StandardizedDeformableDETRWrapper
  StandardizedSAViWrapper
  StandardizedDeformableSAViWrapper

Base model classes (for direct instantiation / testing):
  DETR
  SAVi
"""

from src.models.model_output import ModelOutput
from src.models.base import BaseModelWrapper
from src.models.factory import build_model, register_model, list_models
from src.models.detr import DETR
from src.models.savi import SAVi

# Trigger wrapper registration lazily via the factory's _ensure_wrappers_imported().
# Direct imports here would cause double-registration if factory.py is used before
# this __init__ runs.  Instead, import wrapper classes individually after ensuring
# they're registered.
from src.models.factory import _ensure_wrappers_imported as _load_wrappers
_load_wrappers()

from src.models.wrappers import (
    StandardizedDETRWrapper,
    StandardizedDeformableDETRWrapper,
    StandardizedSAViWrapper,
    StandardizedDeformableSAViWrapper,
)

__all__ = [
    # Core public API
    "build_model",
    "register_model",
    "list_models",
    "BaseModelWrapper",
    "ModelOutput",
    # Base model classes
    "DETR",
    "SAVi",
    # Wrapper classes
    "StandardizedDETRWrapper",
    "StandardizedDeformableDETRWrapper",
    "StandardizedSAViWrapper",
    "StandardizedDeformableSAViWrapper",
]
