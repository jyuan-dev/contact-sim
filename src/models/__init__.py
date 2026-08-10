"""
src.models — Model families for Contact-Sim / Slot-Worldmodel Baselines.

Public API
----------
build_model(cfg)        Build a registered model wrapper from a config dict.
register_model(name)    Decorator to register a new BaseModelWrapper subclass.
list_models()           List all registered model names.
BaseModelWrapper        ABC that all wrappers must inherit from.
ModelOutput             TypedDict defining the standardized forward output contract.

Wrapper classes:
  StandardizedSAViWrapper
  StandardizedDeformableSAViWrapper

Base model classes:
  SAVi
"""

from src.models.model_output import ModelOutput
from src.models.base import BaseModelWrapper
from src.models.factory import build_model, register_model, list_models
from src.models.savi import SAVi

from src.models.wrappers import (
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
    "SAVi",
    # Wrapper classes
    "StandardizedSAViWrapper",
    "StandardizedDeformableSAViWrapper",
]

