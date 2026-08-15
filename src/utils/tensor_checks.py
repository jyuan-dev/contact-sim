"""
Input validation for public model seams.

Two entry points, one contract (bad input raises ValueError naming the
problem):

- :data:`typechecked` — beartype+jaxtyping decorator enforcing shape
  annotations (``Float[Tensor, "B T C H W"]``) at runtime, configured so
  violations raise ``ValueError`` (the train loop's skip-batch recovery
  catches it). Use for purely per-argument shape contracts.
- :func:`check_tensor_shape` — imperative call for checks an annotation
  cannot express (cross-argument relations, conditional shapes, value
  bounds, dict keys).
"""

from typing import Optional

import torch
from beartype import beartype, BeartypeConf

typechecked = beartype(conf=BeartypeConf(
    violation_param_type=ValueError,
    violation_return_type=ValueError,
))


def check_tensor_shape(x, name: str, ndim: Optional[int] = None,
                       shape: Optional[tuple] = None) -> None:
    """
    Validate that ``x`` is a torch.Tensor with the expected rank and shape.

    Args:
        x: value to validate.
        name: caller-facing name of the argument (used in error messages).
        ndim: expected rank, or None to skip the rank check.
        shape: expected shape tuple; None entries are wildcards, or None to
            skip the shape check.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-dimensional, got shape {tuple(x.shape)}")
    if shape is not None:
        if x.ndim != len(shape):
            raise ValueError(
                f"{name} must be {len(shape)}-dimensional (shape {shape}), "
                f"got shape {tuple(x.shape)}")
        for dim, expected in zip(x.shape, shape):
            if expected is not None and dim != expected:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(x.shape)}")


def check_dict_key(data, name: str, key: str) -> None:
    """Validate that ``data`` is a dict containing ``key``."""
    if not isinstance(data, dict):
        raise TypeError(f"{name} must be a dict, got {type(data).__name__}")
    if key not in data:
        raise ValueError(f"{name} must contain the '{key}' key")
