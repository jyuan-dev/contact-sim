"""
Input validation for public model seams.

Two entry points:

- :data:`typechecked` — jaxtyping + beartype decorator enforcing shape and type
  annotations (e.g. ``Float[Tensor, "B T C H W"]``) at runtime, validating both
  individual argument shapes and cross-argument dimension equality. Violations
  raise :class:`jaxtyping.TypeCheckError` (subclass of ``TypeError``).
- :func:`check_tensor_shape` — imperative call for checks an annotation
  cannot express (conditional shapes, dynamic bounds, dict keys, or non-decorated callers).
"""

from typing import Optional

import torch
from beartype import beartype, BeartypeConf
from jaxtyping import jaxtyped

# jaxtyped wraps beartype to provide full runtime shape checking and named axis consistency
typechecked = jaxtyped(typechecker=beartype(conf=BeartypeConf(
    violation_param_type=TypeError,
    violation_return_type=TypeError,
)))


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
