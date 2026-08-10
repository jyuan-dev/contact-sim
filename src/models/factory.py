"""
Model Factory for Contact-Sim / Slot-Worldmodel Baselines.

Provides:
  - ``register_model(name)``  — class decorator that registers a wrapper
  - ``build_model(cfg)``       — constructs a registered wrapper from config

Usage
-----
Register a new model family::

    from src.models.factory import register_model
    from src.models.base import BaseModelWrapper

    @register_model("my_model")
    class MyModelWrapper(BaseModelWrapper):
        @classmethod
        def build(cls, model_cfg: dict) -> "MyModelWrapper":
            ...

Then build via config::

    model = build_model(cfg)  # dispatches via cfg["model"]["type"]

The ``build_model`` call is intentionally kept import-side-effect-free until the
wrappers sub-package is imported.  The ``__init__`` of ``src.models.wrappers``
imports all wrappers which triggers their ``@register_model`` decorators.
"""

from __future__ import annotations

from typing import Type

from src.models.base import BaseModelWrapper

# ── Global registry ──────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, Type[BaseModelWrapper]] = {}


def register_model(name: str):
    """
    Class decorator that registers a ``BaseModelWrapper`` subclass under one or
    more names.  Accepts a single name or a list of aliases.

    Example::

        @register_model("savi")
        class StandardizedSAViWrapper(BaseModelWrapper):
            ...

        @register_model(["deformable_detr", "deformable-detr"])
        class StandardizedDeformableDETRWrapper(BaseModelWrapper):
            ...
    """
    names = [name] if isinstance(name, str) else list(name)

    def decorator(cls: Type[BaseModelWrapper]) -> Type[BaseModelWrapper]:
        if not issubclass(cls, BaseModelWrapper):
            raise TypeError(
                f"@register_model: '{cls.__name__}' must subclass BaseModelWrapper."
            )
        for n in names:
            key = n.lower().replace("-", "_")
            if key in _MODEL_REGISTRY:
                existing = _MODEL_REGISTRY[key]
                if existing is cls:
                    # Same class re-imported (e.g., due to Python module caching).
                    # This is benign — skip silently.
                    continue
                raise ValueError(
                    f"Model name '{key}' is already registered by "
                    f"'{existing.__name__}'. "
                    f"Use a unique name or alias."
                )
            _MODEL_REGISTRY[key] = cls
        return cls

    return decorator


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    _ensure_wrappers_imported()
    return sorted(_MODEL_REGISTRY.keys())


# ── Internal helpers ──────────────────────────────────────────────────────────

_wrappers_imported = False


def _ensure_wrappers_imported() -> None:
    """
    Import the wrappers sub-package on first use to trigger all
    ``@register_model`` decorators without requiring callers to import them
    explicitly.
    """
    global _wrappers_imported
    if not _wrappers_imported:
        import src.models.wrappers  # noqa: F401  — side-effect: registers all wrappers
        _wrappers_imported = True


def _resolve_cfg(cfg) -> dict:
    """Resolve Hydra DictConfig to a plain Python dict if needed."""
    try:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return cfg


# ── Public API ────────────────────────────────────────────────────────────────

def build_model(cfg) -> BaseModelWrapper:
    """
    Construct a model wrapper registered under the config's ``model.type`` key.

    Args:
        cfg (dict | DictConfig): Full experiment config (Hydra-style).
            Must contain a ``model`` sub-dict with at least a ``type`` (or
            ``name``) key identifying the model family.

    Returns:
        An initialized ``BaseModelWrapper`` instance.

    Raises:
        ValueError: If the model type is not in the registry.
    """
    _ensure_wrappers_imported()

    cfg = _resolve_cfg(cfg)
    model_cfg: dict = cfg.get("model", cfg)

    model_type: str = (
        model_cfg.get("type", model_cfg.get("name", "")) or ""
    ).lower().replace("-", "_")

    if not model_type:
        raise ValueError(
            "Could not determine model type from config. "
            "Ensure cfg['model']['type'] (or cfg['model']['name']) is set."
        )

    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(f"'{k}'" for k in sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Registered models: {available}."
        )

    cls = _MODEL_REGISTRY[model_type]

    # Instantiate loss module from Hydra loss config if present
    loss_cfg = cfg.get("loss")
    if loss_cfg is not None:
        from src.losses import build_loss

        loss_fn = build_loss(loss_cfg)
        model_cfg = dict(model_cfg)
        model_cfg["loss_fn"] = loss_fn
        # Merge only actual loss weight values (skip Hydra metadata keys)
        if isinstance(loss_cfg, dict):
            loss_weights = {k: v for k, v in loss_cfg.items()
                            if not k.startswith('_') and isinstance(v, (int, float))}
            if loss_weights:
                weight_dict = dict(model_cfg.get("weight_dict") or {})
                weight_dict.update(loss_weights)
                model_cfg["weight_dict"] = weight_dict

    return cls.build(model_cfg)


