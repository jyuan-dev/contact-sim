"""
src.config — the configuration seam.

One module owns validation, canonicalization, and snapshot I/O. The interface
is one value type and a snapshot entry point:

    RunConfig.from_dict(cfg, permissive=False)   # validate + canonicalize
    load_snapshot(ckpt_dir) -> RunConfig | None  # checkpoint bootstrap
    RunConfig.to_dict()                          # the legacy plain-dict shape

Consumers (factories, wrappers, TrainConfig) keep their plain-dict contract:
``to_dict()`` emits exactly the resolved shape they expect. Hydra composition
stays in the scripts (train.py keeps ``@hydra.main`` for sweeper support);
scripts never import Hydra except to hand this module the composed dict.

Canonicalization rules (single declaration per fact):
- ``num_slots`` is declared once; ``slot_dict.num_slots`` is derived on
  emission. Authoring both with conflicting values is a ConfigError.
- ``slot_size`` is declared as ``slot_dim`` (SAVi family); ``slot_dict.slot_size``
  is derived. Authoring conflicting values is a ConfigError.
- ``dataset.resolution`` defaults from ``model.resolution``; an explicit
  conflict is a ConfigError.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, cast

from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfigError(ValueError):
    """Configuration is invalid; the message carries the offending key."""


# ── Sections ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainSection:
    exp_name: str = "savi_pusht"
    seed: int = 42
    epochs: int = 8
    batch_size: int = 128
    num_workers: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    scheduler: Optional[str] = None
    warmup_steps: Optional[int] = None
    min_lr: Optional[float] = None
    dry_run: bool = False
    max_steps: Optional[int] = None
    ckpt_path: Optional[str] = None
    use_amp: bool = True
    device: str = "cuda"
    mode: str = "deterministic"
    clips_per_ep: int = 2
    use_wandb: bool = True
    wandb_project: str = "pusht-contact-sim"
    auto_dvc: bool = True


@dataclass(frozen=True)
class ModelSection:
    name: str = "savi"
    type: str = "savi"
    resolution: tuple = (64, 64)
    n_sample_frames: int = 6
    num_slots: int = 4
    slot_size: Optional[int] = None  # declared as `slot_dim` or slot_dict.slot_size
    num_iterations: int = 3
    in_channels: int = 3
    use_encoder_bn: bool = False
    use_residual_bn: bool = False
    _has_slot_dim: bool = False  # provenance: was `slot_dim` authored at model level?
    _emit_core: frozenset = frozenset()  # which savi-core keys the source authored
    extra: dict = field(default_factory=dict)  # family-specific keys (stage1_ckpt_path, d_model, ...)


@dataclass(frozen=True)
class DatasetSection:
    name: str = "pusht"
    type: str = "pusht"
    resolution: tuple = (64, 64)
    n_sample_frames: int = 6
    train_frac: float = 0.8
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)  # h5_path, load_masks, gridshapes keys, ...


@dataclass(frozen=True)
class RunConfig:
    train: TrainSection
    model: ModelSection
    dataset: DatasetSection
    loss: dict  # opaque; the `_target_` structure round-trips verbatim
    extra: dict = field(default_factory=dict)

    # ── Construction ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any], *, permissive: bool = False) -> "RunConfig":
        """Validate + canonicalize a resolved plain dict.

        ``permissive=True`` (legacy snapshots): unknown keys are preserved in
        ``extra`` instead of raising, missing sections get schema defaults.
        """
        if not isinstance(cfg, Mapping):
            raise ConfigError(f"config must be a dict, got {type(cfg).__name__}")

        train_names = {f.name for f in fields(TrainSection)}
        model_names = {f.name for f in fields(ModelSection)} - {"extra", "_has_slot_dim", "_emit_core"}
        dataset_names = {f.name for f in fields(DatasetSection)} - {"extra"}

        unknown = [k for k in cfg if k not in train_names
                   and k not in ("model", "dataset", "loss")]
        if unknown and not permissive:
            _reject_if_typo(unknown[0], list(train_names) + ["model", "dataset", "loss"],
                            "top-level")
        extra = {k: cfg[k] for k in unknown}

        train_kwargs = {}
        for f in fields(TrainSection):
            if f.name in cfg:
                train_kwargs[f.name] = _coerce(cfg[f.name], f.name, f.type)
        train = TrainSection(**train_kwargs)

        model_raw = cfg.get("model")
        if model_raw is None and not permissive:
            raise ConfigError("config has no 'model' section")
        model = cls._build_model_section(model_raw or {}, permissive, model_names)

        dataset_raw = cfg.get("dataset")
        if dataset_raw is None and not permissive:
            raise ConfigError("config has no 'dataset' section")
        dataset = cls._build_dataset_section(dataset_raw or {}, permissive,
                                             dataset_names, model)

        loss = cfg.get("loss", {})
        if loss is None:
            loss = {}
        if not isinstance(loss, Mapping):
            raise ConfigError(f"config['loss'] must be a dict, got {type(loss).__name__}")

        return cls(train=train, model=model, dataset=dataset, loss=dict(loss), extra=extra)

    @classmethod
    def _build_model_section(cls, raw: Mapping[str, Any], permissive: bool,
                             names: set[str]) -> "ModelSection":
        if not isinstance(raw, Mapping):
            raise ConfigError(f"config['model'] must be a dict, got {type(raw).__name__}")
        unknown = [k for k in raw if k not in names
                   and k not in ("slot_dict", "enc_dict", "dec_dict", "pred_dict", "loss_dict")
                   and k != "slot_dim"]
        if unknown and not permissive:
            _reject_if_typo(unknown[0], list(names), "model section")
        extra = {k: v for k, v in raw.items()
                 if k not in names and k != "slot_dict" and k != "slot_dim"}

        kwargs = {}
        for name in names:
            if name in raw:
                kwargs[name] = _coerce(raw[name], f"model.{name}", None)
        # resolution may be a list in YAML
        if "resolution" in kwargs and isinstance(kwargs["resolution"], (list, tuple)):
            kwargs["resolution"] = (int(kwargs["resolution"][0]), int(kwargs["resolution"][1]))

        # slot_size: declared as `slot_dim` or as slot_dict.slot_size
        slot_size = None
        has_slot_dim = "slot_dim" in raw
        if has_slot_dim:
            slot_size = int(raw["slot_dim"])
        slot_dict = raw.get("slot_dict")
        if isinstance(slot_dict, Mapping):
            if "slot_size" in slot_dict:
                declared = int(slot_dict["slot_size"])
                if slot_size is not None and slot_size != declared:
                    raise ConfigError(
                        f"conflicting slot_size: model.slot_dim={slot_size} vs "
                        f"model.slot_dict.slot_size={declared}")
                slot_size = declared
            if "num_slots" in slot_dict:
                declared = int(slot_dict["num_slots"])
                if "num_slots" in kwargs and kwargs["num_slots"] != declared:
                    raise ConfigError(
                        f"conflicting num_slots: model.num_slots={kwargs['num_slots']} vs "
                        f"model.slot_dict.num_slots={declared}")
                kwargs["num_slots"] = declared
        if slot_size is not None:
            kwargs["slot_size"] = slot_size

        if not kwargs.get("type") and not kwargs.get("name") and not permissive:
            raise ConfigError("model section has no 'type' or 'name'")

        # Savi-core keys are emitted only when authored, or when this is a
        # savi-family model (stage-2 configs never carry num_slots & co).
        CORE = ("name", "type", "resolution", "n_sample_frames", "num_slots",
                "num_iterations", "in_channels", "use_encoder_bn", "use_residual_bn")
        savi_family = "savi" in str(kwargs.get("type") or kwargs.get("name") or "").lower()
        emit_core = frozenset(CORE) if savi_family else frozenset(
            k for k in CORE if k in kwargs or k in raw)
        return ModelSection(extra=extra, _has_slot_dim=has_slot_dim,
                            _emit_core=emit_core, **kwargs)

    @classmethod
    def _build_dataset_section(cls, raw: Mapping[str, Any], permissive: bool,
                               names: set[str], model: "ModelSection") -> "DatasetSection":
        if not isinstance(raw, Mapping):
            raise ConfigError(f"config['dataset'] must be a dict, got {type(raw).__name__}")
        unknown = [k for k in raw if k not in names]
        if unknown and not permissive:
            _reject_if_typo(unknown[0], list(names), "dataset section")
        extra = {k: v for k, v in raw.items() if k not in names}

        kwargs = {}
        for name in names:
            if name in raw:
                kwargs[name] = _coerce(raw[name], f"dataset.{name}", None)
        if "resolution" in kwargs and isinstance(kwargs["resolution"], (list, tuple)):
            kwargs["resolution"] = (int(kwargs["resolution"][0]), int(kwargs["resolution"][1]))

        # shared key: dataset.resolution defaults from model.resolution
        if "resolution" not in kwargs:
            kwargs["resolution"] = model.resolution
        elif kwargs["resolution"] != model.resolution:
            raise ConfigError(
                f"conflicting resolution: model={model.resolution} vs "
                f"dataset={kwargs['resolution']}")

        if not kwargs.get("name") and not kwargs.get("type") and not permissive:
            raise ConfigError("dataset section has no 'name' or 'type'")
        return DatasetSection(extra=extra, **kwargs)

    # ── Emission (the legacy shape) ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Emit the resolved plain dict consumers expect (legacy shape)."""
        d: dict[str, Any] = {}
        for f in fields(TrainSection):
            # Emit every field including None — legacy snapshots carry
            # explicit nulls (ckpt_path: null, scheduler: null, ...).
            d[f.name] = getattr(self.train, f.name)

        model_d: dict[str, Any] = {
            "name": self.model.name,
            "type": self.model.type,
            "resolution": list(self.model.resolution),
            "n_sample_frames": self.model.n_sample_frames,
            "num_slots": self.model.num_slots,
            "num_iterations": self.model.num_iterations,
            "in_channels": self.model.in_channels,
            "use_encoder_bn": self.model.use_encoder_bn,
            "use_residual_bn": self.model.use_residual_bn,
        }
        # Derived slot emissions (single declaration -> both places):
        # slot_dim is re-emitted only when it was authored at model level;
        # slot_dict.num_slots / slot_size are always derived when known.
        if self.model.slot_size is not None:
            if self.model._has_slot_dim:
                model_d["slot_dim"] = self.model.slot_size
            slot_dict = dict(self.model.extra.get("slot_dict") or {})
            slot_dict["num_slots"] = self.model.num_slots
            slot_dict["slot_size"] = self.model.slot_size
            model_d["slot_dict"] = slot_dict
        for key, value in self.model.extra.items():
            if key != "slot_dict":
                model_d[key] = value
        # Drop savi-core keys the source never had (stage-2 model configs).
        CORE = ("name", "type", "resolution", "n_sample_frames", "num_slots",
                "num_iterations", "in_channels", "use_encoder_bn", "use_residual_bn")
        model_d = {k: v for k, v in model_d.items()
                   if k not in CORE or k in self.model._emit_core}
        d["model"] = model_d

        dataset_d: dict[str, Any] = {
            "name": self.dataset.name,
            "type": self.dataset.type,
            "resolution": list(self.dataset.resolution),
            "n_sample_frames": self.dataset.n_sample_frames,
            "train_frac": self.dataset.train_frac,
        }
        if self.dataset.seed is not None:
            dataset_d["seed"] = self.dataset.seed
        dataset_d.update(self.dataset.extra)
        d["dataset"] = dataset_d

        d["loss"] = dict(self.loss)
        d.update(self.extra)
        return d

    # ── Overrides / snapshots / paths ───────────────────────────────────────

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "RunConfig":
        """Return a new config with dotted-key overrides applied (CLI wins)."""
        d = self.to_dict()
        for key, value in overrides.items():
            _set_dotted(d, key, value)
        return RunConfig.from_dict(d, permissive=True)

    def save_snapshot(self, path: str) -> None:
        """Write the canonical resolved snapshot (plain YAML dict)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        OmegaConf.save(OmegaConf.create(self.to_dict()), path, resolve=True)

    @property
    def exp_dir(self) -> str:
        return os.path.join(REPO_ROOT, "scratch", "checkpoints", self.train.exp_name)

    def as_train_config(self, ckpt_dir: str, model_name: str):
        """Bridge to the training loop's plain TrainConfig dataclass."""
        from src.training.train_loop import TrainConfig
        return TrainConfig.from_cfg(self.to_dict(), ckpt_dir=ckpt_dir, model_name=model_name)


def assert_resume_compatible(snapshot: "RunConfig", live: "RunConfig") -> None:
    """Raise ConfigError when a checkpoint's topology cannot hold live weights."""
    checks = [
        ("model.type", snapshot.model.type, live.model.type),
        ("model.num_slots", snapshot.model.num_slots, live.model.num_slots),
        ("model.slot_size", snapshot.model.slot_size, live.model.slot_size),
        ("model.resolution", snapshot.model.resolution, live.model.resolution),
        ("dataset.name", snapshot.dataset.name, live.dataset.name),
    ]
    mismatches = [f"{key}: checkpoint={snap!r}, live={cur!r}"
                  for key, snap, cur in checks if snap != cur]
    if mismatches:
        raise ConfigError(
            "Resume topology mismatch between checkpoint snapshot and live config:\n  "
            + "\n  ".join(mismatches)
            + "\nFix: pass the experiment/overrides the checkpoint was trained with.")


def load_snapshot(ckpt_dir: str) -> Optional[RunConfig]:
    """Load the saved config from a checkpoint dir (config.yaml, then
    .hydra/config.yaml). Permissive: legacy keys are preserved in `extra`.
    Returns None when no snapshot exists (caller may fall back to sniffing).
    """
    candidates = [
        os.path.join(ckpt_dir, "config.yaml"),
        os.path.join(ckpt_dir, ".hydra", "config.yaml"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                saved_cfg = OmegaConf.load(cand)
            except Exception as e:
                raise ConfigError(f"Failed to load training config file from '{cand}': {e}") from e
            print(f"[bootstrap] Loaded training configuration from: {cand}")
            raw = cast(dict[str, Any], OmegaConf.to_container(saved_cfg, resolve=True))
            return RunConfig.from_dict(raw, permissive=True)
    return None


# ── helpers ──────────────────────────────────────────────────────────────────

def _suggest(key: str, known: list[str]) -> str:
    matches = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
    if matches:
        return f" (did you mean '{matches[0]}'?)"
    return ""


def _reject_if_typo(key: str, known: list[str], where: str) -> None:
    """Strict mode: reject only keys that are suspiciously close to a known
    field name (the typo case). Family-specific passthrough keys
    (d_model, stage1_ckpt_path, h5_path, ...) are not close to any core
    field and flow into `extra` untouched."""
    matches = difflib.get_close_matches(key, known, n=1, cutoff=0.5)
    if matches:
        raise ConfigError(f"unknown key '{key}' in {where} (did you mean '{matches[0]}'?)")


def _coerce(value: Any, name: str, field_type) -> Any:
    """Coerce CLI string values into the declared field type."""
    if value is None:
        return value
    target = getattr(field_type, "__origin__", field_type) if field_type else None
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ConfigError(f"{name}: expected bool, got {value!r}")
    if target is int and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}: expected int, got {value!r}") from None
    if target is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}: expected float, got {value!r}") from None
    return value


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
