"""
Standardized Deformable SAVi wrapper.

Wraps ``DeformableSAVi`` (SAVi with 2-D deformable slot attention) to conform
to the ``BaseModelWrapper`` interface.  Inherits ``forward()`` and
``compute_loss()`` from ``StandardizedSAViWrapper`` — only ``build()`` differs.

Registered under: ``"deformable_savi"``
"""

from __future__ import annotations

from src.models.factory import register_model
from src.models.wrappers.savi_wrapper import StandardizedSAViWrapper


@register_model(["deformable_savi", "deformable-savi"])
class StandardizedDeformableSAViWrapper(StandardizedSAViWrapper):
    """
    Deformable SAVi wrapper conforming to ``BaseModelWrapper``.

    Inherits ``forward()`` and ``compute_loss()`` from
    ``StandardizedSAViWrapper`` — the output contract and loss computation are
    identical.  Only the underlying base model differs.
    """

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedDeformableSAViWrapper":
        """Construct from model config sub-dict."""
        from src.models.deformable_savi import DeformableSAVi

        res = tuple(model_cfg.get("resolution", [64, 64]))
        clip_len = model_cfg.get("n_sample_frames", model_cfg.get("clip_len", 6))

        base_model = DeformableSAVi(
            resolution=res,
            clip_len=clip_len,
            num_slots=model_cfg.get("num_slots", 4),
            slot_dim=model_cfg.get("slot_dim", 64),
            num_iterations=model_cfg.get("num_iterations", 3),
            n_heads=model_cfg.get("n_heads", 4),
            n_points=model_cfg.get("n_points", 4),
            in_channels=model_cfg.get("in_channels", 3),
            slot_dict=model_cfg.get("slot_dict"),
            enc_dict=model_cfg.get("enc_dict"),
            dec_dict=model_cfg.get("dec_dict"),
            pred_dict=model_cfg.get("pred_dict"),
            loss_dict=model_cfg.get("loss_dict"),
        )

        weight_dict = dict(model_cfg.get("weight_dict") or {})
        weight_dict.setdefault("recon", 1.0)
        weight_dict.setdefault("mask", 1.0)
        weight_dict.setdefault("sigreg", 0.1)

        loss_fn = model_cfg.get("loss_fn")

        return cls(base_model, weight_dict=weight_dict, loss_fn=loss_fn, resolution=res)
