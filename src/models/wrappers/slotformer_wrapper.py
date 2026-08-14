"""
Standardized SlotFormer (Stage 2) Wrapper.

Wraps ``SlotFormerModel`` to conform to the ``BaseModelWrapper`` interface.
Registered under: ``"slotformer"``, ``"slot_former"``
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseModelWrapper, ModelOutput
from src.models.factory import register_model, build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@register_model([
    "slotformer", "slot_former", "ocvp_slotformer", "ocvp-slotformer",
    "factorized_slotformer", "cocvp_slotformer", "cocvp-slotformer",
    "cocvp_slotformer_film", "cocvp_slotformer_sum", "cocvp_slotformer_concat",
    "intact_slotformer", "intact_ocvp_slotformer", "ocvp_intact_slotformer",
    "pidm", "pidm_slotformer", "pidm-slotformer", "pidm_pusht",
])
class StandardizedSlotFormerWrapper(BaseModelWrapper):
    """
    SlotFormer Stage 2 wrapper conforming to ``BaseModelWrapper``.
    Supports standard SlotFormer, OCVP Factorized, Action-Conditioned (cOCVP),
    INTACT RobotSlotIntentActionActor, and PIDM (Predictive Inverse Dynamics Model) variants.
    """

    def __init__(
        self,
        model: nn.Module,
        weight_dict: dict | None = None,
        resolution: tuple = (64, 64),
    ) -> None:
        super().__init__()
        self.model = model
        self.resolution = resolution
        self._weight_dict = weight_dict or {"slot_mse": 1.0}

    @classmethod
    def build(cls, model_cfg: dict) -> "StandardizedSlotFormerWrapper":
        """Construct from model config sub-dict."""
        from src.models.slotformer import SlotFormerModel
        from src.models.pidm import PIDMModel

        stage1_ckpt_path = model_cfg.get("stage1_ckpt_path", "scratch/checkpoints/savi_pusht/savi_best.pt")
        if stage1_ckpt_path and not os.path.isabs(stage1_ckpt_path):
            stage1_ckpt_path = os.path.join(REPO_ROOT, stage1_ckpt_path)

        stage1_model_type = model_cfg.get("stage1_model_type", "")
        if not stage1_model_type:
            if stage1_ckpt_path and "deformable" in stage1_ckpt_path.lower():
                stage1_model_type = "deformable_savi"
            else:
                stage1_model_type = "savi"

        res = tuple(model_cfg.get("resolution", [64, 64]))
        stage1_cfg = {
            "model": {
                "name": stage1_model_type,
                "type": stage1_model_type,
                "resolution": res,
            }
        }
        stage1_wrapper = build_model(stage1_cfg)

        if stage1_ckpt_path and os.path.exists(stage1_ckpt_path):
            from src.utils.training_utils import load_checkpoint_state
            load_checkpoint_state(stage1_wrapper, stage1_ckpt_path)
            print(f"[SlotFormer] Loaded pretrained Stage 1 weights from: {stage1_ckpt_path}")
        else:
            print(f"[SlotFormer Warning] Stage 1 checkpoint path '{stage1_ckpt_path}' not found!")

        history_len = model_cfg.get("history_len", 2)
        rollout_len = model_cfg.get("rollout_len", 4)
        d_model = model_cfg.get("d_model", 128)
        num_layers = model_cfg.get("num_layers", 4)
        num_heads = model_cfg.get("num_heads", 8)
        ffn_dim = model_cfg.get("ffn_dim", 512)
        t_pe = model_cfg.get("t_pe", "sin")
        slots_pe = model_cfg.get("slots_pe", "")
        loss_decay_factor = model_cfg.get("loss_decay_factor", 1.0)
        use_img_recon_loss = model_cfg.get("use_img_recon_loss", False)

        model_type_str = str(model_cfg.get("type", model_cfg.get("name", ""))).lower()

        if "pidm" in model_type_str:
            condition_mode = model_cfg.get("condition_mode", "goal_film")
            goal_slot_idx = model_cfg.get("goal_slot_idx", 2)
            raw_action_dim = model_cfg.get("raw_action_dim", model_cfg.get("action_dim", 2))
            action_embed_dim = model_cfg.get("action_embed_dim", model_cfg.get("action_emb_dim", 64))
            action_loss_weight = model_cfg.get("action_loss_weight", 1.0)
            slot_loss_weight = model_cfg.get("slot_loss_weight", 1.0)
            robot_slot_idx = model_cfg.get("robot_slot_idx", 0)
            rollout_consistent = model_cfg.get("rollout_consistent", True)

            inner_model = PIDMModel(
                stage1_model=stage1_wrapper,
                history_len=history_len,
                rollout_len=rollout_len,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                t_pe=t_pe,
                slots_pe=slots_pe,
                loss_decay_factor=loss_decay_factor,
                use_img_recon_loss=use_img_recon_loss,
                condition_mode=condition_mode,
                goal_slot_idx=goal_slot_idx,
                raw_action_dim=raw_action_dim,
                action_embed_dim=action_embed_dim,
                action_loss_weight=action_loss_weight,
                slot_loss_weight=slot_loss_weight,
                robot_slot_idx=robot_slot_idx,
                rollout_consistent=rollout_consistent,
            )
            return cls(inner_model, resolution=res)

        default_rollouter_type = "cocvp" if "cocvp" in model_type_str else (
            "ocvp" if "ocvp" in model_type_str or "factorized" in model_type_str else "standard"
        )
        rollouter_type = model_cfg.get("rollouter_type", default_rollouter_type)

        raw_action_dim = model_cfg.get("raw_action_dim", model_cfg.get("action_dim", 2))
        action_embed_dim = model_cfg.get("action_embed_dim", model_cfg.get("action_emb_dim", 64))
        condition_mode = model_cfg.get("condition_mode", "none")
        use_intact_actor = model_cfg.get("use_intact_actor", model_cfg.get("use_intact", "intact" in model_type_str))
        action_loss_weight = model_cfg.get("action_loss_weight", 1.0)
        robot_slot_idx = model_cfg.get("robot_slot_idx", 0)

        slotformer_model = SlotFormerModel(
            stage1_model=stage1_wrapper,
            history_len=history_len,
            rollout_len=rollout_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            t_pe=t_pe,
            slots_pe=slots_pe,
            loss_decay_factor=loss_decay_factor,
            use_img_recon_loss=use_img_recon_loss,
            rollouter_type=rollouter_type,
            raw_action_dim=raw_action_dim,
            action_embed_dim=action_embed_dim,
            condition_mode=condition_mode,
            use_intact_actor=use_intact_actor,
            action_loss_weight=action_loss_weight,
            robot_slot_idx=robot_slot_idx,
        )

        return cls(slotformer_model, resolution=res)

    def forward(self, x: dict | Tensor) -> ModelOutput:
        out_dict = self.model(x)
        res = {
            "input_img": out_dict.get("input_img"),
            "pred_boxes": None,
            "pred_logits": None,
            "pred_masks": out_dict.get("pred_masks"),
            "recon_img": out_dict.get("recon_img"),
            "post_slots": out_dict.get("post_slots"),
            "gt_slots": out_dict.get("gt_slots"),
            "pred_slots": out_dict.get("pred_slots"),
        }
        if "action_nll_dict" in out_dict:
            res["action_nll_dict"] = out_dict["action_nll_dict"]
        return res

    def compute_loss(
        self, out: ModelOutput, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        return self.model.calc_train_loss(out, batch)

