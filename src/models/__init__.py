from src.models.factory import build_model
from src.models.detr import DETR
from src.models.slot_attention import build_savi_model as StoSAVi

__all__ = ["build_model", "DETR", "StoSAVi"]
