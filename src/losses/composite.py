"""
Composite Loss Module.
Aggregates multiple individual loss modules directly instantiated via Hydra configurations.
"""

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """
    Composite Loss Module that aggregates arbitrary loss components configured in Hydra YAMLs.
    """

    def __init__(self, losses: dict[str, nn.Module] | None = None, **kwargs):
        super().__init__()
        self.losses = nn.ModuleDict()

        items = dict(losses or {})
        items.update(kwargs)

        for name, loss_item in items.items():
            if isinstance(loss_item, dict) and "_target_" in loss_item:
                import hydra

                loss_item = hydra.utils.instantiate(loss_item)
            if isinstance(loss_item, nn.Module):
                self.losses[name] = loss_item

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = None
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            res = loss_fn(out, batch)
            if isinstance(res, tuple):
                sub_weighted, sub_info = res[0], res[1]
            else:
                sub_weighted, sub_info = res, res.item()

            if total_loss is None:
                total_loss = sub_weighted
            else:
                total_loss = total_loss + sub_weighted

            if isinstance(sub_info, dict):
                for k, v in sub_info.items():
                    loss_dict[k] = float(v)
            else:
                loss_dict[f"{name}_loss"] = float(sub_info)

        if total_loss is None:
            device = out.get("recon_img", out.get("input_img")).device if out else "cpu"
            total_loss = torch.tensor(0.0, device=device)

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
