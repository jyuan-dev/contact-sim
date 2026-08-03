import sys
import os
import copy
import torch
import torch.nn as nn

# Add PlaySlot src to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAYSLOT_SRC = os.path.join(REPO_ROOT, 'third_party', 'PlaySlot', 'src')
if PLAYSLOT_SRC in sys.path:
    sys.path.remove(PLAYSLOT_SRC)
sys.path.insert(0, PLAYSLOT_SRC)

from models.SAVi import SAVi


class PlaySlotSAVi(nn.Module):
    """
    Local wrapper for PlaySlot's SAVi (Stage 1 Video Decomposition) model.
    """
    def __init__(
        self,
        num_slots=4,
        slot_dim=64,
        num_iterations=3,
        num_iterations_first=3,
        in_channels=3,
        mlp_hidden=128,
        mlp_encoder_dim=128,
        initializer=None,
        encoder=None,
        decoder=None,
        transition_module_params=None,
        **kwargs
    ):
        super().__init__()
        if initializer is None:
            initializer = "LearnedRandom"
        elif isinstance(initializer, dict):
            init_mode = initializer.get("mode", "LearnedRandom")
            if init_mode in ["learned_random", "learnedrandom", "LearnedRandom"]:
                initializer = "LearnedRandom"
            elif init_mode in ["learned", "Learned"]:
                initializer = "Learned"
            else:
                initializer = init_mode

        if transition_module_params is None:
            transition_module_params = {
                "model_name": "TransformerBlock",
                "num_heads": 4,
                "head_dim": 16,
                "mlp_size": 256
            }
        else:
            transition_module_params = copy.deepcopy(transition_module_params)
            if "model_name" not in transition_module_params:
                t_type = transition_module_params.get("type", "none")
                if t_type in ["identity", "none", "None", ""]:
                    transition_module_params["model_name"] = "none"
                else:
                    transition_module_params["model_name"] = t_type
                if "type" in transition_module_params:
                    del transition_module_params["type"]

        if encoder is None:
            encoder = {
                "encoder_name": "ConvEncoder",
                "encoder_params": {
                    "num_channels": [32, 32, 32, 32],
                    "kernel_size": 5,
                    "resolution": [64, 64],
                    "downsample_encoder": False,
                    "downsample": 2
                }
            }
        else:
            encoder = copy.deepcopy(encoder)

        if decoder is None:
            decoder = {
                "decoder_name": "ConvDecoder",
                "decoder_params": {
                    "num_channels": [64, 64, 64, 64],
                    "kernel_size": 5,
                    "resolution": [64, 64],
                    "downsample_decoder": False,
                    "upsample": 1
                }
            }
        else:
            decoder = copy.deepcopy(decoder)

        self.savi = SAVi(
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
            num_iterations_first=num_iterations_first,
            in_channels=in_channels,
            mlp_hidden=mlp_hidden,
            mlp_encoder_dim=mlp_encoder_dim,
            initializer=initializer,
            encoder=encoder,
            decoder=decoder,
            transition_module_params=transition_module_params
        )

    def forward(self, x, **kwargs):
        """
        Forward pass expecting x as (B, T, C, H, W).
        Returns dictionary formatted for trainer compatibility:
          - recon_combined: (B, T, C, H, W)
          - post_masks: (B, T, num_slots, 1, H, W)
          - slots: (B, T, num_slots, slot_dim)
          - recons_objs: (B, T, num_slots, C, H, W)
        """
        if isinstance(x, dict):
            x = x['img']
        if x.ndim == 4:  # (B, C, H, W) -> add time dim (B, 1, C, H, W)
            x = x.unsqueeze(1)


        B, T, C, H, W = x.shape
        savi_out = self.savi(x, num_imgs=T, decode=True, **kwargs)

        recon_combined = savi_out["recons_imgs"]
        recons_objs = savi_out["recons_objs"]
        masks = savi_out["masks"]
        slot_history = savi_out["slot_history"]

        return {
            "recon_combined": recon_combined,
            "post_masks": masks,
            "slots": slot_history,
            "recons_objs": recons_objs,
            "img": x
        }
