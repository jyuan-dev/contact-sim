import sys
import types

sys.path.extend([
    '/home/jyuan/jyuan-ws/contact-sim',
    '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa',
    '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src',
    '/home/jyuan/jyuan-ws/contact-sim/third_party/cjepa/src/third_party'
])

# Import new modules first
import src.world_models.dinowm_causal as dinowm_causal
sys.modules['stable_worldmodel.wm.dinowm'] = dinowm_causal

import torch
checkpoint_path = '/home/jyuan/.stable-wm/pusht_videosaur_0_epoch_30_object.ckpt'
try:
    print('Starting load...')
    pl_module = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    print('Loaded successfully!', type(pl_module))
    print('Underlying model:', type(pl_module.model))
except Exception as e:
    import traceback
    traceback.print_exc()
