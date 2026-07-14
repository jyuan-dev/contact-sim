import torch

ckpt_path = '/home/jyuan/.stable-wm/detr_pusht/detr_final.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
print("Checkpoint keys:", ckpt.keys())
if 'config' in ckpt:
    print("Config:")
    for k, v in ckpt['config'].items():
        print(f"  {k}: {v}")
