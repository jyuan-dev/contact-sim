import os
import cv2
import h5py
import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

out_dir = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9'
h5_path = '/home/jyuan/.stable-wm/pusht_expert_train.h5'

with h5py.File(h5_path, 'r') as f:
    state = f['state'][0]
    orig_img = f['pixels'][0].copy()

# state is [agent_x, agent_y, block_x, block_y, block_angle, block_vx, block_vy]
agent_x, agent_y, block_x, block_y, block_angle = state[:5]

print(f"Agent coords: ({agent_x:.2f}, {agent_y:.2f})")
print(f"Block coords: ({block_x:.2f}, {block_y:.2f})")

# Draw coordinates on the original image (224x224)
# Note: original image in h5 is 224x224. But simulator dimensions are 512x512!
# Let's map coordinates from 512x512 space to 224x224 space.
scale_x = orig_img.shape[1] / 512.0
scale_y = orig_img.shape[0] / 512.0

agent_px = int(agent_x * scale_x)
agent_py = int(agent_y * scale_y)

block_px = int(block_x * scale_x)
block_py = int(block_y * scale_y)

print(f"Mapped Agent px: ({agent_px}, {agent_py})")
print(f"Mapped Block px: ({block_px}, {block_py})")

# Draw red circle for agent
cv2.circle(orig_img, (agent_px, agent_py), 5, (0, 0, 255), -1)
# Draw blue circle for block
cv2.circle(orig_img, (block_px, block_py), 5, (255, 0, 0), -1)

cv2.imwrite(os.path.join(out_dir, 'coords_debug.png'), orig_img)
print("Saved coords debug image to:", os.path.join(out_dir, 'coords_debug.png'))
