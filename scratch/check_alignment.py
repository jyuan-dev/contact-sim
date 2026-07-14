import os
import cv2
import numpy as np

# Path to the saved artifacts
out_dir = '/home/jyuan/.gemini/antigravity-ide/brain/84ca7224-7271-41ba-9541-8c03fd85f0e9'

orig = cv2.imread(os.path.join(out_dir, 'gt_orig.png'))
block = cv2.imread(os.path.join(out_dir, 'gt_mask_block.png'), cv2.IMREAD_GRAYSCALE)
agent = cv2.imread(os.path.join(out_dir, 'gt_mask_agent.png'), cv2.IMREAD_GRAYSCALE)
goal = cv2.imread(os.path.join(out_dir, 'gt_mask_goal.png'), cv2.IMREAD_GRAYSCALE)

# Create an overlay: original image + colored highlights
# Block mask -> Red channel
# Agent mask -> Green channel
# Goal mask -> Blue channel
overlay = orig.copy()
overlay[block > 127] = [0, 0, 255] # Red highlight
overlay[agent > 127] = [0, 255, 0] # Green highlight

cv2.imwrite(os.path.join(out_dir, 'overlay_debug.png'), overlay)
print("Overlay image successfully written to:", os.path.join(out_dir, 'overlay_debug.png'))
