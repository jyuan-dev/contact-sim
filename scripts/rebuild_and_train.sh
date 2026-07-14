#!/bin/bash
set -e

echo "=== Starting Rebuild and Train Pipeline ==="
echo "Date: $(date)"

# 1. Wait for enrich_pusht_dataset.py to complete
echo "1. Waiting for dataset enrichment to complete..."
while ps -ef | grep "enrich_pusht_dataset.py" | grep -v grep > /dev/null; do
    sleep 30
done
echo "Dataset enrichment completed!"

# 2. Resize dataset to 64x64
echo "2. Resizing enriched dataset to 64x64..."
rm -f /home/jyuan/.stable-wm/pusht_expert_train_64x64.h5
/home/jyuan/miniconda3/envs/contact-sim/bin/python scripts/resize_pusht_dataset.py

echo "Dataset resized successfully!"

# 3. Launch DETR-Style Bipartite Matching Training Session
echo "3. Launching training session with DETR-style bipartite matching..."
tmux kill-session -t savi_detr_train 2>/dev/null || true
rm -f train_savi_detr.log
tmux new-session -d -s savi_detr_train "/home/jyuan/miniconda3/envs/contact-sim/bin/python -u scripts/train_savi.py --dataset pusht > train_savi_detr.log 2>&1"

echo "=== Pipeline successfully scheduled! ==="

