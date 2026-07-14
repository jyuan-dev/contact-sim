#!/bin/bash
set -e

# Configure PYTHONPATH and Python env
export PYTHONPATH=/home/jyuan/jyuan-ws/contact-sim/third_party/PlaySlot/src
PYTHON_BIN=/home/jyuan/miniconda3/envs/contact-sim/bin/python
SCRIPT_PATH=/home/jyuan/jyuan-ws/contact-sim/train_slot_mpc_pusht.py
LOG_DIR=/home/jyuan/.stable-wm

echo "Starting Slot-MPC PushT full training pipeline..."

echo "=========================================="
echo "PHASE 1: Training SAVi Representation (10 Epochs)"
echo "=========================================="
$PYTHON_BIN -u $SCRIPT_PATH --mode train_savi --epochs 10 --batch_size 16 > $LOG_DIR/savi_train.log 2>&1

echo "=========================================="
echo "PHASE 2: Training cOCVP Dynamics (50 Epochs)"
echo "=========================================="
$PYTHON_BIN -u $SCRIPT_PATH --mode train_dynamics --epochs 50 --batch_size 16 > $LOG_DIR/dynamics_train.log 2>&1

echo "=========================================="
echo "PHASE 3: Evaluating Closed-Loop Control in gym_pusht"
echo "=========================================="
$PYTHON_BIN -u $SCRIPT_PATH --mode evaluate > $LOG_DIR/evaluate.log 2>&1

echo "Pipeline finished successfully!"
