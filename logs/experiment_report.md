# Experiment Report: Standard SAVi vs. Deformable SAVi & SIGReg Regularization Benchmark

This report presents a quantitative evaluation and comparative analysis of object-centric visual learning models trained on the **PushT Expert Manipulation Dataset**.

---

## 1. Executive Summary

We evaluate three model variants across 1 full epoch (~14,021 steps per run, 1.8M training clips, 100% deterministic coverage of 1,869 validation episodes / 3,738 clips):

1. **Baseline Model**: `Deformable SAVi` (Default Reconstruction MSE + Mask Segmentation Loss)
2. **Step 1 Model**: `Deformable SAVi + SIGReg` (Le-WM temporal ECF Gaussian regularization)
3. **Step 2 Model**: `Standard SAVi` (Dense Slot Attention Baseline - *Currently Training at 58%*)

### Primary Findings
- **Top Performer**: **Baseline `Deformable SAVi`** achieved the **lowest Validation Reconstruction MSE (0.002365)**, **highest overall mask mIoU (82.68%)**, and **lowest slot swapping transition rate (0.17%)**.
- **Le-WM SIGReg Impact**: Incorporating the Le-WM temporal ECF SIGReg loss (`weight: 0.1`) prevented latent representation collapse and maintained high goal target binding (**95.25%**), but introduced a regularization penalty on RGB reconstruction MSE (`0.004379` vs `0.002365`), reduced Agent slot IoU (**35.97%** vs **71.77%**), and increased temporal slot swapping transition rate to **6.91%**.

---

## 2. Training Dynamics & Convergence

Training progress extracted from execution logs under [`scratch/checkpoints/`](file:///home/jyuan/jyuan-ws/contact-sim/scratch/checkpoints):

| Model Variant | Training Steps | Final Epoch Train Loss | Validation Loss | Peak Speed | Checkpoint Location |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Deformable SAVi (Baseline)** | 14,021 | **0.2326** | **0.1210** | 2.62 step/s | [`scratch/checkpoints/deformable_savi_pusht/`](file:///home/jyuan/jyuan-ws/contact-sim/scratch/checkpoints/deformable_savi_pusht/) |
| **Deformable SAVi + SIGReg (Le-WM)** | 14,021 | 0.7921 | 3.0500 | 2.57 step/s | [`scratch/checkpoints/deformable_savi_pusht_sigreg/`](file:///home/jyuan/jyuan-ws/contact-sim/scratch/checkpoints/deformable_savi_pusht_sigreg/) |
| **Standard SAVi (Dense)** | 8,100 / 14,021 | *0.1088 (In Progress)* | *Pending* | 2.72 step/s | [`scratch/checkpoints/savi_pusht/`](file:///home/jyuan/jyuan-ws/contact-sim/scratch/checkpoints/savi_pusht/) |

---

## 3. Deterministic Evaluation Benchmark & Slot Swapping Matrix

Evaluated deterministically across all 1,869 validation episodes / 3,738 clips (seed = 42):

| Evaluation Metric | Baseline (`Deformable SAVi`) | Step 1 (`Deformable SAVi + SIGReg`) | Top Performer |
| :--- | :---: | :---: | :---: |
| **Val Reconstruction MSE (Selection Metric)** | **0.002365** (std: 0.0005) | 0.004379 (std: 0.0013) | 🏆 **Deformable SAVi** |
| **Overall Mask mIoU (%)** | **82.68%** (median: 83.92%) | 68.35% (median: 67.59%) | 🏆 **Deformable SAVi** |
| **Overall Mask mDice (%)** | **89.36%** (median: 90.82%) | 75.20% (median: 76.51%) | 🏆 **Deformable SAVi** |
| **Slot 0 (Agent / Robot IoU %)** | **71.77%** (std: 19.63%) | 35.97% (std: 32.16%) | 🏆 **Deformable SAVi** |
| **Slot 1 (T-Block Object IoU %)** | **82.77%** (std: 7.05%) | 77.04% (std: 15.76%) | 🏆 **Deformable SAVi** |
| **Slot 2 (Goal Target Area IoU %)** | 95.01% (std: 8.19%) | **95.25%** (std: 7.48%) | 🏆 **Deformable + SIGReg** |
| **Slot Swapping Transition Rate (%)** | **0.17%** (27 / 3,738 clips) | 6.91% (872 / 3,738 clips) | 🏆 **Deformable SAVi** |
| **Sequence Slot Swapping Rate (%)** | **0.72%** | 23.33% | 🏆 **Deformable SAVi** |

---

## 4. Key Observations

1. **Near-Perfect Baseline Binding**: `Deformable SAVi` baseline maintains fixed matching assignment across 99.28% of validation sequences (only 27 out of 3,738 clips experienced a 1-frame boundary swap).
2. **SIGReg Representation Variance Effect**: Enforcing isotropic Gaussian variance across slots via Le-WM SIGReg increases slot latent symmetry. When the robot arm presses into the T-Block, equalized slot variances make it easier for Slot Attention to bleed between Slot 0 (Robot) and Slot 1 (T-Block), resulting in a higher swapping transition rate (6.91%).

---

## 5. Artifacts & Checkpoints

- **Winning Baseline Checkpoint**: [`scratch/checkpoints/deformable_savi_pusht/deformable_savi_best.pt`](file:///home/jyuan/jyuan-ws/contact-sim/scratch/checkpoints/deformable_savi_pusht/deformable_savi_best.pt)
- **Baseline Detailed JSON**: [`scratch/checkpoints/deformable_savi_pusht/eval_results_detailed.json`](file:///home/jyuan/jyuan-ws/contact-sim/scratch/checkpoints/deformable_savi_pusht/eval_results_detailed.json)
- **SIGReg Detailed JSON**: [`scratch/checkpoints/deformable_savi_pusht_sigreg/eval_results_detailed.json`](file:///home/jyuan/jyuan-ws/contact-sim/scratch/checkpoints/deformable_savi_pusht_sigreg/eval_results_detailed.json)
