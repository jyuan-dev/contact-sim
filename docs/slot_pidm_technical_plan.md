# Slot-PIDM Technical Specification & Training Plan

## 1. Executive Summary

This document details the architecture, mathematical formulation, hyperparameter configuration, and training specification for **Slot-PIDM** (**Slot Attention + Predictive Inverse Dynamics Model** with Iterative Relational Interaction).

The primary objective of Slot-PIDM is object-centric visual world modeling and action prediction for physical manipulation tasks (e.g. PushT). The model decouples multi-object dynamics into latent object slots, explicit contact physics via cross-attention, and action conditioning.

---

## 2. Model Architecture

Slot-PIDM consists of four tightly integrated neural network modules:

```
                  ┌───────────────────────────────┐
                  │    Raw RGB Image (B,3,64,64)  │
                  └───────────────┬───────────────┘
                                  │
                       [CNNSlotEncoder (GPU)]
                                  │
                                  ▼
                     Object Slots S_t (B, K, 128)
                       ├─ Slot 0: Agent / Pusher
                       └─ Slot 1..K-1: Passive Objects
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                                                   │
  [Forward Dynamics Module]                        [Inverse Dynamics Module]
  - Action Fusion on Agent Slot 0                  - Agent Delta: Δs_0 = S_{t+1,0} - S_{t,0}
  - Iterative Cross-Attention (L=2)                - Cross-Attention over Object Context
  - Output: S_{t+1} (Pred Future Slots)            - Output: â_t (Predicted Action)
        │                                                   │
        ▼                                                   ▼
  [SIGReg Regularizer]                              Action Loss (MSE)
  - Prevents Latent Collapse
```

### 2.1 Visual Processing Module & Dual Encoder Architectures
- **CNN Slot Encoder (`CNNSlotEncoder`)**:
  - **Feature Extractor**: 4-layer 2D Convolutional network (~20,464 parameters, ~0.02M). Downsamples $64 \times 64$ to $16 \times 16$ spatial feature map.
  - **Slot Attention**: $K=4$ slots of dimension $D_{\text{model}} = 64$. Uses 2 GRU iterations with soft spatial positional embeddings.
- **TinyViT Slot Encoder (`TinyViTEncoder`)**:
  - **Vision Transformer Backbone**: ImageNet-pretrained `tiny_vit_5m_224` (~5.21M parameters, ~260x larger feature extractor capacity than CNN).
  - **Weight Source & Caching**: Pretrained ImageNet-22k weights (`model.safetensors`, 48.4 MB) automatically cached locally under `~/.cache/huggingface/hub/models--timm--tiny_vit_5m_224.dist_in22k/`.
  - **Spatial Feature Grid**: Bilinearly interpolates $64 \times 64$ inputs to $224 \times 224$ internally, extracting $14 \times 14 = 196$ spatial feature tokens projected down to $D_{\text{slot\_in}} = 32$.
  - **Clean Architecture & Submodule Protection**: Implemented via PyTorch subclassing (`StoSAViWithTinyViT` in [src/models/slot_attention.py](file:///home/jyuan/jyuan-ws/contact-sim/src/models/slot_attention.py)) and `TinyViTEncoder` in [src/models/tiny_vit_encoder.py](file:///home/jyuan/jyuan-ws/contact-sim/src/models/tiny_vit_encoder.py), leaving `third_party/slotformer` 100% clean and untouched.
- **`SlotSpatialDecoder`**:
  - Transposed 2D Convolutional network decoding spatial segmentation masks $\hat{M}_t \in \mathbb{R}^{B \times K \times 64 \times 64}$ directly on GPU.

### 2.2 Quantitative Evaluation Metrics Suite
Comprehensive evaluation metrics defined with explicit docstrings and formulas in [src/metrics/eval_metrics.py](file:///home/jyuan/jyuan-ws/contact-sim/src/metrics/eval_metrics.py):
- **PSNR (dB)** ($\uparrow$): Peak Signal-to-Noise Ratio measuring image reconstruction fidelity.
- **SSIM** ($\uparrow$): Structural Similarity Index Measure bounded in $[0.0, 1.0]$.
- **FG-ARI (%)** ($\uparrow$): Foreground Adjusted Rand Index evaluating slot mask object segmentation quality against ground-truth object masks.
- **Latent Std** ($\uparrow$): Average feature standard deviation across slot channels ($~0.10 - 1.00$). Prevents representation collapse (drops to $0.0$ if collapse occurs).
- **SIGReg Stat** ($\downarrow$): Cramér-Wold Epps-Pulley empirical characteristic function distance matching standard Gaussian $\mathcal{N}(0, I)$.

### 2.3 Forward Dynamics Module (`IterativeSlotInteractionBlock`)
- **Kinematic Action Fusion**: Fuses action $a_t \in \mathbb{R}^2$ directly into Agent Slot $0$:
  $$S_{t, 0}^{(0)} = S_{t, 0} + \text{MLP}_{\text{action}}([S_{t, 0} \,;\, a_t])$$
- **Iterative Relational Loop ($L=2$ iterations)**:
  - **Self-Attention**: Captures kinematic inertia and individual slot updates.
  - **Inter-Slot Cross-Attention**: Computes contact forces and collision physics between the Agent Slot and passive object slots.
  - **FeedForward Refinement**: Residual MLP update for latent state prediction $\hat{S}_{t+1}$.

### 2.4 Inverse Dynamics Module (`InverseDynamicsBlock`)
- Computes agent latent displacement delta: $\Delta S_{t, 0} = S_{t+1, 0} - S_{t, 0}$.
- Applies multi-head cross-attention using $\Delta S_{t, 0}$ as Query and target object slots $S_t$ as Key/Value context.
- Passes conditioned representation into a 3-layer MLP predictor to output predicted robot action $\hat{a}_t \in \mathbb{R}^2$.

### 2.5 Latent Slot Regularization (`SIGReg`)
- **Sketch Isotropic Gaussian Regularizer (SIGReg)**: Adapted from LeWorldModel (`leWM`).
- Projects latent slot features along random 1D directions and computes empirical characteristic functions to enforce an isotropic Gaussian distribution, preventing representation collapse.

---


## 3. Loss Formulation

The joint optimization objective combines four loss terms:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{action}} \mathcal{L}_{\text{action}} + \lambda_{\text{slot}} \mathcal{L}_{\text{slot}} + \lambda_{\text{sigreg}} \mathcal{L}_{\text{sigreg}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}$$

| Loss Component | Weight ($\lambda$) | Objective / Formulation |
| :--- | :--- | :--- |
| **Action Loss ($\mathcal{L}_{\text{action}}$)** | $1.0$ | Mean Squared Error between predicted and ground-truth action: $\|\hat{a}_t - a_t\|_2^2$ |
| **Slot Loss ($\mathcal{L}_{\text{slot}}$)** | $1.0$ | Latent slot prediction MSE: $\|\hat{S}_{t+1} - S_{t+1}\|_F^2$ |
| **SIGReg Loss ($\mathcal{L}_{\text{sigreg}}$)** | $0.01$ | Characteristic function discrepancy against Isotropic Gaussian |
| **Mask Loss ($\mathcal{L}_{\text{mask}}$)** | $0.5$ | Binary Cross-Entropy on decoded spatial masks vs. ground-truth segmentation |

---

## 4. Dataset & Hyperparameter Specifications

### 4.1 Dataset Configuration
- **Dataset**: PushT Enriched Dataset (`pusht_expert_train_enriched.h5`)
- **Episodes**: 18,685 expert demonstration episodes
- **Resolution**: $64 \times 64$ RGB
- **Sequence Length**: 6 sampled frames ($T=6$, frame offset $= 1$)
- **Train/Val Split**: 90% Training / 10% Validation

### 4.2 Experiment Configuration (`configs/savi/slot_pidm_pusht_full.yaml`)

```yaml
dataset:
  name: "pusht"
  h5_path: "/home/jyuan/.stable-wm/pusht_expert_train_enriched.h5"
  resolution: [64, 64]
  train_frac: 0.9
  n_sample_frames: 6
  frame_offset: 1

model:
  d_model: 128
  action_dim: 2
  k_slots: 4
  num_heads: 4
  num_iterations: 2
  weight_action_loss: 1.0
  weight_slot_loss: 1.0
  weight_sigreg_loss: 0.01

training:
  batch_size: 128
  lr: 0.001
  weight_decay: 0.0001
  max_epochs: 10
  warmup_epochs: 1
  seed: 42
  num_workers: 8
  log_interval: 50
  checkpoint_interval: 1
  save_dir: "scratch/checkpoints/slot_pidm_pusht_full"
```

---

## 5. Training Pipeline & Execution

### 5.1 Command Entrypoint
Training is executed via the unified slot entrypoint:

```bash
/home/jyuan/miniconda3/envs/contact-sim/bin/python scripts/train_slot_pidm.py \
  --config configs/savi/slot_pidm_pusht_full.yaml
```

### 5.2 Artifacts & Checkpoints Location
- **Checkpoints**: `scratch/checkpoints/slot_pidm_pusht_full/slot_pidm_epoch_*.pt`
- **Final Model**: `scratch/checkpoints/slot_pidm_pusht_full/slot_pidm_final.pt`
- **TensorBoard Logs**: `scratch/checkpoints/slot_pidm_pusht_full/tb_logs/`

---

## 6. Evaluation & Verification Strategy

After training completion, evaluation is executed via `scripts/eval_slot_pidm.py`:

1. **Action Prediction Accuracy**: Evaluate action MSE on held-out 10% test split.
2. **Long-Horizon Slot Rollout**: Unroll forward dynamics over 16-step trajectories and compute latent drift.
3. **Qualitative Visual Verification**: Decode spatial masks from predicted slots and save animated GIFs (`loop=0`) to `scratch/slot_pidm_eval.gif`.
