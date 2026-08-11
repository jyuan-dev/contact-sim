# Contact-Sim

An object-centric video learning codebase supporting **Deformable SAVi** and **SAVi** (Slot Attention for Video) baseline models on manipulation and dynamic visual environments.

---

## 🚀 Quick Start & Environment Setup

### 1. Prerequisites & Environment
Ensure you have PyTorch 2.x and CUDA installed in your environment.

This project uses the `contact-sim` conda environment located at `/home/jyuan/miniconda3/envs/contact-sim`.

```bash
# Activate the conda environment
conda activate contact-sim

# If conda isn't initialized in your shell, use the full path:
#   /home/jyuan/miniconda3/envs/contact-sim/bin/python scripts/train.py

# Install the package in editable mode
pip install -e .
```

### 2. Submodule Setup (`slotformer`)
This repository integrates the third-party **SlotFormer** (`StoSAVi`) codebase located in `third_party/slotformer`.

If initializing a fresh clone of this repository, ensure the submodules are cloned/updated recursively:
```bash
git submodule update --init --recursive
```

#### How SlotFormer Integration Works:
- **Location**: Third-party code resides in [`third_party/slotformer/slotformer/base_slots`](file:///home/jyuan/jyuan-ws/contact-sim/third_party/slotformer/slotformer/base_slots).
- **Import Shim**: `src/models/savi.py` automatically initializes synthetic `nerv` framework stubs in Python's `sys.modules` and binds `third_party/slotformer/slotformer/base_slots` to `sys.path`.
- **Zero Third-Party Code Mutation**: In accordance with project architecture rules, no code within `third_party/` is modified. All wrappers and adapters are cleanly implemented in `src/models/` and `src/models/wrappers/`.

---

## 📊 Experiment Tracking & Data Versioning (W&B & DVC)

This repository is integrated with **Weights & Biases (W&B)** for experiment tracking and **Data Version Control (DVC)** for dataset and model artifact management:

### 1. Weights & Biases (W&B)
- **Default W&B Project**: [`pusht-contact-sim`](https://wandb.ai/jie-yuan/pusht-contact-sim)
- **Enabled by Default**: W&B tracking is active by default in [`configs/config.yaml`](file:///home/jyuan/jyuan-ws/contact-sim/configs/config.yaml) (`use_wandb: true`).
- Training runs automatically log real-time loss curves (`recon_loss`, `mask_bce`, `mask_dice`, `total_loss`) and validation metrics directly to W&B.
- To run offline without W&B syncing:
  ```bash
  python scripts/train.py use_wandb=false
  ```

### 2. Data Version Control (DVC)
- DVC is initialized (`.dvc/`) for managing heavy dataset files and experiment model checkpoints.
- Track large datasets or checkpoint artifacts:
  ```bash
  dvc add /home/jyuan/.stable-wm/pusht_expert_train_64x64.h5
  git add pusht_expert_train_64x64.h5.dvc .gitignore
  ```

---

## 🛠️ Usage & Training Entrypoints

The codebase uses [Hydra](https://hydra.cc/) for configuration management.

### Default Training (Deformable SAVi on PushT Dataset)
```bash
python scripts/train.py
```

### Training Standard SAVi with SIGReg Regularization
```bash
python scripts/train.py model=savi loss=savi_sigreg
```

### Unsupervised SAVi (Reconstruction Only, No Mask Supervision)
```bash
# Reconstruction-only (no ground-truth mask loss)
python scripts/train.py loss=savi_unsupervised

# Unsupervised + SIGReg anti-collapse regularizer
python scripts/train.py loss=savi_unsupervised_sigreg

# Standard SAVi unsupervised
python scripts/train.py model=savi loss=savi_unsupervised
```

### Custom Epochs and Learning Rate
```bash
python scripts/train.py epochs=10 lr=1e-4 batch_size=64
```

### Sanity Debug Run (5 Batches)
```bash
python scripts/train.py dry_run=true
```

---

## 🧪 Testing

Run the full modular unit test suite:
```bash
pytest tests/modular/
```

To run model-specific tests:
```bash
pytest tests/modular/test_models.py
```

---

## 📁 Repository Structure

```
contact-sim/
├── configs/                # Hydra YAML configuration files
│   ├── config.yaml         # Main root Hydra experiment config
│   ├── dataset/            # Dataset configs (pusht, gridshapes)
│   ├── loss/               # Loss configs (savi_default, savi_sigreg, savi_contrastive, savi_unsupervised, savi_unsupervised_sigreg)
│   └── model/              # Model configs (deformable_savi, savi)
├── scripts/                # Execution entrypoints (train.py, eval.py)
├── src/                    # Modular source code
│   ├── datasets/           # PyTorch Datasets & DataLoaders
│   ├── losses/             # Atomic & composite loss modules (SIGReg, Recon MSE, Mask Loss)
│   ├── metrics/            # Evaluation metrics (ARI, MSE, SIGReg quality)
│   ├── models/             # Model architectures and standardized wrappers
│   ├── training/           # Decoupled training loops and BaseTrainer
│   └── utils/              # Visualization, CUDA, and seeding utilities
├── third_party/            # External submodules (slotformer)
└── tests/                  # Modular PyTest test suite
```
