# Hydra Loss Configuration & Extensibility Guide

This directory contains Hydra loss configurations for the `contact-sim` project, powered by [`CompositeLoss`](file:///home/jyuan/jyuan-ws/contact-sim/src/losses/composite.py).

---

## 1. How Loss Configurations Work

Each loss configuration file defines a [`CompositeLoss`](file:///home/jyuan/jyuan-ws/contact-sim/src/losses/composite.py) instance that aggregates arbitrary atomic sub-losses instantiated via `hydra.utils.instantiate`.

- **Base Config** (`configs/loss/savi_default.yaml`):
  Defines core reconstruction MSE loss (`recon`) and mask segmentation loss (`mask`).

- **Extended Configs** (`configs/loss/savi_sigreg.yaml`, `configs/loss/savi_contrastive.yaml`):
  Use **Hydra Defaults Inheritance** to inherit `recon` and `mask` from `savi_default.yaml` and append new terms (`sigreg`, `contrastive`):

  ```yaml
  defaults:
    - savi_default
    - _self_

  losses:
    sigreg:
      _target_: src.losses.sigreg.SIGRegLoss
      weight: 0.1
  ```

---

## 2. How to Extend Loss Configurations

### Method A: Create a New Loss YAML (File-Based Inheritance)
To create a custom loss composition (e.g. `savi_my_custom_loss.yaml`), inherit `savi_default` and define your custom terms:

```yaml
# configs/loss/savi_my_custom_loss.yaml
defaults:
  - savi_default
  - _self_

losses:
  # Override weight of existing reconstruction loss:
  recon:
    weight: 0.5

  # Add custom loss term:
  sigreg:
    _target_: src.losses.sigreg.SIGRegLoss
    weight: 0.05
```

Usage:
```bash
python scripts/train.py loss=savi_my_custom_loss
```

---

### Method B: Dynamic Command-Line Overrides (CLI-Based)
You can dynamically modify loss weights or add new loss terms directly from the terminal without editing files:

```bash
# 1. Override weight of an existing loss term:
python scripts/train.py loss.losses.recon.weight=0.5

# 2. Add a new loss term on the fly to savi_default:
python scripts/train.py loss=savi_default \
    +loss.losses.sigreg._target_=src.losses.sigreg.SIGRegLoss \
    +loss.losses.sigreg.weight=0.1
```
