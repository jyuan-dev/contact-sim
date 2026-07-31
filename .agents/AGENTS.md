# Workspace Rules

## Environment Information
- **Conda Environment**: `contact-sim`
- **Python Executable**: `/home/jyuan/miniconda3/envs/contact-sim/bin/python`
- **Python Version**: `3.10.20`
- **PyTorch Version**: `2.12.1+cu130`
- **CUDA / Hardware**: CUDA 13.0, NVIDIA GeForce RTX 4090

## File & Path Conventions
- **Datasets**: Primary dataset files (HDF5, etc.) reside in `/home/jyuan/.stable-wm/` (with fallback to `scratch/`).
- **Checkpoints & Logs**: Save experiment checkpoints and TensorBoard logs in `scratch/checkpoints/<experiment_name>/`.
- **Scratch & Debug Artifacts**: All temporary debug scripts, intermediate data files, visualizations, PNGs, and GIFs must be saved in `scratch/`.

## Repository Architecture & Code Structure
- **Core Modules**: Keep reusable neural network architectures and dataset classes modular inside `src/` (e.g. `src/models/`, `src/datasets/`).
- **Configurations**: Store experiment hyperparameters exclusively in YAML files under `configs/<model_family>/`.
- **Scripts**: Entrypoint training and validation scripts belong in `scripts/` or `eval/`.
- **Third-Party Submodules**: External repositories and submodules belong in `third_party/`. Never modify or edit third-party submodule code inside `third_party/`; implement all custom wrappers, extensions, and adapters in `src/` or `scripts/`.

## Logging & Experiment Tracking
- **TensorBoard**: Log training metrics, losses, and learning rates using TensorBoard to `scratch/checkpoints/<experiment_name>/tb_logs`.
- **Config-Driven**: Load all hyperparameters dynamically from YAML config files rather than hardcoding in scripts.

## GIF Generation
- When generating animated GIFs using `imageio.mimsave` or PIL, always specify `loop=0` so that the GIF loops infinitely when played.


