# CONTEXT.md — Domain glossary

Shared language for this repo's domain concepts. Architecture reviews use these
names for seams; when a new term appears, add it here.

## Model concepts

- **SAVi** — the public base model class in `src/models/savi.py` (flat kwargs +
  nested dicts, builds the core under `.model`).
- **StoSAVi** — the core model (`src/models/savi.py`): CNN encoder + slot
  attention + decoder + transformer/LSTM predictor. The typed `inner_savi()`
  accessor returns this. Its state-dict keys are pinned by existing checkpoints.
- **slot latents / post-slots** — per-frame slot vectors `[B, T, K, D]`, the
  model's object-centric state. "Post-slots" = slots after slot attention.
- **encode** — the canonical clip→post-slots procedure (`StoSAVi.encode`): the
  only implementation of the per-frame kernel→slot_attention recurrence. All
  callers (rollout, slotformer, pidm, extract_slots) use it.

## Eval concepts

- **ModelOutput** — the wrapper's normalized output contract: `input_img`,
  `pred_masks [B,T,K,H,W]`, `recon_img`, `post_slots [B,T,K,D]`, one shape per
  key. Losses and metrics consume exactly these keys.
- **swap metric** — slot-swap tracking with greedy per-slot argmax assignment
  (canonical; the Hungarian per-class variant also exists in the metrics
  module).
- **EvaluationSuite** — the metrics module's computation home (per-class
  IoU/Dice, swap events, batch-level entry). Scripts own serialization.

## Infrastructure concepts

- **checkpoint bootstrap** — "given a checkpoint path, reconstruct the
  experiment": config discovery (`config.yaml`, then `.hydra/config.yaml`,
  hard-fail), last-resort state-dict shape sniffing, optional `cli_overrides`,
  dataset-path resolution. Lives in `src/utils/checkpoint_bootstrap.py` as
  `bootstrap_checkpoint(ckpt_path, cli_overrides=None) -> (model_wrapper,
  cfg_dict)`. Weight loading stays separate (`load_checkpoint_state`).
- **canonical checkpoint format** — `{"model_state": <wrapper's own
  state_dict>, "epoch": N}`; legacy formats (unprefixed inner, one `model.`
  prefix) are recognized by the loader's documented format table.

## Dataset concepts

- **DeterministicEpisodeEvalDataset** — fixed-seed val coverage sampling;
  batches carry `episode_idx` / `start_frame` (the source of truth for
  per-sequence records).
- **find_dataset_path** — the only dataset-path resolver; raises when nothing
  is found.
