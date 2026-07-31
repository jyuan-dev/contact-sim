#!/usr/bin/env bash
# ==============================================================================
# StoSAVi Ablation Loss & Architecture Benchmark Runner
# Shell-based orchestration script for evaluating model variants and generating
# a side-by-side comparative Markdown report.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="/home/jyuan/miniconda3/envs/contact-sim/bin/python"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval_checkpoint.py"
SCRATCH_DIR="${REPO_ROOT}/scratch/eval_results"
OUTPUT_REPORT="${REPO_ROOT}/scratch/ablation_benchmark_report.md"

mkdir -p "${SCRATCH_DIR}"

echo "======================================================================"
echo "          Starting StoSAVi Ablation Loss Benchmark Runner"
echo "======================================================================"

# Array of evaluation targets: "Name|ConfigPath|CheckpointPath|OutputJson"
VARIANTS=(
  "1) Baseline (Recon + KLD)|configs/savi/unsupervised_ablation.yaml|/home/jyuan/.stable-wm/savi_unsupervised_baseline/savi_best.pt|${SCRATCH_DIR}/1_baseline.json"
  "2) Baseline + SIGReg|configs/savi/unsupervised_ablation.yaml|/home/jyuan/.stable-wm/savi_unsupervised/savi_best.pt|${SCRATCH_DIR}/2_sigreg.json"
  "3) Baseline + Contrast Loss|configs/savi/unsupervised_ablation.yaml|/home/jyuan/.stable-wm/savi_unsupervised_contrast/savi_best.pt|${SCRATCH_DIR}/3_contrast.json"
  "4) Baseline + Mask Loss (DETR Style)|configs/savi/unsupervised_ablation.yaml|/home/jyuan/.stable-wm/savi_unsupervised_mask/savi_best.pt|${SCRATCH_DIR}/4_mask.json"
  "5) Baseline + Contrast Loss + SIGReg|configs/savi/unsupervised_ablation.yaml|/home/jyuan/.stable-wm/savi_unsupervised_full/savi_best.pt|${SCRATCH_DIR}/5_full.json"
)

# Step 1: Run python evaluator per variant
for entry in "${VARIANTS[@]}"; do
  IFS="|" read -r NAME CFG CKPT OUT_JSON <<< "${entry}"

  echo "----------------------------------------------------------------------"
  echo "Evaluating Variant: ${NAME}"
  echo "  Config: ${CFG}"
  echo "  Checkpoint: ${CKPT}"
  echo "----------------------------------------------------------------------"

  PYTHONPATH=. "${PYTHON_EXEC}" "${EVAL_SCRIPT}" \
    --config "${CFG}" \
    --ckpt_path "${CKPT}" \
    --name "${NAME}" \
    --max_batches 15 \
    --out_json "${OUT_JSON}"
done

echo "======================================================================"
echo "          Aggregating Results into Markdown Report"
echo "======================================================================"

# Step 2: Format aggregated results into Markdown table
cat << 'EOF' > "${OUTPUT_REPORT}"
# Loss Function Ablation Study Benchmark Report (CNN Backbone)

Automated quantitative evaluation comparing 5 loss function ablation variants using standard **CNN** backbone:
1. **Baseline** (Recon + KLD)
2. **Baseline + SIGReg**
3. **Baseline + Contrast Loss**
4. **Baseline + Mask Loss (DETR Style)**
5. **Baseline + Contrast Loss + SIGReg**

| Loss Variant | Recon Loss (↓) | PSNR dB (↑) | SSIM (↑) | FG-ARI % (↑) | Latent Std (↑) | SIGReg Stat (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
EOF



for entry in "${VARIANTS[@]}"; do
  IFS="|" read -r NAME CFG CKPT OUT_JSON <<< "${entry}"

  if [ -f "${OUT_JSON}" ]; then
    "${PYTHON_EXEC}" -c "
import json
with open('${OUT_JSON}', 'r') as f:
    data = json.load(f)
print(f\"| **{data['name']}** | {data['recon_loss']:.4f} | {data['psnr']:.2f} dB | {data['ssim']:.4f} | {data['fg_ari']:.1f}% | {data['latent_std']:.4f} | {data['sigreg_stat']:.4f} |\")
" >> "${OUTPUT_REPORT}"
  fi
done

echo "" >> "${OUTPUT_REPORT}"

echo "======================================================================"
echo "Ablation Benchmark Completed Successfully!"
echo "Report generated at: ${OUTPUT_REPORT}"
echo "======================================================================"
