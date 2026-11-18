#!/bin/bash -l
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=ir_hard_selector
#SBATCH --output=logs/hard-selector-%j.out
#SBATCH --error=logs/hard-selector-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
checkpoint="${SELECTOR_CHECKPOINT:-runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt}"
report="${SELECTOR_REPORT:-evaluation/full_validation_66855524/ranking.json}"
output="${SELECTOR_OUTPUT:-runs/hard_selector_${SLURM_JOB_ID:-manual}}"
[[ -f "$checkpoint" && -f "$report" ]] || { echo "Missing checkpoint or baseline report: $checkpoint ; $report" >&2; exit 1; }
module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
resume_args=()
if [[ "${SELECTOR_RESUME:-0}" == "1" ]]; then resume_args=(--resume); fi
printf 'Output: %s\nMining 5,000 training inputs; then two matched selector arms. Four hours is a job limit, not an ETA.\n' "$output"
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u train_hard_selector.py \
  --checkpoint "$checkpoint" --baseline_report "$report" --output_dir "$output" \
  --data data/processed/squirl.npz --splits data/processed/splits.json \
  --device cuda --limit 5000 --epochs 20 --batch_size 64 --projection_dim 128 \
  --seed 78438379 --lr 0.0001 --progress_interval 15 "${resume_args[@]}"
printf 'Read %s/RESULTS.md and download %s/hard_selector_results.zip. No weights promoted.\n' "$output" "$output"
