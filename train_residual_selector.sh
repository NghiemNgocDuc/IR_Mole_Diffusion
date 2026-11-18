#!/bin/bash -l
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=ir_residual_selector
#SBATCH --output=logs/residual-selector-%j.out
#SBATCH --error=logs/residual-selector-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
checkpoint="${RESIDUAL_CHECKPOINT:-runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt}"
report="${RESIDUAL_REPORT:-evaluation/full_validation_66855524/ranking.json}"
mining="${RESIDUAL_MINING:-runs/hard_selector_1361586}"
output="${RESIDUAL_OUTPUT:-runs/residual_selector_${SLURM_JOB_ID:-manual}}"
for path in "$checkpoint" "$report" "$mining/configuration.json" "$mining/training_candidates.json"; do
  [[ -f "$path" ]] || { printf 'Missing input: %s\n' "$path" >&2; exit 1; }
done
module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
resume_args=()
if [[ "${RESIDUAL_RESUME:-0}" == "1" ]]; then resume_args=(--resume); fi
printf 'Output: %s\nReusing saved training candidates; development selection precedes assessment.\nTwo hours is the job limit, not an ETA. Input verification runs before training.\n' "$output"
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u train_residual_selector.py \
  --checkpoint "$checkpoint" --baseline_report "$report" --mining_dir "$mining" --output_dir "$output" \
  --data data/processed/squirl.npz --splits data/processed/splits.json \
  --device cuda --dev_size 2000 --partition_seed 20261110 --epochs 20 --patience 3 \
  --batch_size 64 --seed 78438379 --lr 0.0001 --progress_interval 15 "${resume_args[@]}"
printf 'Read %s/RESULTS.md and download %s/residual_selector_results.zip. No weights promoted.\n' "$output" "$output"
