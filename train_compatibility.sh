#!/bin/bash -l
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=ir_compatibility
#SBATCH --output=logs/compatibility-%j.out
#SBATCH --error=logs/compatibility-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${PILOT_OUTPUT:?Set PILOT_OUTPUT to the prepared spectrum pilot directory}"
if [[ ! -f "$PILOT_OUTPUT/baseline_start.ckpt" || ! -f "$PILOT_OUTPUT/baseline.json" ]]; then
  printf '%s\n' "PILOT_OUTPUT must contain baseline_start.ckpt and the completed baseline.json report." >&2
  exit 1
fi

module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
compatibility_output="${COMPATIBILITY_OUTPUT:-runs/compatibility_${SLURM_JOB_ID:-manual}}"

printf 'Separate compatibility experiment; baseline: %s; fresh output: %s\n' "$PILOT_OUTPUT" "$compatibility_output"
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u train_compatibility.py \
  --checkpoint "$PILOT_OUTPUT/baseline_start.ckpt" \
  --baseline_report "$PILOT_OUTPUT/baseline.json" \
  --data data/processed/squirl.npz \
  --splits data/processed/splits.json \
  --output_dir "$compatibility_output" \
  --device cuda --epochs 5 --batch_size 256 --seed 78438379 \
  --lr 0.0001 --projection_dim 128

printf 'Review %s/RESULTS.md and compatibility_comparison.json. No weights were promoted.\n' "$compatibility_output"
