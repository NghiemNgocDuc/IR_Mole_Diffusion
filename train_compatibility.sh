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
compat_checkpoint="${COMPATIBILITY_CHECKPOINT:-}"
compat_report="${COMPATIBILITY_REPORT:-}"
if [[ -z "$compat_checkpoint" || -z "$compat_report" ]]; then
  : "${PILOT_OUTPUT:?Set PILOT_OUTPUT to the prepared spectrum pilot directory, or set both COMPATIBILITY_CHECKPOINT and COMPATIBILITY_REPORT}"
  compat_checkpoint="${compat_checkpoint:-$PILOT_OUTPUT/baseline_start.ckpt}"
  compat_report="${compat_report:-$PILOT_OUTPUT/baseline.json}"
fi
if [[ ! -f "$compat_checkpoint" || ! -f "$compat_report" ]]; then
  printf '%s\n' "Compatibility experiment requires readable COMPATIBILITY_CHECKPOINT and COMPATIBILITY_REPORT files." >&2
  exit 1
fi

module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
compatibility_output="${COMPATIBILITY_OUTPUT:-runs/compatibility_${SLURM_JOB_ID:-manual}}"

printf 'Separate compatibility experiment; checkpoint: %s; report: %s; fresh output: %s\n' "$compat_checkpoint" "$compat_report" "$compatibility_output"
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u train_compatibility.py \
  --checkpoint "$compat_checkpoint" \
  --baseline_report "$compat_report" \
  --data data/processed/squirl.npz \
  --splits data/processed/splits.json \
  --output_dir "$compatibility_output" \
  --device cuda --epochs 5 --batch_size 256 --seed 78438379 \
  --lr 0.0001 --projection_dim 128

printf 'Review %s/RESULTS.md and compatibility_comparison.json. No weights were promoted.\n' "$compatibility_output"
