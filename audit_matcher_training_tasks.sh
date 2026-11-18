#!/bin/bash -l
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=ir_task_audit
#SBATCH --output=logs/task-audit-%j.out
#SBATCH --error=logs/task-audit-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
run_dir="${MATCHER_RUN:-runs/token_matcher_1426578}"
output="${TASK_AUDIT_OUTPUT:-runs/matcher_task_audit_${SLURM_JOB_ID:-manual}}"
for name in configuration.json selection.json comparison.json negative_plans.json; do
  [[ -f "$run_dir/$name" ]] || { printf 'Missing input: %s\n' "$run_dir/$name" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { printf 'Output exists: %s\n' "$output" >&2; exit 1; }
module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
printf 'Comparing original training choices with generated alternatives on 2000 training molecules.\nTwo hours is an allocation limit, not an ETA.\n'
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u audit_matcher_training_tasks.py \
  --run_dir "$run_dir" --output_dir "$output" --device cuda --limit 2000 --seed 20261112
printf 'Download %s/matcher_task_audit_results.zip\n' "$output"
