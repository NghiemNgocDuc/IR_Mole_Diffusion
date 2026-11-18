#!/bin/bash -l
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=ir_token_probe
#SBATCH --output=logs/token-probe-%j.out
#SBATCH --error=logs/token-probe-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
run_dir="${MATCHER_RUN:-runs/token_matcher_1426578}"
output="${PROBE_OUTPUT:-runs/token_matcher_probe_${SLURM_JOB_ID:-manual}}"
for path in "$run_dir/configuration.json" "$run_dir/selection.json" "$run_dir/comparison.json"; do
  [[ -f "$path" ]] || { printf 'Missing input: %s\n' "$path" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { printf 'Output exists: %s\n' "$output" >&2; exit 1; }
module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
printf 'Diagnostic rescoring only: no training or checkpoint changes.\nOutput: %s\nTwo hours is an allocation limit, not an ETA.\n' "$output"
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u probe_token_matcher.py \
  --run_dir "$run_dir" --output_dir "$output" --device cuda
printf 'Download %s/token_matcher_probe_results.zip\n' "$output"
