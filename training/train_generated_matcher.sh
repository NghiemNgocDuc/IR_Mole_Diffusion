#!/bin/bash -l
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=ir_gen_matcher
#SBATCH --output=logs/generated-matcher-%j.out
#SBATCH --error=logs/generated-matcher-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
checkpoint="${MATCHER_CHECKPOINT:-runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt}"
report="${MATCHER_REPORT:-evaluation/full_validation_66855524/ranking.json}"
output="${GENERATED_MATCHER_OUTPUT:-runs/generated_matcher_${SLURM_JOB_ID:-manual}}"
for path in "$checkpoint" "$report" data/processed/squirl.npz data/processed/splits.json; do
  [[ -f "$path" ]] || { printf 'Missing input: %s\n' "$path" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { printf 'Output exists; choose a fresh GENERATED_MATCHER_OUTPUT: %s\n' "$output" >&2; exit 1; }
module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
printf 'Output: %s\nTwo frozen-encoder arms, ten epochs each; 5000 shared anchors, original versus generated negatives.\nTwelve hours is the job limit, not an ETA. Original decoder stays unchanged.\n' "$output"
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u training/train_generated_matcher.py \
  --checkpoint "$checkpoint" --baseline_report "$report" --output_dir "$output" \
  --data data/processed/squirl.npz --splits data/processed/splits.json --device cuda \
  --training_limit 5000 --mining_seed 20261112 --dev_size 2000 --partition_seed 20261110 --epochs 10 --warmup_epochs 1 \
  --batch_size 64 --seed 78438379 --matcher_layers 2 \
  --lr 0.0001 --encoder_lr 0.00001 --alignment_weight 0.1 --minimum_dev_gain 0.005 --progress_interval 15
printf 'Read %s/RESULTS.md and download %s/generated_matcher_results.zip. No weights promoted.\n' "$output" "$output"
