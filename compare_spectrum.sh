#!/bin/bash -l
#SBATCH --job-name=ir_rank_compare
#SBATCH --output=logs/rank-compare-%j.out
#SBATCH --error=logs/rank-compare-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-ir-diffusion}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
COMPARE_OUTPUT="${COMPARE_OUTPUT:-evaluation/rank_compare_${SLURM_JOB_ID:?}}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-runs/spectrum/checkpoints/best.ckpt}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-runs/spectrum_rank_finetune/checkpoints/best.ckpt}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"
mkdir -p "$COMPARE_OUTPUT"
srun python -u evaluate_spectrum.py --checkpoint "$BASELINE_CHECKPOINT" --device cuda \
    --split val --limit "$EVAL_LIMIT" --beam_size 5 --output "$COMPARE_OUTPUT/baseline.json"
srun python -u evaluate_spectrum.py --checkpoint "$CANDIDATE_CHECKPOINT" --device cuda \
    --split val --limit "$EVAL_LIMIT" --beam_size 5 --output "$COMPARE_OUTPUT/finetuned.json"
python -u scripts/compare_rank_results.py --baseline "$COMPARE_OUTPUT/baseline.json" \
    --candidate "$COMPARE_OUTPUT/finetuned.json" --output "$COMPARE_OUTPUT/comparison.json"
