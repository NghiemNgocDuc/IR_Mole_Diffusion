#!/bin/bash -l
#SBATCH --job-name=ir_prepare
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-highvram
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the diffusion repository root}"
module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
pilot_output="${PILOT_OUTPUT:-runs/spectrum_pilot_${SLURM_JOB_ID:?}}"
echo "Preparation output: $pilot_output"
echo "Decoding comparisons and training-only negative mining; no fine-tuning."
echo "The 12-hour allocation is a limit. Progress logs provide measured rates and ETA per stage."
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u scripts/run_spectrum_pilot.py \
    --audit "${BASELINE_AUDIT:-evaluation/baseline_audit_66782537/baseline.json}" \
    --checkpoint "${SPECTRUM_CHECKPOINT:-runs/spectrum/checkpoints/best.ckpt}" \
    --output "$pilot_output" --epochs 5 --batch_size 64 --seed 78438379 --device cuda \
    --mine_limit "${MINE_LIMIT:-5000}" --decode_widths 5 10 20 --prepare_only
