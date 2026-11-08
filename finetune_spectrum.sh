#!/bin/bash -l
#SBATCH --job-name=ir_rank_ft
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-ir-diffusion}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-runs/spectrum_rank_finetune}"
SPECTRUM_SOURCE="${SPECTRUM_SOURCE:-runs/spectrum/checkpoints/best.ckpt}"
if [[ -f "$FINETUNE_OUTPUT/checkpoints/last.ckpt" ]]; then
    echo "This fine-tune already has a checkpoint. Choose a fresh FINETUNE_OUTPUT or resume explicitly." >&2
    exit 1
fi
if [[ ! -f "$FINETUNE_OUTPUT/ranking_negatives.npz" ]]; then
    python -u prepare_ranking.py --output "$FINETUNE_OUTPUT/ranking_negatives.npz"
fi
srun python -u train_spectrum.py --config configs/spectrum_rank_finetune.json \
    --finetune_from "$SPECTRUM_SOURCE" --accelerator gpu \
    --ranking_negatives "$FINETUNE_OUTPUT/ranking_negatives.npz" --save_dir "$FINETUNE_OUTPUT" "$@"
echo "Fine-tune complete. Compare validation identification before selecting these weights."
