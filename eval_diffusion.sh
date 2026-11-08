#!/bin/bash -l
#SBATCH --job-name=ir_eval
#SBATCH --output=logs/evaluate-%j.out
#SBATCH --error=logs/evaluate-%j.err
#SBATCH --time=04:00:00
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
# This isolates coordinate quality with a known graph. Evaluate IR-only identity separately.
srun python -u evaluate_diffusion.py --checkpoint runs/diffusion/checkpoints/best.ckpt \
    --data data/processed/squirl.npz --splits data/processed/splits.json \
    --split val --limit 500 --num_samples 5 --n_steps 200 --device cuda \
    --use_inference_graph --chemistry --output_dir evaluation/diffusion_val "$@"
