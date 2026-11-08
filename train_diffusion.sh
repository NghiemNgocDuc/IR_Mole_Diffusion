#!/bin/bash -l
#SBATCH --job-name=ir_coords
#SBATCH --output=logs/diffusion-%j.out
#SBATCH --error=logs/diffusion-%j.err
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/run_stage.sh diffusion "$@"
