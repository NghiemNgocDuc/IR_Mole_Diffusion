#!/bin/bash -l
#SBATCH --job-name=ir_graph_eval
#SBATCH --output=logs/graph-eval-%j.out
#SBATCH --error=logs/graph-eval-%j.err
#SBATCH --time=02:00:00
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
python -c "import sys, torch, tqdm, pytorch_lightning; print('Python:', sys.executable, 'PyTorch:', torch.__version__, 'tqdm:', tqdm.__version__)"
srun python -u evaluate_spectrum.py --checkpoint runs/spectrum/checkpoints/best.ckpt \
    --device cuda --split val --limit 500 --beam_size 5 \
    --output evaluation/spectrum_val.json "$@"
