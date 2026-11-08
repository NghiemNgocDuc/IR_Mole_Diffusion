#!/bin/bash -l
#SBATCH --job-name=ir_smoke
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
#SBATCH --time=00:30:00
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
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA device: check allocation and PyTorch wheel'; print(torch.__version__, torch.cuda.get_device_name())"
srun python -u scripts/smoke_test.py --accelerator gpu \
    --output_dir "runs/smoke_gpu_${SLURM_JOB_ID:?}" "$@"
