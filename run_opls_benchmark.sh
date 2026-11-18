#!/bin/bash -l
#SBATCH --job-name=opls-benchmark
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/opls-benchmark-%j.out
#SBATCH --error=logs/opls-benchmark-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load conda/latest
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENMM_CPU_THREADS=1
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u benchmark_opls.py \
  --output "outputs/opls_benchmark_${SLURM_JOB_ID:?Submit with sbatch}" "$@"
