#!/bin/bash -l
#SBATCH --job-name=opls-precision
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/opls-precision-%j.out
#SBATCH --error=logs/opls-precision-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load openmpi/5.0.5-cuda12.8 gromacs/2024.3-cuda12.8
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
gmx_binary=$(command -v gmx_mpi || command -v gmx)
"$PWD/.runtime/opls-tools/bin/python" -u scripts/probe_opls_precision.py \
  --output "outputs/opls_precision_${SLURM_JOB_ID:?Submit with sbatch}" --gmx "$gmx_binary" "$@"
