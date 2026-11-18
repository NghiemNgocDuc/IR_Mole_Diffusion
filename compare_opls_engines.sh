#!/bin/bash -l
#SBATCH --job-name=opls-parity
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/opls-parity-%j.out
#SBATCH --error=logs/opls-parity-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load openmpi/5.0.5-cuda12.8
module load gromacs/2024.3-cuda12.8
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
if command -v gmx_mpi >/dev/null; then gmx=gmx_mpi; else gmx=gmx; fi
"$PWD/.runtime/opls-tools/bin/python" scripts/compare_opls_engines.py \
  --gmx "$gmx" --output "outputs/opls_parity_${SLURM_JOB_ID:?Submit with sbatch}"
