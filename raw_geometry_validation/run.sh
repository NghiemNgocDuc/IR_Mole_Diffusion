#!/bin/bash -l
#SBATCH --job-name=raw-dft
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/raw-dft-%A_%a.out
#SBATCH --error=logs/raw-dft-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load openmpi/4.1.6 orca/6.1.0
export OMP_NUM_THREADS=1
"$PWD/.runtime/opls-tools/bin/python" -u raw_geometry_validation/check.py run \
  --output "${1:?Output directory required}" --index "${SLURM_ARRAY_TASK_ID:?Array required}" \
  --orca "$(command -v orca)"
