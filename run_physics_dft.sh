#!/bin/bash -l
#SBATCH --job-name=physics-dft
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/physics-dft-%A_%a.out
#SBATCH --error=logs/physics-dft-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load openmpi/4.1.6 "${ORCA_MODULE:-orca/6.1.0}"
export OMP_NUM_THREADS=1 OPENMM_CPU_THREADS=1
"$PWD/.runtime/opls-tools/bin/python" -u scripts/validate_opls_physics.py run \
    --output "${1:?Validation output directory required}" \
    --index "${SLURM_ARRAY_TASK_ID:?Array job required}" --orca "$(command -v orca)"
