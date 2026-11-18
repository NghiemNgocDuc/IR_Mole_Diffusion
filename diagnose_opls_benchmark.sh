#!/bin/bash -l
#SBATCH --job-name=opls-diagnose
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/opls-diagnose-%j.out
#SBATCH --error=logs/opls-diagnose-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
./.runtime/opls-tools/bin/python -u scripts/diagnose_opls_benchmark.py --engine \
  --output "outputs/opls_diagnose_${SLURM_JOB_ID:?Submit with sbatch}" "$@"
