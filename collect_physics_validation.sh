#!/bin/bash -l
#SBATCH --job-name=physics-report
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/physics-report-%j.out
#SBATCH --error=logs/physics-report-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
"$PWD/.runtime/opls-tools/bin/python" scripts/validate_opls_physics.py collect \
    --output "${1:?Validation output directory required}"
