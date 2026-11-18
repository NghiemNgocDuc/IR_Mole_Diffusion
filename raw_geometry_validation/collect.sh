#!/bin/bash -l
#SBATCH --job-name=raw-dft-report
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/raw-dft-report-%j.out
#SBATCH --error=logs/raw-dft-report-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
"$PWD/.runtime/opls-tools/bin/python" raw_geometry_validation/check.py collect \
  --output "${1:?Output directory required}"
