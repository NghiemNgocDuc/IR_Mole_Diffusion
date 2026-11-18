#!/bin/bash -l
#SBATCH --job-name=opls-verify
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/opls-verify-%j.out
#SBATCH --error=logs/opls-verify-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENMM_CPU_THREADS=1
loader="${1:?Pass the path to the generated ETH.openmm.py loader}"
"$PWD/.runtime/opls-tools/bin/python" scripts/verify_opls_export.py \
  --loader "$loader" --output "outputs/opls_verification_${SLURM_JOB_ID:-manual}/report.json"
