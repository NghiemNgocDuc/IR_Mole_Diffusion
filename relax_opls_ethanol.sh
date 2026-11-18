#!/bin/bash -l
#SBATCH --job-name=opls-relax
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/opls-relax-%j.out
#SBATCH --error=logs/opls-relax-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENMM_CPU_THREADS=1
"$PWD/.runtime/opls-tools/bin/python" scripts/relax_opls_ethanol.py \
  --loader "${1:-outputs/opls_ethanol_1788228/ETH.openmm.py}" \
  --output "outputs/opls_relaxation_${SLURM_JOB_ID:?Submit with sbatch}"
