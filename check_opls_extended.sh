#!/bin/bash -l
#SBATCH --job-name=opls-extended
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/opls-extended-%j.out
#SBATCH --error=logs/opls-extended-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export QT_QPA_PLATFORM=offscreen
export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1
result="$PWD/outputs/opls_extended_${SLURM_JOB_ID:?Submit with sbatch}"
mkdir "$result"
{
  printf 'GROMACS executable already on PATH:\n'
  command -v gmx || true
  command -v gmx_mpi || true
  printf '\nGROMACS module inventory (no module changes):\n'
  module -t avail gromacs 2>&1 || true
  module spider gromacs 2>&1 || true
} > "$result/gromacs_inventory.txt"
python="$PWD/.runtime/opls-tools/bin/python"
app="$PWD/tools/LigParGen-3.0.1.AppImage"
for name in ACE BEN; do
  if [[ "$name" == ACE ]]; then smiles='CC(=O)C'; else smiles='c1ccccc1'; fi
  "$app" ligpargen -s "$smiles" -n "$name" -r "$name" -c 0 -cgen CM1A-LBCC \
    -p "$result/$name" > "$result/$name-parameterization.log" 2>&1
  "$python" scripts/verify_opls_export.py --loader "$result/$name/$name.openmm.py" \
    --output "$result/$name-verification.json"
done
"$python" scripts/package_opls_extended.py "$result"
printf '\nExtended numerical checks complete. Independent-engine parity is still pending.\n'
printf 'Download: %s/results.zip\n' "$result"
