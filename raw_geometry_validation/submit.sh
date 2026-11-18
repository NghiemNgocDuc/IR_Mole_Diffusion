#!/bin/bash -l
#SBATCH --job-name=raw-dft-prep
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/raw-dft-prep-%j.out
#SBATCH --error=logs/raw-dft-prep-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load openmpi/4.1.6 orca/6.1.0
command -v orca
output="$PWD/outputs/raw_geometry_${SLURM_JOB_ID:?Submit with sbatch}"
"$PWD/.runtime/opls-tools/bin/python" -u raw_geometry_validation/check.py prepare --output "$output"
job=$(sbatch --parsable --array=0-14%2 raw_geometry_validation/run.sh "$output")
job=${job%%;*}
printf 'Fixed-coordinate DFT array: %s\n' "$job" | tee "$output/submission.log"
sbatch --dependency="afterany:$job" raw_geometry_validation/collect.sh "$output" | tee -a "$output/submission.log"
printf 'Results directory: %s\n' "$output"
