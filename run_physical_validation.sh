#!/bin/bash -l
#SBATCH --job-name=physics-prep
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/physics-prep-%j.out
#SBATCH --error=logs/physics-prep-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1 OPENMM_CPU_THREADS=1
output="$PWD/outputs/physical_validation_${SLURM_JOB_ID:?Submit with sbatch}"
python="$PWD/.runtime/opls-tools/bin/python"
gmx_args=()
if module load openmpi/5.0.5-cuda12.8 gromacs/2024.3-cuda12.8; then
    if command -v gmx_mpi >/dev/null; then
        gmx_args=(--gmx "$(command -v gmx_mpi)")
    elif command -v gmx >/dev/null; then
        gmx_args=(--gmx "$(command -v gmx)")
    fi
fi
"$python" -u scripts/validate_opls_physics.py prepare --output "$output" "${gmx_args[@]}"
# Probe ORCA separately so GROMACS MPI libraries do not leak into DFT jobs.
if (module purge; module load openmpi/4.1.6 "${ORCA_MODULE:-orca/6.1.0}"; command -v orca) >"$output/orca_discovery.log" 2>&1; then
    qm_job=$(sbatch --parsable --export=ALL --array=0-11%2 run_physics_dft.sh "$output")
    qm_job=${qm_job%%;*}
    printf 'DFT array job: %s\n' "$qm_job" | tee "$output/submission.log"
    sbatch --dependency="afterany:$qm_job" collect_physics_validation.sh "$output" | tee -a "$output/submission.log"
    printf 'DFT jobs queued; report will be collected automatically. Output: %s\n' "$output"
else
    module spider orca >>"$output/orca_discovery.log" 2>&1 || true
    "$python" scripts/validate_opls_physics.py collect --output "$output"
    printf 'ORCA unavailable with current module setup. DFT inputs are prepared; send physical_validation_results.zip including orca_discovery.log.\n'
fi
