#!/bin/bash -l
# One-time Unity setup. Does not modify the model's base conda environment.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load conda/latest
# Set this AFTER module load: Unity's module can select a shared project cache.
# Scope the override to this script; do not change ~/.condarc or shared caches.
export CONDA_PKGS_DIRS="$PWD/.runtime/opls-conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"
[[ -w "$CONDA_PKGS_DIRS" ]] || { printf 'Package cache is not writable: %s\n' "$CONDA_PKGS_DIRS" >&2; exit 1; }
printf 'Using private Conda package cache: %s\n' "$CONDA_PKGS_DIRS"
prefix="$PWD/.runtime/opls-tools"
app="${1:-$PWD/tools/LigParGen-3.0.1.AppImage}"
[[ -f "$app" ]] || { printf 'Missing Linux LigParGen AppImage: %s\nDownload from https://jorgensenresearch.com/ligpargen/download and upload it here.\n' "$app" >&2; exit 1; }
mkdir -p outputs/opls_setup
if [[ ! -d "$prefix/conda-meta" ]]; then
  conda create -y -p "$prefix" --override-channels -c conda-forge python=3.11 openmm rdkit numpy
fi
conda list -p "$prefix" --explicit > outputs/opls_setup/conda-explicit.txt
conda run -p "$prefix" --no-capture-output python -c 'import openmm, rdkit, numpy; from openmm import unit; s=openmm.System(); s.addParticle(12); s.addParticle(12); f=openmm.HarmonicBondForce(); f.addBond(0,1,0.1,100); s.addForce(f); i=openmm.VerletIntegrator(0.001); c=openmm.Context(s,i,openmm.Platform.getPlatformByName("CPU")); c.setPositions([[0,0,0],[0.2,0,0]]); e=c.getState(getEnergy=True,getForces=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole); assert abs(e-0.5)<1e-6; print("OpenMM",openmm.__version__,"RDKit",rdkit.__version__,"NumPy",numpy.__version__); print("CPU energy smoke test passed (harmonic bond, NOT OPLS)."); print("Available platforms:",[openmm.Platform.getPlatform(n).getName() for n in range(openmm.Platform.getNumPlatforms())])' 2>&1 | tee outputs/opls_setup/openmm-check.txt
chmod u+x "$app"
sha256sum "$app" > outputs/opls_setup/ligpargen-sha256.txt
# This is a headless CLI availability check, not molecule parameterization.
QT_QPA_PLATFORM=offscreen timeout 120 "$app" ligpargen --help > outputs/opls_setup/ligpargen-help.txt 2>&1 || {
  cat outputs/opls_setup/ligpargen-help.txt
  printf '\nLigParGen CLI did not start. Send this output for diagnosis; do not install system drivers or use sudo.\n' >&2
  exit 1
}
cat outputs/opls_setup/ligpargen-help.txt
printf '\nSETUP CHECK COMPLETE. Environment: %s\nOpenMM CPU works and LigParGen CLI starts. OPLS parameterization, import and GPU execution still require separate tests.\n' "$prefix"
