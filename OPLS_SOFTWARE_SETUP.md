# Step 1: software setup on Unity

Use the Linux x86-64 LigParGen 3.0.1 AppImage from:
https://jorgensenresearch.com/ligpargen/download

The current official guide says this distribution includes BOSS, Python and
RDKit and supports command-line invocation. This differs from the legacy Python
LigParGen package, which requires a separate BOSS installation. Do not pip-install
the legacy package expecting BOSS to appear automatically.

Upload the AppImage into ~/diffusion/tools/ and this setup ZIP into ~/diffusion.
Then run:

```bash
cd ~/diffusion &&
unzip -o unity_opls_setup.zip &&
bash setup_opls_tools.sh
```

If the downloaded filename differs, pass its actual full path as the script's
first argument. The setup creates .runtime/opls-tools with conda-forge Python
3.11, OpenMM, RDKit and NumPy. It leaves the model's base environment unchanged.
The solver selects compatible package releases and writes an explicit package
lock to outputs/opls_setup/conda-explicit.txt for reproducibility. This is a
separate tooling environment, not yet the integrated PyTorch inference environment.

Checks: imports, a known two-particle harmonic-bond energy on the OpenMM CPU
platform, and LigParGen's headless command-line help. These do not establish
OPLS parameter correctness or GPU usability. A listed CUDA platform alone does
not mean a GPU test passed. Run OpenMM's testInstallation only on an allocated
compute node for GPU checking.

If the AppImage reports missing FUSE or another runtime dependency, retain the
log and diagnose it before choosing an extraction-based or alternative package
installation. Do not attempt sudo or system driver changes on Unity.

Next milestone: parameterize a small known molecule locally, load its OPLS-correct
OpenMM export (including the generated loader, not XML alone), verify atom mapping,
energies and forces. No trained weights or generation code change in this step.

Sources:
- https://jorgensenresearch.com/ligpargen/guide
- https://docs.openmm.org/latest/userguide/application/01_getting_started.html

The script is prepared locally; actual Unity installation is not yet verified.

## Cache permission fix (2026-11-15)

The supplied Unity log failed while opening a shared package-cache metadata file:
`/work/pi_zhoulin_umass_edu/.conda/pkgs/cache/497deca9.info.json`.
The updated script overrides CONDA_PKGS_DIRS after loading the conda module, using
`~/diffusion/.runtime/opls-conda-pkgs`. This avoids the shared cache without
changing its permissions, deleting packages, disabling plugins or changing the
user's global Conda configuration. The environment prefix remains unchanged.

Upload `unity_opls_setup_cache_fix.zip` to `~/diffusion`, then:

```bash
cd ~/diffusion &&
unzip -o unity_opls_setup_cache_fix.zip &&
set -o pipefail &&
bash setup_opls_tools.sh 2>&1 | tee opls_setup.log
```

Reference: https://docs.conda.io/projects/conda/en/24.7.x/user-guide/configuration/settings.html
