# OPLS export numerical verification

This is the next implementation after software setup. It does not change model
weights, generation, or the existing MMFF path. It needs a successful LigParGen
export from the ethanol job. Use only a trusted local generated loader: importing
it executes Python code.

Upload and extract `unity_opls_verification.zip` in `~/diffusion`. Once job
1788228 successfully generates the ethanol files, inspect the actual loader path:

```bash
cd ~/diffusion
find outputs/opls_ethanol_1788228 -name '*.openmm.py'
```

Then pass that exact path to the launcher, for example if ETH.openmm.py is at
the top level:

```bash
mkdir -p logs
sbatch verify_opls_export.sh outputs/opls_ethanol_1788228/ETH.openmm.py
```

The loader's documented `build()` interface must return `(pdb, system)`. We use
the generated Python loader, not the XML alone, to retain its OPLS handling.
The actual export/interface has not yet been tested on Unity. Errors produce a
JSON report and nonzero exit status; nothing is silently substituted.

Checks use OpenMM Reference on the original and a reproducibly perturbed geometry:
- finite energy and forces;
- each force component against a central finite difference of energy;
- energy invariance under rigid translation/rotation;
- force invariance/covariance under those transforms and near-zero net force.

Coordinates are nm, energies kJ/mol, forces kJ/mol/nm. The finite difference
step is 0.0001 nm, energy absolute tolerance 0.001 kJ/mol, force absolute tolerance
0.05 kJ/mol/nm and relative tolerance 0.001. These are numerical diagnostics,
not calibrated physical acceptance thresholds. This version rejects periodic
systems, constraints and virtual sites rather than claiming to validate them.

Report: `outputs/opls_verification_JOBID/report.json`. It records package version,
export hashes, system hash, force classes and each check. A passing report does
not prove OPLS correctness: reference-engine energy parity, parameter coverage,
charge/atom mapping and chemical identity still need verification before sampler
integration. A consistently wrong force field can pass numerical checks.

Local validation: six analytic tests cover a known harmonic energy, wrong force
sign, both tenfold unit errors, invalid coordinates, and an external restraint
that violates translation invariance. Actual OpenMM/LigParGen integration is
pending; OpenMM is not installed in the local Windows Python environment.

Interface source: https://jorgensenresearch.com/ligpargen/guide
