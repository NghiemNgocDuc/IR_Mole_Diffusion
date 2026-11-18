# Ethanol relaxation diagnostic

Upload unity_opls_relaxation.zip to ~/diffusion, unzip, then submit:

```bash
sbatch --qos=short relax_opls_ethanol.sh
```

Requires the successful ethanol export at outputs/opls_ethanol_1788228/ETH.openmm.py.
Use only a trusted locally generated loader (it executes Python). This test uses
OpenMM Reference and the generated OPLS loader. It repeats prerequisite numerical
checks, then minimizes template and seeded perturbed coordinates independently.
Both before/after PDB and full precision nm NPY coordinates are retained.

A pass requires nonincreasing energy, final RMS force components <=1 kJ/mol/nm,
and RDKit coordinate-inferred ethanol connectivity. The minimizer is capped at
1000 iterations; returning alone is not interpreted as convergence. A directory
already present is refused, preventing overwrite. JSON errors have nonzero exit.

Reports are outputs/opls_relaxation_JOBID/report.json. PDB topology is fixed;
separate RDKit coordinate-based bond perception checks the final geometry rather
than assuming that unchanged PDB topology proves preserved connectivity. This is
an ethanol-only diagnostic, not a general identity/stereochemistry validator.
Neither a local minimum nor these numerical tests establish physical truth,
reference-engine parity, a global minimum, or valid model atom mapping.

Local checks passed: ethanol identity perception for original/perturbed exported
coordinates, known force RMS, NaN rejection, Python compilation, Bash syntax.
Actual OpenMM minimization must run on Unity; it was not executed locally.

OpenMM minimizer documentation:
https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html
