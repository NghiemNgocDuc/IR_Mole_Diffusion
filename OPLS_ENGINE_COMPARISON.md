# OpenMM / GROMACS single-point energy comparison

Upload unity_opls_engine_comparison.zip to ~/diffusion, then:

```bash
cd ~/diffusion &&
unzip -o unity_opls_engine_comparison.zip &&
sbatch --qos=short compare_opls_engines.sh
```

Loads openmpi/5.0.5-cuda12.8 then gromacs/2024.3-cuda12.8, as reported by Unity.
Runs on one CPU task with CPU nonbonded/bonded work; no GPU requested.
Uses existing ethanol job 1788228 and acetone/benzene job 1791099 exports.
Only trusted local generated Python loaders should be supplied.

Six comparisons: template and seeded perturbed coordinates for each molecule.
Atom-name order must agree between OpenMM PDB and GROMACS ITP. Coordinates are
centered, rounded once to eight decimal places in nm, and shared by both engines.
GROMACS runs zero MD steps, with no constraints, thermostat, pressure coupling,
dispersion correction or potential shifts. The 10 nm periodic box and 4 nm
cutoffs enclose all intramolecular interactions, while excluding periodic images:
the runner explicitly requires every intramolecular distance <1 nm. OpenMM uses
the loader's isolated NoCutoff default. This is a small-molecule diagnostic,
not a general solvent/periodic comparison.

Fixed absolute total-energy tolerance: 0.05 kJ/mol. This allows single-precision
arithmetic and rounded exported torsion/angle constants. Failure is reported,
not corrected by adjusting parameters or relaxing the tolerance automatically.
GROMACS warnings are not overridden with maxwarn. Full logs, input files, energy
files, source hashes and report are archived even after a caught comparison error.

Download outputs/opls_parity_JOBID/results.zip. Module-loading failures occur
before Python runs; inspect logs/opls-parity-JOBID.err in that case.

Scope: agreement of total energies for these six geometries only. It does not
establish force parity, universal parameter correctness, model atom mapping,
physical ground truth or the global minimum. Exporter code is shared upstream,
so engine agreement cannot expose every shared parameterization error.

Local validation covers energy-output parsing and malformed/nonfinite/multiple
row rejection, coordinate serialization precision, Python/Bash syntax, and ZIP
integrity. GROMACS/OpenMM execution awaits Unity; neither engine was run locally.

References:
- https://manual.gromacs.org/2024.3/user-guide/mdp-options.html
- https://manual.gromacs.org/2024.3/onlinehelp/gmx-energy.html
