# Independent physical-validation pilot

This package adds bounded parameter-template retries and prepares a small,
selected diagnostic using the saved benchmark 1820024. It does not change model
weights or remove force-field terms. The previous benchmark remains unchanged.

## Run on Unity

Upload `unity_physical_validation.zip` to `~/diffusion`, then:

```bash
cd ~/diffusion &&
unzip -o unity_physical_validation.zip &&
./.runtime/opls-tools/bin/python install_opls_pipeline.py &&
mkdir -p logs &&
sbatch --qos=short run_physical_validation.sh
```

Preparation checks the problematic torsion examples at saved raw and relaxed
coordinates against GROMACS. It preserves the original OpenMM diagnosis and
reports coordinate-rounding effects: single-precision GROMACS can move an almost
collinear triple enough to change its force dramatically. Agreement at rounded
coordinates is not evidence that the original singular geometry is sound.
Original cached exports must match the archived hashes. No reparameterization
is substituted for an old candidate in the comparison.

The previously rejected `CNC12COC1C2C#N` template is retried with up to three
deterministic ETKDG embeddings. Only embedding failure and LigParGen's explicit
unbonded-overlap precheck cause retry. Other failures remain failures. Charge,
graph and mapping checks are retained; each attempt is recorded. A fresh cache
key prevents mixing old and new preparation protocols.

The script discovers ORCA with `module load orca`. If a module name must be
specified, export `ORCA_MODULE` before submission. No software license is
accepted or software downloaded automatically. If ORCA is unavailable,
preparation still produces a downloadable report with module discovery output.

When ORCA is available, the script submits 12 serial DFT tasks, at most two
concurrent, with a 4-hour allocation per task (3h40 process timeout), then an
automatic report collector after all tasks terminate. DFT jobs use the ordinary
queue, not the short QoS. Queue time and completion are not guaranteed. A failed
or preempted task stays visible in the report.

## Fixed pilot design

- Inputs 1, 7, 13; graph 0; conformer 0; both generation arms; raw and relaxed.
- One converged control and two known torsion failures. These are selected
  training examples, not a representative sample or held-out evaluation.
- Gas-phase neutral singlet B3LYP-D4/def2-SVP, TightSCF/TightOpt, optimization
  followed by harmonic frequencies. This is an initial screening level.
- Record termination, optimization convergence, complete 3N frequencies,
  every negative frequency, coordinate-inferred achiral graph preservation,
  aligned displacement and maximum change in pair distances.
- Negative frequencies below -20 cm^-1 are flagged for investigation; smaller
  negatives also require recheck and cannot pass automatically. No arbitrary
  deletion of imaginary modes. Zero modes are handled as reported by ORCA.
- Compare calculated IR peaks to the saved input spectrum only if original
  dataset and spectrum hashes and preprocessing scales are verified. Fixed
  Gaussian FWHM 20 cm^-1, range 400-4000 cm^-1, scale factor 1.0. Cosine is
  descriptive: no tuned cutoff, identity probability or experimental claim.
- Raw and relaxed starts can converge to the same minimum. They are paired
  repair diagnostics, not independent conformers for Boltzmann weighting.

The achiral pilot does not establish stereochemical correctness. Small forces
and no reported imaginary modes support a local minimum under the chosen
method, not a global minimum or molecular identity. Repair displacement must
be considered when assessing the generator itself. No universal repair-size
or spectral-match acceptance cutoff is introduced here.

## Outputs

`outputs/physical_validation_JOBID/physical_validation_results.zip` contains
`RESULTS.md`, `summary.json`, `plan.json`, discovery/submission logs, DFT input,
output and final XYZ files, IR curves, retry records and parity logs. During
preparation an initial ZIP is produced with DFT tasks explicitly pending; the
dependent collector replaces it with final results. Download after the collector
finishes. Each DFT task also writes its own `result.json`.

## Sources

- ORCA optimization/frequency workflow: https://www.faccts.de/docs/orca/6.1/tutorials/prop/freq.html
- ORCA IR output and intensity conventions: https://www.faccts.de/docs/orca/6.1/tutorials/spec/IR.html
- LigParGen usage: https://jorgensenresearch.com/ligpargen/guide

DFT and GROMACS execution must be verified on Unity; the local Windows test
environment does not contain those engines or OpenMM.
