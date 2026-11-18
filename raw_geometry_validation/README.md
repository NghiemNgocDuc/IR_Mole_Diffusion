# Direct checks of generated coordinates

This separate directory can be deployed while physical-validation jobs run.
It changes no existing pipeline modules, launchers, checkpoints, or results.
Use only this directory from `unity_raw_geometry_validation.zip`.

```bash
cd ~/diffusion &&
unzip -n unity_raw_geometry_validation.zip &&
mkdir -p logs &&
sbatch --qos=short raw_geometry_validation/submit.sh
```

The preparation job verifies benchmark 1820024 and its original dataset hash.
It selects the same three diagnostic inputs (1, 7, 13), graph 0, conformer 0:

- Generated coordinates, unguided and OPLS-guided (six structures).
- Their saved OPLS-relaxed coordinates (six structures).
- Corresponding dataset-reference coordinates (three structures), with verified
  graph identity and atom mapping. All selected graphs match these references.

Fifteen ORCA `B3LYP D4 def2-SVP TightSCF EnGrad` jobs calculate energy and
gradients at fixed coordinates. There is no `Opt` or `Freq` keyword. Each job
uses one CPU and 4 GB with a 55-minute process timeout and one-hour allocation;
at most two tasks from this new array run simultaneously. Existing arrays keep
their own concurrency limits. The scheduler may queue tasks.

## Checks and measurements

- Verify every gradient's atom order and evaluated coordinates against the
  submitted XYZ (absolute tolerance 1e-6 angstrom). Preserve input hashes.
- Reject incomplete/nonfinite gradient files and unsuccessful engine output.
- Forces equal minus the energy gradient. Native Eh/bohr values are saved;
  report kJ/mol/nm using Eh = 2625.49963948 kJ/mol and bohr = 0.529177210544 A.
- Report RMS across the 3N force components, largest per-atom force norm,
  force vectors, and net force. Small forces do not establish a local minimum:
  the separate vibrational calculation is needed to distinguish saddle points.
- Report original bond lengths, angles, exact atom overlaps, and nonbonded
  distances relative to RDKit vdW radii, excluding bonded and 1-3 pairs. These
  are descriptive, without arbitrary global rejection thresholds.
- Compare energy and force RMS to the corresponding dataset geometry under
  the same DFT setup/version. This is not an OPLS/DFT energy comparison.
- The dataset reference is a control, not guaranteed to be a DFT minimum or
  experimental ground truth. Its measured forces are reported too.
- These are three selected training examples, not a held-out performance study.
  The pilot is achiral and does not establish stereochemical correctness.

## Combining with the current optimization run

The collector reads completed tasks in `outputs/physical_validation_1849289`
and joins them only after checking method, ORCA version, source archive,
starting coordinates, graph preservation and input/output hashes. It reports
the energy drop to the optimized structure and the recorded repair RMSD.
Pending, failed and incompatible optimization results are explicitly marked.
It never modifies or waits on the existing optimization jobs.

If those optimization jobs finish later, regenerate this report with:

```bash
./.runtime/opls-tools/bin/python raw_geometry_validation/check.py collect \
  --output outputs/raw_geometry_PREPARATION_JOB_ID
```

## Results

The dependent collector produces
`outputs/raw_geometry_PREPARATION_JOB_ID/raw_geometry_results.zip` with
`RESULTS.md`, `summary.json`, original XYZ, bond/angle/contact measurements,
DFT inputs/output, `.engrad` files and force CSVs. An initial ZIP produced
during preparation labels unexecuted tasks as missing results. Use the ZIP
after the final collector finishes.

No DFT calculation is run locally: Windows has no ORCA. Tests use controlled
gradient/output fixtures plus the archived real coordinates. Actual numerical
results require Unity execution. A finite EnGrad result is a measurement,
not a universal physical-validity pass/fail label.

References:
- ORCA EnGrad run type: https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/basics.html
- ORCA gradient ordering/units: https://github.com/faccts/orca-external-tools#output-syntax
- Constants: https://physics.nist.gov/cuu/Constants/
