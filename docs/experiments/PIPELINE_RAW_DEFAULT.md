# Raw coordinates are the primary output

The coordinate pilot now runs unguided by default. It does not start LigParGen
or OpenMM, query OPLS energy, relax coordinates, or reject candidates using
force-field values in this mode. Multiple predicted graphs and conformers are
still supported. Existing checkpoint/provenance checks remain in place.

Each generated conformer has a raw NPY, an NPZ with declared topology, and a
separate raw SDF. The report hashes the NPY. The graph/atom ordering is the
same as the previous pilot; neutral achiral H/C/N/O/F scope is unchanged.
Generation success is not a physical-validity label.

## Installation and ordinary generation

Upload unity_pipeline_raw_default.zip to ~/diffusion, then:

```bash
cd ~/diffusion &&
unzip -o unity_pipeline_raw_default.zip &&
./.runtime/opls-tools/bin/python install_opls_pipeline.py
```

The installer checks known previous versions and backs up changed files.
It does not modify checkpoints, datasets, or existing output directories.

```bash
sbatch --qos=short --export=ALL,CONDA_ENV=base run_opls_pipeline.sh \
  --coordinate-checkpoint runs/diffusion/checkpoints/best.ckpt \
  --spectrum-checkpoint runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt
```

The defaults still use one training spectrum for an engineering pilot. This
is not a new accuracy evaluation or a production validation claim.

## Optional OPLS research mode

Only adding `--experimental-opls` to predict_opls.py/run_opls_pipeline.sh
enables the paired guided arm and OPLS relaxation. The unguided output is saved
before parameterization, so a backend failure cannot remove that baseline.
Generation and assessment outcomes have separate fields. Relaxed SDFs have
`opls_relaxed` in their filenames; raw SDFs have `raw`. Experimental NPZs
retain separate raw_angstrom and relaxed_angstrom arrays for compatibility.

The paired benchmark entry point also requires `--experimental-opls` before
it can run. No old benchmark needs to be repeated to apply this change.
The generic model sampling API is unchanged; this policy applies to these
pipeline and benchmark entry points.

## Focused engine diagnostic

```bash
sbatch --qos=short probe_opls_precision.sh
```

This evaluates only the saved guided seed-0 cases for inputs 1, 7 and 13.
For each molecule it compares raw, relaxed, and two deterministically perturbed
relaxed geometries (per-atom displacement 0.001 and 0.01 angstrom) at two
box locations (0.5 and 5 nm). The original cases are always retained.
It runs 24 single-point engine comparisons, with no model generation, DFT,
parameter fitting, deleted torsions, altered tolerances, or minimization.
It records the GROMACS build version/precision, evaluated positions, forces,
energies, and original parameter hashes. Passing a perturbed case does not
validate the original geometry. Guidance remains experimental regardless of
this diagnostic's outcome.

Output: outputs/opls_precision_JOB_ID/precision_results.zip.
The job uses one CPU, 4 GB, and a 30-minute allocation; queue time is separate.
This requires the original parameter cache from benchmark 1820024 on Unity.

## What is already known

See docs/results/physical_validation_1849289_review.md. All three raw engine
comparisons passed. Only the relaxed geometries of inputs 7 and 13 failed.
Their nonzero torsions involve near-collinear atom triplets. Precision
sensitivity is a supported hypothesis, not yet an established complete cause.
Do not label either molecule physically impossible because OPLS is singular.

Keep four questions separate in reports:
1. Was the original generated geometry preserved and chemically consistent?
2. What are its forces and energy under an independent calculation?
3. How much did optimization repair it, and did the repaired geometry pass
   a complete frequency check?
4. Does its predicted spectrum agree with a reference computed consistently?

The archived DFT checks address the first three only on selected training
examples. IR scores have no calibrated rejection threshold. Molecule 1's
imaginary mode and a matching-method reference IR baseline remain follow-up
work; this package does not silently claim to resolve them.

GROMACS precision documentation:
https://manual.gromacs.org/documentation/current/reference-manual/definitions.html
