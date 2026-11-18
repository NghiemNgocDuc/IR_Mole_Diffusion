# OPLS generation pilot

The readiness archive from job 1807422 passed all six saved energy/force comparisons
and all three coordinate-bridge checks. Saved arrays were independently checked by
`scripts/audit_opls_readiness.py`. Maximum energy discrepancy: 0.0015263 kJ/mol;
maximum force-component discrepancy: 0.049970 kJ/mol/nm. These are implementation
checks on ethanol, acetone and benzene, not physical-truth validation.

This update starts the main generation integration. It does not repeat those jobs.

## One submission

Upload `unity_opls_pipeline.zip` to `~/diffusion` using Unity Files. Then:

```bash
cd ~/diffusion &&
unzip -o unity_opls_pipeline.zip &&
./.runtime/opls-tools/bin/python install_opls_pipeline.py &&
mkdir -p logs &&
sbatch --qos=short --export=ALL,CONDA_ENV=base run_opls_pipeline.sh
```

The installer checks existing file hashes before changing anything. Known previous
versions are backed up under `outputs/opls_pipeline_install_*`. Unexpected local
edits cause it to stop and list conflicts. Do not force past a conflict.

Read `logs/opls-pipeline-JOBID.out` and `.err`. Download
`outputs/opls_pipeline_JOBID/pipeline_results.zip`, including when the job reports
`blocked`. Thirty minutes is the job time limit, not the expected runtime or queue
time. No GPU is required for this small CPU pilot.

## What the run does

1. Inspect checkpoint metadata safely in `ckpt/`, `runs/`, and `lightning_logs/`.
   Select only a uniquely identified format-2 trained coordinate checkpoint with
   `diagnostic=false` and real-spectrum training. Legacy, random, and diagnostic
   weights are not used. If several compatible coordinate checkpoints exist,
   report their paths rather than choose silently. An explicit path can be passed
   with `--coordinate-checkpoint`; its containing directory is also searched.
2. Use the previously evaluated ranking graph checkpoint when compatible, or a
   unique compatible graph model. Verify data, splits and preprocessing match
   both models. Strictly load the architectures and use available EMA weights.
3. Take the first **training** spectrum from the saved split. Decode at beam 5 and
   retain at most two candidate graphs. Reference graph and reference coordinates
   are not inputs to generation. This is an engineering pilot, not an accuracy
   evaluation or model-selection experiment.
4. Create a seeded ETKDG parameterization template for each declared graph. Run
   the pinned local LigParGen 3.0.1 AppImage with CM1A-LBCC. Verify exported graph,
   explicit hydrogens, atom permutation and neutral partial-charge sum. Store and
   hash a reusable parameter set. **Template coordinates never replace samples.**
5. Keep OpenMM in a persistent subprocess running in `.runtime/opls-tools`, while
   the neural model runs in the existing conda environment. Reuse its context.
6. Generate unguided and guided coordinates from the same seed, 50 DDIM steps,
   one conformer per graph. Disable the old geometric guidance for both arms.
   Guidance operates on predicted clean x0 when alpha-bar >= 0.8, with one
   backtracked energy-decreasing step capped at 0.02 Angstrom per atom. Convert
   model coordinate scale explicitly; preserve centering. Recompute DDIM noise
   consistently from xt and corrected x0; use corrected x0 for self-conditioning.
7. Save both arms before relaxation, then run the same bounded OPLS minimization
   on each. Save energy, RMS forces, coordinate-inferred achiral graph match,
   convergence, trace, coordinates, SDFs, parameter files and every failure.

The local checkpoint `ckpt/v4-epoch=156-val_loss=0.2450.ckpt` is legacy (no format
version or model-class metadata). It is **not compatible with the current loader**.
Its presence does not establish that a usable coordinate model exists on Unity.
If discovery finds none, recovering the old architecture and dependencies or
training a compatible coordinate model is the next decision; this script does
neither automatically.

## Scope and limits

- First parameterization scope: one neutral closed-shell H/C/N/O/F molecule,
  3–29 explicit atoms, no isotope labels or explicitly specified stereochemistry.
  Ambiguous or failed export graph matching is reported as unassessed. Unspecified
  stereochemistry remains unspecified; achiral identity checks do not validate it.
- Cache keys contain ordered SMILES, template seed/method, RDKit version,
  executable hash, charge model and backend source hash. Same graph input reuses
  one fixed parameter set. Changed cached files fail hash validation. Stale lock
  directories are reported and must be inspected, never silently overwritten.
- Use only the trusted exported `build()` loader, preserving geometric LJ mixing
  and improper conventions. Nonperiodic, unconstrained OpenMM Reference platform;
  vacuum diagnostic. The first pipeline run on Unity is still required to verify
  the new orchestration with real OpenMM; local tests use analytic/mock backends
  for that interface. Existing cross-engine results cover the three fixtures.
- No active early rejection, no Boltzmann sampling, no conformer population
  estimates, no cross-graph energy ranking, and no checkpoint promotion. Those
  later stages need trajectory evidence and calibrated rules. The guidance gate
  and cap are explicit pilot choices, not established optimal settings.
- Lower energy/forces or a matching perceived graph does not prove a correct
  molecule, a global energy minimum, or agreement with its IR spectrum.
- Parameter failures and sampling failures stay in the denominator. Saved
  post-relaxation success must not be reported as raw generation success.

## Validation performed locally

- 25 tests: OPLS bridge/export numerical checks and new sampler tests, including
  DDIM/DDPM transition consistency, self-conditioning, coordinate scaling,
  masked padding, exact no-op reproducibility, graph mismatch and scope rejection.
- 8 tests: checkpoint discovery/ambiguity, downloadable missing-input reports,
  cache reuse/tamper rejection, persistent worker cleanup, existing sampler and
  basic IR-to-coordinate pipeline regression checks, and guarded upload installation.
- The real local inventory run correctly stopped on missing compatible weights
  and saved a report. It did not run a model or claim physical accuracy.

## Primary implementation references

- [LigParGen guide](https://jorgensenresearch.com/ligpargen/guide): CLI, charge model,
  local execution, exported loader and file conventions.
- [OpenMM local minimizer](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html):
  bounded iteration count and RMS-force stopping criterion in kJ/mol/nm.
