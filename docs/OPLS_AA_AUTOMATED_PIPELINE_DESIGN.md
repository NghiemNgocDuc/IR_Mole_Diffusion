# Automated OPLS-AA guidance and validation pipeline

Design date: 2026-11-12. Status: proposed implementation, not an installed or
validated OPLS backend. No current sampler, checkpoint or scientific result is
changed by this document.

## Objective and user interface

One Unity submission should take an IR input through graph prediction,
parameterization, coordinate generation with physics feedback, final checks,
within-graph conformer weighting, and the existing offline viewer. No manual
website submissions, file conversion or switching applications per molecule.
External physics libraries still run behind the scenes. A one-time environment
setup and backend verification are required.

Proposed command (not implemented):

```bash
sbatch predict_physical.sh --input input_ir.json --config configs/opls_guided.yaml
```

The configuration specifies both model checkpoints, force-field/charge-model
versions, environment, conformer/attempt budgets, sampling seed and guidance
policy. A companion preflight command must explain missing dependencies before
an expensive job starts. Existing runs remain readable and unchanged.

## Existing integration points

- predict_structure.py already predicts graphs, uses Chem.AddHs, builds explicit
  topology and samples coordinates. Attach parameter preparation after graph
  construction, before conformer sampling. Never supply reference graphs at
  IR-only inference.
- diffusion/model.py:sample predicts x0 at each step, but current physics
  guidance acts on the updated noisy x using geometric penalties. Add a separate
  opt-in OPLS policy working on the estimated clean coordinates. Do not silently
  reinterpret the existing physics_guidance option as OPLS.
- Current optional refinement is MMFF. OPLS mode must use an explicitly named
  OPLS refinement path and separate output fields. Unsupported OPLS cases must
  not silently fall back to MMFF and receive an OPLS pass label.

## Architecture

IR -> candidate graphs -> all-atom identity records -> parameter cache
   -> independent coordinate trajectories with physics feedback
   -> final raw/guided/relaxed validation -> deduplicated conformer ensembles
   -> report.json + structures.sdf + viewer.html + results.zip.

Suggested modules:

| Module | Responsibility |
|---|---|
| opls_parameters.py | Local LigParGen invocation, identity mapping, parameter cache |
| opls_backend.py | Verified OpenMM systems, energies, forces, minimization |
| guidance_policy.py | Noise gate, bounded correction, rejection state machine |
| physical_validation.py | Geometry, convergence, stereochemistry and status checks |
| conformer_ensemble.py | Symmetry-aware deduplication and approximate weights |
| predict_physical.py / .sh | Orchestration, restart, provenance, reporting |

The interface between the sampler and backend is deliberately small:
evaluate(positions_angstrom) -> energy_kj_mol, forces_kj_mol_angstrom,
component_energies, status. The backend owns unit conversion and atom ordering.
OpenMM forces can be used directly for inference updates; no differentiation
through OpenMM is needed. Future training losses would need a separately tested
autograd bridge and are outside the first implementation.

## 1. Identity and topology preparation

Create an immutable record of bond orders, formal charges, protonation/tautomer,
specified stereochemistry, atom mapping and explicit hydrogens. Validate valence,
connectivity and supported elements. Do not invent unspecified stereochemistry
or collapse distinct tautomers into a shared parameter cache entry.

Parameterization needs a sensible template, not an overlapping noisy diffusion
state. Build a deterministic template per graph. It is used only to prepare
parameters; its coordinates never replace model-generated coordinates. Archive
the template and seed because charge assignment can depend on geometry.

## 2. Local parameterization and cache

Use a pinned local LigParGen/BOSS distribution via subprocess arguments (no
shell interpolation). The current official guide documents a CLI and OpenMM
exports; test headless execution on Unity before choosing a distribution. The
legacy Python package separately requires BOSS; it is not a complete pip-only
install. Record installed versions, executable hashes and license availability.

Select and name the charge model explicitly: CM1A-LBCC for supported neutral
molecules, a documented ion-compatible CM1A setup for ions. Do not imply every
element, radical or charge state is supported. Record failures individually.

Cache key includes full graph identity, stereo/protonation, total charge, force
field and charge-model versions, template-generation protocol, executable and
configuration hashes. Store template hash, source topology, parameter files,
atom permutation, serialized engine system, charge sum and validation results.
Use locks and atomic writes; verify cached files before reuse. Reuse exactly one
fixed parameter set for all conformers of a given cached identity.

## 3. Energy engine: OpenMM with an OPLS-correct import

OpenMM provides in-process energy and force queries, avoiding process startup at
every denoising step. Keep a context per active graph and reuse it; use bounded
context caching and benchmark CPU versus GPU for these small molecules.

Use a verified exported loader or explicit adapter. OpenMM's ordinary
NonbondedForce uses Lorentz-Berthelot rules; OPLS uses geometric mixing. The
LigParGen guide specifically notes that its OpenMM loader handles mixing and
improper terms that its XML alone cannot express. Validate mixing, exclusions,
1-4 scaling, torsional/improper conventions and absence of double-counted terms.

Initial environment: isolated molecule, nonperiodic and no cutoff, explicitly
labeled a vacuum force-field diagnostic. It is not a solvent free-energy model.
Verify applicability against the dataset's documented simulation conditions.
Any solvent extension needs its own compatible parameters and validation.

Acceptance tests before real guidance: energy-component agreement against a
trusted export/engine on several molecules and distorted conformers; finite-
difference forces; translation/rotation invariance; atom-permutation consistency;
Angstrom/nanometer conversion; charge sum and complete parameter coverage.
Set numerical tolerances based on precision and reference reproducibility.

## 4. Guidance during diffusion

Start with the model's estimated clean geometry x0, converted to physical units,
after an SNR/noise gate. Raw high-noise states are not valid rejection evidence.
Initially the gate and cadence are configurable experimental settings, not
claimed physical constants.

At a checkpoint:
1. Inspect x0 for finite coordinates and basic geometry; obtain energy/forces.
2. Propose a small force-directed update, clipped per atom and bounded in total
   displacement from x0. Center consistently with the existing sampler.
3. Backtrack the step if energy rises; reject the correction if none succeeds.
   Exact overlaps/nonfinite forces need explicit numerical handling, not a
   fabricated low energy. Softened guidance, if later added, must be labeled
   separately from the unmodified OPLS energy used for validation.
4. Use corrected x0 in the sampler's transition. For DDIM, define a consistent
   noise estimate from current xt and corrected x0 when updating the transition;
   for DDPM use the corrected posterior mean. Test both separately and keep
   self-conditioning consistent. Preserve the unguided path exactly at weight 0.
5. Record pre/post correction energy, force norm, movement and accepted step.

This is a proposed inference heuristic, not exact Boltzmann sampling. Strong
guidance may remove diversity or compromise IR conditioning. Benchmark that
tradeoff before increasing guidance strength; no monotonic decrease is required
across all diffusion steps, only within an accepted local correction.

## 5. Early stopping and replacement policy

Distinguish immediate numerical failure, recoverable strain, parameter failure,
and persistent late-stage physical failure. Missing parameters mean unassessed,
not physically impossible. High absolute energy alone is not a graph-independent
rejection rule. Low energy alone also cannot establish correct identity.

First run a shadow policy: log when it would reject but let trajectories finish.
Measure false rejection of trajectories that later pass. Use a training-only
calibration set of trajectories and freeze thresholds before assessment.
Signals include persistent clashes, bond/angle distortion, forces and inability
to improve under a bounded correction budget. Record consecutive checkpoint
violations; do not reject on one intermediate observation. Chirality inversions
must be checked wherever stereo was specified.

When validated, terminate only the failed conformer trajectory and try another
seed up to a fixed per-graph attempt budget. Do not declare the molecular graph
impossible because one or even all sampled trajectories failed. Keep failure
counts and all-attempt denominators; never retry indefinitely until success.

## 6. Final validation, relaxation and Boltzmann weights

Save guided pre-relaxation coordinates first. Run bounded final OPLS minimization
on a copy. Report energy drop, symmetry-aware RMSD/displacement, bond/angle
outliers, nonbonded clashes, maximum/RMS force, graph/stereo checks and convergence
for both copies. A repaired pass must not count as a raw generator pass.

Use distinct statuses: plausible_unrelaxed, plausible_after_relaxation,
failed_physics, unsupported_parameters, numerical_failure, budget_exhausted.
These labels describe the chosen model/checks, not proof of real-world stability.

Deduplicate relaxed conformers within each graph before approximate weighting:
w_i = exp(-(E_i-E_min)/(R*T)); normalize with stable log-sum-exp. Record temperature,
energy units, parameter set and conformer coverage. These weights over discovered
minima omit basin entropy/degeneracy and solvent effects; they are not accurate
equilibrium populations, correctness probabilities, or measures of unseen minima.
Return no weights if no conformers pass. Never compare these normalized weights
across different graphs or let absolute OPLS energies replace the IR graph score.

Future quantum optimization/frequencies and conformer-specific IR calculations
can assess selected cases and improve population/spectral modeling. They are a
separate, more expensive validation layer, not prerequisites for the first pilot.

## 7. Outputs, resumption and user experience

Keep decoder graph order separate from within-graph conformer energy order.
Viewer shows raw/guided/relaxed coordinates, physical status and the reason for
failure, relative energy and explicitly approximate conformer weights. Do not
display a generic green 'correct molecule' badge based on force-field checks.

One result ZIP contains machine-readable records, summary, plots, SDF and offline
viewer. Preserve full traces and parameter provenance locally. Save independent
per-conformer completion records and seeds. Resumption skips only verified
completed attempts with identical input/model/configuration hashes; incomplete
attempts restart with recorded seeds. Parameter workers are isolated and have
timeouts. No manual service uploads or external molecule transmission.

## 8. Validation and implementation sequence

1. Backend milestone: headless local parameterization and cross-engine force/
   energy tests. A missing backend is surfaced before model sampling.
2. Automated post-generation audit on a fixed small set, e.g. 100 training
   molecules selected before outcomes, spanning available atom/ring classes.
3. Frozen-checkpoint comparison: (A) unguided, (B) unguided plus final relaxation,
   (C) guided plus the same final relaxation. Same graphs, seeds and attempt
   budgets. Also compare to the existing geometric-guidance default separately.
4. Shadow rejection calibration, then (D) guided with active rejection/restarts.
5. Frozen-rule assessment and optional quantum subset. No test-driven threshold
   tuning and no model promotion based solely on force-field scores.

Report parameter coverage, convergence and plausibility per attempted conformer
and per graph, raw versus repaired rates, diversity, same-graph reference RMSD
when legitimate, rejection mistakes, runtime including parameterization, time
per accepted conformer and sensitivity to guidance. Early stopping must save
wall time or improve usable yield under equal compute, not just skip steps while
adding a larger force-evaluation overhead. Confidence intervals are clustered by
molecule, not by correlated conformer attempts. Keep wrong graph identity
separate from conformer quality, and report every unsupported/failed case.

## Primary documentation consulted

- https://jorgensenresearch.com/ligpargen/guide (CLI, charge models, coverage and loader)
- https://raw.githubusercontent.com/Isra3l/ligpargen/main/README.md (legacy BOSS dependency)
- https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html (energy/forces)
- https://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html (default mixing)
- https://manual.gromacs.org/2024.6/reference-manual/functions/nonbonded-interactions.html (OPLS mixing)

Documentation availability is not installation verification. The implementation
must pin released versions and pass the backend checks on Unity.
