# Verification — updated October 29, 2026

## Status

Local checks use CPU-only PyTorch. The user subsequently submitted jobs on
Unity and shared the evidence below. Stage 1 has completed, with 70.6% top-1
and 92.4% top-5 graph identity on 500 validation molecules. Stage 2 completed
200 epochs and raw known-graph geometry evaluation. Its 2,500 samples were all
finite, with 96.96% RDKit-valid connected structures. Strict identity/stereo
and 0.5-Angstrom geometry success was 31.4% for the first sample and 69.2%
with reference-based best-of-five selection. End-to-end IR-to-3D, experimental
IR agreement, and full test-set performance remain unmeasured. The trained
Unity checkpoints and arrays have not been downloaded or rescored locally.

The subsequent MMFF comparison converged for all 2,500 samples and raised
RDKit validity to 100%, but reduced strict best-of-five success within
0.5 Angstrom from 69.2% to 66.2%. Both raw and refined outputs should be kept;
force-field convergence is not a guarantee of improved reference geometry.

## Local environment

- Windows, Python 3.12, PyTorch 2.8.0+cpu, Lightning 2.5.5, NumPy 2.2.6.
- RDKit 2025.9.1, SELFIES 2.2.0, NetworkX 3.6.1.
- **58 tests passed** after the November 4 viewer update. The two warnings advise more DataLoader workers; zero
  workers is intentional for the small Windows CPU checks.

## Interactive viewer and optional ranking fine-tune (October 29)

November 4 viewer follow-up:

- Added rotate/zoom/pan/full-screen controls and a perspective camera, removed
  all text overlays from the molecule, and placed element colors and atom or
  distance readouts beneath it. Static 3D pictures also use an outside color key.
- Seven additional tests cover deterministic 2D-SMILES-to-3D generation,
  full-dimensional flexible geometry, explicit generated provenance, charged
  graph fidelity, invalid inputs, and unchanged saved-prediction export.
  `--smiles` is a separate mode; it never replaces saved model coordinates.
- The full suite passed 58 tests. `node scripts/check_viewer_controls.js`
  executes the actual viewer event code with a mocked DOM/rendering adapter,
  checking camera controls, external legend/measurements, precise coordinates,
  export actions, generated/no-IR mode, and failure states.
- Actual offline WebGL rendering was verified with headless Chrome using a
  synthetic saved-coordinate fixture. Screenshot:
  `runs/visualization_validation/viewer-browser.png`. This supersedes the older
  unavailable-browser visual check below. It establishes local rendering;
  user mouse/touch interaction on the user's computer still needs their trial.
- Optional generated conformers use seeded RDKit ETKDGv3 plus MMFF94 when
  supported. They are estimates from a supplied graph, not experimental or
  model-predicted coordinates. The generated candidate-3 HTML is an explicitly
  labeled demonstration, not a new evaluation of Unity's saved prediction.

- Seven new viewer integrity tests verify saved raw/final coordinates and XYZ
  precision, explicit failed frames, empty reports, topology/SMILES consistency,
  rejection of malformed/nonfinite inputs and duplicate IDs, and offline HTML
  export with script-injection escaping. SDF parsing is followed by restoration
  of full saved coordinate precision in the renderer.
- JavaScript syntax passed. The actual control script passed a mocked DOM and
  renderer harness for rotation, zoom, auto-spin, reset, styles, hydrogens,
  raw/final frames, candidate/conformer selection, and XYZ/PNG export wiring.
  This does not establish browser rendering. Native visual WebGL QA remains
  unverified: Computer Use stopped before browser input because it could not
  determine the Chrome URL confidently enough to enforce policy.
- A clearly labeled synthetic ethanol fixture was exported offline. Actual
  Unity prediction arrays remain on Unity; no trained 3D result was invented
  or substituted locally. Viewer HTML preserves candidate order and reads no
  reference identities or coordinates.
- Five new ranking tests cover held-out label exclusion, same-formula
  negatives, data/split hash and row-order rejection, actual encoder/decoder
  gradient changes with the margin objective, and paired validation-only
  comparison. No test example is used for optimization.
- A real tiny CPU fine-tune used 234 training rows and baseline EMA weights,
  took three optimizer steps, strictly reloaded its portable checkpoint, and
  resumed to step four. Optimizer and EMA state restored; source checkpoint
  SHA256 remained unchanged. Evidence is in
  `runs/ranking_validation/verification.json`. Reusing the source output
  directory or an existing fine-tune checkpoint is rejected.
- The full suite passed 51 tests. These are implementation checks, not evidence
  of improved identification. No new Unity fine-tune has run. Mid-epoch shuffled
  sample order remains inexact on resume, as in the existing trainer.
- Active Python files parse, configuration JSON parses, and all Unity shell
  scripts pass Git Bash syntax checks and use LF line endings.

## Full data conversion

Source: `data/SQuIRL/squirl_ir_3d.npz`.

| Result | Rows |
| --- | ---: |
| Checked | 133,795 |
| Accepted | 124,781 |
| Quarantined | 9,014 |
| Train | 99,825 |
| Validation | 12,478 |
| Test | 12,478 |

Rejections: 7,153 source/coordinate connectivity mismatches under the declared
mapping method, 1,844 unsupported formally charged molecules, and 17 invalid
peak rows. This is a conservative filter; it does not establish that every
rejected structure is physically invalid. Every source row/reason is preserved
in `data/processed/squirl.metadata.json`. Originals were not modified.

The prepared NPZ has padded coordinates and atom labels for at most 29 atoms,
81 masked two-feature peaks, explicit bond types, numeric SELFIES sequences of
length at most 23, and an embedded 22-token vocabulary. All 124,781 canonical
achiral identities are unique. A separate pass found no accepted radicals or
isotope-labeled molecules. Native schema validation and grouped manifest
coverage/overlap checks passed. Coordinate units and spectrum/geometry pairing
remain the supplied dataset's convention, not an independent quantum validation.

## Learning and lifecycle checks

- Existing denoiser and physics equivariance, graph, padding, and gradient tests.
- Consistent beta/cumulative-alpha buffers and finite terminal noise precision.
- Centered forward noise, coordinate reconstruction, and the single-atom case.
- Finite diffusion losses/gradients; padded peaks cannot affect conditioning.
- DDIM and full-step DDPM sampling, explicit rejection of shortened DDPM,
  masked outputs, and graph-based guidance.
- Atom-permuted ethanol graph alignment using an independently generated 3D
  structure; a deliberately disconnected geometry is rejected.
- Proper Kabsch rotation aligns a rigidly transformed structure to numerical
  precision, while a reflection is distinguished.
- A tiny spectrum decoder reduces its supervised loss on a fixed batch and
  generates completed SELFIES with the correct BOS/EOS protocol. The resulting
  predicted graph is used to generate finite coordinate arrays without reading
  reference atom labels or coordinates in the prediction function.
- A real Lightning fit, strict checkpoint reload, EMA restoration, optimizer
  resume, and continued global-step advancement. Legacy format rejection.
- Encoder freeze/unfreeze also switches its train/eval behavior correctly.

## Real-data command-line smoke

`python scripts/smoke_test.py --output_dir runs/validation_cpu_v2` completed:

1. Converted 300 source rows: 292 accepted, 8 quarantined; 234/29/29 splits.
2. Trained the small spectrum model for three optimizer updates.
3. Transferred its encoder and trained diffusion for three updates with a small
   physics penalty.
4. Reloaded/resumed diffusion to six total optimizer updates.
5. Evaluated two held-out rows with two conformers each, physics guidance, and
   RDKit chemical scoring; exported numeric samples and JSON reports.
6. Ran held-out spectrum evaluation and the IR-only prediction entry point.

The three-update spectrum model produced no valid graph candidates on those
diagnostic rows. This is recorded as failure, not presented as scientific
success. The learning unit test separately exercises the successful decoded-
graph-to-coordinate path. A useful graph model still needs full training.

A second real-data check used the complete prepared dataset and the production
model widths: the 7,396,118-parameter spectrum model completed two optimizer
updates; its encoder transferred into the 7,937,913-parameter diffusion model,
which completed two optimizer updates with gradient accumulation. Both saved
versioned checkpoints. These remain diagnostic weights and are omitted from
the transfer bundle.

A completed-epoch check confirmed `best.ckpt` and `last.ckpt` are created and
strictly reloadable with EMA. The ETKDG/MMFF baseline completed on two matching
held-out rows and saved its samples and chemistry/geometry report. Those tiny
samples do not establish a performance comparison.

## Unity evidence shared by the user — October 28, 2026

- GPU smoke job 66229263 printed `Smoke complete` with CUDA evaluation enabled.
  The three-update diagnostic spectrum model produced no valid graph candidates
  on its two evaluated rows; those results do not establish trained accuracy.
- Stage 1 job 66229326 reported `COMPLETED`, exit code `0:0`, elapsed
  `03:01:40`. The GPU type, final epoch, training curves, and checkpoint contents
  have not yet been supplied.
- Held-out graph evaluation job 66243338 failed while importing Lightning:
  `ModuleNotFoundError: No module named 'tqdm'`. It did not load or modify the
  trained checkpoint. `tqdm==4.70.1` (the locally tested version) is now explicit
  in requirements, and `eval_spectrum.sh` activates the training environment.
  A subsequent system-Python install attempt hit the externally managed
  environment guard; selecting Conda's base Python explicitly with `conda run`
  allowed the user-only dependency install and import check to succeed.
- Corrected evaluation job 66243541 printed results for 500 validation rows:

  | Metric | Result | Count |
  | --- | ---: | ---: |
  | Top-1 graph identity | 70.6% | 353/500 |
  | Top-5 graph identity | 92.4% | 462/500 |
  | Top-1 formula | 93.0% | 465/500 |
  | No valid candidate | 0.0% | 0/500 |

  These are user-reported terminal results from `best.ckpt`, beam size 5, on
  the first 500 entries in the existing validation split. Identity uses
  canonical non-isomeric SMILES and does not assess stereochemistry. No known
  formula is supplied to generation. A valid candidate means it passed the
  declared graph filters; this does not establish correct 3D geometry or IR
  agreement. The complete JSON report remains on Unity. A manually transcribed,
  explicitly labeled summary is in
  `docs/results/stage1_validation_66243541_summary.json`.

## Stage 2 evidence shared on October 29, 2026

- Training job 66243865: `COMPLETED`, `0:0`, elapsed `16:12:02`; log reached
  epoch index 199 / `max_epochs=200`, displayed final denoising validation loss
  0.108. The user listed `best.ckpt` and `last.ckpt` at 123M each on Unity.
- Evaluation job 66280988 used EMA, DDIM 200 steps, five samples for each of
  500 validation rows, with known atoms/counts/graphs supplied, no sampling
  physics guidance, and no MMFF refinement. The user pasted its complete
  `summary.json`; a labeled transcription is saved in
  `docs/results/stage2_validation_66280988_summary.json`.

  | Raw metric | Reported result |
  | --- | ---: |
  | Finite candidates | 2,500/2,500 |
  | RDKit-valid and connected | 96.96% (2,424/2,500) |
  | Mean / median fixed-correspondence heavy-atom RMSD | 0.836 / 0.855 Angstrom |
  | Strict identity/stereo + RMSD <= 0.5 Angstrom, first sample | 31.4% (157/500) |
  | Strict identity/stereo + RMSD <= 0.5 Angstrom, best of five | 69.2% (346/500) |
  | Strict identity/stereo + RMSD <= 1.0 Angstrom, best of five | 87.8% (439/500) |
  | At least one strict identity/stereo match in five | 92.2% (461/500) |

  Best-of-five is reference-selected coverage, not a learned/available ranking
  method. The chemistry gate uses isomeric SMILES after assigning stereo from
  3D, so failures include stereo disagreement as well as wrong connectivity or
  bond order. It is not the same identity metric as stage 1's achiral SMILES.
  No aggregate end-to-end accuracy follows by multiplying these separate rates.

The existing evaluation CLI's `--refine --refine_iterations 500` path was then
exercised locally on two diagnostic CPU molecules with two samples each. It
saved separate raw/refined metrics and retained all four refinement statuses.
This checks the command path, not the trained Unity model's refined accuracy.

## Unity MMFF comparison shared on October 29, 2026

Evaluation job 66284311 used the same checkpoint/data hashes, split, seed,
five samples per molecule, and 200 DDIM steps, adding MMFF 500 iterations.
The user pasted its full report. All overlapping raw numerical metrics match
the earlier unrefined run, making this a direct raw/refined comparison on
the same generated samples. The labeled report is saved in
`docs/results/stage2_validation_66284311_refined_summary.json`; one terminal
line wrap inside a word was removed for valid JSON, with no numerical changes.

| Metric | Raw | Refined |
| --- | ---: | ---: |
| Finite candidates | 2,500/2,500 | 2,500/2,500 |
| RDKit-valid and connected candidates | 96.96% | 100.0% |
| Candidates with atoms closer than 0.2 Angstrom | 3/2,500 | 0/2,500 |
| Mean fixed-correspondence heavy-atom RMSD | 0.836 Angstrom | 0.841 Angstrom |
| Median fixed-correspondence heavy-atom RMSD | 0.855 Angstrom | 0.883 Angstrom |
| Strict identity/stereo + RMSD <= 0.5 Angstrom, first sample | 31.4% (157/500) | 29.8% (149/500) |
| Strict identity/stereo + RMSD <= 0.5 Angstrom, best of five | 69.2% (346/500) | 66.2% (331/500) |
| Strict identity/stereo + RMSD <= 1.0 Angstrom, best of five | 87.8% (439/500) | 85.6% (428/500) |
| Any strict identity/stereo match among five | 92.2% (461/500) | 92.4% (462/500) |

All 2,500 refinements reported `converged`; no sampling failures were reported.
MMFF improved validity under the declared RDKit check but did not improve
close-reference conformation success on this subset. Keep raw/refined metrics
and outputs separate. This result does not by itself justify further training
or establish end-to-end identification, spectral agreement, or physical
stability. The supplied reference graph still makes this a conditional
coordinate evaluation, and best-of-five still uses reference-based selection.

## Harder held-out example and tautomer audit

Before inspecting its prediction, test row 24778 was selected by structural
criteria: nine heavy atoms, at least two hetero atoms, two rings, and a multiple
bond. Its exact canonical identity, exact model input, and exact RDKit Murcko
scaffold SMILES were absent from train/validation. The selection is retained in
`docs/results/heldout_challenge_24778_selection.json`.

The user shared the picture from prediction job 66285483. Candidate 1 displays
`C1=CCc2[nH]ncc2C1`; the hash-verified test reference is
`C1=CCc2n[nH]cc2C1`. Both have formula C7H8N2 and canonicalize to the same
RDKit tautomer family, but differ in which nitrogen carries the hydrogen.
This is a strict exact-identity miss for the displayed first candidate.
The picture reports a geometry-screen pass and converged MMFF, which do not
check identity against the reference. The complete prediction report was then
provided: five beams yielded four valid graph candidates, with the exact
reference identity at candidate rank 3 (`graph_index=2`). The case is an exact
top-1 miss and top-3/top-5 hit. All twelve returned conformers passed the
geometry screen and reported converged MMFF. No saved prediction arrays were
received, so exact 3D reference agreement remains unverified. Plotting rank 3
using the known answer is diagnostic reference-based selection, not proof
that the deployed ranking chooses it. The full report and local identity
assessment are in `docs/results/prediction_66285483_user_report.json`.

Post-prediction auditing found the predicted alternate tautomer in training
row 24770. The earlier claim of an unseen ring framework was therefore too
strong: the exact identity/input were held out, but the tautomer family was
seen. Exact Murcko-SMILES absence alone does not establish family novelty.
Keep this preselected outcome; future novelty claims require tautomer-family
and tautomer-normalized scaffold checks. This does not retrospectively change
the existing split or its previously declared exact-identity metric. See
`docs/results/heldout_challenge_24778_result.json` for the labeled evidence.

## Evaluation progress update

- Coordinate evaluation now logs phase-specific bars, counts, elapsed time,
  and ETA during sampling, geometry, chemistry, and optional MMFF refinement.
  Flushed newline updates work in Slurm logs, with an atomic `progress.json`
  snapshot in the evaluation output directory.
- Re-ran the real-data diagnostic with two rows, two samples, and MMFF 500.
  Every saved array and raw/refined metric exactly matched the prior diagnostic
  without progress instrumentation. This does not evaluate Unity's trained
  checkpoints locally. All 39 existing tests still pass.
- Checked throttling and phase ETA with a controlled clock, completed/failed
  status handling, and disabled progress. A hard process kill can leave a stale
  status snapshot; Slurm accounting remains the final job-state authority.
- The progress-only transfer archive includes four source/guide files plus
  its own manifest. It needs no new dependency or trained weights. Let active
  evaluation job 66284311 finish before installing the update for later jobs;
  a running Python process does not acquire the new display automatically.

## Remaining scientific and platform validation

- Unity smoke job 66229227 failed before Python started because an ad hoc
  `sbatch --wrap` command ran under `sh`, where `module` was unavailable.
  This was a launcher error, not a model result. The launchers now request login
  Bash, and `smoke_test.sh` provides a batch smoke check with a fresh output
  directory per job. The subsequent user-run GPU smoke check completed.
- Obtain the Unity GPU model, peak memory, and training curves; the local
  machine cannot independently reproduce the CUDA run.
- Retain the 500-row stage 1 results as a preliminary validation result;
  evaluate the full validation set and spectral controls before final model
  selection. The reported stage 2 raw geometry run is a starting validation
  result, not a matched test benchmark.
- Compare known-graph geometry with ETKDG/MMFF and removed/shuffled IR. Keep
  identification error separate from known-graph conformation quality.
- Retain the completed raw/MMFF comparison above and test end-to-end predicted
  graphs separately. Validate final IR agreement
  using a trusted forward spectral method at compatible theory/phase settings.
- Use the test split after model selection. No test examples entered either
  local diagnostic fit.
- Recovery checkpoints restore model/optimizer/scheduler/EMA, but mid-epoch
  sample order is not exactly resumable. Prefer epoch checkpoints when exact
  experiment comparison matters.

## Optional prediction pictures

- Added a CPU postprocessor that plots input peak sticks, a predicted graph,
  and saved sampled coordinates. It reads no reference identities/coordinates
  and does not invent replacement geometries or a reconstructed IR curve.
- Rendered an actual no-valid-graph diagnostic report with explicit failure
  panels. Separately rendered a clearly labeled synthetic layout fixture
  (artificial peaks + RDKit ethanol); it is not a trained prediction result.
- The updated prediction CLI ran with the existing tiny CPU checkpoints and
  exported `input_ir.json`. Its recovered frequencies/intensities match the
  unmasked model input with the declared inverse scales. Saved input makes
  plotting independent of the original dataset path. Original coordinates
  were unchanged after rendering.
- Matplotlib 3.10.7 is optional in `requirements-visualization.txt`. Python and
  shell syntax checks passed. Example pictures were visually inspected for
  labels and overlap. The full prediction + picture batch job still needs
  to run on Unity after stage 2, using the user's trained checkpoints.
