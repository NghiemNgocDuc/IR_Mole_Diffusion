# Molecular generation and physical-validation handoff

Updated: **November 18, 2026**.

## Start here

We are building an automated pipeline that takes IR spectra, predicts molecular graphs, generates 3D coordinates, and assesses whether those **original generated coordinates** are physically plausible. The user wants multiple candidates retained, not necessarily a single winner. The current focus is physical validation, not further Top-1 reranking experiments.

The main result: the pipeline runs, and direct DFT checks have now evaluated saved generated geometries. However, **OPLS guidance/relaxation is not yet reliable enough for automatic acceptance or rejection**. In six tested pairs, OPLS relaxation increased DFT energy. Engine disagreement occurs specifically in two nearly collinear relaxed geometries, not in the corresponding raw geometries.

The latest code update makes **raw unguided coordinates the default primary output** and requires `--experimental-opls` for paired guidance/relaxation. It also supplies a focused precision diagnostic. This update is implemented and packaged locally, with **30 tests passed**, but **the user has not yet confirmed installing/submitting this latest package on Unity**. No new precision-result archive or job ID has been received.

Immediate next action: check whether the user submitted `probe_opls_precision.sh`. If results are available, inspect `precision_results.zip`. Otherwise use the installation/submission command below. Do not rerun completed DFT experiments just to recover reports.

## Workspace and execution

- Local project: `C:\Users\User\Downloads\diffusion`.
- Local shell: PowerShell.
- Local Python: `.runtime/python/python.exe` (PyTorch, NumPy, RDKit, pytest available).
- Unity project: `/home/dnghiem_umass_edu/diffusion`, usually `~/diffusion`.
- Unity physics Python: `.runtime/opls-tools/bin/python`.
- Unity model launcher uses `conda run -n base`; submissions use `--export=ALL,CONDA_ENV=base`.
- No local ORCA/GROMACS/OpenMM execution was used for the latest tests. Real physics calculations run on Unity; local tests use controlled fixtures and archived data.
- There is no established direct remote execution connection. The workflow has been: create ZIP locally, user uploads through Unity OnDemand, user runs provided commands, user downloads results into the local project.
- The working tree contains many existing uncommitted changes. Preserve them; do not reset or broadly replace project files.
- User prefers simple explanations, one concrete next command, and progress without repeated permission questions.

Known working module loads:

```bash
# ORCA
module purge
module load openmpi/4.1.6 orca/6.1.0

# GROMACS
module purge
module load openmpi/5.0.5-cuda12.8 gromacs/2024.3-cuda12.8
```

Use these separately. ORCA requires the OpenMPI 4.1.6 parent, whereas this GROMACS module uses OpenMPI 5.0.5.

## Pipeline components and model checkpoints

Flow: IR spectrum → graph candidates → coordinate diffusion model → saved raw coordinates → optional experimental OPLS assessment/repair → independent DFT assessment.

- Coordinate model: `EquivariantDiffusion3D_v4`.
- Coordinate checkpoint: `runs/diffusion/checkpoints/best.ckpt`.
- Spectrum checkpoint: `runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt`.
- Dataset: `data/processed/squirl.npz`.
- Splits: `data/processed/splits.json`.
- LigParGen: `tools/LigParGen-3.0.1.AppImage` on Unity.
- Parameter cache: `outputs/opls_parameter_cache`; original cache paths are also recorded in benchmark metadata.
- OpenMM version in completed runs: 8.6.1.
- ORCA version in completed runs: 6.1.0.

Checkpoint discovery originally found three eligible coordinate checkpoints, so explicit paths above resolved the ambiguity. Do not replace them with legacy, diagnostic, or randomly initialized weights.

Current pilot scope is connected neutral closed-shell H/C/N/O/F molecules, 3–29 explicit atoms, with achiral graph checks. No experimental stereochemistry or general 3D identity claim has been established.

## Completed experiments and result files

| Experiment | Unity job(s) | Unity output | Local archive |
|---|---|---|---|
| OPLS benchmark | 1820024 | `outputs/opls_benchmark_1820024` | `benchmark_results.zip` |
| OPLS diagnosis | 1820275 | See diagnostic report/archive | `diagnostic_results.zip` |
| Physical-validation preparation | 1849289 | `outputs/physical_validation_1849289` | See refreshed archive below |
| DFT optimization/frequency array | 1849450; collector 1849452 | Same physical-validation directory | `physical_validation_results (1).zip` |
| Fixed-coordinate DFT preparation | 1857206 | `outputs/raw_geometry_1857206` | See refreshed archive below |
| Fixed-coordinate DFT array/collector | 1857217 / 1857218 | Same raw-geometry directory | `raw_geometry_results (1).zip` |

**Use the `(1)` archives.** The unsuffixed `physical_validation_results.zip` and `raw_geometry_results.zip` are initial preparation archives with `missing_result` entries. They are stale snapshots, not evidence that calculations failed.

Complete archive SHA256 values:

- `physical_validation_results (1).zip`: `eb42ef98273b6e0cca08b7fcabda5af82c847d706282dbc671bfb521456b975a`
- `raw_geometry_results (1).zip`: `d2f9cc7e3f7713335edc3058f099cf2030bf9de9a18c24b2068bbcdd999f8884`

The archived ORCA output/final-coordinate hashes were verified. All 15 native gradient files were re-parsed locally; their atom order and evaluated coordinates matched the submitted unchanged XYZ files.

## Physical results

These are **three selected training examples**, graph 0, conformer 0, with unguided and OPLS-guided arms. They are diagnostics, not a held-out performance estimate.

| Input index | Molecular graph |
|---|---|
| 1 | `NC=Nc1n[nH]nc1O` |
| 7 | `CC1=CC2(C#N)NC2C1` |
| 13 | `CNC1C2COC21C#N` |

### DFT at unchanged coordinates

Fifteen calculations: six generated geometries, six OPLS-relaxed counterparts, and three atom-mapped dataset-reference geometries.

Method: `B3LYP D4 def2-SVP TightSCF EnGrad`, isolated vacuum, neutral singlet. There is **no optimization** in these calculations. Forces are minus the gradient, converted from Eh/bohr to kJ/mol/nm. RMS below is across all 3N force components.

| Input | Arm | Raw force RMS | OPLS-relaxed force RMS | Raw energy above dataset reference (kJ/mol) | Relaxed energy above reference (kJ/mol) |
|---|---|---:|---:|---:|---:|
| 1 | unguided | 486.150 | 1157.091 | 30.859 | 50.345 |
| 1 | guided | 1122.907 | 1157.030 | 49.898 | 50.361 |
| 7 | unguided | 195.025 | 433.808 | 2.584 | 22.305 |
| 7 | guided | 263.872 | 428.315 | 3.210 | 23.560 |
| 13 | unguided | 600.693 | 468.034 | 20.646 | 39.313 |
| 13 | guided | 399.781 | 445.387 | 12.237 | 43.789 |

Reference force RMS: approximately 172.943, 170.612, and 177.934 kJ/mol/nm for inputs 1, 7, and 13. The dataset references are **not guaranteed minima under this DFT method**.

Findings:

- All 15 geometries retained the expected coordinate-inferred achiral graph and had no exact atom overlaps.
- OPLS relaxation raised DFT energy in **all six paired cases**.
- Guidance improved raw DFT energy for input 13 but worsened inputs 1 and 7.
- Forces, energy, graph identity, and spectral consistency are different criteria; none alone proves physical correctness.

### DFT optimization and frequencies

Twelve calculations started from raw or OPLS-relaxed coordinates in both arms. Method: `B3LYP D4 def2-SVP TightSCF TightOpt Opt Freq`.

All twelve terminated normally, optimization converged, and inferred achiral graphs were retained.

| Input | Frequency outcome after DFT optimization | Raw-start repair RMSD, unguided / guided (Å) | IR cosine, approximately |
|---|---|---|---:|
| 1 | One imaginary mode in each run; minimum unresolved | 0.2641 / 0.2595 | 0.143 |
| 7 | No imaginary modes at reported precision in all four runs | 0.0638 / 0.0639 | 0.461 |
| 13 | No imaginary modes at reported precision in all four runs | 0.1725 / 0.1700 | 0.443 |

Input 1 imaginary frequencies, in order unguided raw, unguided relaxed, guided raw, guided relaxed: **−13.17, −68.18, −41.49, −68.52 cm⁻¹**. Numerical sensitivity versus a genuine remaining unstable motion is unresolved. Do not automatically discard small negative modes or call these confirmed minima.

Passing frequencies supports a **local minimum of the repaired structure under the selected DFT model**. It does not validate the raw model output, establish a global minimum, or prove experimental identity.

### IR comparison

- Calculated harmonic IR peaks were compared with the supplied dataset spectrum, not a verified experimental spectrum.
- Gaussian FWHM: 20 cm⁻¹; frequency scale: 1.0; grid: 400–4000 cm⁻¹ in 2 cm⁻¹ steps.
- Cosine scores are descriptive, not accuracy/probability or calibrated rejection thresholds.
- Input 1's strongest calculated peak is near 1759.7 cm⁻¹ versus a target peak near 1722.3 cm⁻¹. Peak shifts can strongly reduce overlap at this linewidth.
- A same-method reference-structure IR baseline remains outstanding. Do not attribute the low cosine entirely to generated geometry before establishing it.

## OpenMM–GROMACS disagreement

Earlier broad wording that “molecules 7 and 13 fail parity” was refined: **all three raw geometries passed**. Only relaxed geometries 7 and 13 failed.

| Input | Stage | Energy difference at GROMACS-evaluated coordinates (kJ/mol) | Maximum force difference (kJ/mol/nm) | Passed |
|---|---|---:|---:|---|
| 1 | raw | 0.0000493 | 0.0493 | Yes |
| 1 | relaxed | 0.0000177 | 0.0380 | Yes |
| 7 | raw | 0.0001444 | 0.0602 | Yes |
| 7 | relaxed | 0.6883 | 5.95 × 10⁷ | No |
| 13 | raw | 0.0001170 | 0.0327 | Yes |
| 13 | relaxed | 0.9946 | 3.12 × 10⁷ | No |

Near-collinear nonzero torsions dominate the problematic relaxed structures. Reported minimum torsion-plane sines are about 2.56×10⁻⁸ and 1.56×10⁻¹¹ for inputs 7 and 13. Original relaxed-coordinate torsion force RMS was about 3.52×10⁵ and 1.226×10⁷ kJ/mol/nm.

This supports a numerical-conditioning/precision hypothesis but **does not establish the sole cause**. Do not silently delete torsion terms, relax tolerances until tests pass, or interpret the issue as proof that these molecules are physically impossible. Even perfect engine agreement would not establish agreement with DFT.

## Latest implemented update — deployment not confirmed

Package: `outputs/unity_pipeline_raw_default.zip`.

Changes:

- `predict_opls.py`: default unguided generation without starting the OPLS backend. Saves raw NPY, NPZ with topology, and raw SDF. Hashes the raw NPY.
- Explicit `--experimental-opls` enables paired guided generation and OPLS relaxation. Baseline generation occurs before parameterization; backend failure cannot erase the saved baseline.
- Separate `generation_status` and `assessment_status`; relaxed SDF filenames use `opls_relaxed`. Experimental NPZs retain separate raw and relaxed arrays.
- `benchmark_opls.py`: paired OPLS benchmark now requires the same explicit experimental flag.
- `scripts/compare_opls_engines.py`: adds a bounded optional coordinate-offset argument for diagnosis; default behavior remains 5 nm.
- `scripts/probe_opls_precision.py` and `probe_opls_precision.sh`: focused engine diagnostic described below.
- `PIPELINE_RAW_DEFAULT.md`: installation, use, scope, and limitations.
- `docs/results/physical_validation_1849289_review.md`: detailed local evidence review.
- `scripts/package_pipeline_raw_default.py`: builds a guarded staged update with hashes.

Validation: **30 tests passed** across `tests/test_pipeline_raw_default.py`, `tests/test_opls_pipeline.py`, `tests/test_opls_benchmark.py`, and `tests/test_physical_validation.py`. Bash syntax and ZIP integrity checked. Installation tested over previously shipped benchmark/physical-validation versions, including repeat installation and refusal to overwrite unknown user edits. Actual new Unity execution remains pending.

The generic model sampling API was not changed. This default/opt-in policy applies to the pipeline and benchmark entry points. No automatic energy-based candidate rejection was introduced.

## Next Unity command

If the latest package has not already been deployed, upload `outputs/unity_pipeline_raw_default.zip` into `~/diffusion`, then:

```bash
cd ~/diffusion &&
unzip -o unity_pipeline_raw_default.zip &&
./.runtime/opls-tools/bin/python install_opls_pipeline.py &&
mkdir -p logs &&
sbatch --qos=short probe_opls_precision.sh
```

The installer verifies known previous versions and backs up changed files. If it reports unexpected edits, inspect the conflict rather than bypassing the guard.

The diagnostic performs **24 single-point engine comparisons**: three selected molecules × four coordinate cases × two box offsets. Cases are saved guided raw, saved guided relaxed, and two fixed perturbed relaxed geometries (0.001 and 0.01 Å per atom). Offsets are 0.5 and 5 nm. It preserves original failures and records engine/version information. A passing perturbed geometry never validates its original singular geometry.

Resources: one CPU, 4 GB, 30-minute allocation. No retraining, DFT, minimization, or parameter changes.

After submission, replacing `JOB_ID`:

```bash
cd ~/diffusion
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -n 80 logs/opls-precision-JOB_ID.out logs/opls-precision-JOB_ID.err
```

Download `outputs/opls_precision_JOB_ID/precision_results.zip` when complete.

For later raw-generation smoke testing (not required to repeat the completed experiments):

```bash
sbatch --qos=short --export=ALL,CONDA_ENV=base run_opls_pipeline.sh \
  --coordinate-checkpoint runs/diffusion/checkpoints/best.ckpt \
  --spectrum-checkpoint runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt
```

## Remaining work, in order

1. Obtain/analyze the precision diagnostic; isolate coordinate precision, nearly collinear torsions, and possible parameter/export issues. Keep guidance experimental until resolved and independently useful.
2. Investigate input 1's imaginary mode with tighter numerical settings and mode inspection; retain the original result and compare transparently. This follow-up has not been implemented/submitted in the latest package.
3. Compute a consistent reference-structure IR baseline before using cosine as a geometry-quality test. This is not yet implemented/submitted.
4. Once engineering issues are understood, assess original geometry quality and repaired quality separately on a prespecified broader sample, eventually a genuinely untouched evaluation set.
5. Only then consider calibrated rejection rules or improved guidance. Do not use arbitrary global energy cutoffs, compare absolute energies across different molecular graphs, or claim exact Boltzmann sampling from the current bounded correction heuristic.

## Earlier accuracy work — background only

The original motivation was Top-1 graph accuracy around 64% with Top-5 around 89–90%. A residual selector gave assessment Top-1 63.60% → 63.66% (+0.06 percentage points; paired exploratory interval crossed zero), so no useful improvement was established and no weights were promoted.

Token-matcher IR-shuffling diagnostics showed accuracy dropping from about 35–38% with paired IR to about 16–18% with shuffled IR. That supported dependence on the spectrum, not adequate discrimination among similar candidates. Further matcher experiments were run, but no new accuracy claim should be inferred from this handoff. The user subsequently redirected attention to the physical 3D pipeline.

## Supporting documentation

- `docs/results/physical_validation_1849289_review.md`
- `PIPELINE_RAW_DEFAULT.md`
- `raw_geometry_validation/README.md`
- `PHYSICAL_VALIDATION_PILOT.md`
- `OPLS_DIAGNOSTIC.md`
- ORCA frequency guidance: https://www.faccts.de/docs/orca/6.1/tutorials/prop/freq.html
- ORCA EnGrad: https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/basics.html
- GROMACS precision: https://manual.gromacs.org/documentation/current/reference-manual/definitions.html

Suggested opening message for the next chat:

> Read NEW_CHAT_HANDOFF.md and continue the molecular 3D physical-validation pipeline. First check whether the latest precision diagnostic was submitted and whether precision_results.zip is available. Preserve raw model outputs, keep OPLS guidance experimental, and distinguish unchanged-coordinate quality from DFT-repaired geometry quality.
