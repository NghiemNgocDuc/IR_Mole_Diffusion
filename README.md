# IR to molecular graph and 3D coordinates

**Already trained on Unity?** Start with the current guides below. These workflows
reuse your trained checkpoints and do not require retraining the coordinate model.

| Task | Guide |
| --- | --- |
| Prepare the current accuracy experiments | [Accuracy update v2](ACCURACY_UPDATE_V2.md) |
| Run or monitor the three-variant fine-tuning pilot | [Fine-tuning pilot](FINETUNE_PILOT.md) |
| Rotate and inspect molecular predictions in 3D | [Interactive viewer](VIEWER_3D_UPDATE.md) |
| Try the separate candidate scorer | [Compatibility experiment](COMPATIBILITY_EXPERIMENT.md) |

See the [documentation and repository map](docs/README.md) for results, research,
dataset audits, and the purpose of each directory. The setup instructions below
cover a new installation.

The supported workflow has two trainable stages:

1. **IR → molecular graph:** a masked spectrum transformer predicts SELFIES.
2. **IR + predicted graph → coordinates:** an equivariant diffusion model
   generates conformers, with optional bond/clash guidance and MMFF relaxation.

No known formula or SMILES is required by `predict_structure.py`. Both stages
must be trained. The supplied legacy checkpoints are preserved but incompatible
with this corrected pipeline; they are rejected explicitly rather than partly
loaded. New checkpoints contain the full encoder, vocabulary, preprocessing,
and training state. No external encoder file is needed when loading them.

This is a research baseline. Local checks establish that the code learns, saves,
resumes, and samples. Measured Unity graph accuracy and its scope are recorded in
the [validation-subset baseline](docs/results/stage1_baseline_66782537.md).
See also [research decisions](docs/RESEARCH.md) and
[implementation verification](docs/VALIDATION.md).

## 1. Set up Python on Unity

Run from the project root on UMass Unity. Use a Python 3.12 environment. For a
project environment outside your home quota, replace the environment name with a
path under your group's storage and set `CONDA_ENV` to that path when submitting.

```bash
module load conda/latest
conda create -n ir-diffusion python=3.12 pip -y
conda activate ir-diffusion
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
mkdir -p logs
```

CUDA 12.6 is an example of an official PyTorch 2.8.0 wheel. Check the allocated
node's driver with `nvidia-smi`; use a compatible wheel from the
[official version instructions](https://pytorch.org/get-started/previous-versions/).
Do not train on a login node. The scripts use one generic GPU and FP32 for broad
compatibility; mixed precision is an optional optimization after a GPU smoke run.

For CPU development, install the CPU wheel instead:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
```

The Windows-local runtime is `.runtime/python/python.exe`. Do not upload
`.runtime`, `__pycache__`, or `.pytest_cache` to Unity; create a Linux environment.

For transfer, `python scripts/package_unity.py` creates
`outputs/unity_training_bundle.zip` with supported source, the original SQuIRL
input for the smoke check, and prepared data/splits. It omits old weights,
Windows dependencies, caches, and diagnostic runs. Extract it into a fresh
project directory on Unity. Its file hashes are in `BUNDLE_MANIFEST.json`.

## 2. Prepare and inspect the data once

Conversion uses CPU only and can run in a CPU allocation. If the already
validated `data/processed/squirl.npz`, `splits.json`, and metadata report are
uploaded together, reuse them instead of converting again.

```bash
srun --partition=cpu --cpus-per-task=4 --mem=32G --time=01:00:00 \
  python -u prepare_squirl.py --trust_legacy
python audit_dataset.py --data data/processed/squirl.npz \
  --splits data/processed/splits.json --output data/processed/audit.json
```

`--trust_legacy` applies only to your supplied object-array source. All training
and evaluation use numeric/unicode arrays with `allow_pickle=False`.

Read `data/processed/squirl.metadata.json` before training. It records every
excluded source row, the reason, source/data hashes, preprocessing, and graph
alignment method. The supported domain is connected, closed-shell H/C/N/O/F
molecules with no formal atomic charges and at most 29 atoms including H.
Connectivity or source/geometry mismatches are quarantined, not relabeled.
The connectivity check is conservative and may exclude valid unusual geometries;
exclusion is not a definitive claim that the source molecule is invalid.
Conversion fails if rejection exceeds 10%; raising that limit should follow an
inspection of the report. Original files are never overwritten.

Both stages share one molecule-grouped 80/10/10 split. Canonical achiral identity
keeps alternative orderings and enantiomers together. Do not regenerate splits
between stages. The larger `ir_dataset.npz` lacks 3D labels and is not mixed into
this recipe; doing so requires a separate leakage-aware pretraining experiment.

Peak preprocessing is shared by conversion and prediction: frequencies / 4000,
intensities / 1000, sort by frequency, at most 81 peaks, and an explicit mask.
These are the new recipe's declared scales, not a claim about the unknown
preprocessing of the old checkpoints. Peak frequencies must be in cm^-1 and
intensity units must match the supplied simulated dataset.

## 3. Run a short GPU check before a long job

Submit the short check as a batch job, so closing the browser does not stop it:

```bash
mkdir -p logs
sbatch --partition=gpu-highvram smoke_test.sh
# Replace JOBID with the returned number:
squeue -j JOBID
tail -F logs/smoke-JOBID.out logs/smoke-JOBID.err
```

The launchers use a login Bash shell to initialize Unity's `module` command.
`sbatch --wrap` uses `sh`, which may lack that command; use the provided scripts.
If dependencies were installed successfully into `base` with user installs,
add `--export=ALL,CONDA_ENV=base` to **each** `sbatch` command, including both
training stages. The default remains the dedicated `ir-diffusion` environment.
For example, `sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base smoke_test.sh`.

The smoke test converts 300 source rows, trains both small models, resumes the
coordinate model, evaluates held-out samples, and runs the IR-only entry point.
Its weights and accuracy are diagnostic only. Each job uses a fresh output
directory, `runs/smoke_gpu_JOBID`. Wait for `Smoke complete` before full training.
Ctrl+C leaves the log viewer without cancelling the job. Local CPU results are
in `docs/VALIDATION.md`; CUDA/SLURM must be checked on Unity because this computer
has no CUDA device.

For a measured time estimate on your allocation, run
`python scripts/benchmark_training.py` inside a separate interactive GPU allocation. It writes
`runs/benchmark.json`. See [training-time and accuracy assessment](docs/PERFORMANCE_EXPECTATIONS.md)
for workload calculations, published comparison results, and the current
`gpu-highvram` partition option for A100/H100/L40S devices.

## 4. Train stage 1, then inspect held-out graph predictions

```bash
sbatch train_spectrum.sh
```

Defaults are in `configs/spectrum.json`: encoder width 256, four encoder and four
decoder layers, batch size 64, learning rate 2e-4, and up to 100 epochs. The
encoder and decoder train together from scratch. BOS/EOS/PAD and vocabulary are
embedded in the checkpoint. No original encoder implementation is needed.

After `sbatch` returns a job ID, the training job runs independently of your
computer. Monitor with `squeue -u "$USER"` and
`tail -F logs/spectrum-JOBID.out logs/spectrum-JOBID.err`. The progress bar is
enabled and Python output is unbuffered. Outputs:

- `runs/spectrum/checkpoints/best.ckpt`: lowest validation loss, measured with EMA.
- `runs/spectrum/checkpoints/last.ckpt`: latest epoch/end-of-run training state.
- `runs/spectrum/recovery/last.ckpt`: periodic recovery state every 1000 steps.
- `runs/spectrum/metrics/version_*/metrics.csv`: training curves.
- `resolved_config.json` and `split_manifest.json`: source hashes, settings, data audit, and split evidence.

Assess identification in a GPU batch job before investing in stage 2:

```bash
sbatch --partition=gpu-highvram eval_spectrum.sh
# For the user's existing base + user-install setup:
# sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base eval_spectrum.sh
tail -F logs/graph-eval-JOBID.out logs/graph-eval-JOBID.err
```

This launcher activates the same environment as training. `tqdm` is explicitly
pinned in `requirements.txt` because it is needed when importing Lightning as
well as for progress bars. If it was supplied only by a login node's module
installation, it may be missing on the compute node. For the existing
`base` + user-install setup, install a personal copy without changing the
shared module installation:

```bash
module load conda/latest
conda run -n base --no-capture-output python -m pip install --user --ignore-installed --no-deps tqdm==4.70.1
conda run -n base --no-capture-output python -c "import tqdm, pytorch_lightning; print('Imports OK', tqdm.__version__)"
```

Loading the module alone does not activate `base`; the explicit `conda run`
selects its Python rather than the system Python. Do not bypass the system's
externally managed environment guard.

No training or checkpoint replacement is needed for this dependency repair.

This reports top-1/top-K achiral molecular identity, top-1 formula accuracy, and
the fraction with no valid candidate. Failed rows stay in the denominator.
Teacher-forced token accuracy alone is insufficient. If graph accuracy stays
poor, inspect preprocessing, training curves, and candidate sequences before
training a large coordinate model.

## 5. Train stage 2

After stage 1 produces `best.ckpt`:

```bash
sbatch train_diffusion.sh
```

Defaults in `configs/diffusion.json`: width 256, six layers, spectrum-to-node
attention, graph conditioning, 1000 diffusion steps, batch size 32 with two-step
accumulation, denoiser LR 1e-4, encoder LR 1e-5, encoder frozen for five epochs,
and up to 200 epochs. The transferred encoder must share the same dataset,
splits, preprocessing, and spectral control; mismatches fail early.

Start with the pure denoising baseline (`physics_weight=0`). Both training
coordinates and Gaussian targets are masked and centered. Covalent edges are
retained, schedules are consistent, losses are averaged per molecule, nonfinite
losses/gradients stop the run, and EMA is saved for evaluation.

The provided jobs request less than 48 hours. If a run needs more time, resume
the latest checkpoint. [Unity's partition documentation](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/)
also describes `--qos=long`. Check queue policy and resources before using it.

### Resume an interrupted run

Prefer the epoch checkpoint if recent; use the recovery checkpoint if it saves
substantial work. Substitute `spectrum` for the first stage.

```bash
sbatch --export=ALL,RESUME_CHECKPOINT=runs/diffusion/checkpoints/last.ckpt train_diffusion.sh
# Or, if the periodic recovery checkpoint is newer:
sbatch --export=ALL,RESUME_CHECKPOINT=runs/diffusion/recovery/last.ckpt train_diffusion.sh
```

Resume restores optimizer, scheduler, model, and EMA. A recovery checkpoint from
inside an epoch can repeat or skip some shuffled batches; it is not a bitwise
continuation of data order. Use epoch checkpoints for reproducibility-sensitive
comparisons. Resume keeps the checkpoint's model hyperparameters; config changes
do not silently redesign it. Use `--finetune_from` for a fresh optimizer and a
new experiment directory.

### Optional physics fine-tuning

After measuring the baseline, run a separate experiment with a small bond/clash
penalty. This coefficient is a starting point to tune on validation data.

```bash
sbatch train_diffusion.sh \
  --finetune_from runs/diffusion/checkpoints/best.ckpt \
  --save_dir runs/diffusion_physics --physics_weight 0.001 \
  --unfreeze_at_epoch 0 --lr 0.00003 --encoder_lr 0.000003 --epochs 50
```

Physics training uses only sufficiently low-noise timesteps. Angle, planarity,
torsion, and chirality weights remain zero unless explicit verified targets are
provided. No reference-derived angles are smuggled into prediction. Compare
raw outputs and refinement separately; a lower denoising loss does not establish
better chemistry.

## 6. Evaluate coordinate quality and the benefit of IR

```bash
sbatch eval_diffusion.sh
```

This evaluates validation molecules with their graph supplied and explicitly
labels that task. It does not measure IR-only identification. Reports contain
raw samples, finite/failure rates, proper-rotation RMSD, and optional RDKit
chemical reconstruction and symmetry-aware same-graph RMSD. No failed samples
are silently removed. SDF-based atom symmetry matters when interpreting RMSD.

Coordinate evaluation now logs a flushed progress bar, counts, elapsed time,
and phase-local ETA for sampling, geometry/chemistry checks, and optional MMFF
refinement. Updates appear in `logs/evaluate-JOBID.out` approximately every
15 seconds while work advances. The evaluation output directory also has an
atomically updated `progress.json` status snapshot. ETA applies to the current
phase, not the whole job or scheduler wait. Append `--progress_interval 5` for
more frequent updates or `--no_progress` to disable them.

For an older Unity upload, create a small update with:

```bash
python scripts/package_unity.py --progress_only --output outputs/unity_progress_update.zip
```

Let an active evaluation finish, then upload the ZIP into `~/diffusion` and
run `unzip -o unity_progress_update.zip` there. Subsequent jobs use progress
automatically; no dependency installation, retraining, or checkpoint transfer
is needed. See [the update guide](EVALUATION_PROGRESS_UPDATE.md). Already running
processes do not acquire the display from a source update.

For comparison with the default prediction pathway's MMFF relaxation, evaluate
the same rows and seed in a separate run, retaining the original raw report:

```bash
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base eval_diffusion.sh \
  --refine --refine_iterations 500 --output_dir evaluation/diffusion_val_refined
```

This is evaluation, not retraining. `summary.json` contains separate `raw` and
`refined` metrics and convergence/failure status counts. Refinement uses the
declared graph and sampled coordinates, not reference coordinates. The chemistry
identity check also includes stereo assigned from 3D; it differs from stage 1's
achiral identity metric. Best-of-five uses the reference for scoring and does
not provide an automatic conformer-ranking method.

Run ablations on the same held-out rows with `--spectrum_control removed` and
`--spectrum_control shuffled`. For example:

```bash
python evaluate_diffusion.py --checkpoint runs/diffusion/checkpoints/best.ckpt \
  --data data/processed/squirl.npz --splits data/processed/splits.json \
  --use_inference_graph --chemistry --device cuda --limit 500 \
  --num_samples 5 --n_steps 200 --spectrum_control shuffled \
  --output_dir evaluation/diffusion_shuffled
```

If removing or shuffling spectra barely changes performance, do not claim the
coordinate model uses IR effectively. Compare the graph-only conformer baseline:

```bash
python evaluate_rdkit.py --split val --limit 500 --num_samples 5 \
  --output_dir evaluation/rdkit_val
```

Use validation for model choice; evaluate
`--split test` only after fixing the configuration. Reports keep reference
graphs used for scoring distinct from graphs available during generation.

## 7. Predict from IR alone

On a GPU allocation:

```bash
python predict_structure.py \
  --spectrum_checkpoint runs/spectrum/checkpoints/best.ckpt \
  --diffusion_checkpoint runs/diffusion/checkpoints/best.ckpt \
  --peaks_json my_peaks.json --device cuda --beam_size 5 \
  --conformers 5 --n_steps 200 --output_dir predictions/my_sample
```

`my_peaks.json` contains measured/simulated frequencies and intensities:

```json
{"frequencies": [1050.0, 1720.0, 2950.0], "intensities": [42.0, 160.0, 25.0]}
```

That three-peak example illustrates the file format, not an identified molecule
or a realistic complete training spectrum. Alternatively use `--data
data/processed/squirl.npz --row 123`; generation reads only that row's spectrum,
not its reference structure. Prediction exports `prediction.json`, all raw/final
coordinates in numeric `coordinates.npz`, and molecules in `candidates.sdf`.
New predictions also save `input_ir.json` with the actual retained input peaks
in physical frequency/training intensity units, so figures remain portable.

Graph candidates are ordered by the spectrum decoder's sequence score.
Conformers are unranked; MMFF energies may be compared within one graph. The
report exposes bond/clash screening and refinement status. No valid completed
graph produces an explicit failure report rather than fabricated atoms.

**Physical plausibility is not proof of spectral agreement.** Bond constraints,
RDKit valence checks, and force-field relaxation do not compute an IR spectrum.
Validate promising candidates with a trusted forward spectrum model or matched
quantum frequency/dipole-derivative calculations before claiming they reproduce
the observed IR. Simulated-to-experimental transfer, missing peaks, phase,
intensity normalization, and absolute stereochemistry need separate validation.

## 8. View an IR and molecular prediction picture

After stage 2 finishes, install the optional plotting dependency in the same
environment. For the current `base` + user-install setup:

```bash
module load conda/latest
conda run -n base --no-capture-output python -m pip install --user -r requirements-visualization.txt
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base predict_example.sh
```

For an already uploaded older checkout, `python scripts/package_unity.py
--visualization_only --output outputs/unity_visualization_update.zip` creates
a small update containing only `plot_prediction.py`, `predict_example.sh`, and
`requirements-visualization.txt`. Upload and unzip that archive in the Unity
project root. It requires no dataset/checkpoint upload or replacement of the
currently running training code. The plotter supports the existing prediction
CLI's output format.

The job reads the first validation row's IR input, generates candidates with
both trained checkpoints, and writes `predictions/example_JOBID/result.png`
and `result.svg`. View/download the PNG through OnDemand **Files**. The panels
show input IR peak sticks, the first predicted graph, and its first saved 3D
sample. Conformers are unranked; the picture reports geometry-screen/refinement
status and retains explicit failure panels if a prediction fails.

To plot another candidate from an existing prediction directory:

```bash
python plot_prediction.py --prediction_dir predictions/my_sample --graph_index 1
```

The graph index is zero-based. The plotter uses saved numerical coordinates;
it does not replace failed coordinates with RDKit-generated conformers. An old
prediction without `input_ir.json` can still be plotted from the project root
if its source peaks JSON or processed NPZ is available. Ground-truth molecular
identities and coordinates are not read for plotting.

This picture does not show reconstructed IR. That would require a separate
forward spectrum model or compatible vibrational frequency calculation and
comparison with the original IR. No synthetic curve is labeled as a prediction
of the generated molecule's IR.

## Repository layout

| Path | Purpose |
| --- | --- |
| `diffusion/` | Supported encoder, graph decoder, denoiser, physics, data, and trainer |
| `prepare_squirl.py` | Verified conversion and shared molecule splits |
| `train_spectrum.py`, `train_diffusion.py` | Supported stage entry points |
| `evaluate_spectrum.py`, `evaluate_diffusion.py` | Separate held-out identification and geometry evaluation |
| `predict_structure.py` | IR-only graph-to-coordinate prediction |
| `plot_prediction.py`, `predict_example.sh` | IR/graph/3D result pictures from actual prediction outputs |
| `configs/`, `*.sh`, `scripts/` | Training recipes, Unity jobs, real-data smoke check |
| `tests/` | Component and pipeline regression tests |
| `docs/` | Documentation index, research, audits, and recorded results |
| `outputs/` | Upload bundles, generated reports, and archived local diagnostics |
| `legacy/` | Original source before repairs; not supported launch scripts |
| `data/`, `ckpt/`, `models_formula/` | Original inputs and legacy checkpoints, preserved |

Old `train_diffusion_v2.py` and `inference_v2.py` now route to the corrected
training/evaluation entry points and use the new CLI. The old `src_v2` encoder
source is preserved as `legacy/spectrum_models.py`; it is not a supported model.
`REPO_AUDIT.md` documents the historical problems. This folder arrived without
Git metadata; initialize/version it yourself before recording cluster runs.
