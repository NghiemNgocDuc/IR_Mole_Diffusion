# Repository audit — October 27, 2026

> Historical findings from before the repair. See `README.md`,
> `docs/RESEARCH.md`, and `docs/VALIDATION.md` for the supported replacement
> workflow and current verification. Original code was preserved in `legacy/`.

The main data and checkpoint files are present, but this folder is not ready for end-to-end training or evaluation. It mixes model wrappers, newer shared components, and newer command-line scripts that do not yet share the same interfaces. Several items need integration fixes; adding more data alone will not resolve them.

## What is present and verified

- SQuIRL geometry/IR dataset: `data/SQuIRL/squirl_ir_3d.npz`, 133,795 rows.
- All rows were checked for matching coordinate/atom shapes, matching frequency/intensity shapes, finite values, and nonempty atoms/spectra. These checks passed. Molecules contain 3–29 atoms and elements H, C, N, O, F. No negative intensities were found.
- All 133,795 SMILES strings are distinct as literal strings. This does not verify canonical molecular identity or atom ordering.
- General IR dataset: `data/ir_dataset.npz`, 198,091 rows. Its archive contains spectra, SMILES, and decoder strings, but no coordinates or atom types, so it cannot directly train coordinate diffusion.
- Encoder checkpoint: `models_formula/squirl/best-epoch=033-val_loss=0.102336.ckpt`. Metadata and state dictionary read successfully with `torch.load(weights_only=True, mmap=True)`.
- Diffusion checkpoint: `ckpt/v4-epoch=156-val_loss=0.2450.ckpt`. Metadata and state dictionary read successfully with the same method.
- Vocabulary files and SQuIRL bond statistics are present; their JSON parses successfully.
- Every project Python file parses successfully.
- Existing tests: **30 passed** with `.runtime/python/python.exe -m pytest -q`. These cover the denoiser and physics components, not a complete training/checkpoint/inference workflow.

## Blocking findings

### 1. The v2 model wrapper calls an incompatible denoiser API

Locations: [diffusion_v2/model.py](diffusion_v2/model.py), [diffusion/denoiser.py](diffusion/denoiser.py).

The v2 constructor passes `use_bond_graph` and `bond_emb_dim`; the shared denoiser accepts `graph_conditioning` instead. Training and sampling also pass `bond_adj`, while the denoiser expects `graph={"bond_types": ...}`.

Direct smoke checks reproduced both errors:

```text
TypeError: EquivariantDenoiser.__init__() got an unexpected keyword argument 'use_bond_graph'
TypeError: EquivariantDenoiser.forward() got an unexpected keyword argument 'bond_adj'
```

This affects the workflow selected by `train_diffusion.sh` and `eval_diffusion.sh`. Disabling the bond-graph flag alone does not fix it: the wrapper still passes the unsupported keyword arguments.

Good news: the diffusion checkpoint's denoiser keys and tensor shapes exactly match the current denoiser when explicitly configured with `cond_dim=768`, `hidden_dim=512`, `n_layers=8`, `graph_conditioning=True`, and `readout='residual'`. This establishes structural compatibility only; it does not establish identical numerical behavior.

### 2. The supplied encoder implementation omits trained checkpoint modules

Location: [src_v2/models.py](src_v2/models.py).

Instantiating its `Transformer` from the supplied encoder checkpoint's hyperparameters and comparing state dictionaries found **28 unexpected checkpoint keys**, belonging to `peak_self_attn` and `formula_prefix_proj`. The checkpoint enables `peak_self_attn=True` and `formula_prefix_to_decoder=True`, but the supplied class does not implement those components. `strict=False` can silently discard these trained weights.

The diffusion checkpoint also embeds encoder weights containing these modules. Obtain the complete encoder implementation used to train the checkpoints, including its peak self-attention and formula-prefix dependencies, or implement and validate an explicit compatibility layer. Relaxed loading is insufficient to reproduce the trained model.

### 3. Diffusion checkpoint loading depends on paths absent here

Locations: [diffusion_v2/model.py](diffusion_v2/model.py), [inference_v2.py](inference_v2.py).

The diffusion checkpoint records this encoder source:

```text
./models_formula/stage1_v7/Transformer_ir_g3584_peak_mlp_pk81c2_ir_768x6/best-epoch=034-val_loss=0.092967.ckpt
```

That file is absent. The supplied encoder is a different filename/run: `squirl/...epoch=033...0.102336.ckpt`. The constructor reloads the recorded external encoder before restoring the diffusion state, and the inference loader offers no path override.

The checkpoint also records bond statistics at `/project/pi_zhoulin_umass_edu/multimodel_spec/data/SQuIRL/bond_stats.json`; this needs to resolve to the local file. The current physics loader otherwise warns and uses fallbacks.

The diffusion checkpoint already contains 236 `pretrained.*` state entries. A portable loader may reconstruct the correct encoder architecture and restore those embedded weights without needing a second encoder file. If retaining the present constructor, the original stage1 checkpoint must be supplied or its path explicitly overridden. Do not assume the supplied standalone encoder is the original stage1 run.

### 4. Physics loss is not connected to the required graph targets

Locations: [diffusion_v2/model.py](diffusion_v2/model.py), [diffusion/model.py](diffusion/model.py), [diffusion/physics.py](diffusion/physics.py).

Both wrappers call `self.physics_loss(coords, atom_types, mask)` without a graph. The current physics implementation requires explicit `bond_types` whenever any physics term is active. A direct call reproduced:

```text
ValueError: Active physics losses require an explicit graph with bond_types
```

Default angle and ring terms also require explicit angle/cosine and planarity targets. The shipped NPZ does not contain these arrays. Supply verified targets in coordinate atom order and pass them through, or explicitly disable unavailable terms. A SMILES-derived adjacency alone does not provide all required targets.

Checkpoint physics buffers also use the old `bond_mean`, `bond_std`, and `bond_mask` names; the current implementation uses bond-type-indexed buffers. Their migration needs an explicit policy rather than silent omission.

### 5. The newer native workflow is only partially integrated

Locations: [train_diffusion.py](train_diffusion.py), [evaluate_diffusion.py](evaluate_diffusion.py), [diffusion/model.py](diffusion/model.py), [diffusion/data.py](diffusion/data.py).

- The trainer passes `chirality_weight`, `conditioning_mode`, `readout`, `graph_conditioning`, and optionally `encoder_spec`; `diffusion/model.py` does not accept these options. The `encoder_spec` constructor smoke check fails with `TypeError`.
- The model still imports missing `src.models_with_formula_acc_v2` / `src.models_with_formula_acc`. Direct construction reproduces `ModuleNotFoundError: No module named 'src'`. The available package is `src_v2`.
- The native loader returns dictionary batches; the model's training/validation steps still unpack tuple batches.
- Native evaluation can pass `graph` to `sample()`, but the model's sampler does not accept it.
- The scripts advertise `checkpoint_policy`, but the model has no migration/validation implementation for that policy.
- Both shipped NPZ files contain object arrays. Native loading rejects them, as directly verified. SQuIRL needs conversion to padded numeric coordinates/atoms, packed or dense spectra, counts/masks, molecule IDs, and any required graph targets. No conversion tool or converted dataset is included.
- External mode imports absent `dataset_utils_qmugs`, although the available file is `dataset_utils_qmugs_v2.py`. It also requires a molecule-ID `.npy` sidecar, which is absent.

### 6. Required packages and reproducible setup are missing

The bundled `.runtime` has NumPy, CPU PyTorch, Lightning, and pytest, but lacks **SciPy, RDKit, and SELFIES**.

Both `train_diffusion_v2.py --help` and `inference_v2.py --help` currently fail at import with `ModuleNotFoundError: No module named 'scipy'`. RDKit is required by SMILES-to-bond conversion; SELFIES is required for predicted SELFIES decoding. The latter currently catches import/decoding errors and returns an empty string, potentially hiding graph-generation failure.

There is no `requirements.txt`, `pyproject.toml`, conda environment file, or README explaining a supported setup. The shell scripts assume a Linux SLURM cluster, `module load conda/latest`, and an environment named `ir`; that environment is not supplied. No CUDA device is available in the bundled local runtime. SLURM log output paths also require the `logs` directory to exist before submitting a job.

## Additional correctness and reproducibility gaps

1. **Incorrect RMSD rotation in legacy inference.** `inference_v2.py::_kabsch_align` calculates a column-vector rotation and applies it to row vectors without transposing it. On a four-atom structure transformed by a pure rotation and translation, it reports approximately **1.93649 Å**, while `diffusion.evaluation.kabsch_rmsd` reports approximately **7.9e-16 Å**. Repair this before interpreting legacy evaluation metrics.
2. **No held-out selection in the legacy evaluator.** `inference_v2.py` evaluates the first N rows of the full dataset. `train_diffusion_v2.py` makes a 90/10 random train/validation split and saves no row manifest or test split. The encoder summary reports an 80/10/10 split, but only counts are present. Obtain the original row IDs/manifests and dataset fingerprint to establish which rows are unseen by both encoder and diffusion model. Exact SMILES uniqueness does not establish absence of canonical or cross-stage overlap.
3. **SELFIES training loader expects the wrong type.** SQuIRL `decoderin` and `decoderout` contain strings, such as `[START][C]` and `[C][STOP]`. `_selfies_item()` casts these directly to integer arrays and fails. Tokenization and padding are missing for `mode='selfies'`; ordinary diffusion-mode loading does not use this branch.
4. **Greedy decoder omits token protocol.** `predict_selfies_greedy()` initializes with token zero (`[nop]`) instead of the vocabulary's `[START]` token, and does not stop at/remove `[STOP]`. The 75-token vocabulary has START=55 and STOP=56. Decoding needs to match the training convention.
5. **Vocabulary mapping needs provenance.** The checkpoint declares 75 classes, matching the size of `data/symbol_to_idx.json`. The SQuIRL-specific mapping has 33 classes and different token IDs, so it is not interchangeable. Checkpoint metadata does not include the token mapping; obtain confirmation of the exact mapping used in training.
6. **Peak encoding and padding need alignment.** The v2 peak path returns only `pre.feature_extractor(packed)`, ignores the supplied mask, and skips the checkpoint's peak self-attention. Condition pooling also includes padded peaks. Confirm the original trained forward path and mask propagation.
7. **Intensity preprocessing is ambiguous.** Both shell scripts divide intensities by 1000, but the standalone encoder checkpoint records `ir_peak_int_scale=None`. Raw SQuIRL intensities range from 0.5 to about 2546.6. Obtain the resolved training preprocessing before deciding which scaling is correct; metadata alone does not prove the scripts are wrong.
8. **Corrected diffusion utilities are unused by wrappers.** Both wrappers still generate uncentered noise and use their own schedule/sampling logic rather than `centered_randn_like`, `build_diffusion_schedule`, and `sampling_timesteps`. The denoiser returns centered predictions, so its target should use the same coordinate subspace. The wrappers clamp cumulative alpha to `1e-5`, breaking consistency with their beta products for the final three cosine steps. Reduced-step DDPM also skips adjacent transitions using coefficients for adjacent steps. These paths need deliberate integration and checkpoint compatibility checks.
9. **Atom ordering/provenance remains unverified.** SMILES-derived bonds must match coordinate atom order, including hydrogens. Predicted SELFIES atom order is not guaranteed to match the supplied atom-type sequence. No atom mapping or dataset-generation script is provided. RDKit-dependent validation was not run because RDKit is absent.
10. **Evaluation failures can inflate reported success.** Legacy evaluation catches per-row exceptions and divides success by the number of finite RMSDs, excluding failed rows. Track all attempted rows and report failures explicitly.

## What to request from Siqi

1. The exact source revision used for both checkpoints, especially the full encoder implementation with peak self-attention and formula-prefix support, its dependencies, and the matching diffusion/physics interfaces.
2. Original encoder and diffusion split row IDs/manifests, the exact vocabulary mapping, and resolved preprocessing configuration, including intensity scaling and START/STOP handling.
3. Dataset-generation details confirming coordinate units, spectrum/geometry pairing, atom ordering, and any explicit graph/angle/planarity targets or atom mappings.
4. The original `stage1_v7/...epoch=034...0.092967.ckpt` only if keeping the current external-file loading design; the embedded encoder weights may make that extra file unnecessary after a loader fix.
5. A pinned Python/conda environment and supported launch instructions.

## Recommended order of work

Choose and document one supported workflow; restore checkpoint-compatible encoder behavior; connect the denoiser and physics APIs; make checkpoint loading portable; resolve data/preprocessing/split provenance; then add a small real-data training-step and checkpoint-to-sampling integration check before a long run. Fix the legacy RMSD calculation before comparing results.

This was an audit. Project source, datasets, and checkpoints were not edited, and no training run was started. This report is the only intentional deliverable added; test/import execution may update ignored caches. There is no `.git` directory in this folder, so additions and changes could not be compared against repository history. Full numerical training/inference and chemical validation remain unverified.
