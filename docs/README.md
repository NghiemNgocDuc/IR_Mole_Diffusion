# Documentation and repository map

## Current workflow

- [Accuracy update v2](../ACCURACY_UPDATE_V2.md): prepare decoding comparisons and training-only negative pairs before fine-tuning.
- [Fine-tuning pilot](../FINETUNE_PILOT.md): run and compare the three controlled training variants, then review the results.
- [Optional compatibility experiment](../COMPATIBILITY_EXPERIMENT.md): train a separate scorer for existing graph candidates.
- [Interactive 3D viewer](../VIEWER_3D_UPDATE.md): rotate, zoom, and inspect saved conformers offline.
- [Project README](../README.md): environment setup and the complete training, evaluation, and prediction workflow.

The root update guides remain at their existing paths because the Unity packaging
scripts and earlier instructions refer to them. Older update guides describe
individual releases; use the current workflow links above for the accuracy pilot.

## Results, data, and research

- [Measured stage 1 baseline](results/stage1_baseline_66782537.md): the recorded 1,000-row validation subset audit and its interpretation limits.
- [Dataset and outcome audit](DATASET_AND_OUTCOME_AUDIT.md): dataset contents, split checks, supported domain, and outcome analysis.
- [Accuracy literature review](ACCURACY_LITERATURE_REVIEW.md): research evidence and the motivation for the controlled experiments.
- [Research decisions](RESEARCH.md): model design and source references.
- [Validation record](VALIDATION.md): implementation checks and diagnostic runs.
- [Performance expectations](PERFORMANCE_EXPECTATIONS.md): workload considerations and how to measure runtime.
- [Original repository audit](../REPO_AUDIT.md): the historical review of the supplied implementation.

Files in `docs/results/` preserve supplied run reports and selection records.
Validation-subset graph identity, coordinate quality, and individual viewer
examples measure different things; keep their scope explicit when reporting them.

## Repository layout

| Location | Purpose |
|---|---|
| Root `*.py` and `*.sh` | Supported command-line entry points and Unity launchers; keep these paths stable. |
| `diffusion/` | Current models, data handling, training, evaluation, and shared utilities. |
| `configs/` | Training and diagnostic configurations. |
| `scripts/` | Audits, experiment orchestration, comparisons, packaging, and smoke checks. |
| `tests/` | Automated checks for supported behavior. |
| `viewer_assets/` | Offline viewer code and bundled library provenance/license. |
| `diffusion_v2/` and root compatibility wrappers | Deliberate compatibility paths that route older names to supported code. |
| `legacy/` | Historical source retained for reference; not supported training entry points. |
| `data/`, `ckpt/`, `models_formula/` | Preserved datasets and supplied artifacts, including legacy checkpoints. |
| `.runtime/` | Local Windows Python environment; excluded from Unity uploads. |
| `runs/`, `outputs/`, `evaluation/`, `predictions/`, `logs/` | Generated experiments, deliverables, reports, and logs; some directories exist only after a run. |

The compatibility wrappers include `train_diffusion_v2.py`, `inference_v2.py`,
and `dataset_utils_qmugs_v2.py`; their small size does not make them redundant.

Experiment provenance records source, configuration, input, and library-version
information. Resume checks can reject changed settings or code. While a Unity
pilot is running, repository cleanup therefore keeps runtime source and
configurations unchanged and preserves the uploaded release archive. Documentation
and generated-artifact organization can be cleaned independently.
