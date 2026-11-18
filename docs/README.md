# Documentation and repository map

## Current workflow

- [Matcher training-task audit](experiments/MATCHER_TRAINING_TASK_AUDIT.md): next runnable diagnostic; compare original training groups with generated four-choice groups on the same training molecules.
- [Completed IR diagnostic and next decision](results/token_matcher_probe_1444546.md): verified spectrum dependence; difficult candidate comparisons remain the problem. Includes the proposed training-comparison study.
- [Full-token matcher outcome](results/token_matcher_1426578.md): completed frozen/fine-tuned comparison; the small combined gain is inconclusive.
- [Top1 research before the matcher trial](TOP1_RESEARCH_AND_NEXT_EXPERIMENT.md): paper/code review and the motivation for the now-completed experiment.
- [Development-selected decoder corrections](experiments/RESIDUAL_SELECTOR_EXPERIMENT.md): completed experiment; no reliable assessment improvement. Retain the existing decoder.
- [Hard-negative selector pilot](experiments/HARD_SELECTOR_EXPERIMENT.md): compare generated confusing alternatives with matched random negatives after the failed compatibility experiment.
- [Accuracy update v2](guides/ACCURACY_UPDATE_V2.md): prepare decoding comparisons and training-only negative pairs before fine-tuning.
- [Fine-tuning pilot](guides/FINETUNE_PILOT.md): run and compare the three controlled training variants, then review the results.
- [Optional compatibility experiment](experiments/COMPATIBILITY_EXPERIMENT.md): train a separate scorer for existing graph candidates.
- [Interactive 3D viewer](guides/VIEWER_3D_UPDATE.md): rotate, zoom, and inspect saved conformers offline.
- [Project README](../README.md): environment setup and the complete training, evaluation, and prediction workflow.

Update guides are grouped under `docs/guides/` and experiment protocols under
`docs/experiments/`. Older guides describe individual releases; use the current
workflow links above for the accuracy pilot.

## Results, data, and research

- [Top1 research, November 11](TOP1_RESEARCH_AND_NEXT_EXPERIMENT.md): primary papers, official matching-model code, fresh formula-filter diagnostics and a staged experimental plan.
- [Residual selector outcome](results/residual_selector_1389118.md): epoch selection and early stopping reproduced; assessment gain was six correct answers, with an interval spanning zero.
- [Hard-negative selector outcome](results/hard_selector_1361586.md): both selectors fit the training comparisons but lose validation accuracy; retain decoder ordering.
- [Compatibility experiment outcome](results/compatibility_1361218.md): the helper reduced Top1 to 54.23%; retain the 64.02% ranking model. Verified per-row changes and next hypotheses.
- [Completed full-validation review](results/full_validation_66915416.md): verified 12,478-row baseline/CE/ranking comparison, remaining errors, and the next controlled experiment.
- [Measured stage 1 baseline](results/stage1_baseline_66782537.md): the recorded 1,000-row validation subset audit and its interpretation limits.
- [Dataset and outcome audit](DATASET_AND_OUTCOME_AUDIT.md): dataset contents, split checks, supported domain, and outcome analysis.
- [Accuracy literature review](ACCURACY_LITERATURE_REVIEW.md): research evidence and the motivation for the controlled experiments.
- [Research decisions](RESEARCH.md): model design and source references.
- [Validation record](VALIDATION.md): implementation checks and diagnostic runs.
- [Performance expectations](PERFORMANCE_EXPECTATIONS.md): workload considerations and how to measure runtime.
- [Original repository audit](guides/REPO_AUDIT.md): the historical review of the supplied implementation.

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

The compatibility wrappers include `training/train_diffusion_v2.py`, `inference_v2.py`,
and `dataset_utils_qmugs_v2.py`; their small size does not make them redundant.

Experiment provenance records source, configuration, input, and library-version
information. Resume checks can reject changed settings or code. While a Unity
pilot is running, repository cleanup therefore keeps runtime source and
configurations unchanged and preserves the uploaded release archive. Documentation
and generated-artifact organization can be cleaned independently.
