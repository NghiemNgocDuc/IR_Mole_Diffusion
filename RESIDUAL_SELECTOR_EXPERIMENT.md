# Development checks and bounded decoder corrections

**Completed run 1389118:** the selected epoch improved assessment Top1 from
63.60% to 63.66%, with a paired exploratory 95% interval of -0.35 to +0.44
percentage points. This does not establish an improvement. Keep the existing
decoder; no weights were promoted. See the [audited outcome](docs/results/residual_selector_1389118.md).
The launch instructions below preserve the experiment protocol.

The previous selectors fitted their repeated training comparisons but reduced
validation Top1 to 33.76% and 26.59%. This additive experiment retains the ranking
decoder's scores, checks development accuracy after every epoch, and allows the
unchanged decoder to win. It does not overwrite any existing model or viewer.

## Upload and launch

Upload `unity_residual_selector_update.zip` to **`~/diffusion`**, then run:

```bash
cd ~/diffusion &&
unzip -o unity_residual_selector_update.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base train_residual_selector.sh
```

If the upload went into `~/diffusion/diffusion`, use
`unzip -o diffusion/unity_residual_selector_update.zip` while staying in the project
root. No installation, new checkpoint upload, or candidate regeneration is needed.

The defaults use your completed Unity files:

- Ranking checkpoint: `runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt`
- Baseline report: `evaluation/full_validation_66855524/ranking.json`
- Cached training inputs: `runs/hard_selector_1361586/configuration.json` and
  `runs/hard_selector_1361586/training_candidates.json`

Source, checkpoint, dataset, split and cache hashes are checked before training.
Changed source files cause a stop rather than silent mixing of versions. Keep
existing runtime sources unchanged during the run. The upload adds new files; it
does not replace those whose hashes appear in previous reports.

Replace `JOB_ID` with the submitted number:

```bash
squeue -j JOB_ID
tail -F logs/residual-selector-JOB_ID.out logs/residual-selector-JOB_ID.err
```

The initial input verification can take time before epoch logs appear. Later logs
show phase progress plus each epoch's training loss, development accuracy, and
selected epoch. The two-hour allocation is a limit, not a runtime estimate.

When complete, press Ctrl+C to exit `tail` and run:

```bash
cat runs/residual_selector_JOB_ID/RESULTS.md
```

Download `runs/residual_selector_JOB_ID/residual_selector_results.zip` for review.
`selected.pt` and `last.pt` remain in that directory. No weights are automatically
promoted. If selected epoch is **0**, development did not justify replacing the
decoder: the assessment output preserves the original candidate ordering exactly.

## Fixed protocol

**Data roles.** The saved 5,000 training inputs remain the only source of gradient
updates. A seeded split of existing validation reserves 2,000 rows for development
and 10,478 for assessment. Development labels select checkpoints and determine
early stopping. Assessment labels never enter those decisions. Selection and the
selected checkpoint hash are written before assessment feature extraction/scoring.
The test split is neither trained on nor evaluated.

These validation inputs have been inspected in earlier experiments. Calling part
of them assessment does not make them a fresh independent holdout. Report the
assessment comparison as exploratory, and compare with the paired baseline on
those same 10,478 rows, not with 64.02% on the earlier full split. Development
accuracy is a tuning metric, not a generalization estimate.

**Natural pools.** Use the actual saved candidates, with no injected target.
Training skips inputs whose target is absent or that have fewer than two candidates;
their number is reported. All development and assessment rows remain in accuracy
denominators, including missing targets and empty pools. Missing targets count as
failures. Conditional candidate loss is reported separately and excludes absent
targets because the selector cannot choose a molecule outside its pool.

**Decoder evidence.** Reconstruct the original final beam score as
`log_probability / (SELFIES token count + 2)^0.6`, including BOS/EOS in the length.
Verify that this reproduces candidate ordering. The frozen spectrum encoder and
candidate Morgan fingerprints feed a small head with 32 hidden units and 0.2
dropout. It adds a correction in `[-0.25, +0.25]` to each decoder score. The final
layer starts at zero, so the initial model exactly matches the decoder. A decoder
score gap larger than 0.5 cannot be reversed by this head. This is a fixed design
choice, not a coefficient sweep over validation results.

**Training and selection.** Train with candidate cross entropy on combined scores,
AdamW learning rate 0.0001, weight decay 0.01, batch size 64, fixed seed 78438379,
up to 20 epochs. Check development Top1 every epoch; stop after three consecutive
epochs without a strictly higher development correct count than the best observed
count (including the baseline). Among checkpoints, choose the highest development
correct count that improves over baseline by at least `ceil(0.005 * dev_rows)`:
10 additional correct answers for the default development size. Ties retain the
earlier choice. Otherwise retain epoch 0. Save training/development history each
epoch, the last checkpoint, and the selected checkpoint. The threshold is an
operating rule, not a significance test or guarantee of assessment improvement.

**Remaining limitations.** Mining inputs were already used to train the generator,
so the earlier training/evaluation distribution mismatch can persist. This pilot
tests whether development selection and conservative score corrections help; it
does not remove every possible cause of failure. Exact identity ignores stereo.
No experimental-IR or 3D accuracy claim follows. Fixed candidate coverage still
bounds achievable Top1; 90% is not promised.

## Restart after interruption

```bash
cd ~/diffusion &&
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,RESIDUAL_OUTPUT=runs/residual_selector_JOB_ID,RESIDUAL_RESUME=1 \
  train_residual_selector.sh
```

Use the old job's output directory. Matching interrupted runs restart this small
head deterministically from initialization and reuse the saved mining inputs;
they do not resume an optimizer mid-epoch. Configurations, sources, inputs, and
package versions must match. Completed runs refuse to restart. The development
partition is fixed; restarting never searches for a more favorable split.

## Software verification

Run `python -m pytest tests/test_residual_selector.py tests/test_hard_selector.py
tests/test_compatibility.py -q` as one command. Tests cover decoder-score recovery,
zero-initialized fallback, bounded corrections, disjoint partitions, absent-target
denominators, best-checkpoint restoration, early stopping, and a real tiny CPU
pipeline that freezes selection before assessment. These tests establish software
behavior, not scientific accuracy. The real checkpoint is on Unity.
