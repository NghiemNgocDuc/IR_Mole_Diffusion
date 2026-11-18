# Controlled decoder-negative token matcher pilot

## Why this experiment

Audit 1475520 found 96–97% strict accuracy on original four-choice training
questions, versus 44–50% on the same 1,775 molecules with generated alternatives.
This supports investigating the training comparisons; it does not isolate
memorization from negative-distribution mismatch or promise a validation gain.

## Fixed comparison

- Sample 5,000 training anchors with seed 20261112 before scoring outcomes.
- Generate legacy beam-5 candidates using the existing ranking generator's EMA.
- Original arm: existing random, formula and fingerprint-based training negatives.
- Generated arm: first three distinct wrong decoder candidates, in decoder order.
  If fewer than three exist, fill from that epoch's original negatives without
  duplicates or the correct graph. Keep every sampled anchor in both arms.
- Same stored positive SELFIES, three negatives, initialization, architecture,
  row order, ten epochs, batch size 64, learning rate 0.0001, and loss in both arms.
- Freeze the IR encoder in both arms. Train fresh identical token-matcher heads;
  this does not resume or replace earlier matcher checkpoints.
- Matching cross-entropy plus 0.1 times identity-aware contrastive alignment.
  The generator's decoder score remains the base score at inference; development
  chooses a matching correction from 0, 0.1, 0.3, 1, 3, or matcher-only. Zero is
  the fallback. A combined checkpoint needs at least 0.5 percentage points of
  development improvement. Matcher-only results are separately reported.
- Use the existing 2,000-row development partition, then seal both selections
  before scoring the 10,478 assessment rows. No test evaluation or promotion.

Compare generated against the newly trained original-negative control, not just
against an old run with a different training budget. The matched update budget
is a bounded pilot, not an exhaustive training schedule. Decoder-generated
negatives here are not adaptive mining by the matcher's learned scores.

All mining spectra and positive identities come from training. A generated
structure can coincidentally equal a held-out identity, but no held-out labels
are consulted to generate, filter or select training negatives. Original-arm
negative identities may come from the entire training split. Only the sampled
5,000 spectra are gradient anchors.

The generator has seen these anchors, so this is not out-of-fold mining.
Development and assessment have both been inspected in previous experiments;
assessment is not a fresh independent holdout. Existing candidate coverage
bounds reranking accuracy. No claim of 90% Top1, stereochemistry, experimental
IR or 3D correctness follows.

## Run on Unity

Upload `unity_generated_matcher_update.zip` into `~/diffusion` (the outer project
folder, not `~/diffusion/diffusion`). Keep the original project dependencies,
processed data, ranking checkpoint and full-validation ranking report.

```bash
cd ~/diffusion &&
unzip -o unity_generated_matcher_update.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base training/train_generated_matcher.sh
```

The output is `runs/generated_matcher_JOBID`. Logs are
`logs/generated-matcher-JOBID.out` and `.err`. The 12-hour allocation is a limit,
not a runtime estimate. Mining prints a phase ETA. This initial pilot has no
resume mode; a failed run requires a fresh output directory.

When complete, read `RESULTS.md` and download `generated_matcher_results.zip`
from that output directory. Preserve checkpoint files and candidate_bank.npy
on Unity: the result archive contains reports, histories and candidate metadata,
not model weights or numeric negative-plan files.

Existing checkpoints and original token-matcher source files are not overwritten.
The new runner verifies protected input and source hashes and rejects reused
output directories. Local tests exercise real tiny-model training, selection,
evaluation, unchanged encoder/input hashes and distinct-negative fallback.
