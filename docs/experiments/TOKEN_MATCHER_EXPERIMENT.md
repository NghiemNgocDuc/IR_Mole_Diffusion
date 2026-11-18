# Full-token spectrum/molecule matcher

This is the next controlled experiment after the pooled-feature selectors failed.
It is an adaptation of token-level cross-attention and joint alignment/matching,
not a reproduction of Vib2Mol or a claim of improved accuracy. The research and
measured bottlenecks are in [the research review](../TOP1_RESEARCH_AND_NEXT_EXPERIMENT.md).

## Upload and start

Upload **`unity_token_matcher_update.zip` to `~/diffusion/`** on Unity. That is the
project root containing `training/train_spectrum.py`, not the inner `~/diffusion/diffusion/`
Python package directory. This additive update assumes the previous accuracy and
residual-selector updates are installed. It uses existing Python dependencies.

Paste into the Unity terminal:

```bash
cd ~/diffusion &&
unzip -o unity_token_matcher_update.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base training/train_token_matcher.sh
```

If the file was uploaded to the inner directory, use
`unzip -o diffusion/unity_token_matcher_update.zip` from `~/diffusion` instead.

The launcher checks these existing inputs:

- `runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt`
- `evaluation/full_validation_66855524/ranking.json`
- `data/processed/squirl.npz` and `data/processed/splits.json`

Override `MATCHER_CHECKPOINT`, `MATCHER_REPORT` or `MATCHER_OUTPUT` in `sbatch
--export` only when intentionally using another matching checkpoint/report pair.
The program verifies their hashes, source compatibility, preprocessing, vocabulary,
EMA status, split membership and all candidate identities before training.
It rejects mismatches rather than silently combining incompatible code/results.

After Slurm gives a job number, replace `JOBID` below with that number:

```bash
squeue -j JOBID
tail -F logs/token-matcher-JOBID.out logs/token-matcher-JOBID.err
```

Each stage prints elapsed time and a stage-specific ETA. Two arms run sequentially,
each for ten epochs on all training rows, followed by selected-checkpoint assessment.
The 12-hour allocation is a limit, not a runtime prediction. Use the first complete
epoch to judge training speed. Ctrl+C stops `tail`, not the Slurm job.

## What is trained

The decoder checkpoint is loaded with EMA weights and remains unchanged. Each arm
starts from exactly the same private spectral encoder and new matcher parameters.
The molecule encoder reads SELFIES bidirectionally. Cross-attention lets molecule
tokens examine all valid spectral tokens. It produces a learned compatibility logit;
it does not calculate a physical IR spectrum or a calibrated correctness probability.

| Arm | Spectral encoder | Molecule/cross-attention layers |
|---|---|---|
| frozen | copied baseline, fixed, evaluation mode | trained |
| finetuned | copied baseline; fixed for epoch 1, trained at lower LR thereafter | trained |

Both arms use all 99,825 training pairs and exactly the same stored, epoch-specific
negative plans and row orders. Every positive competes against three distinct
training-only wrong identities: a random molecule, a same-formula molecule when
available, and a structurally similar molecule found in a bounded search. The
structural search prefers same-formula candidates, then equal-heavy-atom candidates,
then other training molecules. Fallbacks and the search cap are explicit in code.
Fingerprint similarity is used only to choose negatives; fingerprints are not the
matcher's molecular input. No validation reference formula is supplied at inference.

The loss combines four-way candidate cross entropy and symmetric contrastive
alignment. The default matching-layer LR is `1e-4`, copied-encoder LR `1e-5`,
alignment weight `0.1`, batch size 64, ten epochs. Training logs report both losses.
These are provisional experiment settings, not literature-optimal settings.

The frozen encoder has dropout disabled. The trainable encoder uses its pretrained
dropout setting when training after warmup. Thus this is a practical adaptation
comparison, not an isolated gradient-only ablation. Neither arm changes the generator.

## Selection and interpretation

The same seed as the prior residual experiment partitions the already inspected
validation split into 2,000 development and 10,478 assessment rows. These are not fresh
holdouts. The test split is not evaluated by this experiment.

Development chooses checkpoint epoch and a small, predeclared score combination:
original length-normalized decoder score plus `weight * matcher_logit`, with weights
0, 0.1, 0.3, 1, 3, or matcher-only. Ties favor decoder-only, then smaller weights and
earlier epochs. The combined branch retains the original decoder unless development
Top1 improves by at least 0.5 percentage points. A separate matcher-only checkpoint
is selected for diagnosis; its failures must not be hidden by the fallback.

Both arms' choices and checkpoint hashes, and the winning arm chosen on development,
are sealed in `selection.json` **before any assessment scoring**. Assessment compares
decoder-only, both combined branches and both matcher-only branches on identical rows.
Do not choose a different winner after looking at assessment. Paired intervals are
exploratory and do not include training-seed or repeated-development-selection uncertainty.

All original candidates and denominators are retained, including empty pools and
cases where the correct identity is absent. Reranking the same five candidates keeps
Top5 unchanged. Their full-validation coverage was 89.26%; this experiment cannot
deliver 90% overall Top1 from those fixed pools. If useful matching is demonstrated,
wider candidate generation is a separate next experiment.

## Outputs and recovery

Read `runs/token_matcher_JOBID/RESULTS.md` and download
`runs/token_matcher_JOBID/token_matcher_results.zip` for review. The ZIP contains
configuration, training histories, sealed selection, assessment predictions and
paired comparisons. It excludes the neural weights and negative-plan arrays to keep
review downloads small; these remain in the run directory. Preserve that directory
and the original checkpoint. Selected `.pt` files contain matcher state, architecture
settings and the configuration hash; reconstruction also requires the original
spectral encoder specification/checkpoint recorded in the configuration.

No weights are automatically promoted and the 3D viewer is unchanged. A completed or
interrupted output directory is never overwritten. This first version does not resume
optimizer state; after a failure, retain its logs/artifacts and launch with a fresh
`MATCHER_OUTPUT`. Check the failure before repeating expensive training.

Local CPU tests use tiny synthetic fixtures to verify software behavior, masking,
gradients, split boundaries, selection timing and protected files. They are not a new
scientific accuracy measurement. The full experiment must run on Unity before we can
report whether this approach helps.

Local verification on November 11, 2026: 18 automated checks passed, including both
training arms and the complete selection/assessment path on a tiny fixture. A separate
preflight decoded and checked all 99,825 real training identities and generated one
complete negative plan. All three negatives were distinct and excluded their positive.
The middle negative used a same-formula alternative for 99,724 rows; structural search
used a same-formula pool for 99,636 rows, a same-heavy-atom pool for 188, and the broader
training pool for one. This verifies data compatibility, not learned accuracy.

On the local CPU the complete data preflight and one negative plan took about 4.3
minutes; this is not a Unity runtime estimate. The job prepares ten shared negative
plans before starting the two training arms, and logs progress for each plan.
