# Next experiment: choose better candidates

This small update is for the existing Unity v2 installation that completed
full-validation job 66915416. It uses the ranking model's saved 12,478 validation
candidate pools (64.02% Top1; 89.26% Top5) and trains a separate compatibility
head on training data. It does not retrain or overwrite the molecular generator.

Upload `unity_compatibility_fullval_update.zip` to **`~/diffusion/`**, the outer
project directory containing `training/train_compatibility.py`. Then run:

```bash
cd ~/diffusion &&
unzip -o unity_compatibility_fullval_update.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,COMPATIBILITY_CHECKPOINT=runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt,COMPATIBILITY_REPORT=evaluation/full_validation_66855524/ranking.json \
  training/train_compatibility.sh
```

If the archive was uploaded inside `~/diffusion/diffusion/`, use
`unzip -o diffusion/unity_compatibility_fullval_update.zip` while remaining in
the outer project directory. Do not unzip the archive from the inner directory.

The checkpoint path above comes from the uploaded full-validation manifest.
Its expected SHA256 is
`da3f14964d84c3195ac838b99a8ea579161bb34e4c5f90f914637990e845a751`.
The Python runner verifies the checkpoint against the saved report, data, splits,
vocabulary and feature-extraction source before training. A mismatch requires
inspection; do not disable the checks.

Use the submitted job number:

```bash
squeue -j NEW_JOB_ID
tail -F logs/compatibility-NEW_JOB_ID.out logs/compatibility-NEW_JOB_ID.err
```

Fingerprint preparation happens before the first progress display. Subsequent
phases report elapsed time and ETA. The four-hour allocation is a limit;
runtime for this experiment has not been measured. Pressing Ctrl+C stops log
viewing and leaves the submitted job running. Do not submit duplicates while waiting.

After completion:

```bash
cat runs/compatibility_NEW_JOB_ID/RESULTS.md
```

Download `RESULTS.md`, `compatibility_comparison.json`,
`compatibility_predictions.json`, `configuration.json`, and
`training_history.json` from that directory for review. The head checkpoint is
`compatibility.pt`; preserve it on Unity. Every run uses a fresh job-specific
output directory. Interrupted runs cannot resume in the current implementation.

Top5 should stay at 89.26% because the candidate pool is fixed. The experiment
succeeds only if ordering improves; it can also make first choices worse.
The original ranking order remains available. No weights are promoted
automatically and the test split is not evaluated.

See [the reviewed protocol](COMPATIBILITY_EXPERIMENT.md) and
[the completed full-validation audit](../results/full_validation_66915416.md).
