# Optional IR–molecule compatibility experiment

This separate experiment trains two small projection networks to associate IR features with molecular fingerprints. The original baseline EMA encoder stays frozen. Only training-split molecular labels enter the objective; duplicate canonical identities are treated as positives rather than incorrect alternatives.

It then reranks the **exact existing baseline candidate pools** on the same validation subset. It cannot recover a molecule absent from those pools. Top1 and fixed Top5 can change; coverage across the full candidate pool cannot. Scores are experimental associations, not calculated IR spectra or calibrated probabilities. Accuracy gains are not guaranteed.

With the prepared beam-5 baseline specifically, every pool has at most five candidates, so Top5 coverage stays unchanged. This experiment tests whether first-choice ordering improves.

Run this only after the pilot preparation has produced a complete `baseline.json` and `baseline_start.ckpt`. It always uses that original baseline, not a fine-tuned arm. This job is optional and is never submitted automatically.

Replace `PILOT_JOB_ID` with the preparation job number:

```bash
cd ~/diffusion
mkdir -p logs
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,PILOT_OUTPUT=runs/spectrum_pilot_PILOT_JOB_ID \
  train_compatibility.sh
```

Use the new job number below:

```bash
squeue -j NEW_JOB_ID
tail -F logs/compatibility-NEW_JOB_ID.out logs/compatibility-NEW_JOB_ID.err
```

Progress reports include phase-specific elapsed time and ETA. Four hours is the scheduler time limit, not a runtime estimate. Press **Ctrl+C** to leave the log viewer; the submitted job continues.

After completion:

```bash
cat runs/compatibility_NEW_JOB_ID/RESULTS.md
cat runs/compatibility_NEW_JOB_ID/compatibility_comparison.json
```

The output also contains the head checkpoint, training history, predictions, and pinned provenance. No baseline weights are overwritten or automatically promoted. The comparison reports paired gains/losses and uncertainty; confirm any improvement on broader validation before final testing.

Every run requires a fresh output directory. To choose another location, additionally export `COMPATIBILITY_OUTPUT=your/new/directory` in the submission command. Interrupted experiments are preserved and must be restarted in a new directory.
