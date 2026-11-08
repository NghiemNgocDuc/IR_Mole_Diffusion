# Accuracy update v2: start with preparation

Use **unity_accuracy_update_v2.zip** for your existing Unity repository. It replaces the earlier pilot upload instructions. Your viewer, data and trained checkpoints are already present; this update supplies the new experiment code.

Current measured result remains **64.0% Top1 and 89.7% Top5 on 1,000 validation molecules**. No improvement from this update has been measured on Unity yet.

## Upload and run this first

Upload the ZIP into `~/diffusion`, then run:

```bash
cd ~/diffusion &&
unzip -o unity_accuracy_update_v2.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base prepare_spectrum_pilot.sh
```

This job **stops before fine-tuning**. It:

1. Verifies your original checkpoint and reproduces the saved baseline on the exact same 1,000 validation rows.
2. Compares legacy and consistent length scoring with beam widths 5, 10 and 20. Top5 always means five candidates, regardless of beam width.
3. Searches 5,000 training inputs for valid wrong molecular predictions and saves them as training pairs. Other training rows retain the existing static pairs.
4. Writes a preparation report and ZIP for review. The baseline checkpoint remains unchanged.

Record the returned job number. Replace `PREP_JOB_ID` below with that number:

```bash
squeue -j PREP_JOB_ID
tail -F logs/prepare-PREP_JOB_ID.out logs/prepare-PREP_JOB_ID.err
```

Progress logs show the stage, processed rows, rate and ETA. Twelve hours is the allocation limit, not a runtime estimate. Press Ctrl+C to leave the log viewer; the submitted job continues.

After completion:

```bash
cat ~/diffusion/runs/spectrum_pilot_PREP_JOB_ID/PREPARATION.md
```

Send that report, or download `runs/spectrum_pilot_PREP_JOB_ID/preparation_results.zip`. It contains the decoding comparison and mining counts. It does not contain model weights or spectral arrays.

## What comes after preparation

The complete training workflow is included in [FINETUNE_PILOT.md](FINETUNE_PILOT.md): ordinary cross-entropy, cross-entropy with static negatives, and cross-entropy with the hybrid model-generated negatives. All three use the same starting EMA weights, five epochs, training seed and evaluation protocol. Nothing is automatically promoted.

An optional, separate [compatibility experiment](COMPATIBILITY_EXPERIMENT.md) trains small IR/molecule projection networks while keeping the original encoder frozen. It reranks the original candidate pools without using held-out targets to score them. This association score is not a calculated IR spectrum or a probability of correctness.

These experiments can improve or worsen results. The 257 rows with a correct candidate below first place identify an opportunity; they do not guarantee an accuracy gain. The 103 missing from the original Top5 also require attention to candidate generation.

## If preparation stops

Reuse verified completed work by submitting the same directory:

```bash
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,PILOT_OUTPUT=runs/spectrum_pilot_PREP_JOB_ID \
  prepare_spectrum_pilot.sh
```

Changed code, settings, inputs or library versions require a fresh output directory. Keep the same update installed throughout preparation and training. Do not run multiple jobs against the same pilot directory at once.

Default inputs are `evaluation/baseline_audit_66782537/baseline.json`, `runs/spectrum/checkpoints/best.ckpt`, `data/processed/squirl.npz` and `data/processed/splits.json`. A missing or incompatible input stops the job with an error; do not start training until that error is resolved.
