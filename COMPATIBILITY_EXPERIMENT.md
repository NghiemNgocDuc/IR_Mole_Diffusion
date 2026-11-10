# Optional IR–molecule compatibility experiment

This separate experiment trains two small projection networks to associate IR features with molecular fingerprints. The original baseline EMA encoder stays frozen. Only training-split molecular labels enter the objective; duplicate canonical identities are treated as positives rather than incorrect alternatives.

It then reranks the **exact existing baseline candidate pools** on the same validation subset. It cannot recover a molecule absent from those pools. Top1 and fixed Top5 can change; coverage across the full candidate pool cannot. Scores are experimental associations, not calculated IR spectra or calibrated probabilities. Accuracy gains are not guaranteed.

With the prepared beam-5 baseline specifically, every pool has at most five candidates, so Top5 coverage stays unchanged. This experiment tests whether first-choice ordering improves.

Run this only after the pilot preparation has produced a complete `baseline.json` and `baseline_start.ckpt`. By default it uses that original baseline, not a fine-tuned arm. A ranking-base variant for the completed full validation is documented at the end. This job is optional and is never submitted automatically.

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

## Ranking-base variant (full 12,478-molecule validation)

The pilot completed on Unity (jobs around 66855524) and full validation measured CE and ranking fine-tuning both at **64.02% Top1**, with ranking at **89.26% Top5** and the original baseline at 62.91% / 89.10%. This variant applies the same frozen-encoder compatibility head to the **ranking** checkpoint and the unchanged ranking candidate pools over all 12,478 validation rows. It answers whether first-choice ordering can improve on the ranking model without touching candidate generation. It changes nothing about training-arm selection.

Predeclared protocol (fixed before looking at any new results):

1. Freeze the ranking model's EMA encoder and its exact saved beam-5 candidate pools. Train only the two projection heads on the training split's matched IR/structure pairs. Never add validation mistakes, reference formulas, or test structures to the training pairs.
2. Keep the implemented settings: Morgan radius 2, 2,048 bits, no chirality; masked mean IR pooling; 128-dimensional normalized embeddings; symmetric identity-aware InfoNCE at temperature 0.1; AdamW with learning rate 1e-4, weight decay 1e-5; batch 256; five epochs; final-epoch head; seed 78438379. No hyperparameter search on this validation selection.
3. Score candidates with input IR and candidate SMILES only. The decoder's existing order breaks equal-score ties. No score blending and no weight chosen on the validation data.
4. Compare the compatibility order with the unchanged ranking order on all 12,478 rows. Report Top1, formula match, paired gains/losses, the paired 95% interval, per-class errors, and displaced correct predictions. The pools are identical (at most five candidates), so Top5 is invariant at 89.26%; the runner raises an error if reranking changes candidate-pool coverage.
5. Treat a gain as development evidence only. Confirm training-seed stability before fixing the method and performing a single final test-split evaluation. Do not tune on this inspected validation set.

Submit on one GPU:

```bash
cd ~/diffusion
sha256sum runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt
# Expected: da3f14964d84c3195ac838b99a8ea579161bb34e4c5f90f914637990e845a751
mkdir -p logs
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,COMPATIBILITY_CHECKPOINT=runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt,COMPATIBILITY_REPORT=evaluation/full_validation_66855524/ranking.json \
  train_compatibility.sh
```

Use the new job number for logs:

```bash
squeue -j NEW_JOB_ID
tail -F logs/compatibility-NEW_JOB_ID.out logs/compatibility-NEW_JOB_ID.err
```

The launcher verifies that the checkpoint, report, data, splits, vocabulary, and the encoder/model/data source hashes match the report before training; a mismatch stops the job early rather than reranking a different pool. `encoder.py`, `model.py`, and `data.py` must be unchanged on the machine that runs this job. Fingerprint construction and head training are CPU-light; the frozen encoder pass dominates GPU time. Expect roughly 15–45 minutes of GPU time on `gpu-highvram` (not measured; well under the 4-hour allocation limit), plus queue wait. Fingerprint construction for the ~99,825 training identities is single-threaded and its progress is not shown until feature caching begins.

After completion:

```bash
cat runs/compatibility_NEW_JOB_ID/RESULTS.md
cat runs/compatibility_NEW_JOB_ID/compatibility_comparison.json
```

Decision rule: `recommendation` is `promising_pending_broader_validation` only if the paired Top1 interval clears zero with nondecreasing Top5 (Top5 must remain 89.26%); otherwise keep the ranking order. A single seed decides nothing by itself; confirm with additional seeds or a development partition drawn from training data before any test-split evaluation. No weights are promoted automatically.

Download `compatibility_predictions.json`, `compatibility_comparison.json`, `training_history.json`, `configuration.json`, and `RESULTS.md` for review; the head checkpoint stays on Unity. The compatibility score is an association signal, not a calculated IR spectrum or a calibrated probability.
