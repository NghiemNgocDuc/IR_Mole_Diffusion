# Controlled stage-1 fine-tuning pilot

First follow [ACCURACY_UPDATE_V2.md](ACCURACY_UPDATE_V2.md) and inspect `PREPARATION.md`. The first job compares decoding and prepares pairs without training. This guide describes the later training job included in the same ZIP.

## Three training arms

| Setting | CE | Static ranking | Mined ranking |
|---|---|---|---|
| Initial weights | Original baseline EMA | Same EMA | Same EMA |
| Objective | Cross-entropy | CE + 0.1 x hinge ranking | CE + 0.1 x hinge ranking |
| Negative structure | None | Static training identity | Model-generated wrong alternative where available; same static fallback elsewhere |
| Epochs / batch | 5 / 64 | 5 / 64 | 5 / 64 |
| Learning rate | 0.00002 | 0.00002 | 0.00002 |
| Inference | Legacy beam 5 | Legacy beam 5 | Legacy beam 5 |
| Evaluation checkpoint | Final-epoch EMA | Final-epoch EMA | Final-epoch EMA |

Each arm starts a fresh optimizer and EMA with seed 78438379. All use FP32, label smoothing 0.05, EMA decay 0.999 and weight decay 0.00001. Ranking margin is 0.1. For 99,825 training rows, every arm must finish exactly **7,800 optimizer updates**. Early stopping and the learning-rate scheduler's patience cannot trigger within the five-epoch budget. Extra ranking decoder passes mean runtime and dropout draws need not match CE.

Static negatives are different training identities selected by Morgan similarity from seeded pools of at most 32, preferring the same formula, then heavy-atom count. Mining searches a seeded 5,000-row training subset with the original baseline EMA and legacy beam 5, selecting the highest-ranked valid wrong identity that fits the vocabulary and sequence length. It can select wrong alternatives even when the first choice is correct. Unsupported tokens and overlength sequences are rejected rather than truncated.

At most about 5% of the training pairs change under default mining. The third arm tests this bounded hybrid intervention, not exhaustive mining. Generated alternatives need not be dataset molecules. Only training inputs and their labels construct pairs; held-out identities are not used to filter or train on them. Reference formulas help static pair preparation; inference still receives IR only.

The decoding study is kept separate: training-arm evaluation always uses legacy beam 5, even if a different decoder looks better in preparation. That isolates training effects. Wider beams are still compared using fixed Top5.

## Launch after reviewing preparation

Replace `PREP_JOB_ID` with the preparation job number:

```bash
cd ~/diffusion
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,PILOT_OUTPUT=runs/spectrum_pilot_PREP_JOB_ID \
  finetune_spectrum_pilot.sh
```

Keep the **preparation directory name**, even though Slurm returns a new training job number. Use that new number for the log files:

```bash
squeue -j TRAIN_JOB_ID
tail -F logs/pilot-TRAIN_JOB_ID.out logs/pilot-TRAIN_JOB_ID.err
```

The launcher verifies and reuses completed preparation before training. Logs include epoch/batch progress and evaluation row counts/rate/ETA. The twelve-hour allocation is a limit, not a completion estimate. Ctrl+C exits the log viewer without cancelling the batch job.

## Results

```bash
cat runs/spectrum_pilot_PREP_JOB_ID/RESULTS.md
```

Download `pilot_results.zip` from the same directory. It includes four evaluation reports (baseline, CE, static ranking and mined ranking), paired comparisons, error audits and configurations. Arrays and checkpoints stay on Unity. The original spectrum and diffusion checkpoints are preserved.

All comparisons use the frozen 1,000-row validation selection. A provisional candidate must improve Top1 without lowering Top5; the mined arm is also compared against both CE and static ranking. The report includes gained/lost cases and paired 95% bootstrap intervals. No model or decoding setting is automatically selected for predictions.

This is validation-based development, not a final unbiased test estimate. Training still monitors validation loss, but evaluation uses the final checkpoint instead of selecting the best one. Intervals describe row-sampling uncertainty, not training-seed variability, and are not corrected for multiple experiments. Confirm promising results with broader validation and additional seeds before evaluating the reserved test split. Do not use the previously inspected challenge molecule as a training target.

Graph identity accuracy here ignores stereochemistry and preserves tautomer identity. It does not establish 3D accuracy or agreement between a calculated spectrum and measured IR.

## Restart behavior

Resubmit the same command with the same `PILOT_OUTPUT`. Completed stages are reused after hash/provenance checks. Partial evaluation and mining resume from verified records. An unfinished training arm restarts from the original baseline in a fresh attempt directory; its old files are preserved. The trainer cannot reproduce an interrupted data-loader shuffle exactly.

Static pairs are saved atomically. If saving their JSON metadata is interrupted, the next run reconstructs it from verified embedded settings. Corrupt archives or older incomplete files without embedded settings require a fresh output directory; they are not silently replaced.

Changing sources, settings, inputs or library versions requires new preparation in a fresh directory. A zero-pair mining result stops the training workflow for review. The optional `MINE_LIMIT` export must be the same in preparation and training; changing it requires a fresh experiment. `BASELINE_AUDIT` and `SPECTRUM_CHECKPOINT` can select another verified matching input pair.

## Research context

[FIRMPO](https://arxiv.org/html/2608.16082v1) motivates studying model-generated alternatives and chemical feedback, but uses a different formulation with formula/spectral information. This pilot does not reproduce its method. The optional [compatibility experiment](COMPATIBILITY_EXPERIMENT.md) separately explores IR/molecule alignment with a frozen encoder and two projection networks; it does not calculate forward IR or guarantee a gain.
