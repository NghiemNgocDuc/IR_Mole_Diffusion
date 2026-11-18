# Compare the matcher's training task with generated candidates

This follows the completed IR probe. The matcher responds to IR, but has not
reliably improved the original decoder's first choice. This diagnostic asks
whether its original training comparisons are easier than generated alternatives.

## Upload and run

Upload **`unity_matcher_task_audit_update.zip` into `~/diffusion/`** (the project
root, not its inner `diffusion/` Python package). Run:

```bash
cd ~/diffusion &&
unzip -o unity_matcher_task_audit_update.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base audit_matcher_training_tasks.sh
```

Keep `runs/token_matcher_1426578` intact, including its selected `.pt` checkpoints,
`negative_plan_*.npy`, `negative_plans.json`, histories and saved predictions.
The original generator checkpoint, dataset and report must remain at the paths
recorded in that run. The script checks hashes before generating candidates.
It uses the earlier token-matcher implementation and the probe replay helper.
No model weights or dataset are included in this update.

Replace `JOBID` with the submitted number:

```bash
cd ~/diffusion
squeue -j JOBID
tail -F logs/task-audit-JOBID.out logs/task-audit-JOBID.err
```

When complete, press Ctrl+C to leave the log viewer and run:

```bash
cat runs/matcher_task_audit_JOBID/RESULTS.md
```

Download `runs/matcher_task_audit_JOBID/matcher_task_audit_results.zip` for review.
Queue waiting time is separate from runtime. Candidate generation is the main
workload; its progress and phase ETA are printed. The two-hour allocation is a
limit, not a runtime estimate. A stopped run preserves partial candidates but is
not resumable; do not treat partial output as a completed diagnostic.

## Fixed protocol

- Select 2,000 rows from the original training split, seed 20261112, before scoring.
  Each row was seen by both the generator and matcher. This is an in-sample audit.
- Use the four previously selected checkpoints: frozen/fine-tuned, combined/pure.
  Skip a combined branch if it previously retained the decoder alone. Do not choose
  another checkpoint or mixture from the diagnostic.
- Recreate each checkpoint's four-choice training groups from the saved negative
  plan for its selected epoch. All three negative source identities are training
  identities; reject duplicate or self-negative choices.
- Generate natural candidates once using the preserved EMA decoder, beam 5 and
  original length normalization. Use exactly the same pools across all matchers.
- Build a second four-choice task using the exact stored training positive and the
  first three distinct wrong generated molecules. The positive is deliberately
  inserted for this diagnostic. Exclude rows with fewer than three negatives and
  compare the original four-choice task on precisely the same remaining rows.
- Both four-choice tasks use the same positive serialization. Score all molecules
  independently with the same checkpoint. Report strict Top1 (ties fail), matching
  cross-entropy, same-formula pairwise comparisons and paired correct/incorrect counts.
- Separately report natural-pool coverage, matcher Top1, decoder Top1, and the saved
  fixed combination. Missing targets remain failures; never inject the reference
  into these natural predictions.
- Replay up to 64 already saved assessment predictions per checkpoint solely as
  an inference check, requiring original first choices and logits within 0.001.
  This does not create another validation score or change any selection.

Outputs include all sampled row IDs, original plan hashes, candidate structures,
logits, selections, source/input hashes and interpretation limits. No training,
test evaluation or weight promotion is performed.

## How to interpret the result

If the matcher scores its original groups highly but struggles with the generated
four-choice groups on those same rows, that supports investigating the training
comparisons. It does not prove that changing negative sampling will improve unseen
molecules. Original negative plans were already encountered during training, so
memorization remains an alternative explanation.

If both training tasks score highly, the problem may primarily involve generalization
to unseen molecules. Training candidates from the current generator can be unusually
easy because it already saw these molecules. Truly out-of-fold generation would
require a generator trained without the relevant fold; this job does not provide it.

If both tasks score poorly, inspect optimization, objective balance and checkpoint
behavior before blaming candidate distribution. The reported cross-entropy is the
four-choice matching component in evaluation mode, not the historical total training
loss that also included alignment and dropout.

Same-formula pair accuracy uses different sets of negative pairs across tasks;
compare its denominators as well as its value. The four-choice paired comparison
is the main controlled diagnostic. Tie handling and empty-group exclusions are
explicit. None of these training metrics is a new validation-accuracy claim.

## Local software checks

The tests exercise training-split and saved-plan guards, missing targets, same-formula
pairs, ties, empty matched subsets, unchanged natural coverage, and a real tiny
CPU train-then-audit run that checks input/checkpoint hashes stay unchanged. Launcher
and archive checks verify the upload command and package contents. Scientific task
metrics still require the saved Unity weights and data.
