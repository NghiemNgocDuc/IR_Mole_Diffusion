# Hard-negative selector pilot

**Completed run 1361586 did not improve accuracy.** The hard selector scored
26.59% Top1, the random control 33.76%, and the retained baseline 64.02% on the
same full validation split. Both selectors reached 100% on their repeated training
comparisons. See [the outcome review](../results/hard_selector_1361586.md).
The instructions below preserve the experiment protocol; they are not a
recommendation to rerun or promote these heads.

This is experiment 2: teach a selector to distinguish the correct training molecule
from the wrong molecules generated for that same IR input. It does not retrain the
generator or change the 3D viewer.

The last compatibility run reduced full-validation Top1 from 64.02% to 54.23%.
Retain the ranking model. This new experiment tests a possible cause of that failure;
it does not assume the cause is proven or promise 90% accuracy.

## Upload and run

Upload `unity_hard_selector_update.zip` into **`~/diffusion`**, the project root.
The file browser should end in `/diffusion`, not `/diffusion/diffusion`.

```bash
cd ~/diffusion &&
unzip -o unity_hard_selector_update.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base training/train_hard_selector.sh
```

If you uploaded into the inner package folder again, the unzip line is instead:
`unzip -o diffusion/unity_hard_selector_update.zip` while staying in `~/diffusion`.

Defaults use the already-existing ranking checkpoint and full-validation report:

- `runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt`
- `evaluation/full_validation_66855524/ranking.json`

Their hashes must match. Encoder, decoder, and evaluation source hashes must also
match the report. If a recent code edit changed those files, the job stops before
training; inspect the mismatch and regenerate the baseline with the intended code.
Do not bypass a mismatch or retrain the original model to fix it.

Use the submitted number in these commands, replacing `JOB_ID`:

```bash
squeue -j JOB_ID
tail -F logs/hard-selector-JOB_ID.out logs/hard-selector-JOB_ID.err
```

Progress prints every 15 seconds with a phase ETA. Mining 5,000 training inputs is
likely the main cost; use its measured ETA. The four-hour allocation is a limit,
not a runtime estimate. The two training arms each run 20 fixed epochs.

When complete, press Ctrl+C to leave `tail`, then:

```bash
cat runs/hard_selector_JOB_ID/RESULTS.md
```

Download `runs/hard_selector_JOB_ID/hard_selector_results.zip` for review. The
archive contains reports, training groups, candidate pools, and histories. Trained
heads remain beside it as `random.pt` and `hard.pt`; no model is promoted.

## Controlled comparison

1. Select 5,000 training inputs using a fixed seed. Generate legacy beam-5
   candidates using the frozen ranking checkpoint's EMA weights.
2. For each input, keep all valid distinct wrong candidates (up to five), excluding
   its canonical exact identity. Add its known training target as the positive.
   Rows without a wrong candidate are excluded from **both** training arms.
3. Train two new projection heads with identical initialization, input rows,
   negative counts, batch order, architecture, and optimizer budget. The random
   control draws distinct wrong identities from the selected training inputs. The
   hard arm uses the actual generated wrong candidates.
4. Both use candidate cross entropy with temperature 0.1. Unlike the earlier
   experiment, this objective compares each spectrum with its own positive and
   negatives, rather than every other spectrum's target in a batch. The matched
   control isolates negative construction within this new objective; it does not
   isolate every difference from the old 99,825-input experiment.
5. Score the existing validation pools with each final head. Validation references
   are used for metrics only, after ordering. There is no target injection,
   checkpoint selection, coefficient tuning, or test evaluation at this stage.

Read `hard_vs_random` to assess the negative-training change and
`hard_vs_baseline` to assess whether it actually beats the current decoder.
Improvement over the random control alone is insufficient. Reports include paired
intervals and gains/losses; these are exploratory comparisons with one training
seed and repeatedly used validation data.

This pilot deliberately keeps the existing masked-mean IR representation and
Morgan fingerprint heads. Failure can indicate that changing negatives alone is
insufficient. Candidate coverage remains a ceiling: a fixed pool with 89.26%
coverage cannot yield 90% Top1 just by reordering it. These scores measure molecular
graph identity, not stereochemistry, 3D accuracy, or experimental-IR performance.

## Interrupted runs

Mining saves every ten inputs and on ordinary exceptions. Resume requires the
same output directory, arguments, inputs, sources, and package versions:

```bash
cd ~/diffusion &&
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,SELECTOR_OUTPUT=runs/hard_selector_JOB_ID,SELECTOR_RESUME=1 \
  training/train_hard_selector.sh
```

Use the **old** job's output directory. Resume reuses verified candidate generation
and restarts both small heads from the same initialization; it does not resume an
optimizer mid-epoch. Completed runs refuse to restart. Do not edit runtime sources
while a job runs. Source files, checkpoints, reports, and data are checked again at
completion.

## Local verification

`python -m pytest tests/test_hard_selector.py tests/test_compatibility.py -q`
checks exact-identity exclusion, matched budgets, training-only row membership,
learning on a synthetic problem, reference-blind reranking, real tiny-model
generation/training/evaluation, protected inputs, and deterministic interrupted-run
recovery. These software tests do not establish scientific accuracy. The new
scientific experiment must run on Unity because its trained checkpoint is there.
