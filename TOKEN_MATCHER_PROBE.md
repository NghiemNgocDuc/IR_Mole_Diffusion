# Diagnose the matcher before another training run

This additive update performs controlled inference using the saved models from
`runs/token_matcher_1426578`. It does not train, select new checkpoints, change score
weights, or promote a model. Keep the existing decoder available.

## What the completed audit found

The local audit recomputed all four methods from the uploaded predictions, checked
original candidate pools and references, and confirmed the dataset hash. On 10,478
assessment rows, the development-selected frozen combination fixed 314 predictions
and broke 287: net +27 (+0.26 percentage points, interval includes zero).

Of those changes, 217 fixes and 201 regressions switched between molecules with the
same formula. Fixes and regressions had similarly small original decoder margins
(median 0.066 versus 0.075). There is no demonstrated rule here for safely choosing
which changes to accept. The model cannot be improved by using the correct reference
formula at inference unless that is independently available and part of the task.

The correct molecule was present in 9,324 pools. In 8,860 of those (95.0%), its generated
SELFIES exactly matched that row's stored dataset serialization. On these matching
serializations, the frozen matcher alone was correct on 3,787 rows, versus 6,450 for
the decoder. Alternative serialization is a real possible difficulty for some rows,
but cannot explain most failures. The same five candidates still limit achievable Top1.

These are descriptive findings on previously inspected assessment data, not causal
proof or a fresh validation set. Detailed outcomes are in the included result note.

## Upload and run

Upload **`unity_token_matcher_probe_update.zip` to `~/diffusion/`**, the project root.
Paste:

```bash
cd ~/diffusion &&
unzip -o unity_token_matcher_probe_update.zip &&
mkdir -p logs &&
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base probe_token_matcher.sh
```

The existing checkpoint files must remain under `runs/token_matcher_1426578`.
They were intentionally excluded from the earlier small results download; the ZIP
of predictions alone cannot run this test. The original decoder checkpoint, data,
split manifest and full-validation report must also be at their recorded paths.

After submission replace `JOBID` below:

```bash
squeue -j JOBID
tail -F logs/token-probe-JOBID.out logs/token-probe-JOBID.err
```

Read `runs/token_matcher_probe_JOBID/RESULTS.md` and download
`runs/token_matcher_probe_JOBID/token_matcher_probe_results.zip` when complete.
There are no training epochs. The job evaluates four previously selected checkpoints,
each with real IR and four shuffled-IR conditions. Stage progress is printed; the
two-hour allocation is a limit, not a runtime estimate.

## What is tested

1. **Real-input replay:** reconstruct the original saved matcher scores within 0.001
   absolute logit tolerance and reproduce every saved first choice. Stop on mismatch.
2. **Unrestricted shuffling:** give each candidate pool another assessment molecule's
   spectrum, using three predetermined seeded derangements. No self-donors are allowed.
3. **Same-formula shuffling:** swap spectra only among assessment molecules sharing a
   reference formula. This isolates a harder distinction than exchanging unrelated
   compositions. Formula-singleton rows are excluded from the paired comparison.

Reference formula is used only to construct the diagnostic control. It never enters
the matcher or prediction ranking. Donor mapping and eligible row IDs are saved.
The decoder scores and candidate pools stay fixed in every condition, so this probes
the matcher rather than rerunning the full generator with wrong inputs.

Each condition reports matcher-only and fixed-combination accuracy, paired changes,
first-choice changes and both raw and within-pool-centered score changes. Centering
removes score offsets that affect every candidate equally and do not affect ranking.

- If wrong IR barely changes candidate ordering, that supports weak dependence on
  paired spectral information for ranking.
- If wrong IR changes rankings and lowers accuracy, the matcher uses IR; its poor
  original accuracy still indicates inadequate discrimination among close candidates.
- Similar aggregate accuracy alone is insufficient: many predictions can change in
  both directions while the total stays similar.

These controls cannot by themselves prove whether the main cause is training-negative
distribution, optimization, representation, or missing spectral information. They
narrow that question without another speculative fine-tune. Do not select different
weights or checkpoints based on these controls. Test data remains unused by this job.

## Verification and reproducibility

The probe verifies recorded source/input/checkpoint hashes and the saved selection,
then checks all original files remain unchanged afterward. It creates only a new
probe directory and rejects an existing output directory. Override `MATCHER_RUN` or
`PROBE_OUTPUT` through `sbatch --export` only when intentionally inspecting another run.
Absolute original input paths recorded in the configuration must remain accessible.

Local software verification covers grouped derangements, exclusion of singletons,
separation of logit offsets from rank changes, replay mismatch detection, and a full
CPU train-then-probe fixture with unchanged checkpoint hashes. This verifies the
diagnostic code; the actual trained model's IR dependence is still unmeasured locally.
