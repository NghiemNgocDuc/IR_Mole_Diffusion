# IR dependence probe: verified result and next decision

Reviewed 2026-11-11. Inputs: `token_matcher_probe_results.zip` and
`token_matcher_results.zip`. Reproduce with:

```powershell
.runtime/python/python.exe scripts/audit_token_matcher_probe.py
```

Machine-readable verification: `outputs/token_matcher_probe_audit_1444546/verified_analysis.json`.
This audit checks saved predictions; matcher weights remain on Unity.

## What was verified

Both ZIP CRCs passed. Original configuration and selection hashes agree across
archives; the probe source hash matches local code. All 10,478 assessment row IDs,
candidate pools, replay scores, donor permutations, eligibility lists, Top1 counts,
paired gains/losses, first-choice changes and mean score changes were checked across
all 16 controls. Same-formula comparisons contain 10,397 rows; 81 singleton rows
are excluded. Saved real-input replay errors are at most 0.00000477 logits.

## Results

The labels below distinguish the scoring method from the checkpoint selection.
The original probe's `frozen/combined` row reports matcher-only accuracy at the
checkpoint chosen for combination. It does not report the combined accuracy in
that table. The JSON contains both, reproduced here.

| Scoring method | Correct IR | Unrestricted shuffled IR, three seeds |
|---|---:|---:|
| Original decoder, fixed throughout | 63.60% | 63.60% |
| Frozen matcher plus decoder, selected epoch 9 / weight 0.1 | 63.86% | 62.39–62.51% |
| Matcher alone at that same epoch 9 | 38.01% | 16.41–16.73% |
| Fine-tuned matcher plus decoder, selected epoch 7 / weight 0.1 | 63.64% | 62.32–62.83% |
| Matcher alone at that same epoch 7 | 34.95% | 16.59–17.18% |

The separately selected pure-matcher checkpoints score 37.64% (frozen) and 36.39%
(fine-tuned). They are different epochs, so their comparison with the combined
models is not a fixed-checkpoint weight ablation.

On the same-formula subset, the frozen combination declines from 63.78% to 61.83%;
its pure matcher declines from 38.03% to 17.59%. First choices change on about
78–82% of rows across all pure-matcher controls. Within-pool-centered logit changes
are nonzero, so the effect is not merely a common offset applied to all candidates.

These results support useful dependence on the paired spectrum, including distinctions
within formula groups. They do not establish physical interpretability, resolve
which spectral features are used, or identify the cause of poor generalization.
The shuffled spectra can also change peak counts and other input properties.

## Why using IR is insufficient

The frozen combination adds 27 correct predictions over the decoder: 314 fixes
and 287 regressions. Its earlier exploratory 95% interval for the net gain includes
zero. Wrong-IR controls support an IR contribution, but do not turn that marginal
decoder improvement into an established gain.

For the epoch-9 matcher, on 2,660 rows where the correct answer is at decoder ranks
2–5, its true-IR logit exceeds the wrong first candidate's logit in only 1,383 cases
(51.99%). This is a two-candidate diagnostic, not five-candidate accuracy or a
random-chance test. Among 6,660 multi-candidate rows where the decoder is correct,
the matcher strictly scores the target above every alternative in only 3,144 cases
(47.21%). Four additional decoder-correct rows have only one candidate and are
excluded from this latter diagnostic. Strict comparisons count ties as failures.

The correct graph appears in 9,324/10,478 pools (88.99%). Even an ideal selector
restricted to these exact pools cannot reach 90% overall Top1. This is a ceiling
for this candidate set, not a general limit on IR identification. Achieving 90%
requires both better candidate coverage and better selection.

## Next experiment specification

Keep the existing decoder as the reference. The next candidate experiment should
test the training comparisons, holding the matcher architecture fixed. The current
`TrainingPairs.negative_plan` samples one random training identity, one same-formula
identity when available, and one structurally similar identity from a bounded search
of 32 identities. `train_epoch` learns four-way classification plus alignment. It
does not select negatives by how strongly the spectrum matcher confuses them.

1. Measure the saved matcher's accuracy on its original four-way training groups
   and on decoder-generated training candidates. Use a predetermined sample and
   report target-present coverage, same-formula pairwise accuracy, and loss. This
   directly checks the suspected task mismatch; neither high training accuracy nor
   a distribution gap alone proves causation.
2. If the gap is present, compare the current sampling against negatives chosen
   by the current spectrum–molecule similarity or matching scores, using training
   identities only. Hold architecture, initialization, updates and selection protocol
   constant. Retain decoder scores for the combined prediction and report pure
   matching separately. Do not simply repeat the older fingerprint-head experiment.
3. For generated-negative supervision, prefer candidates from a generator trained
   without the relevant training fold. The existing generator already saw all
   original training inputs; its memorized training candidates are easier and can
   misrepresent deployment. This requires additional generator training or suitable
   provenance-verified checkpoints. Do not label the current candidates out-of-fold.
4. Preserve test data for a final locked comparison. The current development and
   assessment rows have been inspected repeatedly, so further results there remain
   exploratory. Do not select a new mixture or checkpoint from the shuffle controls.

This is a proposed controlled study, not an implemented training change or a promise
of improvement. The archive lacks training candidate logits and selected weights,
so step 1 cannot be measured from this results download alone.

## Research cross-check

[Vib2Mol, methods 4.5](https://arxiv.org/html/2503.07014v3#S4.SS5) describes sampling
hard negatives based on learned contrastive similarity and training a binary
spectrum–structure matching objective. Our current structural-similarity sampling
and four-way objective differ. This supports testing model-confusing negatives;
the paper does not establish that this difference causes our measured deficit.
Its benchmark accuracy should not be transferred to this dataset or task.

No training, checkpoint replacement or score tuning was performed in this audit.
Graph identity ignores stereochemistry. These results establish neither 3D accuracy
nor experimental-IR accuracy.
