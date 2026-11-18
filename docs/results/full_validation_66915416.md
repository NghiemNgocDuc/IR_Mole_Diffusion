# Full-validation review: Unity job 66915416

Review completed 2026-11-10. Source: user-supplied `full_validation_results.zip`,
received 2026-11-08. ZIP SHA256:
`b29cdb4249f59aadac9334c6bd90f97890f75d9337019d5d8cab0bd1fb9afca1`.

## Confirmed results

All 12,478 validation molecules; EMA weights; legacy beam 5; canonical
non-isomeric, tautomer-specific graph identity. No reference formula is supplied
to prediction. The test split remains unevaluated.

| Model | Top1 | Top5 | Formula match | Top1 correct |
|---|---:|---:|---:|---:|
| Original baseline | 62.91% | 89.10% | 89.64% | 7,850 |
| CE fine-tuning | 64.02% | 89.04% | 90.27% | 7,988 |
| Ranking fine-tuning | 64.02% | 89.26% | 90.19% | 7,988 |

Each fine-tuned model adds 138 net correct first predictions (+1.11 percentage
points) over the baseline. The uploaded paired 95% intervals are +0.62 to +1.59
points for CE and +0.61 to +1.59 for ranking. CE versus ranking is tied: 299 gains
and 299 losses. Ranking's Top5 edge over CE is 27 molecules (+0.216 points);
a separate exploratory 20,000-replicate paired bootstrap gives -0.040 to +0.473
points. These findings do not establish a ranking advantage over CE.

Outside the original 1,000-row pilot, the additional 11,478 validation rows score
62.82% baseline, 63.84% CE, and 63.90% ranking Top1. Thus the gain extends beyond
the pilot rows. This complement is still validation, not a newly reserved test set.

## Remaining mistakes

| Ranking model first-choice outcome | Molecules |
|---|---:|
| Correct identity | 7,988 |
| Correct formula, different graph (excluding alternate tautomers) | 3,216 |
| Correct formula, alternate tautomer | 50 |
| Wrong formula | 1,224 |
| Missing or invalid first candidate | 0 |

Of 4,490 first-choice mistakes, **3,150 (70.16%) already contain the correct
identity at ranks 2-5**. Another 1,340 lack it in the candidate pool. Selecting
the correct candidate is the larger measured opportunity. The 89.26% pool
coverage is an oracle ceiling for this fixed pool, not promised achievable Top1.

Structures with three or more rings are particularly difficult: ranking Top1
is 44.55% on 2,339 molecules, versus 68.51% on 10,139 with zero to two rings.
The three-or-more-ring group contributes 595 of 1,340 pool misses. Ring classes
differ in other properties; this is a diagnostic association, not a causal claim.
Fluorinated molecules score 63.94% Top1 on 208 examples, so the earlier tiny
pilot fluorine group does not justify prioritizing a fluorine-specific change.

CE and ranking choose different first candidates on 1,089 molecules. In 297 of
each model's 299 wins, the answer was already in the other's pool. This supports
testing a different ordering signal. A post-hoc larger-score-gap selector reaches
64.39% on this inspected validation set; it remains an exploratory diagnostic,
not a promoted model or independently established accuracy. Model score scales
are uncalibrated. Oracle choices using known answers are likewise not deployable.

## Next controlled experiment

Use the existing separate compatibility head with the frozen ranking encoder and
exact saved ranking candidate pools. It learns associations between IR features
and candidate fingerprints from training pairs only. Keep the five-epoch preset,
seed, candidate pools and pure cosine scoring fixed. This isolates ordering from
candidate generation and cannot increase the fixed Top5 of 89.26%.

The existing head uses random in-batch negatives; it does not yet train against
the decoder's explicit wrong candidates. If it cannot distinguish these
confusions, a later matched-budget experiment can introduce multiple
training-generated negatives. The current pilot's mined arm changed only about
5% of training pairs and did not establish an added gain. Do not add these
validation mistakes to training.

Keep the existing ranking order unless the new result supports an improvement.
Before investing in additional seeds, require positive paired Top1 gain, a
paired interval above zero, and unchanged Top5. Then examine seed stability
before fixing the method and using the test set. These are research decision
criteria, not automatic model promotion.

The code's historical `promising_pending_broader_validation` flag only requires
positive Top1 and nondecreasing Top5; it does not check interval exclusion of
zero. The guide now distinguishes that flag from the stricter review criterion.

Launch instructions: [compatibility experiment](../../COMPATIBILITY_EXPERIMENT.md)
and [upload quick start](../../START_COMPATIBILITY_RANKING.md).

## Verification and code changes

- ZIP CRC and all seven embedded file hashes checked.
- Local dataset and split SHA256 values match the reports. All validation row
  identities, formula references, complete membership, and ordered selections checked.
- The three full chemistry audits reparse candidates and recompute identities,
  formulas, error categories and scored tautomer diagnostics. No invalid
  candidate entries or formula annotation corrections were found.
- The independently rerun strict comparison exactly matches uploaded model
  summaries, paired comparisons and recommendation.
- The six evaluation source files still match the remote report hashes.
- Recent commit `05a1d1e` parameterized the compatibility launcher and updated
  its guides. This change is preserved. No model architecture, loss, checkpoint,
  evaluation code or training data was modified during this review.
- Nine existing compatibility tests pass, including a diagnostic training run.
  Bash syntax, LF line endings, explicit checkpoint/report routing, legacy pilot
  fallback, and rejection of missing inputs were checked. Shell routing checks
  used stubs and submitted no training job.
- Windows blocked an RDKit extension on November 8; the unchanged import worked
  on November 10, allowing full chemistry checks. No security policy was altered.
  Earlier saved-string audit limitations are superseded by these chemistry checks.

Detailed artifacts are in `outputs/full_validation_review_2026-11-08/`:
`baseline_audit/`, `ce_audit/`, `ranking_audit/`, `comparison_recomputed.json`,
`saved_result_audit.json`, `independent/`, and `launcher_verification.json`.
The existing dataset cache supplies training tautomer-family membership; scored
tautomer keys are recomputed. The scientific checkpoints remain on Unity and
were not loaded locally; their reported hashes are preserved.

Full validation includes the pilot subset and has been inspected repeatedly.
Intervals describe paired row sampling, not training-seed variability, and are
not adjusted for multiple comparisons. Graph accuracy does not establish
stereochemistry, geometry, experimental-IR or new-family performance.
