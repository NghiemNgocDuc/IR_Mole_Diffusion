# Hard-negative selector pilot: run 1361586

The uploaded `hard_selector_results.zip` reports a failed experiment. Retain the
ranking generator's original candidate ordering. No weights were promoted.

## Measured outcome

All methods use the same 12,478 validation inputs and fixed candidate pools.

| Method | Top1 identity | Top5 identity | First-choice formula |
|---|---:|---:|---:|
| Ranking decoder baseline | 64.02% | 89.26% | 90.19% |
| Matched random-negative selector | 33.76% | 89.26% | 76.09% |
| Generated hard-negative selector | 26.59% | 89.26% | 68.91% |

The hard arm fixes 829 baseline mistakes but breaks 5,499 correct baseline answers:
4,670 fewer correct predictions overall. Its Top1 change is -37.43 percentage
points (paired exploratory 95% interval -38.49 to -36.40). It is also 7.16 points
worse than the matched random control. Equal Top5 is expected for reordering a
fixed pool of at most five candidates; it is not evidence that selection improved.

## What the training logs show

Both arms train on the same 4,996 retained inputs, with 79 updates per epoch for
20 epochs. Four of the 5,000 selected training inputs produced no wrong candidate.

- Random control: loss falls from 1.0563 to 0.0090; online training accuracy first
  reaches 100% in epoch 4 and remains there.
- Hard negatives: loss falls from 1.5149 to 0.0249; online training accuracy first
  reaches 100% in epoch 13, fluctuates slightly, and remains 100% in epochs 17–20.
- The frozen generator itself gets 4,724/5,000 selected training inputs correct
  first (94.48%), with the correct identity present in 4,967/5,000 pools (99.34%).
  Validation accuracy and coverage are only 64.02% and 89.26%, respectively.

These are online scores on repeated, fixed training comparisons, with the correct
target injected into each training group. They are not validation accuracy. The
large training/validation gap supports poor generalization, consistent with
overfitting to these comparisons and a mismatch between training and evaluation
candidates. It does not identify a single cause or prove that a particular earlier
epoch would have worked: no per-epoch held-out selector results were saved.

The training set for the generator is not an independent development set for the
whole pipeline. Mining mistakes on inputs it has already learned can produce a
different problem from choosing candidates for unseen inputs. This experiment also
discards decoder confidence and uses only the new head's cosine scores for ranking.
The damage caused by that replacement is directly visible in the gains/losses.

## Audit and limits

The reproducible local audit is
`outputs/hard_selector_review_1361586/analyze.py`. Its `analysis.json` records input
and source-hash checks, regenerated training selections and negative groups,
recomputed metrics and paired intervals, candidate-pool preservation, score order,
fingerprint collision counts, and training/inference scoring consistency.
`row_changes.csv` records individual first-choice changes.

The audit completed successfully: all 12 recorded source hashes match, the seeded
training selection and both sets of 17,060 negative pairs reproduce exactly, and
every reported metric and paired interval reproduces. No selected training input
belongs to validation or test. All candidate pools and descending score order are
preserved. Training/inference scoring agrees within 1.8e-7 in a numerical software
check. No selected wrong first choice shares an identical Morgan fingerprint with
the correct candidate when that candidate is present. These checks found no
reporting, row-alignment, score-direction, or exact-fingerprint-collision explanation
for the drop; they do not prove that every modeling or implementation choice is sound.

Hard negatives were genuinely more confusable by formula: 8,861/17,060 (51.94%)
share the target formula, versus 279/17,060 (1.64%) in the random control. The hard
arm changed 8,978 of 12,478 first choices. The expected Top1 from choosing uniformly
among each existing candidate pool is 20.31%; 26.59% is above that reference but
far below the decoder's 64.02%.

The archive contains predictions and training histories, but not trained head or
generator weights. It therefore cannot support independent recomputation of the
saved neural scores. Graph identity ignores stereochemistry. The test split has
not been evaluated. Repeatedly used validation results are exploratory.

## Decision before another run

Do not promote either selector or simply extend this run's epochs. Before another
selector trial, establish a separate development evaluation with natural candidate
pools, record development performance each epoch, and preserve decoder scores as
an explicit input or baseline. A generator-unseen development set would diagnose
the whole-pipeline generalization problem more directly than another random split
of inputs the generator already saw. Any checkpoint or score-mixing choice must be
made without tuning on the final test set.

This experiment does not show that hard negatives are universally ineffective; it
shows that this particular small, fixed-pool projection-head setup is ineffective.
Even perfect reordering of the current pools cannot exceed 89.26% Top1 coverage.
