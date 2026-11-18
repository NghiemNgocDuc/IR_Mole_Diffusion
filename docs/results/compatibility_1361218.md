# Compatibility experiment 1361218: keep the ranking model

Reviewed 2026-11-10 from the supplied `compatibility_predictions.json`,
`compatibility_comparison.json`, `configuration.json`, and `training_history.json`.
The comparison reference is the fine-tuned ranking checkpoint, not the original
62.91% model. All 12,478 validation rows were evaluated; the test split was not used.

| Metric | Ranking model | Compatibility ordering |
|---|---:|---:|
| Top1 identity | 7,988 / 12,478 (64.02%) | 6,767 / 12,478 (54.23%) |
| Top5 identity | 11,138 / 12,478 (89.26%) | 11,138 / 12,478 (89.26%) |
| First-choice formula | 11,254 / 12,478 (90.19%) | 10,930 / 12,478 (87.59%) |

Top1 falls by 9.785 percentage points: 1,487 gains and 2,708 losses, a net loss
of 1,221 correct answers. The paired 95% interval is -10.843 to -8.832 points.
This is a clear regression on this validation selection. No checkpoint was promoted.

## What happened

The helper changed the first choice on 5,965 rows. It preserved 5,280 correct
answers but displaced another 2,708. Of those losses, 1,947 chose a wrong identity
with the correct formula and 761 chose a wrong formula. The regression occurs
across ring-count groups zero through five, not just one unusual molecular class.

The training history contains all five epochs, each with 99,825 examples and
390 optimizer steps. Loss decreases from 2.12779 to 0.46419. Recorded final-batch
gradient norms are finite. This establishes optimization progress on the training
objective; it does not demonstrate convergence or successful candidate selection.
No per-epoch validation results or training-candidate ranking diagnostics were supplied.

The implementation trains against other molecules in randomly shuffled batches.
It does not explicitly use the generator's closely competing wrong candidates.
At inference it replaces decoder ordering entirely with cosine similarity; decoder
order is retained only for exact-score ties. This training/inference mismatch is
a plausible explanation to test, not a proven sole cause. Frozen mean-pooled IR
features and the fingerprint representation could also limit discrimination.

Exactly three of the 4,371 wrong first choices whose pool contains the answer
have a Morgan fingerprint identical to the answer's fingerprint under the scorer's
radius-2, 2,048-bit, nonchiral representation. Exact fingerprint equality therefore
does not explain most of the regression. This does not rule out broader limitations
of that representation or of the learned projections.

## Verification

The uploaded configuration references the correct ranking checkpoint hash and
the exact full-validation report from the earlier ZIP. Data/split hashes match
local files, and the reported training row list exactly equals the 99,825-row
training split, disjoint from validation. All nine recorded source hashes match
the current local source. Candidate-pool provenance and ordered validation row
identities agree across the uploaded files.

The original report's candidate chemistry was reparsed. Every reranked candidate
retains its original SMILES, formula, SELFIES and log probability; only rank and
compatibility score were added. All pools are unchanged. Scores are finite,
within cosine bounds, and sorted in descending order with the documented decoder
tie-break. Per-row flags, aggregate metrics, gains/losses and paired intervals
were independently recomputed and exactly match the supplied comparison.

No data/pool/order mismatch was found in these checks. This is not a proof of
absence of every software issue: model weights were not supplied or loaded,
and training/inference were not replayed locally. Dataset class annotations come
from the existing chemistry audit.

Reproducible audit and input hashes:
`outputs/compatibility_review_1361218/analyze.py`, `analysis.json`, and
`row_changes.csv`. No scientific training, test evaluation or promotion was run
during this review.

## Decision and next hypothesis

Retain the 64.02% ranking model. Do not use this pure-cosine helper as the default
or repeat the same run merely because its training loss decreased.

A next bounded experiment can train a selector on the generator's actual wrong
alternatives from training inputs, especially same-formula confusions, with a
matched-budget random-negative control. Preserve decoder evidence as an explicit
control or feature rather than assuming the association score can replace it.
Any score mixing, calibration or hyperparameter choice needs a development
partition drawn from training, not optimization against these inspected validation
answers. Start with a small measured experiment before exhaustive negative mining.

These changes are proposed, not implemented or measured. Additional epochs alone
are not supported as a fix by the supplied results. Keep the test split untouched
until the method is fixed; report all predeclared training seeds rather than select
the best validation seed. Validation identity metrics do not establish 3D accuracy.
