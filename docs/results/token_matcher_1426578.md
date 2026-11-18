# Token matcher 1426578: audit and next diagnostic

Source: supplied `token_matcher_results.zip`; numerical counts recomputed from its
four assessment prediction files using `scripts/audit_token_matcher.py`. The dataset
hash matched the original configuration, and original candidate identities,
serializations and decoder probabilities agreed across methods. Local source hashes
also matched the original run configuration. The archive does not contain neural
weights, so these checks do not independently reproduce neural inference.

Development: 2,000 rows. Assessment: 10,478 rows from previously inspected validation.
All methods use the same assessment rows; do not directly compare these percentages
to the 64.02% full-validation result. No test evaluation or automatic model promotion.

| Method | Correct first choices | Top1 | Fixes | Regressions |
|---|---:|---:|---:|---:|
| Decoder | 6,664 | 63.60% | — | — |
| Frozen combination | 6,691 | 63.86% | 314 | 287 |
| Fine-tuned combination | 6,668 | 63.64% | 292 | 288 |
| Frozen matcher alone | 3,944 | 37.64% | 861 | 3,581 |
| Fine-tuned matcher alone | 3,813 | 36.39% | 822 | 3,673 |

Development selected the frozen combination, epoch 9, mixing weight 0.1. Its paired
assessment change is +0.2577 percentage points, exploratory 95% interval
[-0.2007, +0.7253]. This does not demonstrate a reliable gain. Matcher-only checkpoints
were separately selected at epoch 10, so pure versus combined results are not a
same-checkpoint mixing-weight ablation.

The frozen combination changed first choice on 1,083 rows. In addition to 314 fixes
and 287 regressions, 482 changes left the prediction wrong. Most fixes (217) and
regressions (201) occurred between structures with the same formula. Among regressions,
86 changed from the correct formula to an incorrect one. Median decoder top-two score
margins were 0.0659 for fixes and 0.0749 for regressions. These descriptive values do
not establish a safe post-hoc confidence threshold.

The correct graph appears in 9,324 pools (88.99%). Of those, 8,860 correct candidates
use the exact dataset SELFIES sequence; only 464 use a different sequence. Even on
the matching-sequence group, frozen matcher-alone accuracy is 3,787/8,860 (42.74%),
compared with decoder 6,450/8,860 (72.80%). Thus serialization mismatch cannot explain
most of the observed deficit.

Training loss fell from 0.5953 to 0.1470 (frozen) and to 0.1722 (fine-tuned). This shows
improvement on the training objective, not generalization to the decoder's candidate
pools. A training-versus-inference comparison mismatch remains a hypothesis, not a
proven cause. The raw scores with one spectrum per pool cannot establish whether
the trained scorer uses IR effectively.

Follow-up completed: [IR dependence probe 1444546](token_matcher_probe_1444546.md)
verifies a large loss of matcher accuracy with shuffled spectra. It confirms IR
dependence while leaving the cause of weak candidate discrimination unresolved.
The linked report records the next controlled comparison and preserves this run's
original results.
