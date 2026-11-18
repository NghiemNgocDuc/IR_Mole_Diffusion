# Residual selector outcome: Unity job 1389118

The bounded residual selector did not demonstrate a reliable assessment gain.
Retain the existing ranking decoder. No weights were promoted and no additional
training was launched during this review.

## Paired results

| Partition | Rows | Decoder Top1 | Selected Top1 | Top5, both |
|---|---:|---:|---:|---:|
| Development | 2,000 | 66.20% | 66.70% | 90.70% |
| Assessment | 10,478 | 63.60% | 63.66% | 88.99% |

Assessment correct counts were 6,664 and 6,670: 220 fixes and 214 newly incorrect
answers, a net gain of six. The paired exploratory 95% interval for the Top1
change is **-0.353 to +0.439 percentage points**, spanning zero. First choices
changed on 826 rows; 392 changes replaced one wrong answer with another wrong
answer. Formula accuracy changed from 89.96% to 90.03%.

Compare against the paired 63.60% assessment baseline. The earlier 64.02%
ranking-model result covers all 12,478 validation rows and has a different
denominator.

## What the training history adds

| Epoch | Training loss | Development Top1 |
|---|---:|---:|
| 0, unchanged decoder | — | 66.20% |
| 1 | 0.89199 | 66.20% |
| 2 | 0.89138 | 66.10% |
| 3 | 0.88565 | 66.55% |
| 4, selected | 0.86362 | 66.70% |
| 5 | 0.83256 | 65.90% |
| 6 | 0.80764 | 65.10% |
| 7, stopped | 0.78843 | 63.90% |

The fixed selection rule selected epoch 4, which reached the required ten extra
development matches. Three later epochs failed to improve development correct
count, triggering early stopping at epoch 7. Training loss continued falling
while development Top1 declined. This is consistent with overfitting or a
mismatch between the training objective and Top1 accuracy; it does not identify
one definitive cause. Longer training of this configuration is not supported by
these results. Only the selected epoch was assessed on the assessment partition.

Training used 4,963 eligible natural candidate groups from the previously mined
5,000 training inputs. Targets were not injected. The generator had already
trained on these inputs, so candidate-distribution mismatch remains possible.

## Audit performed

The reproducible [audit script](../../outputs/residual_selector_review_1389118/analyze.py)
produced [analysis.json](../../outputs/residual_selector_review_1389118/analysis.json)
and [changed_predictions.csv](../../outputs/residual_selector_review_1389118/changed_predictions.csv).
It verified:

- Archive CRC, processed data and split hashes, baseline report hash, mining
  provenance, and all 14 recorded source hashes against local files.
- Training-only mining membership; exact seeded development/assessment
  partitions; disjointness and full validation coverage; no test rows.
- The eligible training count, epoch-selection rule and early-stop decision.
- Unchanged candidate pools and candidate payloads mapped through decoder rank.
- Decoder length-normalized scores, finite corrections within +/-0.25,
  combined-score arithmetic, stable ordering, and all reported identity/formula
  metrics, paired intervals and gain/loss counts.

No inconsistency was detected in these checks. Learned model weights were not
included in the archive, so neural corrections were not independently recomputed
from the selected checkpoint. Matching metadata is not independent verification
of the checkpoint bytes.

## Limits and decision

The selector avoided the severe degradation of earlier replacement scorers,
but did not establish better molecular identification. Assessment has 1,154
rows whose correct identity is absent from the fixed candidate pool; reordering
cannot fix those rows. Top5 therefore remains unchanged.

A post-hoc oracle calculation allowing arbitrary corrections within +/-0.25
gives an assessment Top1 ceiling of approximately 82.74% for these saved scores
and pools. This is a mathematical upper bound for this constrained experiment,
not an expected learned accuracy or a limit on other model designs.

Development was used for checkpoint selection. Both partitions come from
previously inspected validation data, not fresh independent holdouts. One
training seed was used; paired intervals do not measure seed variability or
correct for repeated experiments. Test data remain untouched. Graph identity
ignores stereochemistry and does not establish 3D or experimental-IR accuracy.

Keep the existing decoder. A further experiment needs a specific hypothesis
about candidate generation, representations or training-distribution mismatch,
rather than simply more epochs of this selector. No 90% Top1 claim follows.

## Artifact provenance

- Archive: `residual_selector_results.zip`
- SHA256: `26574750049654488b444cbdcd39f872ea159ec2e6cd4834b42326ee898f0115`
- Unity output: `runs/residual_selector_1389118`
- Selected checkpoint SHA256, reported only:
  `d0c3f9595abc085ad885fcf3d4b12d1ddf8aabea591bfcab11cf6168d5b43b1c`

This review changed documentation and generated audit artifacts only. Runtime
sources, datasets, checkpoints and the previously supplied release ZIP were
preserved.
