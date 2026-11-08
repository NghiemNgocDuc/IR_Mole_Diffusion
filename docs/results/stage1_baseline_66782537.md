# Stage-1 validation baseline: Unity job 66782537

Source: user-pasted `evaluation/baseline_audit_66782537/error_audit/analysis.md`, received 2026-11-05. The complete report and trained checkpoint are on Unity; they have not been independently downloaded here. The pilot verifies them on Unity before training.

Scope: 1,000 evaluated validation rows out of 12,478; canonical non-isomeric graph identity, with tautomer distinctions preserved.

| Metric | Result |
|---|---:|
| Strict Top1 | 640 / 1000 (64.0%) |
| Strict Top5 | 897 / 1000 (89.7%) |
| First-choice formula match | 892 / 1000 (89.2%) |
| Correct identity at ranks 2–5 | 257 |
| Correct identity absent from first five | 103 |

First-choice errors: 247 other graphs with the same formula, 5 alternate tautomers with the same formula, and 108 wrong formulas. No invalid or missing first choices were reported. Thus 257/360 (71.4%) of first-choice errors have an existing beam candidate to recover; this is an opportunity count, not an achievable-accuracy prediction.

Multiple-ring structures are a useful follow-up diagnostic: Top1 falls from 81.7% for acyclic molecules (109 rows), to 71.5% for one ring (389), 61.3% for two rings (315), 45.8% for three (142), 41.4% for four (29), and 13.3% for five (15). These groups differ in composition and have unequal sizes; no causal claim follows. The fluorinated group has only 13 rows, too few to justify a separate training strategy from this audit alone. Aromatic molecules did not perform worse here (67.9% Top1 on 159 rows).

The older first-500 result (70.6% Top1) used a different subset. Its difference from this result does not demonstrate a model regression.

Dataset SHA256: `49c832a638c46a8420bd8a0187d18dfb757e9b9289cf4b4d9db0a3b3fbac71dd`.

Split manifest SHA256: `9e46e1949dfcd88181a64691ba407bc4ab8ea5fa98e0b1f741b7bb69242c8d62`.

This baseline does not establish full-validation, test, experimental-IR, stereochemical or 3D accuracy. Fine-tuning gains have not been measured yet.
