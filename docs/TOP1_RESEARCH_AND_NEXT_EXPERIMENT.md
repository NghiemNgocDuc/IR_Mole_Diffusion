# Why Top1 stalls, and what to test next

Update: the proposed token matcher and IR probe have now completed. See
[the verified probe review](results/token_matcher_probe_1444546.md) for current
findings and the next decision. The research and proposal below record the
reasoning before those experiments; their proposed gains are not measured outcomes.

Research and local diagnostics completed **2026-11-11**. This document replaces
the older proposed-next-experiment recommendations; the historical results remain
unchanged. No model was trained, downloaded, promoted or modified in this review.

**Conclusion:** learned selection among generated molecules is a credible task,
and published systems explicitly implement it. The unsuccessful local helpers
were small frozen-feature experiments, not reproductions of those systems.
The next useful experiment should change the representation and supervision,
rather than extend the same selector's training.

## 1. Quantify the problem correctly

The audited ranking decoder evaluated all 12,478 validation molecules:

| Location of the correct identity | Rows | Share of all rows |
|---|---:|---:|
| First candidate | 7,988 | 64.02% |
| Second candidate | 1,835 | 14.71% |
| Third candidate | 750 | 6.01% |
| Fourth candidate | 372 | 2.98% |
| Fifth candidate | 193 | 1.55% |
| Absent from these candidates | 1,340 | 10.74% |

There are **3,150 recoverable ranking mistakes**, accounting for 70.16% of the
4,490 Top1 errors. Conditional on the answer being present, the decoder selects
it correctly **71.72%** of the time. This conditional selection accuracy and
end-to-end identification accuracy are different metrics.

For a selector restricted to this fixed pool:

`end-to-end Top1 = pool coverage × selection accuracy when the answer is present`

| Hypothetical conditional selector accuracy | End-to-end Top1 with today's pools |
|---|---:|
| 80% | 71.41% |
| 90% | 80.33% |
| 95% | 84.80% |
| 100% | 89.26% |

These are arithmetic scenarios, not forecasts. To exceed 90% end-to-end, the
generator must also increase candidate coverage. For example, 95% coverage and
95% conditional selection would yield 90.25%; neither has been demonstrated here.

The new [diagnostic script](../outputs/research_top1/analyze_candidate_bottleneck.py)
reparsed every saved candidate with RDKit, verified the original ranking-report
hash, and reproduced Top1 and Top5. Its [JSON results](../outputs/research_top1/candidate_bottleneck.json)
also show:

- Of the 3,150 recoverable errors, **2,395 (76.03%)** have a wrong first candidate
  with the correct formula. Detailed connectivity discrimination is essential.
- Supplying the **true reference formula** and selecting the first matching
  candidate gives **68.68%**, versus 64.02% originally. This is an oracle
  diagnostic, not IR-only performance or an upper bound for every formula-aware
  model. It shows that a simple formula filter alone leaves many errors.
- Only nine target-present pools contain a distinct candidate with an identical
  2048-bit radius-2 fingerprint. Exact fingerprint collisions cannot explain the
  broad failure. A representation can still be inadequate without exact collisions.

## 2. Evidence for the causes, and what remains uncertain

### Training candidate distribution differs sharply from deployment

The generator scored **94.48% Top1 and 99.34% coverage** on the 5,000 inputs used
for mining, compared with **64.02% and 89.26%** on validation. It had already
trained on those mining inputs. The hard-selector experiment repeated roughly
5,000 fixed comparisons and reached 100% online training accuracy, yet achieved
only 26.59% validation Top1. This is strong evidence of poor generalization and
consistent with overfitting and candidate-distribution mismatch; it does not
prove which mechanism contributes most.

The residual experiment retained decoder information and selected epoch 4, but
assessment gains and losses were 220/214. Later training reduced development
Top1 while training loss continued decreasing. Additional epochs of that
configuration are not supported by the observed history.

### The selectors had much less access to spectral detail than the generator

In [compatibility.py](../diffusion/compatibility.py), `cache_spectrum_features`
freezes the spectrum encoder and averages all live output tokens into one vector.
`molecular_fingerprints` provides fixed radius-2 fingerprints. Both the compatibility
and hard heads use small projected-vector similarity functions. The residual
head in [residual_selector.py](../diffusion/residual_selector.py) reduces these
representations to width 32 and adds a bounded correction.

The head cannot ask different parts of a candidate structure to attend to
different spectral tokens. It also cannot update the encoder to preserve features
useful for distinguishing these alternatives. This is a concrete architectural
restriction. That restriction causes a particular number of errors is a
**hypothesis requiring an ablation**, not a proven conclusion. Mean pooling can
work when representations are trained for it; it is not inherently invalid.

The residual bound is another restriction: differences in corrections cannot
exceed 0.5. Its previously audited ideal-correction ceiling on the assessment
partition was about 82.74%. Increasing the bound alone is not an evidence-based
fix, because the unconstrained replacement scorers already degraded accuracy.

### Sequence likelihood is an imperfect proxy for molecular identity

[spectrum.py](../diffusion/spectrum.py) primarily trains next-token cross-entropy
on one saved SELFIES representation. Fine-tuning adds a small pairwise hinge
term. That training signal differs from explicitly choosing the correct complete
graph among several close alternatives. The generator uses the full spectral
token sequence; the latest helper trials only trained heads on its frozen output.

Length-normalized sequence likelihood, representation syntax and graph identity
are not interchangeable. Molecular-level matching and representation augmentation
are plausible remedies. Neither a length-normalization adjustment nor wider beam
search raised Top1 in the earlier 1,000-row decoding experiment, so those results
do not support presenting a decoder setting as the main solution.

### Chemistry is difficult, but 64% is not an established information limit

The full-validation audit finds 44.55% Top1 for structures with at least three
rings, versus 68.51% for zero to two rings. This is an association, not a causal
test. The data audit found no exact duplicate packed spectra, no observed
held-out vocabulary gap, and no truncation by the 81-peak capacity. We have not
established that the remaining errors are physically unresolvable from IR.

The supplied source lists appear to omit weak modes: no stored intensity is
below 0.5, and many lists have fewer modes than a complete normal-mode catalog.
The upstream filtering rule is undocumented. Recovering additional original
modes may help, but requires the original source and controlled retraining;
increasing the current padding limit will not recover absent measurements.

## 3. Papers that directly inform the next step

The following descriptions keep each benchmark's task and inputs explicit.
Their percentages are not directly comparable with our SQuIRL validation score.

### Vib2Mol: a dedicated candidate matcher exists

Lu et al., *Chemical Science*, 2026, DOI 10.1039/D6SC04559F. The model combines
spectrum/structure alignment with a matching loss and uses the matcher to rerank
retrieved or generated structures. Its VB-Mols **Raman retrieval** ablation goes
from 89.57% after matching-loss training to 93.20% with reranking without formula.
This demonstrates the mechanism in that benchmark, not 93% IR-only generation.
The article also reports experimental IR generation at 34.07% without formula
and 55.14% with formula. [Published article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13573156/),
[accessible methods, sections 4.3–4.5](https://arxiv.org/html/2503.07014v3).

The official implementation's `MolecularEncoder` has a multimodal branch with
cross-attention to the spectrum's hidden states; `matching()` calls that branch.
Its configuration includes an IR-only/no-formula generation experiment. I checked
the code at commit `732806b99abf4aedac07cf56ac34fd670948e338`:
[model](https://github.com/X1nyuLu/vib2mol/blob/732806b99abf4aedac07cf56ac34fd670948e338/models/vib2mol.py),
[inference settings](https://github.com/X1nyuLu/vib2mol/blob/732806b99abf4aedac07cf56ac34fd670948e338/scripts/configs/generation.yaml).

### Forward IR prediction supplies another scoring signal

Rowlands et al., *Digital Discovery*, 2026, DOI 10.1039/D6DD00281A, evaluate
structure-to-spectrum models by ranking candidates against an observed spectrum.
On their fixed 1,357-query experimental benchmark, Top1 is 62.9% for a 2D MPNN,
73.9% for a 3D PaiNN model and 76.0% for their ensemble. The library consists of
the held-out fold, not our decoder's top five. They also find that chemically
similar distractors make retrieval harder. The force-field geometry variant is
relevant to practical candidate scoring. [Methods and results](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667908/),
[official code](https://github.com/Goodman-lab/pasgen).

### Fine-tune the spectral representation, not only an attached head

Mick et al., arXiv:2607.26164v1, 2026, is a preprint on formula-free prediction.
It jointly trains a structural encoder and attentional spectral pooling with
contrastive alignment alongside sequence cross-entropy. Section 5.2 reports
improved Top1 from alignment. Its separate architecture comparison reports
52.15% versus 63.28% QM9S Top1 for dense and MoE-LOSN variants with other shared
components. This is evidence for testing representation learning, not a reason
to copy every architectural change at once. [Methods and ablations](https://arxiv.org/html/2607.26164v1).

### Preference fine-tuning is promising but requires appropriate feedback

Tan et al., FIRMPO, arXiv:2608.16082v1, 2026, is a preprint using reference-model
regularized preference optimization, formula agreement and forward IR similarity.
Table 1 reports IR-Bench NIST Top1 rising 50.99% to 62.38%, but only 69.48% to
70.79% on IBM. Gains depend on the setting. Formula is supplied as input and
Chemprop-IR provides the spectral score, so our current pipeline cannot reproduce
this by changing a loss coefficient. [Paper, sections 3–4](https://arxiv.org/html/2608.16082v1).

Access note: the published Vib2Mol full-page endpoint challenged automated
access, but indexed publisher-deposited text, the complete accessible preprint
methods and official code were available. The Rowlands and two other preprint
methods/results were read directly. Some older RSC pages returned 403; they
are not treated as newly verified full-text evidence here.

## 4. Recommended first experiment: token-level spectrum/candidate matching

This is a proposed local adaptation, not a claimed paper reproduction or a
finished training script.

**Question:** does jointly trained, candidate-conditioned access to spectral
tokens distinguish the current close alternatives better than a frozen pooled
representation?

1. **Preserve the generator and existing evaluation pools initially.** Copy the
   encoder into an independent matcher. The matcher can learn without silently
   changing which candidates the existing generator supplies. Fixed pools isolate
   ranking improvements; missing targets stay in the accuracy denominator.
2. **Represent the complete candidate and spectral tokens.** Start with a
   bidirectional candidate SELFIES encoder and cross-attention to masked peak
   tokens, reusing compatible embeddings as initialization. Canonicalize candidate
   identities; validate vocabulary and sequence lengths. A molecular graph encoder
   is a later controlled alternative, especially for ring topology.
3. **Train with all 99,825 original training pairs.** Use identity-aware alignment
   plus a spectrum/candidate matching objective. Mix random negatives with
   training-only same-formula and structurally similar negatives, refreshing them
   over epochs. Never treat another serialization of the same graph as a negative.
   Keep tautomer-specific labels consistent with the evaluation definition.
4. **Compare frozen and trainable spectral encoders at matched budgets.** A short
   new-layer warmup followed by a lower learning rate for the copied encoder is a
   reasonable starting design. Choose numeric budgets on development data and
   record them in advance of assessment. The critical comparison is whether
   learning spectral features helps, not whether a larger model fits training.
5. **Include decoder-only, matcher-only and calibrated-combination results.**
   Decoder scores are useful evidence; replacing them blindly already failed.
   Any score calibration or mixing is fitted on development predictions, with
   the unchanged decoder available as the fallback. All score scales and
   candidate-pool definitions must be recorded.

A practical ablation sequence is: unchanged decoder; full-token matcher with
frozen encoder; the same matcher with encoder fine-tuning. Use the same target
pairs, negative sampling recipe, candidate pools and training budget for the last
two. Comparing with the existing frozen-mean results is additional historical
context, not a matched causal test of pooling alone.

### Handle the mining mismatch explicitly

For the first matcher, all-training-pair alignment and changing structural
negatives avoid relying solely on the old 5,000 in-sample decoder pools. These
negatives still need not match deployment difficulty; measure that limitation.

If a second stage uses decoder-generated mistakes, use out-of-fold generation
inside the original training split: a fold's candidates must come from a
generator that never trained on that fold. Simply splitting those rows now or
fine-tuning the current all-training checkpoint on fewer rows does not undo its
prior exposure. True out-of-fold generation therefore requires separately trained
generators, and its cost should be measured before committing to all folds.
Do not substitute validation mistakes as gradient-training examples.

## 5. Complementary experiment: predict candidate spectra

Train a graph-to-IR model using original training structures and their retained
spectra. Start with 2D graphs; a 3D version should use candidate-generated
conformers available equally for every candidate, with consistent geometry
preparation in training and inference. Never give only the correct candidate its
reference geometry.

Convert retained peaks to a documented broadened target on a fixed physical
wavenumber grid. Preserve the full available frequency range and distinguish
shape normalization from absolute intensity. Broadening represents existing
information; it does not reconstruct missing weak modes. Verify forward errors
on held-out structures and, crucially, accuracy in choosing among actual decoder
candidates. A good average spectrum reconstruction score alone is insufficient.

Use spectral mismatch as an additional score, or later as a preference-training
signal, only if it improves molecular selection on development. Training on this
SQuIRL representation is more defensible than directly trusting a model trained
on a different quantum method or experimental phase. Check any external
pretraining for overlap with our validation/test molecular identities: QM9-derived
checkpoints are particularly likely to overlap, and cannot be assumed clean.

This is the strongest complementary route if token matching remains weak. It
also makes errors inspectable through predicted-versus-input spectral plots.

## 6. Generator fine-tuning and evaluation order

Once an ordering signal is useful, expand generation to a larger deduplicated
pool, for example 20 candidates, and measure actual coverage. Keep fixed Top1 and
Top5 reporting, with Top20 coverage reported separately. Wider beams previously
improved fixed Top5 only on a 1,000-row pilot; that does not establish full-set
Top20 coverage or Top1 gains.

For a direct generator fine-tune, test moderate alternate molecular
serializations and joint alignment as separate changes against an equal-budget
CE control. Convert any alternate SMILES to validated SELFIES, require identical
canonical graph identity, and reject unsupported tokens or excessive lengths
explicitly. Do not silently grow the checkpoint vocabulary or assume a Raman
SMILES result transfers numerically to this model. Preference optimization should
follow a validated scoring signal, with reference regularization and retained
positive supervision to control forgetting.

Use the existing development/assessment partition consistently for exploratory
selection, report gains/losses and conditional selection accuracy, and record
development performance every epoch. It is previously inspected validation data,
not a new holdout. The original generator has also used validation for selection;
head-level partitioning does not remove that history. Confirm a promising recipe
across seeds before freezing the protocol and evaluating untouched test data.
Prior inspected test examples must be disclosed rather than presented as unseen.

Success means a reproducible net Top1 gain on paired inputs, without concealing
missing candidates, validity failures, seed instability or reduced Top5. An
exploratory paired interval above zero is a screening signal, not sufficient
evidence by itself after repeated tuning. No 90% result is promised.

**Recommended priority:** implement and evaluate the trainable full-token matcher
first; retain a matching-domain forward-spectrum model as the independent next
signal; expand candidate generation to address the missing 10.74%. Keep the current
64.02% full-validation decoder available throughout.
