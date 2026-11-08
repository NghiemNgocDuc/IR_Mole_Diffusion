# IR-to-structure accuracy: literature review and proposed experiments

**2026-11-06 implementation update:** [Accuracy update v2](../ACCURACY_UPDATE_V2.md)
adds a preparation-only decoding comparison, model-generated alternatives from
training inputs, three matched-budget training arms, and a separate optional
frozen-encoder compatibility scorer. These are controlled local experiments,
not reproductions of the papers below. The Unity baseline remains 64.0% Top1
and 89.7% Top5 on the same 1,000 validation rows until new results are measured.

**2026-11-05 update:** The [representative 1,000-row audit](results/stage1_baseline_66782537.md)
now reports 64.0% Top1 and 89.7% Top5. Its 257 correct lower-ranked candidates
motivate the [controlled CE versus ranking pilot](../FINETUNE_PILOT.md).
This supersedes the earlier first-500 sample as the pilot baseline. No fine-tuning
accuracy gain has been measured yet; the research discussion below records the
earlier investigation.

Reviewed 2026-11-04. This is a research and experiment plan, not a finalized
training recipe. No training job was submitted and no accuracy gain is claimed.

Follow-up: the [full local dataset audit](DATASET_AND_OUTCOME_AUDIT.md) found
that the existing first-500 validation subset contains only 1.0% aromatic
molecules versus 17.8% in full validation, and 43.8% acyclic molecules versus
10.8%. A representative baseline and per-row error audit now take priority
before selecting the fine-tuning objective. Keep the old subset for historical
paired comparison, not as the only screening benchmark.

The immediate objective is better **IR-only, first-choice molecular identity**
on the existing supported chemistry. Experimental IR transfer, unfamiliar
molecular families, and 3D geometry require separate evidence.

## What we know about this project

The recorded Unity graph evaluation covers 500 validation molecules: 353 correct
first predictions (70.6%) and 462 correct identities among five candidates
(92.4%). Thus 109 cases contain the answer below first place, while 38 lack the
answer in the retained candidate list. The gap motivates ranking experiments;
92.4% is coverage of this particular candidate set, not a promised attainable
first-choice accuracy. See [the recorded results](PERFORMANCE_EXPECTATIONS.md).

The model uses peak tokens and SELFIES. The draft fine-tune retains positive
token cross-entropy and adds a pairwise sequence-score hinge loss. It selects
one negative from a sampled pool of training molecules, preferring the same
formula and higher fingerprint similarity. It does not mine the model's actual
wrong predictions, implement contrastive embedding alignment, or predict a
candidate's IR spectrum.

## Papers and what transfers

**Kanakala, Sridharan and Priyakumar, 2024: SMEN.** This peer-reviewed work learns
a shared space for IR spectra and 3D molecules through contrastive training,
then uses embedding similarity for ranking. A decoder extends the system to
structure generation. Its prominent approximately 81% Top-1 result is a library
ranking result; it must not be compared directly with our free generation
score. The reviewed publisher text also reports only 7/27 first matches in a
small experimental transfer exercise. The useful idea is an independently
learned spectrum-candidate compatibility score. We would need to establish its
benefit on our generated candidates and available coordinates. Publisher search
excerpts were accessible; direct full-text/PDF retrieval was blocked during
this review. [Paper](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00135d)

**Alberts, Laino and Vaucher, 2024: Leveraging infrared spectroscopy for
automated structure elucidation.** The authors pretrain on simulated spectra
and fine-tune on experimental spectra. Their principal model also receives the
molecular formula. Horizontal shifts and smoothing helped individually, while
adding noise and combining augmentations did not consistently help. They
also evaluate checkpoint ensembles. This supports controlled augmentation
experiments, not an assumption that stronger perturbations always help. Their
continuous spectral representation differs from our retained peaks, and
experimental transfer requires matching-domain data. The primary article text,
including Methods and augmentation experiments, was available through search.
[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11569215/)

**Alberts, Zipoli and Laino, 2025: Setting new benchmarks in AI-driven infrared
structure elucidation.** Sections 2.2-2.3 and Table 3 test equivalent SMILES
representations, spectral augmentation, pseudo-experimental spectra, and
formula-constrained generation. Representation augmentation improved Top-1 in
their setting but could reduce Top-5/Top-10. Their simulated-to-experimental
transfer function uses paired data; this is more than arbitrary noise. Known
formula constraints supply additional information that our IR-only evaluation
does not have. Full primary article reviewed. Adapting representation
augmentation to SELFIES is a new experiment, not a demonstrated result for our
model. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12203481/)

**Mick et al., July 2026: Data Fusion and Contrastive Alignment for Unconstrained
IR Molecular Structure Elucidation.** This preprint is directly relevant to
formula-free prediction. Section 3.1 combines autoregressive cross-entropy with
symmetric spectrum-structure InfoNCE using a separate structure encoder;
Section 5.2 evaluates the contrastive contribution. It also investigates
larger architecture changes. Appendix C.3 reports that excessive equivalent
SMILES augmentation can hurt and discusses false negatives when equivalent
molecules share a contrastive batch. Our hinge loss is not this objective.
Full preprint reviewed; results not reproduced. A small alignment branch is a
reasonable later experiment, with identity-aware positive/negative handling.
[Preprint](https://arxiv.org/html/2607.26164v1)

**Tan et al., August 2026: FIRMPO.** Sections 3.3-3.4 construct preferences from
beam-generated candidates using target identity, known formula agreement, and
Chemprop-IR forward-spectrum similarity. The objective compares updated and
reference model probabilities with chemistry-dependent weighting. This
supports investigating candidate mistakes produced by the model. It does not
validate our current fingerprint-selected hinge objective. Formula is part of
their input, and their feedback requires a forward IR model that we do not
have. Full preprint and Appendix B reviewed; exact numerical training settings
are insufficient there to copy a complete recipe. Gains vary substantially
by backbone/dataset. [Preprint](https://arxiv.org/html/2608.16082v1)

**Krishnadas et al., 2026: SQuIRL dataset descriptor.** The release supplies
raw harmonic frequencies and intensities for QM9 molecules, with spectra
computed using omegaB97X-D/aug-cc-pVTZ on QM9 geometries. It permits downstream
broadening/scaling choices. This confirms that our raw peak workflow differs
from experimental absorbance curves and from papers trained on broadened
spectra. Full Methods and dataset description reviewed. Any forward-spectrum
scoring or augmentation must match our representation and spectral domain.
[Dataset paper](https://www.nature.com/articles/s41597-026-07240-0)

Additional context: Zhang and Ha's peer-reviewed *Toward Complete Molecular
Structure Prediction from Infrared Spectroscopy Using Deep Learning* describes
a dual-loss IR-only model. The primary abstract was reviewed; full methods were
not verified, so no implementation parameters are taken from it.
[Abstract](https://pubmed.ncbi.nlm.nih.gov/41403343/)

## Changes proposed before finalizing the script

These are project recommendations inferred from the research and local code,
not settings established by those papers for this model.

1. **Establish a reproducible decoding baseline first.**
   `diffusion/spectrum.py` currently uses different length normalizations for
   pruning completed beams and ordering the final list. Define one documented
   score, including BOS/EOS and length convention, and use it consistently for
   candidate scoring. Make a controlled validation comparison against legacy
   decoding before attributing any gain to fine-tuning. Expose decoding
   settings in reports and retain legacy results.

2. **Separate search width from reported Top-K.**
   `evaluate_spectrum.py` currently evaluates all retained valid candidates as
   Top-K; increasing beam width therefore changes the metric. Always report
   Top-1 and Top-5, even with search width 10 or 20. Also report Top-10 where
   applicable, validity, and number of distinct identities. Canonicalize and
   deduplicate equivalent structures before counting candidates.

3. **Select weights using generated molecular accuracy.**
   `diffusion/training.py` selects and stops on `val_loss`, while validation in
   `diffusion/spectrum.py` measures positive token loss. Retain that diagnostic,
   but periodically decode a fixed validation subset and record molecular
   Top-1/Top-5. Preserve candidate checkpoints for molecular-accuracy-based
   selection. Loss improvements alone are insufficient to select the winner.

4. **Compare negative sources under matched conditions.**
   Start with three training arms: ordinary low-rate fine-tuning; the existing
   static-negative hinge recipe; and the same objective using model-generated
   hard negatives. Keep initialization, positive examples, optimizer-step
   budget, and decoding identical. This distinguishes extra training from
   ranking benefits and isolates the effect of negative selection.

5. **Mine only from training inputs and labels.**
   Decode a fixed training subset with the frozen baseline, retain valid
   incorrect unique structures, and prioritize high-scoring confusions. Use
   the known training target as the preferred molecule. Train-only
   same-formula/similarity negatives can supplement rows without useful
   generated mistakes. Validate SELFIES conversion, vocabulary, length, and
   molecular identity; never silently truncate targets. Store checkpoint,
   dataset, split, and generation-setting hashes. A generated candidate need
   not be an existing dataset row, so the current negative-row file format
   needs extension. Do not consult validation/test labels to construct pairs.

6. **Make progress and experiment results visible.**
   Record mining progress and candidate counts, training loss and margin
   success, periodic validation Top-1/Top-5, throughput, and elapsed time.
   Print measured ETA after sufficient progress, with queue time separate.
   Save new experiments to unique output directories and preserve baseline
   weights. Checkpoint, resume, and provenance checks remain required.

The existing `2e-5` learning rate, 15-epoch maximum, and ranking weight/margin
`0.1` are provisional local choices. They are not a paper-reproduced recipe.
A small pilot should establish throughput and stability before fixing the
full budget. Avoid combining every proposed change in one run.

## What to test afterward

| Experiment | Requirement and decision |
| --- | --- |
| Equivalent molecular representations | Generate alternate SMILES only from training molecules, convert to SELFIES, verify the same identity and supported tokens/length. Preserve canonical examples. Evaluate Top-5 as well as Top-1. |
| Modest spectral augmentation | Test independently on peak tokens; preserve masks, valid frequencies, and intensity conventions. Broadened-curve smoothing cannot simply be copied onto a list of peaks. |
| Contrastive compatibility branch | Add an independent molecule representation and identity-aware batch handling; compare against the simpler ranking recipe before adopting it. |
| Forward IR feedback | Validate a forward model on matching-domain spectra and splits first. Predicted spectral agreement is a surrogate, not proof of correct identity. |
| Larger architecture or SMILES decoder | Treat as a new model experiment with separate cost and compatibility, rather than a small fine-tune of the existing checkpoint. |

## Evidence required before choosing the new model

- Use the same 500 validation rows for inexpensive paired screening. Report
  gained and lost molecules, not just aggregate percentages.
- Confirm promising recipes on all 12,478 validation rows; repeat promising
  runs with additional seeds. Estimate uncertainty on the paired improvement,
  grouping repeated identities where appropriate.
- Count failures in the denominator. Separate wrong formula, wrong
  connectivity, alternate tautomer, and missing-candidate errors. Keep exact
  identity primary; a relaxed tautomer-family score is additional context.
- Tune decoding and training only on validation. Use a final test evaluation
  after decisions are fixed. Previously inspected test example 24778 is not
  an unbiased final demonstration.
- The existing split can share tautomer families. A new-family/scaffold claim
  needs a new grouped benchmark and training that respects it from the start;
  merely moving previously seen molecules to a new test split is insufficient.
- Keep IR-only results separate from a future mode supplied with independently
  known formula. Do not inject reference formulas or reference 3D coordinates
  into inference-time ranking.
- Evaluate end-to-end graph-plus-geometry separately. Our known-graph geometry
  results and MMFF validity are not IR-only identification accuracy. The
  recorded MMFF run worsened close-reference geometry, so increasing
  refinement is not the first accuracy experiment.

Recommended next implementation scope: consistent and versioned candidate
scoring, fixed Top-K reporting, molecular-accuracy checkpoint evaluation, and
training-only generated-negative mining with matched fine-tuning controls.
The script should be finalized only after these experiment choices are agreed.
