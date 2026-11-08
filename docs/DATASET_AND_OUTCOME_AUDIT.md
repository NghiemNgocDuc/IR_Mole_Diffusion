# What is actually in our IR dataset, and what the results support

Audited November 4, 2026. This review inspected all 133,795 local source records
and all 124,781 prepared records. It includes full chemical-family enumeration,
spectral statistics, source reconstruction, and comparison with the recorded
Unity outcomes. No training job was launched and no fine-tuning recipe was
changed.

The main finding is that **the existing 500-molecule evaluation is strongly
unrepresentative of the full validation set**. Establishing a representative
baseline and examining its actual mistakes should precede choosing the
fine-tuning objective. The prepared data otherwise passes the numerical and
source-correspondence checks performed here.

## 1. The local data and its provenance

| Item | What the files contain |
| --- | --- |
| Source archive | `data/SQuIRL/squirl_ir_3d.npz`: 133,795 records |
| Prepared archive | `data/processed/squirl.npz`: 124,781 records |
| Training / validation / test | 99,825 / 12,478 / 12,478 |
| Molecular identity | One unique canonical non-isomeric SMILES per prepared row |
| Elements | H, C, N, O, F; no formally charged atoms in accepted graphs |
| Size | At most 9 heavy atoms and 29 atoms including hydrogens |
| Size concentration | 103,729 molecules, or 83.1%, have 9 heavy atoms |
| IR | Retained frequency/intensity pairs, explicit padding mask |
| Structure | One atom-aligned coordinate array and bond-order matrix per row |
| Decoder label | SELFIES tokens; 22-token vocabulary including 3 special tokens |
| Target length | At most 21 molecular tokens, plus BOS/EOS |

The principal prepared arrays are `coords` [124781,29,3], `atom_types`
[124781,29], `bond_types` [124781,29,29], `spectra_packed` [124781,81,2],
`spectra_mask` [124781,81], and `selfies_tokens` [124781,23]. `source_rows`
preserves the row mapping back to the local archive. Zero padding is explicit.

The published SQuIRL description reports 133,885 records, distributed as HDF5,
with raw harmonic frequencies/intensities and quality-warning metadata. It
uses QM9 geometries and a specified DFT frequency calculation. Our local NPZ
has 90 fewer records and lacks those original identifiers, warning flags,
method metadata, and the upstream export/filter recipe. Therefore the local
conversion is verified against the supplied NPZ; its equivalence to the
complete published release is not established. This is a provenance gap to
resolve before replacing data or claiming exact reproduction of the release.
[Published dataset description](https://www.nature.com/articles/s41597-026-07240-0)

The local archive is not a collection of experimentally measured continuous
absorbance curves. It also does not provide a conformational ensemble for each
identity, repeated measurements, or spectral uncertainty labels.

## 2. Filtering and source correspondence

The converter accepted 93.3% of source rows. Its saved first-failure reasons are:

| Reason | Rows |
| --- | ---: |
| Coordinate connectivity did not match source SMILES under the converter's check | 7,153 |
| Formally charged atoms | 1,844 |
| Invalid peak values | 17 |
| Total rejected | 9,014 |

The connectivity category is a failed validation criterion, not proof that
every rejected molecule is intrinsically bad. Distance-based bond perception
and source annotations need examination before deciding whether any can be
recovered. Do not simply relax the filter to increase training size.

Across **every accepted row**, reapplying peak packing and coordinate centering
gave **zero mismatches** in peak values, masks, atom counts, atom labels, and
centered coordinates. Accepted and rejected source-row sets are disjoint and
account for the complete local source archive.

There are 22 source rows with negative frequencies. All are excluded: 17 fail
the peak check, 4 fail the charge check earlier, and 1 fails connectivity
earlier. No nonfinite frequencies/intensities or negative intensities were
found in the source. Reasons are sequential, so a rejected row can have more
than one problem.

Prepared coordinates are finite, zero-padded, and centered to within
8.7e-7 angstrom. No reference row has two atoms within 0.2 angstrom. These are
numerical checks, not quantum stability certification.

## 3. What information reaches the model

There are 5,586,460 active prepared peaks. Per molecule, the retained peak count
ranges from **3 to 71**, with median **45**. Thus **the 81-peak capacity truncates
none of this dataset**. Increasing that limit cannot recover information already
absent from the supplied archive.

Recovered frequencies span approximately **21.4 to 3917.5 cm^-1**. Stored
intensities span **0.5 to 1757.94** in the supplied training units. The converter
divides frequencies by 4000 and intensities by 1000; it does not normalize each
molecule to the same maximum intensity.

No raw intensity is below 0.5. Only 18,245 prepared rows have exactly 3N-6
retained peaks. The local lists therefore should not be treated as a complete
normal-mode catalog. A preceding intensity filter is strongly suggested by the
lower bound, but its exact rule is not documented locally. The 3N-6 comparison
is descriptive: linear molecules have a different mode count.

Peak count carries substantial molecular-size information in this dataset:
its correlation with total atom count is **0.916**. A simple lookup derived
only from training rows, predicting the most common atom count for each peak
count, gets **55.7%** exact atom counts on validation, versus **13.4%** for a
constant majority guess. This is a diagnostic of the inputs, **not an accuracy
result for our neural model**, and does not establish that the model uses this
cue. It motivates testing sensitivity to weak-peak removal and experimental
peak extraction before assuming real-spectrum transfer.

All 124,781 exact packed inputs are unique. No validation or test packed input
is duplicated in training. Near-duplicate spectra were not exhaustively
compared; this result concerns exact arrays only.

Every held-out token occurs in training. There is no observed out-of-vocabulary
problem to fix by expanding the vocabulary. Some tokens are rare, however:
`[#Branch2]` occurs only 164 times in training. Representation augmentation
would need to check vocabulary and sequence-length compatibility explicitly.

## 4. Chemical coverage and the evaluation-subset problem

The prepared data includes carbonyl motifs in 37.3% of molecules, O-H in 32.7%,
amide motifs in 10.9%, nitriles in 12.7%, and aromatic atoms in 18.4%.
These are overlapping substructure counts with explicit SMARTS definitions,
not mutually exclusive molecule classes. Only 1.68% contain fluorine.

The charge restriction also removes all 141 source molecules matching the
charge-separated nitro pattern. The supported domain therefore does not cover
every chemistry involving the five allowed elements. Larger molecules,
additional elements, and excluded charge patterns require data/schema/model
work rather than an expectation that a small fine-tune will cover them.

The split memberships are randomized by identity, but their row lists are
stored in ascending source order. Both evaluators take the first `limit` rows.
The evaluated 500 validation molecules all come from original source rows
23-5499, out of 133,795 source records.

| Property | Evaluated 500 | All 12,478 validation rows |
| --- | ---: | ---: |
| Aromatic molecules | 1.0% (5) | 17.8% (2,221) |
| Acyclic molecules | 43.8% (219) | 10.8% (1,353) |
| Contains nitrogen | 44.0% | 61.2% |
| Contains fluorine | 0% | 1.67% |
| Contains O-H | 58.2% | 32.7% |
| Mean ring count | 0.71 | 1.63 |
| Mean rotatable bonds | 2.31 | 0.95 |
| Mean total atoms | 20.57 | 18.08 |
| Mean retained IR peaks | 52.16 | 44.94 |

![Measured validation subset composition](../outputs/dataset_review/validation_subset_comparison.png)

Consequently, **70.6% Top-1 and 92.4% Top-5 remain measured results for those
500 molecules**, but they are not established full-validation accuracies.
The direction or size of the accuracy bias cannot be calculated from input
composition alone. More flexible molecules might also affect the 3D benchmark
differently from rigid aromatic molecules. We must measure those effects.

## 5. What “unseen” means for this split

Exact non-isomeric identities are disjoint across train, validation, and test.
That does not make the split disjoint by related chemistry:

| Held-out grouping also present in training | Validation | Test |
| --- | ---: | ---: |
| Exact molecular identity | 0% | 0% |
| Molecular formula | 99.89% | 99.97% |
| RDKit canonical tautomer family | 7.68% (958 rows) | 7.30% (911 rows) |
| Exact Murcko scaffold, among cyclic molecules only | 90.15% | 89.54% |
| Tautomer-normalized Murcko scaffold, cyclic only | 90.53% | 90.12% |

Empty acyclic scaffolds are excluded from scaffold-overlap denominators. All
tautomer enumerations completed within RDKit's default caps of 1000 tautomers
and 1000 transforms; the grouping is still software-definition dependent.

The original 500-row validation subset contains 281 cyclic molecules, and
**all 281 have an exact scaffold represented in training**. Formula/scaffold
overlap is not itself accidental label leakage. It establishes that the
current benchmark mostly tests new identities within familiar chemistry.

The inspected challenge, row 24778, has an exact held-out target but an alternate
tautomer in training. Its correct answer at candidate rank 3 is a genuine
ranking failure under the strict identity metric. It is not evidence of
unseen-family performance. Keep strict identity as the main metric and add
tautomer-family diagnostics; do not relabel wrong tautomers as exact successes.

## 6. What the available outcomes do and do not tell us

Stage 1's existing 500 cases decompose into:

- 353 correct first choices.
- 109 additional correct identities later among the retained candidates.
- 38 cases without the correct identity in that list.

Of the 147 first-choice errors, 112 have the correct formula and 35 have an
incorrect formula. These are a separate partition: without per-row results we
cannot cross-tabulate formula errors with missing-candidate errors. Better
ranking is a plausible priority because 109/147 errors have an answer already
available, but this does not promise 92.4% first-choice performance.

The supplied known-graph 3D evaluation uses the correct graph and atom counts.
Its fields named `graph_match` also compare geometry-assigned stereochemistry.
They should not be described as pure connectivity accuracy. Source SMILES have
no `@`, `/`, or `\\` stereochemical markers, and the stage-1 vocabulary has no
stereo tokens. Stage-1 achiral identity and stage-2 stereo-aware scores measure
different things.

MMFF refinement raised sanitized/connected validity from 96.96% to 100%, while
strict identity/stereo plus symmetry-aware RMSD <=0.5 angstrom fell from 31.4%
to 29.8% for the first sample, and 69.2% to 66.2% for the reference-selected
best of five. More refinement is not supported as a way to improve the measured
close-reference geometry. Best-of-five uses the known answer and is not an
inference-time selection method.

We have aggregate Unity reports and one complete candidate report locally.
We do **not** yet have the full stage-1 per-row validation report or the full
predicted coordinate archives. Thus statements such as “aromatic molecules
are the main errors” or “the model mostly confuses tautomers” are hypotheses,
not findings from this audit.

## 7. How to use this information before fine-tuning

1. **Measure a representative baseline.** Use full validation, or a frozen
   seeded sample stratified across size, rings/aromaticity, N/O/F presence and
   peak count for a cheap screen. Preserve the old 500-row result for paired
   historical comparison. Report class counts and fixed Top-1/Top-5 separately
   from search width. Do not change the existing training/test membership.
2. **Join actual errors to the audited molecules.** Retrieve the existing
   `evaluation/spectrum_val.json` from Unity. It already stores row IDs,
   reference identities, candidates, and hit flags. Join these to
   `chemical_rows.csv` to distinguish wrong formula, tautomer, ring/topology,
   and absent-candidate errors. New full-validation output is needed to
   diagnose underrepresented classes reliably.
3. **Resolve decoding before attributing gains to learning.** The current
   pruning and final ordering use different length-normalization conventions.
   Compare a consistent documented score against the frozen old baseline.
   Score unique identities with fixed reported Top-K.
4. **Then compare matched fine-tuning experiments.** Ordinary low-rate CE is
   the control. Compare the draft same-formula negative loss and negatives
   generated from the model's actual training-set mistakes, with the same
   starting checkpoint and update budget. Almost every training molecule has
   same-formula alternatives, but formula agreement alone does not make a
   negative spectrally difficult. Mine only from training inputs/labels.
5. **Select by generated molecular accuracy.** Keep token validation loss as
   a diagnostic, but select promising runs using actual decoded identity
   results. Inspect gains and regressions by chemical class and by training
   tautomer/scaffold overlap. Preserve first-choice and candidate-coverage
   scores so a ranking gain cannot hide reduced coverage.
6. **Test spectral robustness separately.** First resolve the local peak
   selection provenance. Then compare modest intensity/frequency perturbations
   and weak-peak dropout individually. A fixed 81-peak limit, more padding, or
   arbitrary noise is not currently supported as the main bottleneck.
7. **Evaluate the 3D stage independently and end to end.** Compare to graph-only
   conformer generation and shuffled/removed IR controls with equal sample
   budgets. Report connectivity and stereo-aware geometry separately. Also
   test the complete pipeline with predicted graphs and a real inference-time
   selector. A geometry PASS label does not verify spectral identity.

A future novel-family benchmark needs family-aware splitting and training
that respects it from the start. Repartitioning already-trained examples into
a new “test” set cannot establish unseen-family performance. Experimental IR
transfer likewise needs matched experimental evidence. These are separate
objectives from improving the current first-choice simulated-spectrum result.

The paper review and architectural alternatives remain in
[ACCURACY_LITERATURE_REVIEW.md](ACCURACY_LITERATURE_REVIEW.md). This audit changes
the immediate priority: establish the evaluation coverage and real error
distribution before finalizing ranking or augmentation choices.

## 8. Reproducibility and saved evidence

Data SHA-256:
`49c832a638c46a8420bd8a0187d18dfb757e9b9289cf4b4d9db0a3b3fbac71dd`

Split SHA-256:
`9e46e1949dfcd88181a64691ba407bc4ab8ea5fa98e0b1f741b7bb69242c8d62`

Run from the repository root with the project Python environment:

```text
python scripts/audit_source_preprocessing.py
python scripts/audit_spectral_geometry_data.py
python scripts/audit_chemical_splits.py
python scripts/audit_functional_coverage.py
python scripts/render_dataset_audit.py
```

Only `audit_source_preprocessing.py` and `audit_functional_coverage.py` read the
explicitly trusted legacy object archive with pickle enabled. Other audits
load numerical/string arrays without pickle. Scripts write new analysis files
under `outputs/dataset_review`; they do not modify data, split memberships,
model checkpoints, or training configuration.

Saved outputs include source reconstruction results, spectral/geometry
statistics, chemical split statistics, explicit motif definitions/counts,
`chemical_rows.csv` with all 124,781 row-level descriptors, and two figures.
The row CSV supports later joins to Unity evaluation results. The scan of
held-out labels here measures benchmark composition; it supplies no new
training targets or mined negatives.
