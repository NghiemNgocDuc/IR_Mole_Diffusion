# Research and implementation decisions

Reviewed before implementation on October 27, 2026. This is a new baseline that
borrows established principles; it is not a reproduction of the cited results.
No third-party model implementation was vendored. The optional viewer added on
October 29 vendors the licensed 3Dmol.js browser renderer, described below.

| Primary source | What was inspected | Decision in this repository |
| --- | --- | --- |
| [Hoogeboom et al., EDM, ICML 2022](https://proceedings.mlr.press/v162/hoogeboom22a.html), [paper](https://proceedings.mlr.press/v162/hoogeboom22a/hoogeboom22a.pdf) | Equivariant noise prediction and zero-center-of-gravity coordinate subspace | Center coordinates, Gaussian targets, reverse noise, and outputs with the same atom mask. |
| [EDM official implementation](https://github.com/ehoogeboom/e3_diffusion_for_molecules), especially [diffusion](https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py) and [utilities](https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/utils.py) | Consistent marginal schedules, finite noise precision, masked Gaussian sampling, and EMA | Derive all buffers from one beta schedule; use a finite terminal SNR; preserve EMA in checkpoints. Our loss averages masked coordinate MSE per molecule. |
| [Xu et al., GeoDiff, ICLR 2022](https://arxiv.org/abs/2203.02923), [official code](https://github.com/MinkaiXu/GeoDiff), [network/sampling](https://github.com/MinkaiXu/GeoDiff/blob/main/models/epsnet/dualenc.py), [trainer](https://github.com/MinkaiXu/GeoDiff/blob/main/train.py) | Graph-conditioned conformation generation, local/global geometric relations, gradient clipping, and separate validation | Keep atom-aligned graph conditioning explicit; retain covalent edges; clip gradient norms; separate IR-only identification from known-graph conformation evaluation. We do not adopt its older PyTorch Geometric dependency stack. |
| [Alberts et al., Leveraging infrared spectroscopy for automated structure elucidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11569215/), [official IR code repository](https://github.com/rxn4chemistry/rxn-ir-to-structure) | Autoregressive spectrum-to-structure translation; the paper also supplies molecular formula as an input | Use a separate autoregressive graph stage. Our IR-only task omits a known formula, so its difficulty and expected accuracy differ from formula-assisted results. |
| [SELFIES primary paper](https://arxiv.org/abs/1905.13741), [official library and API](https://github.com/the-matter-lab/selfies), [library paper](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00044c) | Explicit tokenization, graph representation, encode/decode round trips | Build a new embedded vocabulary with PAD/BOS/EOS; use teacher forcing and masked decoding; validate resulting graphs with RDKit. Representation validity does not establish energetic stability or an IR match. |

## Practical design

1. Convert the supplied trusted object NPZ once. Match the source molecular
   graph to coordinate connectivity by element-preserving graph isomorphism
   before transferring bond orders. Preserve source row IDs and quarantine
   failures in a report. Do not assume SMILES and XYZ use the same atom order.
2. Group canonical non-isomeric molecular identities into one shared 80/10/10
   train/validation/test split. Train both stages on the same training groups.
   Group enantiomers together because ordinary achiral IR is generally unable
   to select absolute handedness.
3. Train a masked peak-token encoder and an autoregressive SELFIES decoder.
4. Initialize the diffusion spectrum encoder from that new checkpoint. Train
   centered coordinate noise prediction, conditioned on the spectrum and graph.
   Ground-truth graphs are used during coordinate training; prediction uses
   graphs decoded from IR. These are distinct evaluation tasks.
5. Establish the pure denoising baseline first. Then compare explicit bond/clash
   loss fine-tuning and sampling guidance on validation data. Higher-order terms
   require verified targets; no arbitrary angle or torsion targets are invented.
6. Use held-out identity/formula accuracy, same-graph symmetry-aware RMSD,
   chemical reconstruction validity, and raw-versus-refined outputs. Keep failed
   candidates in metric denominators. Compare against shuffled/removed IR and
   an RDKit conformer baseline before claiming the spectrum improves geometry.

## Physical interpretation and limits

Bond/clash penalties and MMFF relaxation screen molecular geometry. They do not
calculate normal modes or dipole derivatives and therefore do not prove that a
candidate reproduces an observed IR spectrum. Strong spectral consistency needs
a validated forward spectrum predictor or a quantum frequency calculation at a
compatible theory level, phase, scaling, and broadening convention. That is a
separate validation step, not a guarantee provided by this training code.

The supplied legacy encoder's missing modules and unknown preprocessing are not
reconstructed by guessing. New models embed their complete encoder specification,
vocabulary, preprocessing, split/data fingerprints, optimizer state, and EMA.
Legacy checkpoints remain unchanged and require their original source revision
for faithful reproduction.

## October 29: interactive viewing and graph ranking experiment

The November 4 viewer follow-up adds an explicit 2D-graph-to-3D mode using
RDKit ETKDGv3 embedding with explicit hydrogens, a fixed random seed, and
optional MMFF94 refinement, following the [official RDKit 3D guide](https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules).
This mode is separate from saved-prediction display and is labeled as generated
conformers, not IR-model outputs or experimentally established geometry.

The browser viewer uses [3Dmol.js](https://3dmol.org/doc/GLViewer.html)
([primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4393526/)) to rotate,
zoom, restyle, and measure the actual saved conformers. Version 2.5.5 is embedded
offline with its BSD license and bundled dependency notices; the npm tarball
integrity and individual asset hashes are recorded in
`viewer_assets/3dmol_provenance.json`. No reference coordinates, new embeddings,
or forward IR calculations enter this display.

For graph selection, the motivating failure was a correct reference graph at
beam rank 3. That test example is illustrative only and is excluded from the
fine-tuning objective. The optional experiment retains token cross entropy
and adds a small margin between teacher-forced sequence scores for a positive
and a training-only negative graph. Same-formula negatives are preferred;
same-heavy-atom-count and other distinct training identities provide fallbacks.
Within a seeded capped pool, Morgan similarity chooses the negative. This is a
sequence-ranking adaptation, not a second molecular embedding encoder.

[SMEN's primary paper](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00135d)
and its [official CombinedLoss implementation](https://github.com/devalab/Spectra2Structure/blob/master/train_utils.py)
combine spectrum/molecule alignment and molecular reconstruction. We inspected
both. Our margin objective uses the existing architecture and is inspired by
that alignment principle; it does not reproduce SMEN or transfer its reported
library-retrieval accuracy to unconstrained molecular generation.

Fine-tuning initializes from the baseline EMA inference weights with a fresh
optimizer at learning rate 2e-5. Token cross entropy remains the validation-loss
checkpoint criterion. Matched validation identification reports are therefore
required before selecting the candidate checkpoint; no automatic promotion is
implemented. Test labels and this example's identity are not training targets.
The existing split groups exact non-isomeric identities, not all tautomer
families. A future novel-family benchmark would need a new split and retraining
of both stages. Improvement is an empirical question, not established by local
software checks or the research citation.
