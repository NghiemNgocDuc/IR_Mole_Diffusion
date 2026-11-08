# Training time and accuracy assessment — updated October 29, 2026

## What is measured

The full supported configurations have 7,396,118 parameters in the spectrum
model and 7,937,913 in the coordinate model. Tests and short CPU fits establish
functionality. The user subsequently reported Unity stage 1 job 66229326
completed with exit code `0:0` in **03:01:40**. The GPU type and final epoch are
not yet known, so this observed elapsed time is not a per-epoch throughput
benchmark or a forecast for stage 2.

Stage 2 job 66243865 subsequently completed all 200 epochs with exit code
`0:0` in **16:12:02**. Combined allocated training-job elapsed time was
**19:13:42**, excluding queue waiting and evaluation. This is evidence for the
user's two runs, not a hardware-independent training-time guarantee.

Unity evaluation job 66243541 reported **70.6% top-1**, **92.4% top-5**,
**93.0% top-1 formula accuracy**, and **0/500 rows without a valid candidate**
on 500 validation molecules. These preliminary results use IR only, beam size
5, and canonical non-isomeric graph identity. They do not assess
stereochemistry, experimental IR, full test-set accuracy, or 3D prediction.
The checkpoint and complete report remain on Unity; see
[recorded evidence](VALIDATION.md) and the
[transcribed summary](results/stage1_validation_66243541_summary.json).

The November 4 [dataset audit](DATASET_AND_OUTCOME_AUDIT.md) found substantial
source-order bias in those first 500 validation rows: 1.0% aromatic versus
17.8% across full validation, and 43.8% acyclic versus 10.8%. The reported
accuracies remain measurements of this subset; full-validation accuracy and
the direction of the subset's accuracy bias have not been measured.

Raw known-graph 3D evaluation on the same size validation subset reported
96.96% sanitized connected candidates and 0.836-Angstrom mean fixed-order
heavy-atom RMSD. Strict perceived identity/stereo plus symmetry-aware RMSD
<= 0.5 Angstrom succeeded on 31.4% of first samples and 69.2% of reference-
selected best-of-five samples. The corresponding best-of-five rate at
1.0 Angstrom was 87.8%. See the
[reported stage 2 metrics](results/stage2_validation_66280988_summary.json).
This run supplied the correct graph and atom counts; it is not end-to-end
IR-to-3D accuracy. Best-of-five is oracle selection and the stereo-aware gate
differs from stage 1's achiral graph metric. A graph-only baseline, spectral
ablations, and a matched test benchmark are still needed.

MMFF comparison job 66284311 subsequently converged for all 2,500 candidates,
increasing sanitized connected validity from 96.96% to 100%. Strict identity/
stereo plus symmetry-aware RMSD <= 0.5 Angstrom decreased from 31.4% to 29.8%
for the first sample and 69.2% to 66.2% for reference-selected best-of-five.
Best-of-five success at 1.0 Angstrom decreased from 87.8% to 85.6%. The raw
metrics match the previous run exactly. These results support keeping both
outputs: force-field cleanup improved validity but not close-reference
geometry on this subset. See the
[reported MMFF comparison](results/stage2_validation_66284311_refined_summary.json).
These percentages still do not measure complete IR-to-3D prediction accuracy.

## Exact training workload

For the prepared 99,825 training and 12,478 validation molecules:

| Stage | Train batch | Train batches/epoch | Validation batches/epoch | Optimizer updates/epoch | Maximum epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spectrum | 64 | 1,560 | 195 | 1,560 | 100 |
| Coordinates | 32, accumulation 2 | 3,120 | 390 | 1,560 | 200 |

The 1000-step diffusion schedule samples one timestep per training example. It
does not require 1000 denoiser calls per training batch. Sampling with 200 steps
does require 200 sequential denoiser passes per candidate.

For each stage, calculate hours as
`epochs * (training_batches * seconds_per_training_batch + validation_batches * seconds_per_validation_batch) / 3600`.

These are **assumed-throughput scenarios**, not measurements or GPU guarantees:

| Scenario | Spectrum train/val batch | Coordinate train/val batch | Spectrum maximum | Coordinate maximum | Combined, with 15% allowance |
| --- | --- | --- | ---: | ---: | ---: |
| Faster | 40/20 ms | 80/40 ms | 1.84 h | 14.73 h | 19.06 h |
| Middle | 100/40 ms | 150/70 ms | 4.55 h | 27.52 h | 36.88 h |
| Slower | 200/80 ms | 300/140 ms | 9.10 h | 55.03 h | 73.75 h |

For an initial compute reservation, roughly 1–3 GPU-days is a provisional budget
for a modern GPU if its measured throughput lies in this range. Older hardware
or slower execution can exceed it. Queueing, evaluation, candidate sampling,
physics fine-tuning, and hyperparameter searches are additional. Early stopping
can reduce training time. This range is not derived by scaling CPU timing.

[Unity's current GPU inventory](https://docs.unity.rc.umass.edu/documentation/cluster_specs/gpu_summary/)
lists A100/H100/L40S devices under `gpu-highvram`; `gpu` includes older cards.
The supplied jobs default to `gpu`. Override it when requesting modern devices:

```bash
sbatch --partition=gpu-highvram train_spectrum.sh
# After stage 1 is trained and evaluated:
sbatch --partition=gpu-highvram train_diffusion.sh
```

## Replace assumptions with an allocated-GPU measurement

After activating the environment, run:

```bash
srun --partition=gpu-highvram --gpus=1 --cpus-per-task=8 --mem=64G \
  --time=00:20:00 python scripts/benchmark_training.py
```

The benchmark warms up, measures forward/backward/AdamW, gradient accumulation,
EMA, transfers, and validation, and writes `runs/benchmark.json`. It uses default
batch sizes and an unfrozen coordinate encoder, representative of most training
epochs. Its manual loop excludes trainer logging/checkpoint I/O, covered only
by an explicit configurable 15% allowance. It saves no weights. BF16 is an
optional separate benchmark on supported hardware; FP32 matches the default
recipe. The script's CPU path was exercised with small batches; CUDA throughput
must be measured on Unity.

## Relevant published accuracy reference points

Top-1 means the first candidate has the correct molecular graph; Top-5 means
the correct graph occurs among five candidates. Formula accuracy, token
accuracy, validity, and 3D RMSD are separate metrics.

| Reference | Data/input | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| Our new pipeline, preliminary | Filtered SQuIRL; 500 validation rows; IR only | 70.60% | 92.40% |
| Mick et al., 2026, dense comparison | Simulated QM9S; IR only | 52.15% | 69.53% |
| Mick et al., 2026, MoE-LOSN | Simulated QM9S; IR only | 63.28% | 86.13% |
| Mick et al., 2026, MoE-LOSN | Experimental NIST after fine-tuning; IR only | 31.83% | 50.13% |
| Alberts et al., 2025 | Experimental NIST, 6–13 heavy atoms; IR + known formula | 63.25% | 79.15% |

Sources: [Mick et al., Tables 1 and 3](https://arxiv.org/html/2607.26164v1),
[Alberts et al., Tables 4 and 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC12203481/).
The 2026 work is a preprint with an [official code release](https://github.com/ethanmick8/ir2structure);
its reported experiments have not been reproduced here.

QM9S shares the H/C/N/O/F, up-to-nine-heavy-atom domain with the prepared data,
but spectra, filtering, splits, tokenization, and identity metrics are not
verified identical. These figures cannot be transferred to our setup. A matched
comparison requires training/evaluating both methods on the same data and split.
Our 500-row validation result is not evidence of outperforming these papers:
it uses a different evaluation population and has not been confirmed on the
full held-out test set.

The 2026 comparison uses about 33 million parameters plus contrastive alignment,
encoder fusion, and sequence augmentation; the strongest variant adds MoE.
Our 7.4-million-parameter graph model is a simpler baseline. The measured
validation subset above is a starting result for this recipe. The approximately
50–63% Top-1 range on the related simulated benchmark remains an external
reference with different evaluation conditions. Experimental performance also
requires matching-domain experimental fine-tuning.

## Physical checks and 3D success

The default coordinate-training physics coefficient is zero for the baseline;
bond/clash guidance and MMFF are available during prediction, and optional
physics fine-tuning is documented. These do not calculate IR frequencies.

[FIRMPO, 2026](https://arxiv.org/html/2608.16082v1) studies formula and forward-IR
consistency feedback. That mechanism is not implemented by a bond/clash loss.
It is a relevant future direction once baseline identity and geometry quality
are established.

There is no verified single percentage for the complete IR-to-3D pipeline.
Report graph Top-1/Top-5 first, then symmetry-aware same-graph RMSD and chemical
validity, with all failures retained. Compare coordinate quality against the
provided ETKDG/MMFF baseline and removed/shuffled IR. End-to-end correct-graph
and correct-geometry success cannot exceed correct-graph recovery for the same
candidate-selection rule.
