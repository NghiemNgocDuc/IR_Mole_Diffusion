# Review of physical validation 1849289

Local review: 2026-11-17. Selected training examples; no generalization estimate.
All twelve DFT jobs completed and retained the inferred achiral graph. Archived output and final-coordinate hashes were verified.

## Original geometry versus DFT repair

The unchanged-coordinate calculations and subsequent optimized structures answer separate questions. No repaired result is a validity label for the original geometry.

| Input | Arm | Stage | Original DFT force RMS (kJ/mol/nm) | DFT repair energy drop (kJ/mol) | Repair RMSD (A) | Repaired minimum supported |
|---|---|---|---:|---:|---:|---|
| 1 | unguided | raw | 486.150 | 32.039 | 0.2641 | False |
| 1 | unguided | relaxed | 1157.091 | 51.509 | 0.1158 | False |
| 1 | opls_guided | raw | 1122.907 | 51.068 | 0.2595 | False |
| 1 | opls_guided | relaxed | 1157.030 | 51.517 | 0.1159 | False |
| 7 | unguided | raw | 195.025 | 3.879 | 0.0638 | True |
| 7 | unguided | relaxed | 433.808 | 23.604 | 0.1183 | True |
| 7 | opls_guided | raw | 263.872 | 4.510 | 0.0639 | True |
| 7 | opls_guided | relaxed | 428.315 | 24.860 | 0.1274 | True |
| 13 | unguided | raw | 600.693 | 21.905 | 0.1725 | True |
| 13 | unguided | relaxed | 468.034 | 40.554 | 0.1966 | True |
| 13 | opls_guided | raw | 399.781 | 13.491 | 0.1700 | True |
| 13 | opls_guided | relaxed | 445.387 | 45.028 | 0.2041 | True |

## Frequency and IR checks

Input 1 has one imaginary mode in each optimized structure: -13.17, -68.18, -41.49, and -68.52 cm^-1. The current checks do not support a local minimum for those structures. Inputs 7 and 13 have no imaginary modes at reported precision in all four starts; this supports local minima under the chosen DFT model, not global minima or experimental identity.

IR cosine is about 0.143, 0.461, and 0.443 for inputs 1, 7, and 13 respectively. These are descriptive scores with fixed 20 cm^-1 Gaussian FWHM, no frequency scaling, and a 400-4000 cm^-1 grid. They are not probabilities or calibrated failure labels. A dataset-reference geometry calculation at the same settings remains needed to separate method and geometry effects.

## Engine comparison is stage-specific

| Input | Stage | Energy difference at GROMACS coordinates (kJ/mol) | Maximum force difference (kJ/mol/nm) | Passed |
|---|---|---:|---:|---|
| 1 | raw | 4.9303e-05 | 0.0492591 | True |
| 1 | relaxed | 1.77149e-05 | 0.0379963 | True |
| 7 | raw | 0.000144436 | 0.0602125 | True |
| 7 | relaxed | 0.688276 | 5.94878e+07 | False |
| 13 | raw | 0.000117001 | 0.0326969 | True |
| 13 | relaxed | 0.994608 | 3.11718e+07 | False |

All raw cases passed; only relaxed inputs 7 and 13 failed. Their original relaxed-coordinate torsion force RMS was approximately 3.52e5 and 1.226e7 kJ/mol/nm. The smallest reported torsion plane sines were approximately 2.56e-8 and 1.56e-11. This implicates near-collinear nonzero torsions and numerical conditioning. It does not prove that precision is the only cause, or that force-field parameters can safely be deleted.

OPLS relaxation raised DFT energy in all six matched pairs in this pilot. This is a distinct finding from engine parity: engine agreement alone cannot establish DFT accuracy.

## Pipeline decision

Raw unguided coordinates are the default primary output. Paired OPLS guidance/relaxation requires explicit experimental opt-in. Preserve each raw file and record generation, assessment and repaired-geometry outcomes separately. No automatic physics rejection, parameter alteration, or guidance promotion is authorized by these diagnostics.

## Source archive hashes

- physical_validation_results (1).zip: eb42ef98273b6e0cca08b7fcabda5af82c847d706282dbc671bfb521456b975a
- raw_geometry_results (1).zip: d2f9cc7e3f7713335edc3058f099cf2030bf9de9a18c24b2068bbcdd999f8884
