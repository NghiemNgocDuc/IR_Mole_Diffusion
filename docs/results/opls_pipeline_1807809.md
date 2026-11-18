# OPLS coordinate pilot 1807809

The attached `pipeline_results (1).zip` completed all four attempts on one
training spectrum (row 0). Archive SHA256:
`54e36322b56e96848fa774c29299276f85c2cf403c93e485c03aabc65e384301`.

| Predicted graph | Arm | Raw energy kJ/mol | Raw RMS force kJ/mol/nm | Relaxed energy kJ/mol |
|---|---|---:|---:|---:|
| C | Unguided | 34.13751 | 607.34201 | 0.704820 |
| C | Guided | 14.48085 | 371.80534 | 0.704819 |
| C=N | Unguided | 292.47142 | 3916.85512 | 23.185380 |
| C=N | Guided | 201.91818 | 2993.82334 | 23.185380 |

Both arms used identical coordinate seeds within each graph. All four local
minimizations converged below RMS force 1 kJ/mol/nm; raw and relaxed inferred
achiral graphs matched the declared candidates. Each guided attempt accepted
15 bounded corrections. Guidance reduced raw strain in these examples; both
arms reached essentially equal relaxed energies. Two tiny candidates from one
training input do not establish general performance or physical correctness.

Review checked archive integrity, parameter hashes, saved arrays, atom/bond
metadata, paired seeds, correction gates/caps, and reported convergence.
Energies were not recomputed locally. The fixed broader experiment is documented
in `OPLS_BENCHMARK.md`.
