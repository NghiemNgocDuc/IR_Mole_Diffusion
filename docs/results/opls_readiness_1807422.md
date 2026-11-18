# OPLS readiness job 1807422

Archive: `readiness_results.zip`, SHA256
`997f32ebf15b07f09bd2f0722d7f7e6caefe78ad4fb1520ab3b8a3c5294df874`.

Saved-array audit: `outputs/opls_readiness_review_1807422/audit.json`.

| Molecule | Geometry | Energy difference (kJ/mol) | Maximum force difference (kJ/mol/nm) |
|---|---|---:|---:|
| Ethanol | template | 0.00009574 | 0.03270 |
| Ethanol | perturbed | 0.00117333 | 0.04350 |
| Acetone | template | 0.00152626 | 0.02603 |
| Acetone | perturbed | 0.00072312 | 0.02060 |
| Benzene | template | 0.00012036 | 0.04997 |
| Benzene | perturbed | 0.00071342 | 0.02922 |

All pass energy absolute tolerance 0.05 kJ/mol and force component tolerance
0.2 kJ/mol/nm plus relative tolerance 0.001. Forces compared at GROMACS's stored
coordinates to account for single-precision coordinate rounding.

All three cases also passed explicit graph mapping, Angstrom-unit gradient,
permuted atom-order equivalence, and capped energy-decreasing correction checks.
Ten corrections reduced energies as follows (same molecule/parameters only):
ethanol 44.43769 -> 21.95884; acetone -45.40903 -> -69.85333;
benzene 51.79992 -> 31.15915 kJ/mol.

The archive's CRC, numerical force arrays, energy output arithmetic, mapping
permutations and correction traces were independently rechecked locally. No
engine rerun was required. This supports controlled sampler integration for the
tested implementation. It does not prove physical truth, a global minimum, or
correct generated molecular identity. No generator inference ran in this job.

The Unity inventory found no checkpoint under `ckpt/`. Broader discovery is now
included in the pipeline pilot, with explicit legacy/diagnostic exclusions.
