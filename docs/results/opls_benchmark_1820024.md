# OPLS benchmark 1820024

Reviewed the downloaded `benchmark_results.zip` on 2026-11-16.
SHA256: `6f75f279c30d8149ff838fb02a27553d50de42eba9f6f4f533b19e966631557f`.
The independent saved-artifact analyzer found no consistency errors. It checked
selection/candidate manifests, parameter hashes, paired seeds, saved coordinates,
guidance traces, and report arithmetic. Engines were not rerun locally.

## Observed results

- Runtime: 734.4 seconds, excluding scheduler queue time.
- 24 training spectra, 48 distinct predicted graphs, three seeds per graph.
- 282/288 attempts completed; one graph failed parameterization, affecting six
  attempts. There were 141/144 assessable paired comparisons.
- Guided raw energy was lower in 140/141 pairs. Raw RMS force was lower in
  137/141 pairs. Every input had a lower mean raw energy across its assessed pairs.
- Median guided-minus-unguided raw energy: -27.1646 kJ/mol.
- Median raw RMS force difference: -516.752 kJ/mol/nm.
- Median generation time ratio: 1.03645 (about 3.6% overhead). This excludes
  graph parameterization and setup, reported separately.
- Median raw-to-relaxed displacement difference: -0.00454339 Angstrom.
- Median accepted relaxed energy difference: 0.000000926811 kJ/mol. A near-zero
  median does not imply equal energies or identical conformers for every pair.

Both arms have the same outcome counts:

| Recorded outcome | Count per arm |
|---|---:|
| Completed attempt | 141/144 planned |
| Raw inferred graph matches declared candidate | 129/141 assessed |
| Relaxation accepted and force converged and inferred graph matches | 106/141 assessed |
| Relaxation accepted, graph matches, but force not converged | 21/141 assessed |
| Relaxation accepted and force converged, but inferred graph check fails | 14/141 assessed |

All assessed relaxations lowered energy enough to be accepted. Acceptance alone
therefore must not be used as the final success criterion.

## Follow-up diagnosis

1. Parameterization failed for input 13, candidate `CNC12COC1C2C#N`.
   The report records a LigParGen failure, but its detailed failed-work log is
   outside the downloaded results archive. The underlying cause is undetermined.
   Unity retained it at:

   `/home/dnghiem_umass_edu/diffusion/outputs/opls_parameter_cache/40b428daf7e5414c172b5340c2a0077dc8c87284f809b2adb68166c3ef39e1c9.work-4c4cj3_o/parameterization.log`

2. The 21 force-nonconverged attempts per arm cover seven graphs, each with
   three seeds. Some reported final forces are extremely large despite reduced
   energy. Examples include input 7's bridged-ring candidates and input 13's
   successful parameterization candidate. These require saved-geometry/force-term
   diagnosis; this report does not establish whether the cause is a difficult
   starting geometry, a force-field issue, or minimizer behavior. Increasing the
   iteration limit alone is not yet justified by these results.

3. Six of the 14 inferred-graph failures per arm are representation mismatches
   in input 11. The inferred strings retain an explicit imine hydrogen:

   - `CCOC(=N)C(C)=NO` versus `[H]N=C(OCC)C(C)=NO`
   - `CCOC(=NO)C(C)=N` versus `[H]N=C(C)C(=NO)OCC`

   Local RDKit comparison after explicitly adding hydrogens and removing
   stereochemistry confirmed matching atom counts (19) and bidirectional graph
   matches for both pairs. The original string-equality check flags these as
   failures even though the reported molecular graphs are equivalent. Preserve
   the original scores and report this correction separately; it does not prove
   geometry quality or experimental identity.

4. The remaining eight inferred-graph failures per arm report bond-perception
   errors, chiefly on ring systems. They are unresolved diagnostics, not proof
   of incorrect chemistry. Check known-topology bond lengths and nonbonded
   contacts against the saved coordinates before assigning a cause.

## Next work

The bounded guidance improves raw force-field strain on this training batch but
has not improved final convergence or the original graph-check pass count. Focus
the next engineering change on graph-equivalence checking and investigation of
the seven nonconverged graphs. Keep early rejection disabled while those failure
modes are unresolved. Do not select new guidance settings using this batch and
then describe it as an independent validation set.

Machine-readable analysis and the short report are in
`outputs/opls_benchmark_analysis_1820024/analysis.json` and `ANALYSIS.md`.
No model weights were trained or promoted. No Top1, physical-correctness,
stereochemistry, global-minimum, or Boltzmann-population claim is supported.
