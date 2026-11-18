# OPLS diagnostic 1820275

Reviewed the attached `diagnostic_results.zip`. Archive CRC passed; status is
`completed`, engine evaluation enabled, eight selected graphs inspected.
Source benchmark report SHA-256 matches
`7eb6f5d2bba6fa3c013e67a9f29902d811349b826a34bf99ed2719d37f2c6439`.

All 96 saved-geometry energy and RMS-force replay checks passed. All 104
force-group and energy-group sum checks passed (including eight templates).
These checks establish reproduction of the saved calculations, not physical
accuracy of the force field.

## Findings

- Hydrogen-normalized graph comparison raises converged, graph-matching
  conformers from 106/141 to 112/141 in each arm. These six additional matches
  per arm correct representation mismatches; they are not new generated samples.
- The most extreme relaxed forces are dominated by PeriodicTorsionForce near
  collinear atom triples in nitrile-containing candidates. For example,
  input 13 graph 0 has a guided sample with RMS force about 12.3 million
  kJ/mol/nm and a nonzero torsion with plane sine about 1.56e-11. Finite
  differences are strongly step-size dependent in such cases. This identifies
  a numerical pathology to investigate; it does not yet establish whether
  parameter assignment, export, or evaluation is responsible.
- Some other nonconverged samples are bond/angle dominated. One fix cannot
  be assumed to resolve every failure. The converged control has large
  individual force contributions that cancel to a small total force, so
  a large force-group contribution alone is not a failure criterion.
- LigParGen rejected one parameter-template geometry because an unbonded
  atom pair was too close. This template is generated before model sampling;
  this is not evidence that the candidate molecular graph is invalid.

## Pipeline priorities

1. Retry rejected parameter templates with a bounded, deterministic set of
   alternative embeddings, retaining atom mapping and topology validation.
2. Check the offending nonzero torsions against the parameter/export source
   and an independent engine at identical geometries. Do not silently remove
   torsions or bend nitriles to obtain passing results.
3. Keep nonconverged conformers flagged and retain other passing candidates.
   Continue reporting generation, parameterization, and relaxation failures
   separately. This pipeline supports multiple candidates.

No new sampling or model training was performed for this review. The results
do not establish molecular identity, a global minimum, or physical correctness.
