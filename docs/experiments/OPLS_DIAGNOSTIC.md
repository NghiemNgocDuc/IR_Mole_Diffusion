# Graph-check fix and saved-geometry diagnostic

This update fixes a representation error in the coordinate graph checker.
Equivalent explicit and implicit hydrogen notation now compares through a
canonical, fully hydrogenated achiral graph. Charge, isotope, connectivity, and
bond-order differences remain distinct; atom map labels and stereo are ignored
as intended for this experiment. Original benchmark results are preserved.

Local rescoring of recorded inferred graphs changes the final converged-plus-
matching count from 106 to 112 out of 141 assessed attempts in **each** arm.
It does not change coordinates, force convergence, or the conclusion that the
two arms have equal pass counts. Six false failures per arm were hydrogen
notation differences. Other bond-perception failures remain unresolved.

The diagnostic selects all seven graphs with nonconverged attempts plus the
first fully force-converged graph as a control. It reads their existing raw and
relaxed coordinates and the original LigParGen parameter exports. It does not
generate new structures or rerun minimization/parameterization.

For every selected geometry it reports bonded distances, angles, nonbonded
distances excluding 1-2 and 1-3 pairs, per-force-class energies/forces, dominant
force components, and collinearity of nonzero periodic torsions. A finite-
difference probe at three step sizes checks sensitivity at the dominant force
component. Template coordinates are also inspected. Near-linear angles may be
chemically expected; these measurements are not automatic rejection rules.

The implementation uses OpenMM's documented force groups and state queries:
[Force](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html)
and [Context](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html).
Grouped forces/energies must sum to the total, and saved energies/force RMS are
replayed. These checks are diagnostics; discrepancies need review, not threshold
tuning on these results. The script also collects the failed parameterization
log if it is still present on Unity.

## Run

Upload `unity_opls_diagnostic.zip` into `~/diffusion`, then:

```bash
cd ~/diffusion &&
unzip -o unity_opls_diagnostic.zip &&
./.runtime/opls-tools/bin/python install_opls_pipeline.py &&
mkdir -p logs &&
sbatch --qos=short diagnose_opls_benchmark.sh
```

The script defaults to the saved `outputs/opls_benchmark_1820024` directory and
requires its exact reviewed report hash. Parameter hashes are verified before
loading the trusted local LigParGen exports. No downloaded executable is run.
The installer accepts only the known previous backend or the new version and
backs up changes. Future parameter-cache entries use the updated backend hash;
this diagnostic bypasses parameter generation and uses the original exports.

Download `outputs/opls_diagnose_JOBID/diagnostic_results.zip` when finished.
Logs are `logs/opls-diagnose-JOBID.out` and `.err`.

Local tests cover graph comparison, geometry measurements, and rescoring actual
saved benchmark reports. OpenMM force decomposition must run on Unity because
OpenMM is not installed in the local development environment.
