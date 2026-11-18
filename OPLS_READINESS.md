# Consolidated OPLS integration checks

Run one job and return one archive:

```bash
cd ~/diffusion &&
unzip -o unity_opls_readiness.zip &&
sbatch --qos=short check_opls_readiness.sh
```

Uses existing exports from ethanol job 1788228 and acetone/benzene job 1791099.
No additional installation, parameterization, model inference or training is
required. OpenMPI and GROMACS modules are loaded automatically. Per-molecule
errors are collected so that one failed comparison does not prevent the other
checks. A module-loading failure can still terminate before the runner starts.

## Checks in this job

1. Repeat the six same-coordinate energy comparisons and extract GROMACS atom
   forces from a zero-step TRR file. Compare against OpenMM Reference at the
   actual GROMACS stored coordinates. Explicit continuation=yes and no constraints
   prevent startup coordinate constraints. Force tolerance is fixed at 0.2
   kJ/mol/nm absolute plus 0.001 relative per component. Energy tolerance remains
   0.05 kJ/mol. These allow rounded exported constants and single precision.
2. Match the predictor's AddHs(MolFromSmiles()) atom convention to exported
   atom order for all three declared neutral graphs. PDB explicit connectivity
   and hydrogens determine bond orders; no proximity bonds are invented. This
   fixture mapper is not a general charged/stereo/radical parameterization mapper.
   Symmetric alternatives are resolved by a recorded fixed permutation.
3. Test model-to-engine coordinate conversion (Angstrom to nm), inverse force
   mapping (kJ/mol/nm to kJ/mol/Angstrom), finite differences in model coordinates,
   and an intentionally permuted atom order.
4. Apply ten bounded energy-decreasing force corrections on perturbed templates.
   Each step caps maximum atom displacement at 0.02 Angstrom and backtracks up
   to twelve times. An unsuccessful correction retains its input. Save before/
   after coordinates and energy traces. This tests the adapter that will support
   later predicted-clean-coordinate guidance, not the diffusion model itself.

The shared adapter is in diffusion/opls_bridge.py. Existing samplers and weights
are unchanged. Guidance during noisy diffusion, early rejection thresholds,
Boltzmann weighting and production acceptance are not enabled by these checks.
The job also records source hashes and the names of checkpoint files under ckpt
to help prepare the next controlled integration without another inventory request.

## Result

Download outputs/opls_readiness_JOBID/readiness_results.zip. It contains the
complete report, source hashes, GROMACS inputs/outputs, force arrays and correction
trajectories. Pass means ready for controlled integration on these fixtures;
it does not prove physical truth, global minima, correctness for arbitrary
molecules, or better IR identification accuracy. No promise that a failed check
can be skipped; remaining problems will be diagnosed from this archive.

Local validation: 14 unit tests pass (mapping/units, finite differences,
correction bounds, invalid mappings/nonfinite input, retaining input on a wrong
force direction, vector parser). The three actual archived molecules map
successfully. Full engine and adapter integration still needs Unity execution.

GROMACS force extraction reference:
https://manual.gromacs.org/2024.3/onlinehelp/gmx-traj.html
