# Extended OPLS numerical checks

Upload unity_opls_extended.zip to ~/diffusion, then:

```bash
cd ~/diffusion &&
unzip -o unity_opls_extended.zip &&
sbatch --qos=short check_opls_extended.sh
```

This creates new acetone (ACE) and benzene (BEN) parameter sets and repeats the
energy-gradient and rigid-transform checks on template and perturbed geometries.
It explicitly requires nonempty improper tables in both generated loaders. These
molecules extend coverage beyond ethanol, which has no improper terms.

The job also collects GROMACS module availability without installing or loading
another engine. This prepares a later comparison; no independent-engine energy
calculation is performed here. Numerical consistency can pass even with incorrect
force-field parameters. Neither physical correctness nor global minima are claimed.

Download outputs/opls_extended_JOBID/results.zip when the job finishes.
If it fails before packaging, send logs/opls-extended-JOBID.out and .err and the
per-molecule parameterization log in that output directory. Earlier ethanol
results, model weights, and generation code are not modified.

Locally checked: Bash syntax, archive integrity, and improper-table parsing.
Actual parameterization and OpenMM checks run on Unity and remain pending.
