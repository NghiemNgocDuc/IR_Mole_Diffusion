# Fixed OPLS training-spectrum experiment

This extends successful pilot 1807809 without changing its sampler, guidance,
parameterization, or minimizer. Run one batch to assess whether its raw-geometry
improvement persists across more inputs and seeds. No training is performed.

## Frozen default plan

- 24 unique training identities: eight each with 3–10, 11–18, and 19–29 explicit
  atoms. Choose one spectrum per identity, using seed 20261116, before decoding.
  Reference atom counts are used only for stratification; reference bonds and
  coordinates never condition generation or score the output. This is an
  engineering stress test, not an independent accuracy benchmark.
- First two valid unique predicted graphs from beam size five, as in the pilot.
  Save all decoded beams. Generated sizes can differ from selection sizes.
  Save actual predicted atom counts and the number of distinct predicted graphs.
- Three coordinate seeds per candidate, shared across both arms. Stable seed
  allocation does not change when another candidate fails. Alternate which arm
  runs first to reduce timing order effects.
- 288 intended attempts: 24 inputs × two candidates × three seeds × two arms.
  Missing candidates, unsupported OPLS graphs, and failed attempts are retained;
  no replacement candidate is chosen after an energy result.
- Same 50 DDIM steps and EMA checkpoints for both arms. Guidance starts at
  alpha-bar >= 0.8 and caps each correction at 0.02 Angstrom per atom.
- Scope remains neutral connected closed-shell H/C/N/O/F, 3–29 explicit atoms,
  without explicitly specified stereo or isotope labels. The graph decoder
  uses achiral canonical identities as in the pilot.
- No early rejection, new tuning, checkpoint selection, or Boltzmann weights.

## Run on Unity

Upload `unity_opls_benchmark.zip` into `~/diffusion`, then:

```bash
cd ~/diffusion &&
unzip -o unity_opls_benchmark.zip &&
./.runtime/opls-tools/bin/python install_opls_pipeline.py &&
mkdir -p logs &&
sbatch --qos=short --export=ALL,CONDA_ENV=base run_opls_benchmark.sh
```

The installer checks hashes before edits and backs up recognized changed files.
It carries the same tested pilot dependencies. Other project files are untouched.
Default checkpoints are the pair used successfully in 1807809:

```
runs/diffusion/checkpoints/best.ckpt
runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt
```

The launcher requests two CPUs, 12 GB, and 30 minutes. The program stops starting
new work after an 18-minute budget and writes a partial report if needed. This
budget is a stopping rule, not a duration estimate. Each worker request has its
existing timeout. A Slurm warning requests a stop between attempts; forced
termination can still interrupt a request. Progress JSON is saved before and
after attempts; the ZIP is refreshed after each graph and at normal exit.
There is no automatic resume or duplicate job submission.

Check the returned job number (replace JOBID):

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
tail -n 40 logs/opls-benchmark-JOBID.out logs/opls-benchmark-JOBID.err
```

Download `outputs/opls_benchmark_JOBID/benchmark_results.zip` once finished.
An exit code of 1 can mean a partial run or reported candidate failures; read
`RESULTS.md` and `report.json` before deciding whether another run is needed.

## What the report answers

1. Does guidance reduce **raw** energy and force for the same graph and seed?
2. How often do raw coordinates reconstruct the declared achiral graph?
3. How much coordinate repair does relaxation require? How often does it both
   converge and preserve the inferred graph?
4. Do guided and unguided samples end at similar relaxed energies?
5. What fails, on which predicted sizes, and how long do setup, generation,
   assessment, and relaxation take?

Do not compare absolute energies across different molecules. Paired differences
use the identical graph and parameter set. The report averages within each input
before any interpretation across inputs; seeds are not independent molecules.
It supplies descriptive summaries without confidence intervals or significance
claims. Failure counts and planned denominators must accompany conditional rates.

Raw coordinates, relaxed coordinates, SDFs, mapping/parameter manifests, hashes,
seeds, and guidance traces are retained. Bond-perception agreement is a useful
diagnostic, not proof of identity, stereochemistry, experimental stability, or
correct IR assignment. Low OPLS energy is not proof of physical correctness.
Raw-to-relaxed displacement is fixed-order, unaligned RMS atom displacement;
it is not a conformer-diversity estimate. A rejected relaxation retains raw
coordinates; its attempted final energy is not counted as an accepted result.

Local tests cover selection, pairing, orchestration, failure accounting, partial
reports, and installation. Actual OpenMM/LigParGen calculations run on Unity.
