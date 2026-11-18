# Live evaluation progress on Unity

Let the current job finish. An already running Python process cannot acquire
this progress display from a source-file update. No restart or retraining is
needed for that job to finish normally.

For subsequent coordinate evaluations, upload `unity_progress_update.zip` to
`~/diffusion` and extract it from that directory:

```bash
cd ~/diffusion
unzip -o unity_progress_update.zip
```

The update contains only the evaluation CLI, metric progress callbacks, a small
progress logger, and this guide. It contains no model, checkpoint, dataset, or
new dependency, and preserves sampling and metric calculations.

Submit evaluations with the same `eval_diffusion.sh` command as before.
Progress is enabled by default. Logs show a newline-based bar, counts, elapsed
time, and an estimated remaining time **for the current phase**. Phases include
sampling, raw geometry/chemistry, MMFF refinement if requested, refined metrics,
and output writing. ETA appears after work completes and may vary early on.

Watch the output log, replacing `JOBID` with the submitted job number:

```bash
tail -F logs/evaluate-JOBID.out logs/evaluate-JOBID.err
```

The evaluation output directory also contains `progress.json` with the latest
phase, counts, estimated time, and `running`, `completed`, or `failed` status.
For example, for a refined evaluation:

```bash
cat evaluation/diffusion_val_refined/progress.json
```

Updates are flushed approximately every 15 seconds while work advances, plus
phase starts and completions. The sampling bar advances after each candidate
batch finishes; it cannot advance while an individual GPU call is unfinished.
The file is replaced atomically. It is a status snapshot, not resumable state.
Time in the scheduler queue is not included. A hard process kill can leave a
stale `running` status, so use `sacct` to confirm final job state.

Optional flags, appended to the evaluation submission command:

- `--progress_interval 5`: log more frequently.
- `--progress_interval 0`: log every candidate batch or completed molecule.
- `--no_progress`: disable the progress lines and status file.

Ctrl+C exits `tail`; it does not stop the submitted job. Closing the browser or
turning off the local computer also leaves the Slurm job running.
