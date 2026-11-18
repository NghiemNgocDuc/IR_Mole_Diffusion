# Measure the baseline before fine-tuning

This update evaluates the existing stage-1 checkpoint on **1,000 randomly selected validation molecules**, with a fixed seed and beam width 5. It saves the selected rows so later comparisons use the same examples. It also analyzes the earlier `evaluation/spectrum_val.json` when that full report is present.

It reports strict Top1 and fixed Top5 identity accuracy, formula errors, tautomer alternatives, and results by molecular class. Tautomer alternatives remain errors under the strict identity metric. A random subset gives an estimate; only evaluating every validation row gives full-validation accuracy.

The job reads the trained weights without modifying them. It does not fine-tune a model or use test examples for training.

The selected sample was checked against the dataset before any new predictions were scored:

| Property | Earlier first 500 | New random 1,000 | Full validation |
|---|---:|---:|---:|
| Aromatic molecules | 1.0% | 15.9% | 17.8% |
| Acyclic molecules | 43.8% | 10.9% | 10.8% |
| Contains nitrogen | 44.0% | 59.5% | 61.2% |
| Mean ring count | 0.712 | 1.642 | 1.626 |

This is a composition check, not a new accuracy measurement. The seed was fixed before inspecting this composition; it was not searched for a favorable result. These graph metrics do not measure 3D coordinate accuracy.

## Upload and run on Unity

Upload **`unity_baseline_audit_update.zip`** into `/home/dnghiem_umass_edu/diffusion` using the Unity file browser. Then paste:

```bash
cd ~/diffusion
unzip -o unity_baseline_audit_update.zip
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=base audit_spectrum_baseline.sh
```

The existing environment must contain the packages used for your earlier evaluation. This update does not reinstall them. The job requires these existing files:

- `runs/spectrum/checkpoints/best.ckpt`
- `data/processed/squirl.npz`
- `data/processed/splits.json`

## Watch progress

Replace `JOB_ID` with the number returned by `sbatch`:

```bash
cd ~/diffusion
squeue -j JOB_ID
tail -F logs/audit-JOB_ID.out logs/audit-JOB_ID.err
```

Progress messages show completed molecules, elapsed time, and an estimated remaining time after evaluation begins. Queue waiting is separate. Press **Ctrl+C** to leave the log viewer; that does not cancel the batch job.

After it ends, confirm the job status:

```bash
sacct -X -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

The default output folder is `evaluation/baseline_audit_JOB_ID`. Open `error_audit/analysis.md` inside that folder for the findings. `errors.csv` contains first-choice mistakes; `rows.csv` contains every evaluated example. `baseline.json` retains all ranked candidates and references. `selection.json` records the frozen selection.

The final log prints **`DOWNLOAD RESULTS:`** followed by a ZIP path beside the output folder. Download that ZIP from the Unity file browser and share it for review. It includes the new reports, analyses, and the earlier full validation report when available. It excludes dataset arrays and checkpoint files.

## Resume an interrupted evaluation

Keep the same output folder, selection settings, seed, and checkpoint. For the default 1,000-example job, replace `ORIGINAL_JOB_ID` below:

```bash
cd ~/diffusion
sbatch --export=ALL,CONDA_ENV=base,EVAL_OUTPUT=evaluation/baseline_audit_ORIGINAL_JOB_ID audit_spectrum_baseline.sh
```

The evaluator validates saved progress and continues from completed records. Completed evaluation files are validated before reuse; completed analyses are reused only when their input hashes match. Conflicting or incomplete analysis files produce an error rather than being overwritten. A partial report must not be presented as a final accuracy result.

Reports are saved every ten molecules and on handled interruptions. A hard termination may repeat up to nine completed molecules. The decoder, preprocessing, weights, software versions, numerical settings, and selection are checked before resuming. Resume equality was tested on CPU; different GPU hardware can introduce numerical differences. Submit only one job per output folder; an OS-held lock prevents simultaneous evaluators from writing the same report. ETA describes remaining generation work, and later error analysis is a separate phase.

## Evaluate the whole validation split

For full-validation accuracy instead of the initial random subset:

```bash
cd ~/diffusion
sbatch --export=ALL,CONDA_ENV=base,EVAL_SELECTION=all audit_spectrum_baseline.sh
```

This evaluates all 12,478 validation rows and takes longer. The default wall-time request is four hours; if it times out, resume the same folder with `EVAL_SELECTION=all` included. A job's wall-time limit is not a runtime prediction.

Optional environment settings are `EVAL_LIMIT` (default `1000` for random selection), `EVAL_SEED` (default `20261105`), `EVAL_OUTPUT`, and `SPECTRUM_CHECKPOINT`. Keep these fixed across a resumed run. Save experiments using different settings in different output folders.

## How we will use the results

We will first distinguish cases where the correct molecule is already at ranks 2–5 from cases where it is absent. The molecular-class breakdown will show which errors merit targeted training experiments. Training examples will come from the training split; these validation errors guide evaluation and experiment design without becoming training labels. Fine-tuning remains a separate next step after reviewing this baseline.
