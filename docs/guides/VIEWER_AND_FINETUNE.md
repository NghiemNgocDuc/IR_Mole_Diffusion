# Rotate saved predictions and test a graph fine-tune

The update adds an offline, interactive molecule viewer and an optional stage 1
fine-tuning experiment. The viewer works on existing predictions immediately.
Fine-tuning has not run on the user's Unity checkpoints, and no accuracy gain is
claimed. Existing trained checkpoints and data are reused.

## Install once on Unity

Upload `outputs/unity_improvement_update.zip` from this computer to
`/home/dnghiem_umass_edu/diffusion` using OnDemand **Files > Upload**. Then run:

```bash
cd ~/diffusion
unzip -o unity_improvement_update.zip
module load conda/latest
```

This archive contains source and viewer assets only. Its hashes are in
`IMPROVEMENTS_MANIFEST.json`. It does not contain trained weights or data.

## Rotate the existing harder example

```bash
conda run -n base --no-capture-output python make_viewer.py \
  --prediction_dir predictions/example_66285483
```

In OnDemand Files, open `diffusion/predictions/example_66285483`, download
**viewer.html**, and open the downloaded file in Chrome or Edge on your computer.
Downloading avoids restrictions on active HTML in OnDemand's preview.

- Drag the molecule to rotate; use the wheel or pinch to zoom.
- Select **Graph candidate 3** to inspect the candidate that matched this
  example's reference identity. Choose any saved conformer or raw/final frame.
- Rotate, zoom, move, or expand to full screen. Toggle hydrogens and
  ball-and-stick/stick/space-filling styles. Element names appear in the color
  legend below the model; no atom letters are drawn on the 3D model.
- Click two visible atoms for their distance; use Reset view if needed.
- Save XYZ exports the selected saved coordinates. Save PNG exports the view.

Everything needed is embedded in the HTML, including the licensed 3Dmol.js
viewer. It needs no GPU, server, notebook, or internet connection after export.
No new Python dependency is required beyond the project's NumPy and RDKit.
The 3D coordinates are the saved model/refinement output; no replacement
conformer is generated. Failed or unavailable frames are identified explicitly.
The IR chart displays the input; no forward IR spectrum is calculated.
Conformers remain unranked. A geometry pass does not establish correct identity
or agreement with the IR spectrum.

The updated `predict_example.sh` automatically creates `viewer.html` for future
predictions as well as the existing static `result.png`.

For the smaller viewer-only update and explicit 2D-graph-to-3D generation,
see [the 3D viewer guide](VIEWER_3D_UPDATE.md).

## Optional: fine-tune stage 1 ranking

The harder example's correct graph was candidate 3 rather than candidate 1.
This motivates a ranking experiment, not training on that test example. The
experiment retains token prediction loss and adds a small positive-versus-
negative sequence margin. Negative identities come only from training rows;
same-formula candidates are preferred. The learning rate is reduced from
`2e-4` to `2e-5`, for at most 15 epochs, with early stopping.

```bash
cd ~/diffusion
mkdir -p logs
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base finetune_spectrum.sh
```

Replace `JOBID` with the returned number to watch preparation and training:

```bash
squeue -j JOBID
tail -F logs/finetune-JOBID.out logs/finetune-JOBID.err
```

The new run writes to `runs/spectrum_rank_finetune`; existing baseline weights
are preserved. The batch job continues if the browser or computer is closed.
Ctrl+C stops following the log without cancelling training. Runtime depends on
the assigned GPU and early stopping; the 12-hour allocation is a limit, not an
estimate. Training progress gives measured epoch times once it starts.

After it finishes successfully, submit a matched validation comparison:

```bash
sbatch --partition=gpu-highvram --export=ALL,CONDA_ENV=base compare_spectrum.sh
```

Watch `logs/rank-compare-JOBID.out` and `.err`. The report is
`evaluation/rank_compare_JOBID/comparison.json`. It compares both checkpoints
on the same 500 validation rows with beam size 5, reporting Top-1/Top-5 gains
and losses. It does not automatically replace the baseline. If promising,
confirm on the full validation set before selecting weights, for example:

```bash
sbatch --partition=gpu-highvram --time=24:00:00 \
  --export=ALL,CONDA_ENV=base,EVAL_LIMIT=12478 compare_spectrum.sh
```

Use the test split only after model selection. The current split can share
tautomer families between partitions; this is not a novel-family benchmark.
This experiment targets graph selection, not a demonstrated improvement in
3D geometry or experimental IR interpretation. Stage 2 remains unchanged.

Once validation supports choosing the fine-tuned graph model, an example job
can explicitly select it:

```bash
sbatch --partition=gpu-highvram \
  --export=ALL,CONDA_ENV=base,SPECTRUM_CHECKPOINT=runs/spectrum_rank_finetune/checkpoints/best.ckpt \
  predict_example.sh
```

To repeat a prediction on a chosen row, also add `PREDICTION_ROW=ROW_NUMBER`
to the comma-separated export list. Do not use outcomes on a repeatedly
inspected test example to tune the model.
