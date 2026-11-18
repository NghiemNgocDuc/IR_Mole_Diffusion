# One upload for the pending Unity updates

Upload **`unity_complete_update.zip`** into
`/home/dnghiem_umass_edu/diffusion` using Unity's file browser.
This replaces the separate viewer, progress, and baseline-audit ZIP uploads.
The dataset, trained models, and saved predictions already on Unity are reused.

This bundle includes:

- The interactive 3D viewer: drag to rotate, zoom, move, auto-rotate, full screen,
  candidate/conformer selection, and a color legend below the model. Element
  letters are kept off the 3D atoms.
- The option to generate estimated 3D conformers from an explicit SMILES
  structure. Saved prediction mode retains the saved model coordinates.
- Updated static pictures and automatic viewer creation for future prediction
  jobs through `predict_example.sh`.
- Progress and phase ETA for coordinate evaluation.
- Random 1,000-molecule validation with progress, resume support, fixed Top-5
  scoring, and an error report by molecular class.
- The dataset-audit findings and cached chemical annotations used in analysis.

Fine-tuning will be selected after reviewing this validation report. The old
experimental fine-tuning launchers are not part of this update.

## Install, create your viewer, and submit the audit

Paste this complete block in the Unity terminal. Each step runs only after the
previous step succeeds:

```bash
cd ~/diffusion &&
unzip -o unity_complete_update.zip &&
module load conda/latest &&
conda run -n base --no-capture-output python make_viewer.py \
  --prediction_dir predictions/example_66285483 &&
mkdir -p logs &&
sbatch --export=ALL,CONDA_ENV=base audit_spectrum_baseline.sh
```

The existing environment used for your successful predictions supplies the
dependencies. No package reinstall is needed for these commands. The ZIP
contains the offline renderer and its license.

## Open the movable 3D result

In Unity's file browser, download:

```text
/home/dnghiem_umass_edu/diffusion/predictions/example_66285483/viewer.html
```

Open the downloaded **HTML file** in Chrome or Edge on your computer. It is
self-contained and works offline. Drag the molecule to rotate it and use the
wheel to zoom. The existing PNG stays a static image. Unity's HTML preview may
restrict scripts, so open the downloaded file locally.

Viewer creation uses the existing prediction and finishes separately from the
GPU accuracy audit. You can use the viewer while that audit runs.

## Follow the accuracy audit

Replace `JOB_ID` with the number printed by `sbatch`:

```bash
cd ~/diffusion
tail -F logs/audit-JOB_ID.out logs/audit-JOB_ID.err
```

The log shows progress and estimated remaining generation time. Press Ctrl+C
to leave the log viewer. The batch job continues. At completion, the log prints
`DOWNLOAD RESULTS:` and a results ZIP path. Download that results ZIP for review.

For job status, resume instructions, or full-validation evaluation, read
[BASELINE_AUDIT_UPDATE.md](BASELINE_AUDIT_UPDATE.md). Its separate ZIP-install
step is already covered by this combined upload. The same applies to the
installation steps in [VIEWER_3D_UPDATE.md](VIEWER_3D_UPDATE.md) and
[EVALUATION_PROGRESS_UPDATE.md](EVALUATION_PROGRESS_UPDATE.md).

## Existing files required on Unity

- `data/processed/squirl.npz` and `data/processed/splits.json`
- `runs/spectrum/checkpoints/best.ckpt` for the accuracy audit
- The original `diffusion` Python package from the training bundle
- `predictions/example_66285483/prediction.json` and `coordinates.npz`
  for your saved example; its input IR is read from the saved input or dataset

`COMPLETE_UPDATE_MANIFEST.json` lists every included file and its SHA-256 hash.
Extraction updates the listed source, guide, and annotation files. Dataset
arrays, model checkpoints, prediction JSON, and prediction coordinates are
not included in the archive.
