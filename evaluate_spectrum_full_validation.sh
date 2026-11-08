#!/bin/bash -l
#SBATCH --job-name=ir_full_validation
#SBATCH --output=logs/fullval-%j.out
#SBATCH --error=logs/fullval-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=gpu-highvram
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the diffusion repository root}"
module load conda/latest
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PILOT_OUTPUT="${PILOT_OUTPUT:-runs/spectrum_pilot_66855524}"
export FULL_VALIDATION_OUTPUT="${FULL_VALIDATION_OUTPUT:-evaluation/full_validation_66855524}"
conda run -n "${CONDA_ENV:-base}" --no-capture-output python -u - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from diffusion.data import file_sha256
from diffusion.spectrum_audit import atomic_json, exclusive_report
from scripts.run_spectrum_pilot import source_hashes

read = lambda path: json.loads(path.read_text(encoding="utf-8"))
pilot = Path(os.environ["PILOT_OUTPUT"]).resolve()
output = Path(os.environ["FULL_VALIDATION_OUTPUT"]).resolve()
if output == pilot or output in pilot.parents or pilot in output.parents:
    raise ValueError("Use a separate evaluation directory outside the pilot directory")
recipe = read(pilot / "pilot.json")
if not (pilot / "RESULTS.md").is_file():
    raise ValueError("Complete and review the pilot before full validation")
if source_hashes() != recipe["source_sha256"]:
    raise ValueError("Runtime sources differ from the completed pilot")
data = Path(recipe["arguments"]["data"]).resolve()
splits = Path(recipe["arguments"]["splits"]).resolve()
for path, expected in ((data, recipe["data_sha256"]), (splits, recipe["split_sha256"])):
    if file_sha256(path) != expected:
        raise ValueError(f"Pilot input changed: {path}")
count = len(read(splits)["splits"]["val"])
models = {"baseline": (pilot / "baseline_start.ckpt", recipe["checkpoint_sha256"])}
for arm in ("ce", "ranking"):
    marker = read(pilot / f"{arm}_complete.json")
    path = Path(marker["checkpoint"]).resolve()
    if not path.is_relative_to(pilot):
        raise ValueError(f"{arm} checkpoint is outside the completed pilot")
    if file_sha256(path.parent.parent / "resolved_config.json") != marker["config_sha256"]:
        raise ValueError(f"{arm} training configuration changed")
    models[arm] = (path, marker["checkpoint_sha256"])
for arm, (path, expected) in models.items():
    if file_sha256(path) != expected:
        raise ValueError(f"{arm} checkpoint changed")

with exclusive_report(output / "full_validation.json"):
    for arm, (path, expected) in models.items():
        print(f"FULL VALIDATION: {arm}; {count} validation molecules; legacy beam 5; EMA", flush=True)
        subprocess.run([sys.executable, "-u", "evaluate_spectrum.py",
            "--checkpoint", str(path), "--data", str(data), "--splits", str(splits),
            "--split", "val", "--selection", "all", "--limit", str(count),
            "--seed", "20261105", "--beam_size", "5", "--score_mode", "legacy",
            "--length_penalty", "0.6", "--device", "cuda", "--resume",
            "--save_every", "100", "--progress_interval", "15",
            "--selection_manifest", str(output / "selection.json"),
            "--output", str(output / f"{arm}.json")], check=True)
        report = read(output / f"{arm}.json")
        if not report["complete"] or report["attempted_rows"] != count or report["checkpoint_sha256"] != expected:
            raise ValueError(f"{arm} full-validation report differs from the requested checkpoint or row count")
        if file_sha256(path) != expected or source_hashes() != recipe["source_sha256"]:
            raise ValueError("Checkpoint or runtime sources changed during evaluation")
    subprocess.run([sys.executable, "-u", "scripts/compare_spectrum_pilot.py",
        "--baseline", str(output / "baseline.json"), "--ce", str(output / "ce.json"),
        "--ranking", str(output / "ranking.json"), "--output", str(output / "comparison.json")], check=True)
    comparison = read(output / "comparison.json")
    lines = ["# Full validation comparison", "", f"Validation molecules: {count}. EMA weights; legacy beam 5.", "",
        "| Model | Top1 | Top5 | Formula |", "|---|---:|---:|---:|"]
    for arm, metrics in comparison["arms"].items():
        values = [metrics[key + "_fraction"] for key in ("top1_identity", "top5_identity", "top1_formula")]
        lines.append(f"| {arm} | " + " | ".join(f"{100 * value:.2f}%" for value in values) + " |")
    lines += ["", "| Comparison | Top1 change | Paired 95% interval | Gains / losses |", "|---|---:|---:|---:|"]
    for name, pair in comparison["comparisons"].items():
        m = pair["metrics"]["top1_identity"]
        low, high = m["paired_bootstrap_ci95_percentage_points"]
        lines.append(f"| {name} | {m['delta_percentage_points']:+.2f} pp | {low:+.2f} to {high:+.2f} pp | {m['gained_rows']} / {m['lost_rows']} |")
    lines += ["", "This broader validation includes the original pilot subset. It is not an independent holdout.",
        "The test split was not evaluated. No training was performed or weights promoted.",
        "Single training seed; graph identity ignores stereochemistry. Intervals are exploratory and do not measure seed variability.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(output / "full_validation.json", {"reports_complete": True, "pilot": str(pilot), "rows": count,
        "checkpoints": {arm: {"path": str(path), "sha256": digest} for arm, (path, digest) in models.items()}})
    names = ("baseline.json", "ce.json", "ranking.json", "comparison.json", "selection.json", "RESULTS.md", "full_validation.json")
    with zipfile.ZipFile(output / "full_validation_results.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(output / name, name)
        bundle.writestr("SHA256SUMS.json", json.dumps({name: file_sha256(output / name) for name in names}, indent=2) + "\n")
    with zipfile.ZipFile(output / "full_validation_results.zip") as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Result ZIP failed its integrity check")
    print("\n".join(lines), flush=True)
    print(f"FULL VALIDATION: Complete. Results: {output / 'full_validation_results.zip'}", flush=True)
PY
