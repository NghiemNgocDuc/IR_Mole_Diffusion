#!/bin/bash -l
#SBATCH --job-name=ir_baseline_audit
#SBATCH --output=logs/audit-%j.out
#SBATCH --error=logs/audit-%j.err
#SBATCH --time=04:00:00
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

conda_runner=(conda run -n "${CONDA_ENV:-base}" --no-capture-output)
audit_output="${EVAL_OUTPUT:-evaluation/baseline_audit_${SLURM_JOB_ID:?}}"
audit_checkpoint="${SPECTRUM_CHECKPOINT:-runs/spectrum/checkpoints/best.ckpt}"
audit_selection="${EVAL_SELECTION:-random}"
audit_limit="${EVAL_LIMIT:-1000}"
audit_seed="${EVAL_SEED:-20261105}"
audit_data="data/processed/squirl.npz"
audit_splits="data/processed/splits.json"

for audit_input in "$audit_checkpoint" "$audit_data" "$audit_splits"; do
    if [[ ! -s "$audit_input" ]]; then
        echo "Missing required input: $audit_input" >&2
        exit 1
    fi
done
mkdir -p "$audit_output"
echo "Baseline audit output: $audit_output"
echo "Selection: $audit_selection; requested limit: $audit_limit; seed: $audit_seed; beam width: 5"
echo "Model weights are read-only. This job evaluates the existing stage-1 checkpoint."
"${conda_runner[@]}" python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in the selected environment'; print('GPU:', torch.cuda.get_device_name(0), flush=True)"

"${conda_runner[@]}" python -u evaluate_spectrum.py \
    --checkpoint "$audit_checkpoint" --device cuda \
    --data "$audit_data" --splits "$audit_splits" --split val \
    --selection "$audit_selection" --limit "$audit_limit" --seed "$audit_seed" \
    --selection_manifest "$audit_output/selection.json" --beam_size 5 \
    --output "$audit_output/baseline.json" --resume --progress_interval 15

# Optional cached chemical annotations enable train-tautomer-family diagnostics.
# The analyzer checks dataset/split hashes, RDKit version, and row identities.
audit_chemical_rows=""
if [[ -s outputs/dataset_review/chemical_rows.csv && -s outputs/dataset_review/chemical_split_audit.json ]]; then
    audit_chemical_rows="outputs/dataset_review/chemical_rows.csv"
fi

analyze_saved_report() {
    "${conda_runner[@]}" python - "$1" "$2" "$audit_data" "$audit_splits" "$audit_chemical_rows" <<'PY'
import json
from pathlib import Path
import sys

from scripts.analyze_spectrum_errors import OUTPUT_NAMES, analyze_report, file_sha256, write_outputs

source, destination, data, splits, chemical_rows = sys.argv[1:]
destination = Path(destination)
source_hash, data_hash, split_hash = map(file_sha256, (source, data, splits))
existing = destination / "analysis.json"
if existing.exists():
    metadata = json.loads(existing.read_text(encoding="utf-8"))
    expected = {"report_sha256": source_hash, "data_sha256": data_hash, "split_sha256": split_hash}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"Existing analysis has different input hashes: {destination}. Choose a new EVAL_OUTPUT.")
    if not all((destination / name).is_file() for name in OUTPUT_NAMES):
        raise SystemExit(f"Existing analysis is incomplete: {destination}. Choose a new EVAL_OUTPUT; no files were removed.")
    if chemical_rows and (metadata.get("chemical_cache") or {}).get("sha256") != file_sha256(chemical_rows):
        raise SystemExit(f"Existing analysis used different chemical annotations: {destination}. Choose a new EVAL_OUTPUT.")
    print(f"Verified existing analysis: {destination}", flush=True)
else:
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    analysis, rows = analyze_report(report, data, splits, chemical_rows or None)
    analysis.update(report_path=source, report_sha256=source_hash)
    write_outputs(destination, analysis, rows, (source, data, splits, chemical_rows))
    print(f"Wrote {destination / 'analysis.md'}", flush=True)
    print(json.dumps(analysis["overall"], indent=2), flush=True)
PY
}

analyze_saved_report "$audit_output/baseline.json" "$audit_output/error_audit"
if [[ -s evaluation/spectrum_val.json ]]; then
    echo "Auditing the earlier validation report for comparison."
    analyze_saved_report evaluation/spectrum_val.json "$audit_output/legacy_audit"
else
    echo "Earlier full report evaluation/spectrum_val.json was not found; the new baseline audit is available."
fi

# The archive is a sibling of the output directory and contains an explicit
# report allowlist, so model weights and datasets are never added recursively.
"${conda_runner[@]}" python - "$audit_output" "${SLURM_JOB_ID:?}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from scripts.analyze_spectrum_errors import OUTPUT_NAMES

output, job_id = Path(sys.argv[1]).resolve(), sys.argv[2]
archive = output.with_name(f"{output.name}_results_{job_id}.zip")
report = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
if report.get("complete") is not True:
    raise SystemExit("Baseline evaluation is not complete; resume it before creating the final results archive.")
sources = [(output / name, name) for name in ("baseline.json", "selection.json")]
sources += [(output / "error_audit" / name, f"error_audit/{name}") for name in OUTPUT_NAMES]
legacy = Path("evaluation/spectrum_val.json")
if legacy.is_file():
    sources.append((legacy, "legacy/spectrum_val.json"))
    sources += [(output / "legacy_audit" / name, f"legacy_audit/{name}") for name in OUTPUT_NAMES]
if any(not source.is_file() for source, _ in sources):
    raise SystemExit("An expected report is missing; final results archive was not created.")
payloads = {name: source.read_bytes() for source, name in sources}
checksums = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
if json.loads(payloads["error_audit/analysis.json"]).get("report_sha256") != checksums["baseline.json"]:
    raise SystemExit("Baseline report changed after analysis; final results archive was not created.")
if legacy.is_file() and json.loads(payloads["legacy_audit/analysis.json"]).get("report_sha256") != checksums["legacy/spectrum_val.json"]:
    raise SystemExit("Legacy report changed after analysis; final results archive was not created.")
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
    for name, content in payloads.items():
        bundle.writestr(name, content)
    bundle.writestr("SHA256SUMS.json", json.dumps(checksums, indent=2) + "\n")
    bundle.writestr("README.txt", "Completed stage-1 validation baseline and per-molecule error audit.\n"
                    "Read error_audit/analysis.md first. Legacy results cover an earlier subset.\n"
                    "No training was performed. No model weights or dataset arrays are included.\n")
with zipfile.ZipFile(archive) as bundle:
    if bundle.testzip() is not None:
        raise SystemExit("Results archive failed its integrity check.")
print(f"DOWNLOAD RESULTS: {archive}", flush=True)
PY

echo "Baseline audit complete. Open $audit_output/error_audit/analysis.md or download the printed results ZIP."
