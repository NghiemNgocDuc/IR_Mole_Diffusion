"""Package the supported source and data without Windows runtime/old weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/unity_training_bundle.zip")
    contents = parser.add_mutually_exclusive_group()
    contents.add_argument("--source_only", action="store_true", help="Omit all data")
    contents.add_argument("--visualization_only", action="store_true", help="Only the three optional picture files; no training source, data, or weights")
    contents.add_argument("--progress_only", action="store_true", help="Only coordinate-evaluation progress files; no data, model, or weights")
    contents.add_argument("--improvements_only", action="store_true", help="Interactive viewer and optional graph fine-tuning update; no data or weights")
    contents.add_argument("--viewer_only", action="store_true", help="Only rotating color-key viewer and plotting source; no training source or weights")
    contents.add_argument("--baseline_audit_only", action="store_true", help="Resumable representative stage-1 evaluation and error audit; no model weights or spectral arrays")
    contents.add_argument("--complete_update", action="store_true", help="One cumulative viewer, evaluation-progress, and baseline-audit upload for an existing trained Unity installation")
    contents.add_argument("--finetune_pilot_only", action="store_true", help="Controlled CE versus ranking fine-tuning pilot for an existing trained installation; no data or weights")
    contents.add_argument("--accuracy_update_v2", action="store_true", help="Preparation-first decoding, mined-negative fine-tuning and optional compatibility scorer; existing installation required")
    args = parser.parse_args()
    files = [*Path(".").glob("*.py"), *Path(".").glob("*.sh"),
             Path("README.md"), Path("REPO_AUDIT.md"), Path("requirements.txt"),
             Path("EVALUATION_PROGRESS_UPDATE.md"),
             Path("VIEWER_AND_FINETUNE.md"),
             Path("VIEWER_3D_UPDATE.md"),
             Path("requirements-visualization.txt"),
             Path("pytest.ini"), Path(".gitignore"), Path(".gitattributes")]
    for folder in ("diffusion", "diffusion_v2", "configs", "docs", "scripts", "tests", "legacy", "viewer_assets"):
        files.extend(path for path in Path(folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    if args.finetune_pilot_only or args.accuracy_update_v2:
        files = [Path(name) for name in (
            "FINETUNE_PILOT.md", "finetune_spectrum_pilot.sh", "scripts/run_spectrum_pilot.py",
            "scripts/compare_spectrum_pilot.py", "scripts/analyze_spectrum_errors.py",
            "prepare_ranking.py", "train_spectrum.py", "evaluate_spectrum.py",
            "diffusion/ranking.py", "diffusion/spectrum.py", "diffusion/training.py",
            "diffusion/progress.py", "diffusion/spectrum_audit.py",
            "docs/results/stage1_baseline_66782537.md",
            "tests/test_spectrum_pilot.py", "tests/test_spectrum_pilot_comparison.py", "tests/test_ranking.py")]
        # The pilot runner now includes preparation and the third training arm.
        files += [Path(name) for name in (
            "prepare_spectrum_pilot.sh", "mine_ranking.py", "scripts/compare_decoding.py",
            "train_compatibility.py", "train_compatibility.sh", "diffusion/compatibility.py",
            "COMPATIBILITY_EXPERIMENT.md", "ACCURACY_UPDATE_V2.md",
            "tests/test_spectrum_decoding.py", "tests/test_mined_ranking.py", "tests/test_compatibility.py", "tests/test_prepare_ranking.py",
            "scripts/compare_rank_results.py", "pytest.ini", "docs/ACCURACY_LITERATURE_REVIEW.md")]
    elif args.complete_update:
        files = [Path(name) for name in (
            "START_HERE_UNITY_UPDATE.md",
            "make_viewer.py", "plot_prediction.py", "predict_example.sh",
            "requirements-visualization.txt", "VIEWER_3D_UPDATE.md",
            "evaluate_diffusion.py", "diffusion/evaluation.py", "EVALUATION_PROGRESS_UPDATE.md",
            "evaluate_spectrum.py", "diffusion/spectrum_audit.py", "diffusion/progress.py",
            "scripts/analyze_spectrum_errors.py", "audit_spectrum_baseline.sh",
            "BASELINE_AUDIT_UPDATE.md", "docs/DATASET_AND_OUTCOME_AUDIT.md",
            "outputs/dataset_review/chemical_rows.csv", "outputs/dataset_review/chemical_split_audit.json")]
        files.extend(path for path in Path("viewer_assets").rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    elif args.baseline_audit_only:
        files = [Path(name) for name in (
            "evaluate_spectrum.py", "diffusion/spectrum_audit.py", "diffusion/progress.py",
            "scripts/analyze_spectrum_errors.py", "audit_spectrum_baseline.sh",
            "BASELINE_AUDIT_UPDATE.md", "docs/DATASET_AND_OUTCOME_AUDIT.md",
            "outputs/dataset_review/chemical_rows.csv", "outputs/dataset_review/chemical_split_audit.json")]
    elif args.viewer_only:
        files = [Path(name) for name in ("make_viewer.py", "plot_prediction.py", "VIEWER_3D_UPDATE.md")]
        files.extend(path for path in Path("viewer_assets").rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    elif args.visualization_only:
        files = [Path("plot_prediction.py"), Path("predict_example.sh"), Path("requirements-visualization.txt")]
    elif args.progress_only:
        files = [Path("evaluate_diffusion.py"), Path("diffusion/evaluation.py"),
                 Path("diffusion/progress.py"), Path("EVALUATION_PROGRESS_UPDATE.md")]
    elif args.improvements_only:
        files = [Path(name) for name in (
            "make_viewer.py", "plot_prediction.py", "predict_example.sh",
            "diffusion/ranking.py", "diffusion/spectrum.py", "diffusion/training.py",
            "diffusion/progress.py", "prepare_ranking.py", "train_spectrum.py",
            "configs/spectrum_rank_finetune.json", "finetune_spectrum.sh",
            "compare_spectrum.sh", "scripts/compare_rank_results.py",
            "evaluate_spectrum.py", "VIEWER_AND_FINETUNE.md", "VIEWER_3D_UPDATE.md")]
        files.extend(path for path in Path("viewer_assets").rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    elif not args.source_only:
        files.extend(Path(name) for name in ["data/SQuIRL/squirl_ir_3d.npz",
            "data/processed/squirl.npz", "data/processed/splits.json",
            "data/processed/squirl.metadata.json"])
    files = sorted(set(files))
    if any(not path.is_file() for path in files):
        parser.error("Missing source or prepared data; finish conversion and verification first.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    purpose = "Unity two-stage training; no useful trained weights included"
    if args.finetune_pilot_only or args.accuracy_update_v2:
        purpose = "Preparation-first decoding comparison and training-only negative mining; three matched-budget fine-tuning arms and separate compatibility experiment; no data or weights or automatic promotion"
    elif args.complete_update:
        purpose = "Cumulative offline 3D viewer, evaluation progress, and representative baseline audit for existing Unity training; no datasets, checkpoints, or fine-tuning launchers"
    elif args.baseline_audit_only:
        purpose = "Representative stage-1 validation and per-molecule error audit; cached chemical annotations, no spectral arrays or model weights; no fine-tuning"
    elif args.viewer_only:
        purpose = "Interactive color-key molecule viewer and explicit 2D graph to generated 3D conformers; no training data or weights"
    elif args.visualization_only:
        purpose = "Optional picture files compatible with the existing Unity prediction CLI; no data or weights"
    elif args.progress_only:
        purpose = "Live coordinate-evaluation progress; no data, model, or weights"
    elif args.improvements_only:
        purpose = "Offline rotating prediction viewer and experimental train-only graph fine-tuning; no data or weights"
    manifest = {"purpose": purpose,
                "files": {path.as_posix(): digest(path) for path in files}}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.as_posix())
        manifest_name = ("VISUALIZATION_MANIFEST.json" if args.visualization_only else
                         "PROGRESS_MANIFEST.json" if args.progress_only else "BUNDLE_MANIFEST.json")
        if args.accuracy_update_v2:
            manifest_name = "ACCURACY_V2_MANIFEST.json"
        elif args.finetune_pilot_only:
            manifest_name = "FINETUNE_PILOT_MANIFEST.json"
        elif args.complete_update:
            manifest_name = "COMPLETE_UPDATE_MANIFEST.json"
        elif args.baseline_audit_only:
            manifest_name = "BASELINE_AUDIT_MANIFEST.json"
        elif args.improvements_only:
            manifest_name = "IMPROVEMENTS_MANIFEST.json"
        elif args.viewer_only:
            manifest_name = "VIEWER_MANIFEST.json"
        archive.writestr(manifest_name, json.dumps(manifest, indent=2) + "\n")
    # Verify archive contents and each original-file hash before reporting success.
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC failed: {bad}")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive hash failed: {name}")
    print(json.dumps({"bundle": str(output), "bytes": output.stat().st_size,
                      "files": len(files), "sha256": digest(output)}, indent=2))


if __name__ == "__main__":
    main()
