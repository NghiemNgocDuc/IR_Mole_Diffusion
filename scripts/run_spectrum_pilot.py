"""Prepare decoding/mining experiments, then compare three training objectives."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.spectrum_audit import atomic_json, exclusive_report, validate_selection
from scripts.compare_spectrum_pilot import same_json, validate_report


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_json(path, payload):
    """Reuse immutable metadata only when every field matches."""
    path = Path(path)
    if path.exists():
        if not same_json(read_json(path), payload):
            raise ValueError(f"Existing evidence differs: {path}. Choose a new PILOT_OUTPUT.")
    else:
        atomic_json(path, payload)


def invoke(*arguments):
    print("RUN " + " ".join(map(str, arguments)), flush=True)
    subprocess.run([sys.executable, "-u", *map(str, arguments)], cwd=ROOT, check=True)


def source_hashes():
    paths = [*Path(ROOT / "diffusion").glob("*.py")]
    paths += [ROOT / name for name in (
        "train_spectrum.py", "prepare_ranking.py", "mine_ranking.py", "evaluate_spectrum.py",
        "scripts/run_spectrum_pilot.py", "scripts/compare_spectrum_pilot.py",
        "scripts/compare_decoding.py", "scripts/analyze_spectrum_errors.py")]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(paths)}


def check_budget(checkpoint, epochs, expected_steps):
    """A validation-time best/last or max_steps-truncated file is not final."""
    completed = checkpoint.get("loops", {}).get("fit_loop", {}).get("epoch_progress", {}).get("total", {}).get("completed")
    if (completed != epochs or checkpoint.get("epoch") != epochs
            or checkpoint.get("global_step") != expected_steps):
        raise ValueError("Checkpoint did not complete the exact epoch/update budget")
    if not checkpoint.get("ema_state_dict"):
        raise ValueError("Final checkpoint is missing EMA weights")


def baseline_equivalence(previous, current):
    """Ignore paths and floating scores, but require unchanged ordered identities."""
    old, new = validate_report(previous, "previous baseline"), validate_report(current, "current baseline")
    if previous["checkpoint_sha256"] != current["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint differs from the completed audit")
    if old["selection"] != new["selection"] or old["rows"] != new["rows"]:
        raise ValueError("Baseline predictions changed under the updated software. Inspect baseline.json before training; pilot stopped.")


def preflight(args):
    import numpy as np
    import pytorch_lightning as pl
    import selfies
    import torch
    from rdkit import rdBase
    from diffusion.spectrum import SpectrumStructureModel

    report = read_json(args.audit)
    validate_report(report, "completed audit")
    if report["provenance"]["spectrum_control"] != "none" or report["provenance"]["beam_size"] != 5:
        raise ValueError("Use the uncontrolled, beam-5 baseline audit")
    if report.get("checkpoint_training_diagnostic") and not args.diagnostic:
        raise ValueError("Diagnostic checkpoint cannot start an accuracy pilot")
    data = load_native_npz(args.data)
    manifest = validate_split_manifest(read_json(args.splits), data["molecule_ids"], args.data)
    split_hash = file_sha256(args.splits)
    selected = validate_selection(report["selection"], manifest, split_hash, "val")
    if any(record["reference_smiles"] != str(data["molecule_ids"][row]) for row, record in zip(selected, report["rows"])):
        raise ValueError("Baseline references differ from the current data")
    checkpoint_hash = file_sha256(args.checkpoint)
    if checkpoint_hash != report["checkpoint_sha256"]:
        raise ValueError("Baseline weights differ from the audited checkpoint")
    model = SpectrumStructureModel.load_from_checkpoint(args.checkpoint)
    if file_sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during preflight")
    for key, expected in (("data_sha256", manifest["data_sha256"]), ("split_sha256", split_hash),
                          ("preprocessing", data.get("preprocessing"))):
        if not same_json(model.run_metadata.get(key), expected):
            raise ValueError(f"Checkpoint {key} differs from the current data")
    if model.run_metadata.get("spectrum_control", "none") != "none":
        raise ValueError("Baseline was trained with a spectral control")
    if model.run_metadata.get("diagnostic") and not args.diagnostic:
        raise ValueError("Diagnostic training checkpoint rejected")
    if list(model.hparams.vocabulary) != data["vocabulary"] or not model.use_ema_weights():
        raise ValueError("Baseline vocabulary or EMA weights are unavailable/incompatible")
    if float(model.hparams.ranking_length_penalty) != 0.6:
        raise ValueError("Expected ranking length penalty 0.6")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; submit with one GPU")
    if len(manifest["splits"]["train"]) < 2:
        raise ValueError("At least two distinct training identities are required")
    # Preparation and training may be separate Slurm jobs with the same recipe.
    arguments = {key: value for key, value in vars(args).items() if key != "prepare_only"}
    result = {"schema_version": 2, "arguments": arguments, "audit_sha256": file_sha256(args.audit),
        "data_sha256": manifest["data_sha256"], "split_sha256": split_hash,
        "checkpoint_sha256": checkpoint_hash, "selection": report["selection"],
        "training_rows": len(manifest["splits"]["train"]),
        "optimizer_steps_per_arm": math.ceil(len(manifest["splits"]["train"]) / args.batch_size) * args.epochs,
        "initialization": "same baseline EMA; fresh optimizer and EMA for each arm",
        "training": {"lr": 2e-5, "weight_decay": 1e-5, "label_smoothing": 0.05,
            "ema_decay": 0.999, "ranking_weight": 0.1, "ranking_margin": 0.1,
            "ranking_length_penalty": 0.6, "precision": "32-true", "workers": 0,
            "negative_pool_size": 32, "checkpoint_policy": "final epoch EMA, not best validation loss"},
        "source_sha256": source_hashes(),
        "versions": {"python": sys.version, "torch": str(torch.__version__), "lightning": pl.__version__,
            "numpy": np.__version__, "rdkit": rdBase.rdkitVersion, "selfies": selfies.__version__,
            "cuda": torch.version.cuda, "threads": torch.get_num_threads()}}
    del model, data
    gc.collect()
    return result, report, manifest


def evaluate_checkpoint(args, output, checkpoint, selection, name, score_mode="legacy", beam_size=5):
    report_path = output / f"{name}.json"
    invoke("evaluate_spectrum.py", "--checkpoint", checkpoint, "--data", args.data, "--splits", args.splits,
        "--device", args.device, "--split", "val", "--selection", selection["policy"],
        "--limit", len(selection["rows"]), "--seed", selection["seed"], "--beam_size", beam_size,
        "--score_mode", score_mode, "--length_penalty", .6,
        "--selection_manifest", output / "selection.json", "--output", report_path,
        "--resume", "--progress_interval", 15)
    return report_path


def completed_arm(path, evidence, args, recipe, weight, negative_hash):
    import torch
    if not path.is_relative_to(Path(args.output).resolve()) or not path.is_file():
        raise ValueError("Completed arm checkpoint is missing or outside this pilot")
    if file_sha256(path) != evidence["checkpoint_sha256"]:
        raise ValueError("Completed arm checkpoint was modified")
    config_path = path.parent.parent / "resolved_config.json"
    if file_sha256(config_path) != evidence["config_sha256"]:
        raise ValueError("Completed arm configuration was modified")
    config = read_json(config_path)
    if config.get("checkpoint_source_sha256") != recipe["checkpoint_sha256"] or config.get("finetune_initial_weights") != "EMA":
        raise ValueError("Arm was not initialized from the pinned baseline EMA")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    check_budget(checkpoint, args.epochs, recipe["optimizer_steps_per_arm"])
    meta = checkpoint["run_metadata"]
    if any(meta.get(key) != recipe[key] for key in ("data_sha256", "split_sha256")) or meta.get("diagnostic"):
        raise ValueError("Final checkpoint has incompatible training provenance")
    if weight and meta.get("ranking_negatives_sha256") != negative_hash:
        raise ValueError("Ranking checkpoint used different negatives")
    if float(checkpoint["hyper_parameters"]["ranking_weight"]) != weight:
        raise ValueError("Ranking objective differs from the specified arm")


def train_arm(args, output, recipe, name, weight, negatives):
    marker = output / f"{name}_complete.json"
    negative_hash = file_sha256(negatives)
    if marker.exists():
        evidence = read_json(marker)
        path = Path(evidence["checkpoint"]).resolve()
        completed_arm(path, evidence, args, recipe, weight, negative_hash)
        print(f"Reusing verified completed {name} arm: {path}", flush=True)
        return path
    # Never resume a partly consumed shuffle. Preserve the old attempt and start
    # a fresh five-epoch arm from the immutable EMA baseline.
    index = 1
    while (output / f"{name}_attempt_{index}").exists():
        index += 1
    attempt = output / f"{name}_attempt_{index}"
    attempt.mkdir()
    command = ["train_spectrum.py", "--data", args.data, "--splits", args.splits,
        "--finetune_from", output / "baseline_start.ckpt", "--save_dir", attempt,
        "--epochs", args.epochs, "--batch_size", args.batch_size, "--seed", args.seed,
        "--lr", 2e-5, "--weight_decay", 1e-5, "--ema_decay", .999, "--label_smoothing", .05,
        "--patience", args.epochs + 1, "--num_workers", 0, "--devices", 1,
        "--accelerator", "gpu" if args.device == "cuda" else "cpu", "--precision", "32-true",
        "--ranking_weight", weight, "--ranking_margin", .1]
    if weight:
        command += ["--ranking_negatives", negatives]
    invoke(*command)
    checkpoint = attempt / "checkpoints/last.ckpt"
    evidence = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(attempt / "resolved_config.json")}
    completed_arm(checkpoint, evidence, args, recipe, weight, negative_hash)
    atomic_json(marker, evidence)
    return checkpoint


def write_results(output):
    comparison = read_json(output / "comparison.json")
    lines = ["# Stage-1 fine-tuning pilot", "", f"Paired validation rows: {comparison['rows']}. Final-epoch EMA weights.", "",
        "| Approach | Top1 | Top5 | Formula |", "|---|---:|---:|---:|"]
    for name, arm in comparison["arms"].items():
        values = [arm[key + "_fraction"] for key in ("top1_identity", "top5_identity", "top1_formula")]
        lines.append(f"| {name} | " + " | ".join(f"{100 * value:.1f}%" for value in values) + " |")
    lines += ["", "| Comparison | Top1 change | Paired 95% interval | Gains / losses |",
              "|---|---:|---:|---:|"]
    for name, pair in comparison["comparisons"].items():
        value = pair["metrics"]["top1_identity"]
        low, high = value["paired_bootstrap_ci95_percentage_points"]
        lines.append(f"| {name} | {value['delta_percentage_points']:+.2f} pp | {low:+.2f} to {high:+.2f} pp | {value['gained_rows']} / {value['lost_rows']} |")
    lines += ["", comparison["recommendation"]["text"], "", *comparison["limitations"], "",
        "No weights were automatically promoted. Inspect comparison.json and error audits before another run.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    # Explicit small report allowlist; no model weights, arrays, or old attempts.
    paths = [output / name for name in ("pilot.json", "selection.json", "baseline.json", "ce.json", "ranking.json",
        "mined.json", "comparison.json", "RESULTS.md", "ce_complete.json", "ranking_complete.json",
        "mined_complete.json", "negatives.json", "mined_negatives.json", "decoding_comparison.json",
        "decoding_comparison.md", "PREPARATION.md")]
    paths += [path for name in ("baseline", "ce", "ranking", "mined") for path in (output / f"{name}_errors").glob("*") if path.suffix in (".md", ".json", ".csv")]
    for name in ("ce", "ranking", "mined"):
        attempt = Path(read_json(output / f"{name}_complete.json")["checkpoint"]).parent.parent
        paths += [attempt / "resolved_config.json", *attempt.glob("metrics/version_*/metrics.csv")]
    archive = output / "pilot_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        checksums = {}
        for path in paths:
            relative = path.relative_to(output).as_posix()
            bundle.write(path, relative)
            checksums[relative] = file_sha256(path)
        bundle.writestr("SHA256SUMS.json", json.dumps(checksums, indent=2) + "\n")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError("Results archive failed its integrity check")
    print("\n".join(lines), flush=True)
    print(f"DOWNLOAD RESULTS: {archive}", flush=True)


def write_preparation(output):
    mining = read_json(output / "mined_negatives.json")
    lines = ["# Preparation results", "", "Baseline preserved. Preparation does not fine-tune any model.", "",
        (output / "decoding_comparison.md").read_text(encoding="utf-8"),
        "## Training-only negative pairs", "",
        f"Training inputs searched: {mining['attempted_training_rows']:,}.",
        f"Model-generated wrong alternatives: {mining['model_generated_pairs']:,}.",
        f"Remaining static fallback pairs: {mining['static_fallback_pairs']:,}.", "",
        "The third training arm uses these hybrid pairs. A wrong alternative can be mined even when the first choice is correct.",
        "Default mining searches 5,000 of 99,825 training inputs; it is a bounded pilot, not exhaustive mining.",
        "No held-out targets enter pair construction. Wider search and consistent scoring may improve or worsen accuracy.", "",
        "Review these results before launching finetune_spectrum_pilot.sh with PILOT_OUTPUT set to this directory.",
        "The optional train_compatibility.sh experiment trains only separate projection heads; it is not started here.", ""]
    (output / "PREPARATION.md").write_text("\n".join(lines), encoding="utf-8")
    paths = [output / name for name in ("pilot.json", "selection.json", "baseline.json", "negatives.json",
        "mined_negatives.json", "decoding_comparison.json", "decoding_comparison.md", "PREPARATION.md")]
    paths += sorted(output.glob("decode_*.json"))
    # Resumable row/progress files are unnecessary for the small review package.
    paths = [path for path in paths if ".rows." not in path.name and ".progress." not in path.name]
    archive = output / "preparation_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        checksums = {}
        for path in paths:
            name = path.relative_to(output).as_posix()
            bundle.write(path, name)
            checksums[name] = file_sha256(path)
        bundle.writestr("SHA256SUMS.json", json.dumps(checksums, indent=2) + "\n")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError("Preparation archive failed its integrity check")
    print("\n".join(lines), flush=True)
    print(f"DOWNLOAD PREPARATION: {archive}", flush=True)


def run(args):
    output = Path(args.output)
    with exclusive_report(output / "pilot.json"):
        print("PILOT: Checking baseline weights, saved audit, data and experiment settings", flush=True)
        recipe, previous, manifest = preflight(args)
        checked_json(output / "pilot.json", recipe)
        checked_json(output / "selection.json", recipe["selection"])
        snapshot = output / "baseline_start.ckpt"
        if not snapshot.exists():
            temporary = snapshot.with_suffix(".copying")
            shutil.copyfile(args.checkpoint, temporary)
            if file_sha256(temporary) != recipe["checkpoint_sha256"]:
                raise ValueError("Baseline changed while copying")
            os.replace(temporary, snapshot)

        def stage(message):
            if source_hashes() != recipe["source_sha256"] or file_sha256(snapshot) != recipe["checkpoint_sha256"]:
                raise ValueError("Pilot software or pinned baseline changed during the run")
            print(f"\nPILOT: {message}", flush=True)
            atomic_json(output / "status.json", {"stage": message, "updated_unix": time.time()})

        stage("1/8: Rechecking baseline on the frozen validation rows")
        baseline = evaluate_checkpoint(args, output, snapshot, recipe["selection"], "baseline")
        baseline_equivalence(previous, read_json(baseline))
        stage("2/8: Comparing decoding settings without training")
        variants = []
        for width in args.decode_widths:
            for mode in ("legacy", "consistent"):
                if width == 5 and mode == "legacy":
                    continue
                name = f"decode_{mode}_{width}"
                variants.append(evaluate_checkpoint(args, output, snapshot, recipe["selection"], name, mode, width))
        invoke("scripts/compare_decoding.py", "--baseline", baseline, "--variants", *variants,
               "--output", output / "decoding_comparison.json")
        stage("3/8: Preparing static and model-generated training-only negatives")
        negatives = output / "ranking_negatives.npz"
        negative_meta = output / "negatives.json"
        invoke("prepare_ranking.py", "--data", args.data, "--splits", args.splits,
               "--output", negatives, "--seed", args.seed, "--max_pool", 32, "--resume")
        from diffusion.ranking import load_negative_rows
        load_negative_rows(negatives, manifest["splits"]["train"], recipe["data_sha256"], recipe["split_sha256"])
        prepared = read_json(negatives.with_suffix(".json"))
        if prepared["output_sha256"] != file_sha256(negatives) or prepared["configuration"]["seed"] != args.seed or prepared["configuration"]["max_pool"] != 32:
            raise ValueError("Existing negative preparation has different settings or hash")
        checked_json(negative_meta, prepared)
        mined_negatives = output / "mined_negatives.npz"
        mining_command = ["mine_ranking.py", "--checkpoint", snapshot, "--data", args.data, "--splits", args.splits,
            "--fallback", negatives, "--output", mined_negatives, "--limit", args.mine_limit,
            "--seed", args.seed, "--beam_size", 5, "--device", args.device, "--resume"]
        if args.diagnostic:
            mining_command.append("--allow_diagnostic")
        invoke(*mining_command)
        write_preparation(output)
        if args.prepare_only:
            stage("Preparation complete; no fine-tuning performed")
            return output
        if read_json(output / "mined_negatives.json")["model_generated_pairs"] == 0:
            raise ValueError("Mining produced no usable model-generated pairs. Inspect PREPARATION.md before training.")
        stage("4/8: Ordinary CE fine-tuning")
        ce = train_arm(args, output, recipe, "ce", 0., negatives)
        evaluate_checkpoint(args, output, ce, recipe["selection"], "ce")
        stage("5/8: CE plus static-negative ranking fine-tuning")
        ranking = train_arm(args, output, recipe, "ranking", .1, negatives)
        evaluate_checkpoint(args, output, ranking, recipe["selection"], "ranking")
        stage("6/8: CE plus model-generated-negative ranking fine-tuning")
        mined = train_arm(args, output, recipe, "mined", .1, mined_negatives)
        evaluate_checkpoint(args, output, mined, recipe["selection"], "mined")
        stage("7/8: Comparing paired predictions and auditing errors")
        invoke("scripts/compare_spectrum_pilot.py", "--baseline", baseline, "--ce", output / "ce.json",
               "--ranking", output / "ranking.json", "--mined", output / "mined.json", "--output", output / "comparison.json")
        from scripts.analyze_spectrum_errors import analyze_report, write_outputs, OUTPUT_NAMES
        for name in ("baseline", "ce", "ranking", "mined"):
            source, destination = output / f"{name}.json", output / f"{name}_errors"
            if (destination / "analysis.json").exists():
                if read_json(destination / "analysis.json").get("report_sha256") != file_sha256(source) or not all((destination / item).is_file() for item in OUTPUT_NAMES):
                    raise ValueError(f"Existing error analysis differs or is incomplete: {destination}")
            else:
                cache = ROOT / "outputs/dataset_review/chemical_rows.csv"
                use_cache = cache.is_file() and not args.diagnostic
                analysis, rows = analyze_report(read_json(source), args.data, args.splits, str(cache) if use_cache else None)
                analysis.update(report_path=str(source), report_sha256=file_sha256(source))
                write_outputs(destination, analysis, rows)
        stage("8/8: Writing results ZIP")
        write_results(output)
        stage("Complete")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="evaluation/baseline_audit_66782537/baseline.json")
    parser.add_argument("--checkpoint", default="runs/spectrum/checkpoints/best.ckpt")
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prepare_only", action="store_true", help="Compare decoding and mine training pairs, then stop before fine-tuning")
    parser.add_argument("--mine_limit", type=int, default=5000)
    parser.add_argument("--decode_widths", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--diagnostic", action="store_true", help="Allow synthetic smoke-test checkpoints; not an accuracy experiment")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.mine_limit) < 1 or args.seed < 0:
        parser.error("Epochs/batch size/mining limit must be positive; seed nonnegative")
    if 5 not in args.decode_widths or len(set(args.decode_widths)) != len(args.decode_widths) or any(not 5 <= width <= 100 for width in args.decode_widths):
        parser.error("Decode widths must be unique, include 5 and lie between 5 and 100")
    args.decode_widths = sorted(args.decode_widths)
    for name in ("audit", "checkpoint", "data", "splits", "output"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if any(Path(getattr(args, name)).is_relative_to(Path(args.output)) for name in ("audit", "checkpoint", "data", "splits")):
        parser.error("Pilot output must be separate from its inputs")
    try:
        return run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
