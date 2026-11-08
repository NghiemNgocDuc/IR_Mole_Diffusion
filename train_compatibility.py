"""Train a separate frozen-IR compatibility experiment, then rerank a fixed validation pool."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from diffusion.compatibility import (cache_spectrum_features, fit_head, rerank_candidates,
                                      training_fingerprints)
from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.progress import EvaluationProgress
from diffusion.ranking import canonical_identity
from diffusion.spectrum import SpectrumStructureModel
from diffusion.spectrum_audit import atomic_json, row_record, summarize_records, validate_selection
from scripts.compare_spectrum_pilot import (BOOTSTRAP_REPLICAS, BOOTSTRAP_SEED, paired_metrics,
                                            same_json, validate_report)

ROOT = Path(__file__).resolve().parent
FEATURE_SOURCES = ("diffusion/encoder.py", "diffusion/model.py", "diffusion/data.py")


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Compatibility output already exists; use a fresh output directory. Partial runs are not resumed.")
    baseline_bytes = Path(args.baseline_report).read_bytes()
    baseline = json.loads(baseline_bytes.decode("utf-8"))
    checked = validate_report(baseline, "baseline candidate pool")
    if checked["provenance"]["spectrum_control"] != "none":
        raise ValueError("Compatibility training requires an uncontrolled baseline candidate report.")
    data = load_native_npz(args.data)
    manifest = validate_split_manifest(json.loads(Path(args.splits).read_text(encoding="utf-8")), data["molecule_ids"], args.data)
    split_hash, checkpoint_hash = file_sha256(args.splits), file_sha256(args.checkpoint)
    if any(baseline[key] != value for key, value in (("data_sha256", manifest["data_sha256"]),
            ("split_sha256", split_hash), ("checkpoint_sha256", checkpoint_hash))):
        raise ValueError("Baseline report does not match checkpoint/data/split hashes.")
    val_rows = validate_selection(baseline["selection"], manifest, split_hash, "val")
    if any(record["reference_smiles"] != canonical_identity(data["molecule_ids"][row])
           for record, row in zip(checked["rows"], val_rows)):
        raise ValueError("Baseline report references differ from current validation identities.")
    for name in FEATURE_SOURCES:
        if checked["provenance"]["source_sha256"][name] != file_sha256(ROOT / name):
            raise ValueError(f"Baseline feature-extraction source differs: {name}; regenerate the baseline report.")
    model = SpectrumStructureModel.load_from_checkpoint(args.checkpoint)
    if file_sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError("Baseline checkpoint changed during loading.")
    for key, expected in (("data_sha256", manifest["data_sha256"]), ("split_sha256", split_hash),
                          ("preprocessing", data.get("preprocessing"))):
        if not same_json(model.run_metadata.get(key), expected) or not same_json(checked["provenance"][key], expected):
            raise ValueError(f"Checkpoint/report {key} differs from the data.")
    if list(model.hparams.vocabulary) != data["vocabulary"] or checked["provenance"]["vocabulary"] != data["vocabulary"]:
        raise ValueError("Vocabulary mismatch.")
    if model.run_metadata.get("spectrum_control", "none") != "none" or not model.use_ema_weights():
        raise ValueError("Frozen encoder requires an uncontrolled baseline with EMA weights.")
    diagnostic = bool(model.run_metadata.get("diagnostic") or baseline.get("checkpoint_training_diagnostic"))
    if diagnostic and not args.diagnostic:
        raise ValueError("Diagnostic checkpoint rejected; --diagnostic permits software testing only.")
    diagnostic = diagnostic or args.diagnostic
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; submit the experiment with one GPU.")
    model.to(args.device).eval().requires_grad_(False)
    training_rows = manifest["splits"]["train"]
    # The scorer receives no validation identities until training is complete.
    fingerprints, identities = training_fingerprints(data, training_rows, manifest["splits"]["train"])
    sources = {name: file_sha256(ROOT / name) for name in (*FEATURE_SOURCES, "diffusion/compatibility.py",
               "diffusion/ranking.py", "diffusion/spectrum.py", "diffusion/spectrum_audit.py",
               "scripts/compare_spectrum_pilot.py", "train_compatibility.py")}
    import hashlib
    configuration = {"schema_version": 1, "arguments": vars(args), "data_sha256": manifest["data_sha256"],
        "split_sha256": split_hash, "baseline_checkpoint_sha256": checkpoint_hash,
        "baseline_report_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "baseline_candidate_provenance": baseline["provenance"], "selection": baseline["selection"],
        "training_rows": training_rows, "training_labels_split": "train", "held_out_labels_used_for_training": False,
        "encoder": "frozen baseline EMA; masked mean pooling", "temperature": 0.1,
        "fingerprint": {"algorithm": "Morgan", "radius": 2, "bits": 2048, "chirality": False},
        "source_sha256": sources, "diagnostic": diagnostic,
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "rdkit": rdBase.rdkitVersion,
                     "cuda": torch.version.cuda},
        "numerics": {"torch_num_threads": torch.get_num_threads(),
                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "configuration.json", configuration)
    with EvaluationProgress(output / "progress", interval=args.progress_interval) as progress:
        features = cache_spectrum_features(model.encoder, data, training_rows, args.device,
                                          args.batch_size, progress, "Caching training-only IR features")
        head, history = fit_head(features, fingerprints, identities, device=args.device, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, lr=args.lr, projection_dim=args.projection_dim, progress=progress)
        atomic_json(output / "training_history.json", history)
        del features, fingerprints, identities
        temporary = output / "compatibility.pt.tmp"
        torch.save({"schema_version": 1, "model_class": "CompatibilityHead", "spec": head.spec,
                    "state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                    "configuration": configuration, "training_history": history}, temporary)
        temporary.replace(output / "compatibility.pt")
        val_features = cache_spectrum_features(model.encoder, data, val_rows, args.device,
                                              args.batch_size, progress, "Caching held-out IR for inference only")
        progress.start("Reranking fixed validation candidate pools", len(val_rows), "molecules")
        records = []
        for index, record in enumerate(baseline["rows"]):
            candidates = rerank_candidates(head, val_features[index], record["candidates"], args.device)
            # References are used only here, after candidate ordering, for evaluation.
            records.append(row_record(record["row"], record["reference_smiles"], record["reference_formula"], candidates))
            progress.update(index + 1)
        if any(old["topk_identity"] != new["topk_identity"] for old, new in zip(checked["rows"], records)):
            raise ValueError("Reranking changed candidate-pool coverage.")
        for name, digest in sources.items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError(f"Source changed during experiment: {name}; use a fresh output.")
        if file_sha256(args.baseline_report) != configuration["baseline_report_sha256"]:
            raise ValueError("Baseline report changed during experiment.")
        pair = paired_metrics(checked["rows"], records, BOOTSTRAP_SEED)
        summary = summarize_records(records)
        comparison = {"schema_version": 1, "task": "separate experimental IR–molecule compatibility reranker",
            "split": "val", "rows": len(records), "selection": baseline["selection"], "diagnostic": diagnostic,
            "baseline_report_sha256": configuration["baseline_report_sha256"],
            "compatibility_checkpoint_sha256": file_sha256(output / "compatibility.pt"),
            "baseline": checked["summary"], "compatibility": summary, "paired": pair,
            "bootstrap_replicas": BOOTSTRAP_REPLICAS, "candidate_pool_unchanged": True,
            "recommendation": "diagnostic_only" if diagnostic else (
                "promising_pending_broader_validation" if pair["meets_provisional_criterion"] else "keep_baseline"),
            "weights_automatically_promoted": False,
            "limitations": ["Experimental association scorer; no forward IR calculation or physical/3D validation.",
                "Cosine scores are not calibrated probabilities of correctness; fingerprint collisions are possible.",
                "Frozen decoder candidates bound this experiment; missing identities cannot be recovered.",
                "The head is trained on training identities only; molecular features ignore stereochemistry.",
                "The paired interval covers these validation rows, not training-seed variability or multiple-comparison correction.",
                "No validation-based checkpoint selection, score mixing, or hyperparameter search was performed.",
                "Confirm improvements on broader validation and untouched test data before production use."]}
        atomic_json(output / "compatibility_predictions.json", {"schema_version": 1, "complete": True,
            "ranking": "compatibility cosine score; decoder order breaks ties", "selection": baseline["selection"],
            "baseline_report_sha256": configuration["baseline_report_sha256"], **summary, "rows": records})
        atomic_json(output / "compatibility_comparison.json", comparison)
        lines = ["# Experimental compatibility reranking", "", f"Validation rows: {len(records)}. Diagnostic: {diagnostic}.", "",
                 "| Score | Baseline | Compatibility |", "|---|---:|---:|"]
        for key in ("top1_identity", "top5_identity", "top1_formula"):
            lines.append(f"| {key} | {100 * checked['summary'][key + '_fraction']:.1f}% | {100 * summary[key + '_fraction']:.1f}% |")
        metric = pair["metrics"]["top1_identity"]
        lines += ["", f"Top1 change: {metric['delta_percentage_points']:+.2f} percentage points; paired 95% interval: {metric['paired_bootstrap_ci95_percentage_points']}.",
                  pair["uncertainty"], "", f"Recommendation: {comparison['recommendation']}. No weights were promoted.", "", *comparison["limitations"], ""]
        (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Compatibility experiment complete: {output / 'RESULTS.md'}", flush=True)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--baseline_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--progress_interval", type=float, default=15)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 2 or args.seed < 0 or args.projection_dim < 2
            or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.progress_interval) or args.progress_interval < 0):
        parser.error("Use positive epochs/lr, batch/projection size >=2, and nonnegative seed/progress interval.")
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
