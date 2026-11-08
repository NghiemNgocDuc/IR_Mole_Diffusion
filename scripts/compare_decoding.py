"""Paired decoding-only comparison: one checkpoint and frozen validation rows."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.data import file_sha256
from diffusion.spectrum_audit import atomic_json
from scripts.compare_spectrum_pilot import validate_report, same_json, paired_metrics


def compare_decoding(baseline, variants):
    first = validate_report(baseline, "baseline")
    if (first["provenance"]["beam_size"] != 5
            or first["provenance"]["decode_protocol"] != "legacy_generate_v1_valid_unique_candidates"):
        raise ValueError("Decoding comparison requires the legacy beam-5 baseline")
    permitted = {"beam_size", "length_penalty", "decode_protocol"}
    fixed = lambda report: {key: value for key, value in report["provenance"].items() if key not in permitted}
    result = {}
    for index, (name, report) in enumerate(variants.items()):
        arm = validate_report(report, name)
        if not same_json(first["selection"], arm["selection"]) or not same_json(fixed(first), fixed(arm)):
            raise ValueError("Decoding comparisons require the same checkpoint, rows and fixed inference provenance")
        if first["identity_metric"] != arm["identity_metric"]:
            raise ValueError("Identity metric differs")
        if any((a["reference_smiles"], a["reference_formula"]) != (b["reference_smiles"], b["reference_formula"])
               for a, b in zip(first["rows"], arm["rows"])):
            raise ValueError("Reference identities differ")
        if arm["provenance"]["decode_protocol"] not in ("legacy_generate_v1_valid_unique_candidates", "consistent_generate_v1_valid_unique_candidates"):
            raise ValueError("Unrecognized decoding protocol")
        result[name] = {"configuration": {key: arm["provenance"][key] for key in sorted(permitted)},
            "summary": arm["summary"], "paired": paired_metrics(first["rows"], arm["rows"], 20261105 + 10 * index)}
    diagnostic = (bool(baseline.get("checkpoint_training_diagnostic"))
                  or first["provenance"]["spectrum_control"] != "none")
    return {"schema_version": 1, "task": "decoding only, no training", "diagnostic_only": diagnostic,
        "rows": len(first["rows"]),
        "selection": first["selection"], "checkpoint_sha256": first["provenance"]["checkpoint_sha256"],
        "baseline": first["summary"], "variants": result, "fixed_reported_k": 5,
        "automatically_selected_protocol": None,
        "limitations": ["Wider beam search can improve or worsen ordering; compare fixed Top5, not TopK with changing K.",
            "All variants use the same validation rows; exploratory intervals are not adjusted for multiple comparisons.",
            "No decoder or checkpoint is automatically promoted. Fine-tuning controls continue to use legacy beam 5."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    paths = [Path(args.baseline), *map(Path, args.variants)]
    output = Path(args.output)
    if output.resolve() in {path.resolve() for path in paths} or len({path.stem for path in paths[1:]}) != len(paths[1:]):
        parser.error("Output collides with inputs or variant names are not unique")
    result = compare_decoding(json.loads(paths[0].read_text()), {path.stem: json.loads(path.read_text()) for path in paths[1:]})
    result["source_reports"] = {str(path): file_sha256(path) for path in paths}
    if output.exists():
        if not same_json(json.loads(output.read_text()), result):
            raise ValueError("Existing decoding comparison differs; use a new output")
    else:
        atomic_json(output, result)
    lines = ["# Decoding comparison", "", "Same checkpoint and validation rows; no training performed.", "",
        "| Variant | Top1 | Fixed Top5 | Top1 change |", "|---|---:|---:|---:|"]
    if result["diagnostic_only"]:
        lines[2] = "DIAGNOSTIC ONLY: software/control check, not an accuracy experiment. Same checkpoint and validation rows."
    lines.append(f"| legacy beam 5 | {100*result['baseline']['top1_identity_fraction']:.1f}% | {100*result['baseline']['top5_identity_fraction']:.1f}% | — |")
    for name, item in result["variants"].items():
        lines.append(f"| {name} | {100*item['summary']['top1_identity_fraction']:.1f}% | {100*item['summary']['top5_identity_fraction']:.1f}% | {item['paired']['metrics']['top1_identity']['delta_percentage_points']:+.2f} pp |")
    lines += ["", *result["limitations"], ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return result


if __name__ == "__main__":
    main()
