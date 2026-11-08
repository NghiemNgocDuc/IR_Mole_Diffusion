"""Paired validation comparison; never overwrite or automatically promote weights."""
import argparse
import json
from pathlib import Path


def compare_reports(baseline, candidate):
    if baseline["data_sha256"] != candidate["data_sha256"] or baseline["identity_metric"] != candidate["identity_metric"]:
        raise ValueError("Dataset hashes and identity definitions must agree.")
    for key in ("split", "limit", "beam_size", "spectrum_control"):
        if baseline["configuration"].get(key) != candidate["configuration"].get(key):
            raise ValueError(f"Evaluation setting {key} differs.")
    if baseline["configuration"]["split"] != "val":
        raise ValueError("Fine-tuning decisions must use validation results, not the test split.")
    before, after = baseline["rows"], candidate["rows"]
    if not before or len(before) != len(after):
        raise ValueError("Both reports must evaluate the same nonempty rows.")
    for left, right in zip(before, after):
        if (left["row"], left["reference_smiles"]) != (right["row"], right["reference_smiles"]):
            raise ValueError("Evaluated row order or target identities differ.")
    metrics = {}
    for key in ("top1_identity", "topk_identity", "top1_formula"):
        gains = sum(not left[key] and right[key] for left, right in zip(before, after))
        losses = sum(left[key] and not right[key] for left, right in zip(before, after))
        metrics[key] = {"baseline_fraction": sum(left[key] for left in before)/len(before),
            "candidate_fraction": sum(right[key] for right in after)/len(after),
            "gained_rows": gains, "lost_rows": losses, "delta_percentage_points": 100*(gains-losses)/len(before)}
    improved = metrics["top1_identity"]["delta_percentage_points"] > 0 and metrics["topk_identity"]["delta_percentage_points"] >= 0
    return {"rows": len(before), "split": "val", "paired_metrics": metrics,
        "recommendation": ("Preliminary improvement; confirm on the full validation set before selecting this checkpoint."
                           if improved else "Keep the baseline; this comparison does not demonstrate a useful ranking improvement."),
        "weights_automatically_promoted": False,
        "limitations": ["One recipe and seed; improvement is not guaranteed on other data.",
                        "The existing identity split can share tautomer families; this is not a novel-family benchmark.",
                        "No experimental IR or 3D performance claim follows from this graph comparison."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_reports(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text()))
    result["source_reports"] = vars(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
