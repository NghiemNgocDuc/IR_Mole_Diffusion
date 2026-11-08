"""Strict paired comparison of baseline, CE-only, static, and optional mined ranking.

All reports must come from the same evaluation protocol and ordered
validation selection. Checkpoint files are never loaded, changed, or promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.spectrum_audit import METRICS, atomic_json, exclusive_report, row_record, summarize_records


BOOTSTRAP_REPLICAS = 2000
BOOTSTRAP_SEED = 20261105
PAIRED_METRICS = ("top1_identity", "top5_identity", "top1_formula")
CHECKPOINT_PROVENANCE = {"checkpoint_sha256", "checkpoint_training_diagnostic"}
REQUIRED_PROVENANCE = {
    "data_sha256", "split_sha256", "checkpoint_sha256", "ema_weights", "preprocessing", "vocabulary",
    "split", "beam_size", "length_penalty", "decode_protocol", "spectrum_control", "control_seed",
    "device", "torch_num_threads", "cuda_version", "float32_matmul_precision", "cuda_matmul_allow_tf32",
    "versions", "source_sha256",
}
SOURCE_FILES = {"evaluate_spectrum.py", "diffusion/spectrum_audit.py", "diffusion/spectrum.py",
                "diffusion/encoder.py", "diffusion/model.py", "diffusion/data.py"}


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reports must contain finite JSON values") from exc


def same_json(left, right):
    return canonical_json(left) == canonical_json(right)


def require_hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 hash")
    return value


def integer_rows(value, label):
    if not isinstance(value, list) or not value or any(type(row) is not int or row < 0 for row in value):
        raise ValueError(f"{label} must be a nonempty integer row list")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate row IDs")
    return value


def molecule_identity(smiles, label):
    if not isinstance(smiles, str) or not smiles:
        raise ValueError(f"{label} requires a nonempty SMILES")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"{label} is not a valid connected molecule")
    if any(atom.GetAtomicNum() not in (1, 6, 7, 8, 9) or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           for atom in mol.GetAtoms()):
        raise ValueError(f"{label} violates the evaluation's element/charge/radical validity rules")
    return Chem.MolToSmiles(mol, isomericSmiles=False), rdMolDescriptors.CalcMolFormula(mol)


def validate_report(report, label):
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 2:
        raise ValueError(f"{label}: evaluation schema_version 2 is required")
    canonical_json(report)
    if report.get("complete") is not True:
        raise ValueError(f"{label}: report must be complete")
    configuration, provenance, selection = (report.get(key) for key in ("configuration", "provenance", "selection"))
    if not all(isinstance(value, dict) for value in (configuration, provenance, selection)):
        raise ValueError(f"{label}: configuration, provenance, and selection are required")
    if any(value.get("split") != "val" for value in (configuration, provenance, selection)):
        raise ValueError(f"{label}: pilot selection must use validation, never test results")
    if not REQUIRED_PROVENANCE.issubset(provenance):
        raise ValueError(f"{label}: incomplete inference provenance: {sorted(REQUIRED_PROVENANCE - provenance.keys())}")
    for key in ("data_sha256", "split_sha256", "checkpoint_sha256"):
        require_hash(report.get(key), f"{label} {key}")
        if provenance[key] != report[key]:
            raise ValueError(f"{label}: top-level and provenance {key} differ")
    for key in ("data_sha256", "split_sha256"):
        if selection.get(key) != report[key]:
            raise ValueError(f"{label}: selection {key} differs")
    if report.get("ema_weights") is not True or provenance["ema_weights"] is not True:
        raise ValueError(f"{label}: all pilot evaluations must use EMA weights")
    if not same_json(report.get("checkpoint_training_diagnostic"), provenance.get("checkpoint_training_diagnostic")):
        raise ValueError(f"{label}: diagnostic checkpoint annotations differ")
    if provenance.get("checkpoint_training_diagnostic") is not None and type(provenance["checkpoint_training_diagnostic"]) is not bool:
        raise ValueError(f"{label}: diagnostic checkpoint annotation must be boolean or null")
    if not isinstance(report.get("identity_metric"), str) or not report["identity_metric"]:
        raise ValueError(f"{label}: identity_metric is required")
    if not isinstance(report.get("candidate_ranking"), str) or not report["candidate_ranking"]:
        raise ValueError(f"{label}: candidate_ranking is required")
    for key in ("beam_size", "control_seed"):
        if type(provenance[key]) is not int or provenance[key] < (1 if key == "beam_size" else 0):
            raise ValueError(f"{label}: invalid provenance {key}")
    if not isinstance(provenance["length_penalty"], (int, float)) or isinstance(provenance["length_penalty"], bool):
        raise ValueError(f"{label}: invalid length_penalty")
    for key in ("split", "beam_size", "spectrum_control", "control_seed", "device"):
        if key not in configuration or not same_json(configuration[key], provenance[key]):
            raise ValueError(f"{label}: configuration and provenance {key} differ")
    if provenance["spectrum_control"] not in ("none", "removed", "shuffled"):
        raise ValueError(f"{label}: invalid spectrum control")
    if not isinstance(provenance["decode_protocol"], str) or not provenance["decode_protocol"]:
        raise ValueError(f"{label}: decode_protocol is required")
    if not isinstance(provenance["vocabulary"], list) or not provenance["vocabulary"]:
        raise ValueError(f"{label}: vocabulary is required")
    if not isinstance(provenance["versions"], dict) or any(not provenance["versions"].get(key) for key in ("torch", "numpy", "rdkit", "selfies")):
        raise ValueError(f"{label}: software versions are incomplete")
    if not isinstance(provenance["source_sha256"], dict) or not SOURCE_FILES.issubset(provenance["source_sha256"]):
        raise ValueError(f"{label}: evaluation source hashes are incomplete")
    for name, value in provenance["source_sha256"].items():
        require_hash(value, f"{label} source {name}")
    selected = integer_rows(report.get("selected_rows"), f"{label} selected_rows")
    if type(selection.get("schema_version")) is not int or selection["schema_version"] != 1 or not same_json(selection.get("rows"), selected):
        raise ValueError(f"{label}: selection and selected_rows differ")
    ordered_hash = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode("utf-8")).hexdigest()
    if selection.get("rows_sha256") != ordered_hash:
        raise ValueError(f"{label}: ordered selection hash differs")
    if selection.get("policy") not in ("random", "all", "first") or type(selection.get("seed")) is not int or selection["seed"] < 0:
        raise ValueError(f"{label}: selection policy/seed is invalid")
    for config_key, selection_key in (("selection", "policy"), ("seed", "seed")):
        if not same_json(configuration.get(config_key), selection[selection_key]):
            raise ValueError(f"{label}: configuration selection policy/seed differs")
    records = report.get("rows")
    if not isinstance(records, list) or len(records) != len(selected):
        raise ValueError(f"{label}: complete report must cover every selected row")
    if type(report.get("attempted_rows")) is not int or report["attempted_rows"] != len(records):
        raise ValueError(f"{label}: attempted_rows differs from the complete record count")
    rebuilt = []
    for expected_row, record in zip(selected, records):
        if not isinstance(record, dict) or type(record.get("row")) is not int or record["row"] != expected_row:
            raise ValueError(f"{label}: report records do not match ordered selection")
        identity, formula = molecule_identity(record.get("reference_smiles"), f"{label} reference row {expected_row}")
        if formula != record.get("reference_formula"):
            raise ValueError(f"{label}: reference formula mismatch at row {expected_row}")
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"{label}: candidates must be a list at row {expected_row}")
        parsed = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"{label}: malformed candidate at row {expected_row}")
            candidate_identity, candidate_formula = molecule_identity(candidate.get("smiles"), f"{label} candidate row {expected_row}")
            if candidate.get("formula") != candidate_formula:
                raise ValueError(f"{label}: candidate formula mismatch at row {expected_row}")
            parsed.append({"smiles": candidate_identity, "formula": candidate_formula})
        actual = row_record(expected_row, identity, formula, parsed)
        if any(type(record.get(key)) is not bool or record[key] != actual[key] for key in METRICS):
            raise ValueError(f"{label}: reported row flags differ from predictions at row {expected_row}")
        rebuilt.append(actual)
    summary = summarize_records(rebuilt)
    for key, value in summary.items():
        if key not in report or isinstance(report[key], bool) or not isinstance(report[key], (int, float)) or not math.isclose(report[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{label}: reported summary {key} differs from recomputed predictions")
    return {"rows": rebuilt, "summary": summary, "selection": selection, "provenance": provenance,
            "inference_provenance": {key: value for key, value in provenance.items() if key not in CHECKPOINT_PROVENANCE},
            "identity_metric": report["identity_metric"], "candidate_ranking": report["candidate_ranking"],
            "configuration": configuration}


def paired_metrics(before, after, seed):
    """Bootstrap paired row differences; multinomial counts equal row resampling."""
    ids = [record["row"] for record in before]
    results = {}
    for index, metric in enumerate(PAIRED_METRICS):
        left = np.asarray([record[metric] for record in before], dtype=np.int8)
        right = np.asarray([record[metric] for record in after], dtype=np.int8)
        difference = right - left
        gains, losses = difference == 1, difference == -1
        counts = np.array([(difference == value).sum() for value in (-1, 0, 1)])
        draws = np.random.default_rng(seed + index).multinomial(len(ids), counts / len(ids), size=BOOTSTRAP_REPLICAS)
        bootstrapped_delta = (draws[:, 2] - draws[:, 0]) / len(ids)
        lower, upper = np.quantile(bootstrapped_delta, (0.025, 0.975))
        results[metric] = {"reference_correct": int(left.sum()), "candidate_correct": int(right.sum()),
            "reference_fraction": float(left.mean()), "candidate_fraction": float(right.mean()),
            "gained_rows": int(gains.sum()), "lost_rows": int(losses.sum()),
            "gained_row_ids": [row for row, gained in zip(ids, gains) if gained],
            "lost_row_ids": [row for row, lost in zip(ids, losses) if lost],
            "unchanged_correct_rows": int(((left == 1) & (right == 1)).sum()),
            "unchanged_incorrect_rows": int(((left == 0) & (right == 0)).sum()),
            "delta_percentage_points": float(100 * difference.mean()),
            "paired_bootstrap_ci95_percentage_points": [float(100 * lower), float(100 * upper)],
            "ci_includes_zero": bool(lower <= 0 <= upper), "bootstrap_seed": seed + index}
    qualifies = results["top1_identity"]["gained_rows"] > results["top1_identity"]["lost_rows"] and results["top5_identity"]["gained_rows"] >= results["top5_identity"]["lost_rows"]
    return {"rows": len(ids), "metrics": results, "meets_provisional_criterion": qualifies,
            "criterion": "positive paired Top1 gain and nondecreasing fixed Top5",
            "top1_gain_ci_above_zero": results["top1_identity"]["paired_bootstrap_ci95_percentage_points"][0] > 0,
            "uncertainty": "Paired Top1 gain is uncertain: the 95% interval includes zero." if results["top1_identity"]["ci_includes_zero"] else
                           "The paired Top1 interval excludes zero; this remains a single validation selection and training seed."}


def compare_pilot(baseline, ce, ranking, mined=None):
    """Compare in-memory reports without reading files or modifying arguments."""
    reports = {"baseline": baseline, "ce": ce, "ranking": ranking}
    if mined is not None:
        reports["mined"] = mined
    arms = {label: validate_report(report, label) for label, report in reports.items()}
    first = arms["baseline"]
    for label, arm in arms.items():
        if label == "baseline":
            continue
        for field in ("selection", "inference_provenance", "identity_metric", "candidate_ranking"):
            if not same_json(first[field], arm[field]):
                raise ValueError(f"{label}: paired {field} differs from baseline")
        for left, right in zip(first["rows"], arm["rows"]):
            if (left["row"], left["reference_smiles"], left["reference_formula"]) != (right["row"], right["reference_smiles"], right["reference_formula"]):
                raise ValueError(f"{label}: paired reference identity/formula differs")
    comparisons = {}
    pairs = [("rank_vs_baseline", "baseline", "ranking"),
             ("ce_vs_baseline", "baseline", "ce"), ("rank_vs_ce", "ce", "ranking")]
    if mined is not None:
        pairs.extend((f"mined_vs_{reference}", reference, "mined") for reference in ("baseline", "ce", "ranking"))
    for index, (name, reference, candidate) in enumerate(pairs):
        comparisons[name] = {"reference_arm": reference, "candidate_arm": candidate,
            **paired_metrics(arms[reference]["rows"], arms[candidate]["rows"], BOOTSTRAP_SEED + index * 10)}
    rank_ok = comparisons["rank_vs_baseline"]["meets_provisional_criterion"]
    ce_ok = comparisons["ce_vs_baseline"]["meets_provisional_criterion"]
    rank_over_ce = comparisons["rank_vs_ce"]["meets_provisional_criterion"]
    mined_ok = mined is not None and all(comparisons[f"mined_vs_{reference}"]["meets_provisional_criterion"]
                                        for reference in ("baseline", "ce", "ranking"))
    diagnostic = any(arm["provenance"].get("checkpoint_training_diagnostic") is True for arm in arms.values())
    control = first["provenance"]["spectrum_control"]
    if diagnostic or control != "none":
        status, text = "diagnostic_only", "These reports use a diagnostic checkpoint or spectral control; do not select production weights from this comparison."
    elif mined_ok:
        status, text = "mined_ranking_promising", "Mined ranking meets the provisional Top1/Top5 criterion against baseline, CE-only, and static ranking. Confirm matched training settings and broader validation before selecting weights."
    elif rank_ok and rank_over_ce:
        status, text = "ranking_promising", "Ranking meets the provisional Top1/Top5 criterion against both baseline and CE-only. Confirm on broader validation before selecting weights."
    elif ce_ok:
        status, text = "ce_only_promising", "CE-only meets the provisional criterion. This pilot does not establish an additional ranking benefit over CE-only; confirm before selecting weights."
    elif rank_ok:
        status, text = "ranking_gain_not_isolated", "Ranking meets the criterion against baseline, but its added benefit over CE-only is unestablished. Review the paired comparisons before choosing another experiment."
    else:
        status, text = "keep_baseline", "Neither pilot demonstrates a Top1 gain with nondecreasing Top5 against baseline. Keep the baseline for now."
        if mined is not None and comparisons["mined_vs_baseline"]["meets_provisional_criterion"]:
            status, text = "mined_gain_not_isolated", "Mined ranking meets the criterion against baseline, but its added benefit over both CE-only and static ranking is unestablished. Review the paired comparisons before choosing another experiment."
        elif mined is not None:
            text = "No pilot meets the provisional Top1/Top5 criterion against baseline. Keep the baseline for now."
    if mined is not None and not mined_ok and status not in ("diagnostic_only", "mined_gain_not_isolated", "keep_baseline"):
        text += " The mined-negative arm has not demonstrated an additional gain over all three controls."
    uncertain = any(comparison["metrics"]["top1_identity"]["ci_includes_zero"] for comparison in comparisons.values())
    negative_selection = {"mined_arm_included": mined is not None,
        "mined_meets_criterion_against_all_controls": bool(mined_ok),
        "matched_training_settings_verified_by_this_comparator": False,
        "interpretation": "Mined versus static ranking measures the effect of negative selection only when initialization, positive examples, training budget, optimizer, loss settings, and inference protocol are controlled. This comparator verifies evaluation evidence, not training execution."}
    if mined is not None:
        negative_selection.update({
            "mined_vs_static_meets_provisional_criterion": comparisons["mined_vs_ranking"]["meets_provisional_criterion"],
            "mined_vs_static_top1_gain_ci_above_zero": comparisons["mined_vs_ranking"]["top1_gain_ci_above_zero"],
            "mined_vs_static_top1_delta_percentage_points": comparisons["mined_vs_ranking"]["metrics"]["top1_identity"]["delta_percentage_points"],
            "mined_vs_static_top5_delta_percentage_points": comparisons["mined_vs_ranking"]["metrics"]["top5_identity"]["delta_percentage_points"]})
    return {"schema_version": 1, "task": "paired stage-1 fine-tuning pilot comparison", "split": "val", "fixed_k": 5,
        "rows": len(first["rows"]), "selection": first["selection"], "inference_provenance": first["inference_provenance"],
        "comparison_versions": {"numpy": np.__version__, "rdkit": rdBase.rdkitVersion},
        "arms": {label: {"checkpoint_sha256": arm["provenance"]["checkpoint_sha256"],
                          "checkpoint_training_diagnostic": arm["provenance"].get("checkpoint_training_diagnostic"),
                          **arm["summary"]} for label, arm in arms.items()},
        "comparisons": comparisons,
        "negative_selection_assessment": negative_selection,
        "bootstrap": {"replicas": BOOTSTRAP_REPLICAS, "base_seed": BOOTSTRAP_SEED, "confidence": 0.95,
                      "method": "Paired nonparametric percentile bootstrap, implemented by multinomial resampling of observed row differences (-1, 0, 1)."},
        "recommendation": {"status": status, "text": text, "provisional": True,
                           "some_top1_comparisons_include_zero": uncertain,
                           "ranking_gain_over_ce_ci_above_zero": comparisons["rank_vs_ce"]["top1_gain_ci_above_zero"]},
        "weights_automatically_promoted": False,
        "limitations": ["All evaluated rows, including empty candidate lists, remain in every denominator.",
                        "Exact identity is canonical non-isomeric, tautomer-specific graph identity; fixed Top5 ignores rank 6 and beyond.",
                        "Intervals quantify resampling uncertainty on these paired rows, not variability across training seeds or datasets.",
                        f"{len(arms)} arms and multiple metrics are exploratory comparisons; intervals are not corrected for multiple comparisons.",
                        "Shared validation selection, protocol, reference identities, and reported hashes are verified; this command does not reload the dataset or model weights.",
                        "The validation split can share tautomer families with training. No experimental-IR, novel-family, stereochemistry, or 3D accuracy claim follows.",
                        "Repeated tuning on this selection can overfit validation; confirm improvements on broader validation and reserve the test set for final evaluation."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "ce", "ranking", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--mined", help="Optional model-mined-negative ranking report, evaluated with the same protocol")
    args = parser.parse_args(argv)
    paths = {name: Path(getattr(args, name)) for name in ("baseline", "ce", "ranking")}
    if args.mined:
        paths["mined"] = Path(args.mined)
    output = Path(args.output)
    try:
        if output.resolve() in {path.resolve() for path in paths.values()}:
            raise ValueError("Comparison output would overwrite an input report")
        payloads = {name: path.read_bytes() for name, path in paths.items()}
        reports = {name: json.loads(payload.decode("utf-8")) for name, payload in payloads.items()}
        result = compare_pilot(**reports)
        result["source_reports"] = {name: str(path) for name, path in paths.items()}
        result["source_report_sha256"] = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
        result["comparison_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with exclusive_report(output):
            if output.exists():
                existing = json.loads(output.read_text(encoding="utf-8"))
                if not same_json(existing.get("source_report_sha256"), result["source_report_sha256"]) or existing.get("comparison_script_sha256") != result["comparison_script_sha256"]:
                    raise ValueError("Comparison output already exists with different source hashes or comparison code; choose a new output")
                comparable = lambda value: {key: item for key, item in value.items() if key != "source_reports"}
                if not same_json(comparable(existing), comparable(result)):
                    raise ValueError("Existing comparison content differs from recomputation; choose a new output")
                print(f"Reusing verified comparison: {output}", flush=True)
                return existing
            atomic_json(output, result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
