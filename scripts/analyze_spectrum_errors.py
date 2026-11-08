"""Audit saved IR-to-graph predictions without running a model or changing data.

Strict graph identity uses canonical non-isomeric SMILES. Tautomer matching is
only an error diagnostic and never changes accuracy. Invalid candidate slots
retain their original ranks; later candidates are not silently promoted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.data import validate_split_manifest


CATEGORIES = ("exact", "same_formula_alternate_tautomer", "same_formula_other_graph",
              "wrong_formula", "no_valid_candidate")
OUTPUT_NAMES = ("analysis.json", "rows.csv", "errors.csv", "analysis.md")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_identity(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Missing SMILES string")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SMILES does not parse to a nonempty molecule")
    return mol, Chem.MolToSmiles(mol, isomericSmiles=False), rdMolDescriptors.CalcMolFormula(mol)


def candidate_identity(candidate, max_atoms):
    if not isinstance(candidate, dict):
        raise ValueError("Candidate is not an object with a smiles field")
    mol, identity, formula = parse_identity(candidate.get("smiles"))
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Disconnected candidate")
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetAtomicNum() not in (1, 6, 7, 8, 9)
           for a in mol.GetAtoms()):
        raise ValueError("Candidate has unsupported elements, charge, or radicals")
    if Chem.AddHs(mol).GetNumAtoms() > max_atoms:
        raise ValueError("Candidate exceeds the prepared data's total atom limit")
    return mol, identity, formula


def tautomer_key(mol, enumerator):
    result = enumerator.Enumerate(mol)
    if str(result.status) != "Completed":
        raise ValueError("Tautomer enumeration did not complete; cannot classify this error reliably")
    return Chem.MolToSmiles(enumerator.PickCanonical(result), isomericSmiles=False)


def _integer_rows(values, label, sample_count):
    if not isinstance(values, list) or any(type(row) is not int for row in values):
        raise ValueError(f"{label} must be an integer list")
    if len(set(values)) != len(values) or any(row < 0 or row >= sample_count for row in values):
        raise ValueError(f"{label} contains duplicate or out-of-range rows")
    return values


def load_chemical_cache(path, audit_path, identifiers, manifest, data_hash, split_hash):
    """Validate a dataset audit cache before using its training-family metadata.

    Every row identity and split is checked. Dataset/split hashes and the RDKit
    version must agree with the companion audit. Scored rows' tautomer keys are
    additionally recomputed in analyze_report. Training-family keys are cached
    annotations; their origin is recorded rather than presented as new analysis.
    """
    path = Path(path)
    audit_path = Path(audit_path) if audit_path else path.with_name("chemical_split_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("data_sha256") != data_hash or audit.get("splits_sha256", audit.get("split_sha256")) != split_hash:
        raise ValueError("Chemical cache data/split hashes differ")
    if audit.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError("Chemical cache RDKit version differs; rebuild the cache")
    owner = {row: name for name, values in manifest["splits"].items() for row in values}
    cache = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            try:
                row = int(item["row"])
            except (KeyError, ValueError) as exc:
                raise ValueError("Chemical cache has a malformed row index") from exc
            if row in cache or row not in owner:
                raise ValueError("Chemical cache contains duplicate or out-of-range rows")
            if item.get("smiles") != str(identifiers[row]) or item.get("split") != owner[row]:
                raise ValueError(f"Chemical cache identity/split mismatch at row {row}")
            if not item.get("tautomer_family") or item.get("tautomer_status") != "Completed":
                raise ValueError(f"Chemical cache tautomer diagnostic is incomplete at row {row}")
            cache[row] = item
    if len(cache) != len(identifiers):
        raise ValueError("Chemical cache must cover every data row")
    families = {cache[row]["tautomer_family"] for row in manifest["splits"]["train"]}
    provenance = {"path": str(path), "sha256": file_sha256(path),
                  "audit_path": str(audit_path), "audit_sha256": file_sha256(audit_path),
                  "validation": "all identities/splits and source hashes; scored tautomer keys recomputed"}
    return cache, families, provenance


def summarize_rows(rows):
    count = len(rows)
    categories = Counter(row["error_category"] for row in rows)
    return {"rows": count,
            "top1_correct": sum(row["top1_exact"] for row in rows),
            "top5_correct": sum(row["top5_exact"] for row in rows),
            "top1_accuracy": sum(row["top1_exact"] for row in rows) / count if count else None,
            "top5_accuracy": sum(row["top5_exact"] for row in rows) / count if count else None,
            "top1_formula_accuracy": sum(row["top1_formula_match"] for row in rows) / count if count else None,
            "no_valid_candidate_rows": sum(row["valid_candidate_count"] == 0 for row in rows),
            "invalid_top1_rows": sum(row["invalid_top1"] for row in rows),
            "error_categories": {name: categories[name] for name in CATEGORIES},
            "coverage": {label: sum(row["coverage"] == label for row in rows)
                         for label in ("rank1", "rank2_to_5", "absent_from_top5")}}


def analyze_report(report, data_path, splits_path, chemical_rows=None, chemical_audit=None):
    if not isinstance(report, dict) or not isinstance(report.get("rows"), list) or not report["rows"]:
        raise ValueError("A full evaluation report with nonempty per-molecule rows is required; aggregate summaries are insufficient")
    data_hash, split_hash = file_sha256(data_path), file_sha256(splits_path)
    if report.get("data_sha256") != data_hash:
        raise ValueError("Report data hash differs from the prepared data")
    reported_split_hash = report.get("split_sha256", report.get("splits_sha256"))
    if reported_split_hash is not None and reported_split_hash != split_hash:
        raise ValueError("Report split hash differs from the supplied split manifest")
    with np.load(data_path, allow_pickle=False) as archive:
        identifiers = archive["molecule_ids"]
        if identifiers.ndim != 1 or identifiers.dtype.kind not in "US":
            raise ValueError("Prepared molecule_ids must be a one-dimensional string array")
        source_rows = archive["source_rows"] if "source_rows" in archive else np.arange(len(identifiers))
        max_atoms = int(archive["atom_types"].shape[1]) if "atom_types" in archive else 29
    manifest = validate_split_manifest(json.loads(Path(splits_path).read_text(encoding="utf-8")), identifiers, data_path)
    configuration = report.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("Report configuration must be an object")
    split = configuration.get("split")
    if split not in ("val", "test"):
        raise ValueError("Report configuration must declare a val or test split")
    actual_rows = _integer_rows([record.get("row") if isinstance(record, dict) else None
                                for record in report["rows"]], "Report rows", len(identifiers))
    if not set(actual_rows).issubset(set(manifest["splits"][split])):
        raise ValueError("Report contains rows outside its declared held-out split")
    if report.get("attempted_rows") is not None and report["attempted_rows"] != len(actual_rows):
        raise ValueError("Report attempted_rows differs from its record count")
    if "complete" in report and type(report["complete"]) is not bool:
        raise ValueError("Report complete flag must be boolean")
    selection_metadata = report.get("selection")
    if isinstance(selection_metadata, dict):
        if selection_metadata.get("data_sha256", data_hash) != data_hash:
            raise ValueError("Selection data hash differs from the prepared data")
        if selection_metadata.get("split_sha256", selection_metadata.get("splits_sha256", split_hash)) != split_hash:
            raise ValueError("Selection split hash differs from the supplied manifest")
    partial = report.get("complete") is False
    selected = report.get("selected_rows")
    if selected is None and isinstance(report.get("selection"), dict):
        selected = report["selection"].get("rows")
    if selected is not None:
        selected = _integer_rows(selected, "Selected rows", len(identifiers))
        if not selected or not set(selected).issubset(set(manifest["splits"][split])):
            raise ValueError("Selected rows do not belong to the declared split")
        if partial and actual_rows != selected[:len(actual_rows)]:
            raise ValueError("Partial report is not an ordered prefix of selected rows")
        if not partial and actual_rows != selected:
            raise ValueError("Complete report does not cover selected rows in order")
    cache, train_families, cache_provenance = {}, set(), None
    if chemical_rows:
        cache, train_families, cache_provenance = load_chemical_cache(
            chemical_rows, chemical_audit, identifiers, manifest, data_hash, split_hash)
    enumerator = rdMolStandardize.TautomerEnumerator()
    rows, invalid_records = [], []
    formula_disagreements = 0
    for record in report["rows"]:
        row = record["row"]
        mol, identity, formula = parse_identity(str(identifiers[row]))
        _, reported_identity, reported_formula = parse_identity(record.get("reference_smiles"))
        if reported_identity != identity:
            raise ValueError(f"Reference identity mismatch at row {row}")
        if record.get("reference_formula") not in (None, formula) or reported_formula != formula:
            raise ValueError(f"Reference formula mismatch at row {row}")
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"Candidates must be a list at row {row}")
        parsed = []
        invalid = []
        for rank, candidate in enumerate(candidates, 1):
            try:
                entry = candidate_identity(candidate, max_atoms)
            except (ValueError, RuntimeError) as exc:
                entry = None
                invalid.append({"row": row, "rank": rank, "reason": str(exc)})
            parsed.append(entry)
            if entry is not None and candidate.get("formula") not in (None, entry[2]):
                formula_disagreements += 1
        invalid_records.extend(invalid)
        ranks = [rank for rank, item in enumerate(parsed, 1) if item is not None and item[1] == identity]
        first_rank = min(ranks) if ranks else None
        top1 = parsed[0] if parsed else None
        family = tautomer_key(mol, enumerator) if cache or (top1 and top1[1] != identity and top1[2] == formula) else None
        if cache and cache[row]["tautomer_family"] != family:
            raise ValueError(f"Chemical cache tautomer key mismatch at scored row {row}")
        if top1 is None:
            category = "no_valid_candidate"
        elif top1[1] == identity:
            category = "exact"
        elif top1[2] != formula:
            category = "wrong_formula"
        elif tautomer_key(top1[0], enumerator) == family:
            category = "same_formula_alternate_tautomer"
        else:
            category = "same_formula_other_graph"
        atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        rows.append({"row": row, "source_row": int(source_rows[row]), "split": split,
                     "reference_smiles": identity, "reference_formula": formula,
                     "top1_smiles": top1[1] if top1 else "", "top1_formula": top1[2] if top1 else "",
                     "top1_exact": category == "exact", "top5_exact": bool(first_rank and first_rank <= 5),
                     "top1_formula_match": bool(top1 and top1[2] == formula),
                     "first_exact_rank": first_rank, "error_category": category,
                     "coverage": "rank1" if first_rank == 1 else "rank2_to_5" if first_rank and first_rank <= 5 else "absent_from_top5",
                     "candidate_count": len(candidates), "valid_candidate_count": sum(item is not None for item in parsed),
                     "invalid_candidate_count": len(invalid), "invalid_top1": bool(parsed and top1 is None),
                     "duplicate_identity_candidates": sum(item is not None for item in parsed) - len({item[1] for item in parsed if item}),
                     "heavy_atoms": mol.GetNumHeavyAtoms(), "rings": rdMolDescriptors.CalcNumRings(mol),
                     "aromatic": any(atom.GetIsAromatic() for atom in mol.GetAtoms()),
                     "N": atom_numbers.count(7), "O": atom_numbers.count(8), "F": atom_numbers.count(9),
                     "train_tautomer_family_seen": family in train_families if cache else None,
                     "candidate_smiles_by_rank": json.dumps([item[1] if item else None for item in parsed]),
                     "invalid_candidates": json.dumps(invalid)})
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        keys = {"aromatic": str(row["aromatic"]).lower(), "rings": str(row["rings"]),
                "heavy_atoms": str(row["heavy_atoms"]),
                **{f"contains_{element}": str(row[element] > 0).lower() for element in ("N", "O", "F")}}
        if cache:
            keys["train_tautomer_family_seen"] = str(row["train_tautomer_family_seen"]).lower()
        for group, value in keys.items():
            groups[group][value].append(row)
    full_split = not partial and set(actual_rows) == set(manifest["splits"][split])
    analysis = {"schema_version": 1, "task": "IR-only molecular graph identity error audit", "fixed_k": 5,
                "data_sha256": data_hash, "split_sha256": split_hash,
                "checkpoint_sha256_reported": report.get("checkpoint_sha256"),
                "checkpoint_weights_verified_locally": False, "rdkit_version": rdBase.rdkitVersion,
                "configuration": configuration, "complete": report.get("complete"), "partial": partial,
                "scope": "partial_report" if partial else "full_split" if full_split else "evaluated_subset",
                "split": split, "split_rows": len(manifest["splits"][split]),
                "evaluated_rows": len(rows), "selected_rows_count": len(selected) if selected is not None else None,
                "can_report_full_split_accuracy": full_split,
                "selection": report.get("selection"),
                "provenance_notes": ([] if reported_split_hash else ["Legacy report has no split hash; current manifest membership and dataset hash were verified."]),
                "chemical_cache": cache_provenance, "overall": summarize_rows(rows),
                "by_class": {group: {key: summarize_rows(value) for key, value in sorted(buckets.items())}
                             for group, buckets in sorted(groups.items())},
                "invalid_candidates": invalid_records, "candidate_formula_annotations_corrected": formula_disagreements,
                "limitations": ["Accuracy applies to the evaluated rows; a subset does not establish full-validation or test accuracy.",
                    "Exact identity ignores stereochemistry but preserves tautomer-specific graph identity.",
                    "Tautomer-family membership is diagnostic and does not count as an exact match.",
                    "Top5 uses the first five reported rank slots even when beam width is larger; invalid slots are not promoted.",
                    "No valid candidate in the error categories means no valid first-choice candidate; invalid top-ranked entries are counted explicitly.",
                    "Graph identity accuracy does not establish 3D geometry or predicted-spectrum accuracy."]}
    return analysis, rows


def render_markdown(analysis):
    overall = analysis["overall"]
    percent = lambda value: f"{100 * value:.1f}%" if value is not None else "n/a"
    lines = ["# Spectrum prediction error audit", "",
             f"Scope: **{analysis['scope']}** — {analysis['evaluated_rows']:,} evaluated rows out of {analysis['split_rows']:,} in `{analysis['split']}`.", ""]
    if analysis["partial"]:
        lines += ["**This report is partial. Metrics below describe completed rows only and are not the final evaluation result.**", ""]
    elif not analysis["can_report_full_split_accuracy"]:
        lines += ["These are subset metrics. They must not be labeled full-validation or test-set accuracy.", ""]
    lines += [f"Strict Top1: **{percent(overall['top1_accuracy'])}** ({overall['top1_correct']}/{overall['rows']}).",
              f"Strict Top5: **{percent(overall['top5_accuracy'])}** ({overall['top5_correct']}/{overall['rows']}).",
              f"First-choice formula match: **{percent(overall['top1_formula_accuracy'])}**.", "",
              "## First-choice outcomes", "", "| Outcome | Rows |", "|---|---:|"]
    lines += [f"| {name.replace('_', ' ')} | {overall['error_categories'][name]} |" for name in CATEGORIES]
    lines += ["", f"The correct identity appears at ranks 2–5 for **{overall['coverage']['rank2_to_5']}** rows and is absent from the first five for **{overall['coverage']['absent_from_top5']}** rows.",
              "This separates ranking opportunities from candidate-generation misses; neither count guarantees a fine-tuning gain.", "",
              "## Results by molecular class", "", "| Class | Value | Rows | Top1 | Top5 |", "|---|---|---:|---:|---:|"]
    for group, values in analysis["by_class"].items():
        for value, metrics in values.items():
            lines.append(f"| {group} | {value} | {metrics['rows']} | {percent(metrics['top1_accuracy'])} | {percent(metrics['top5_accuracy'])} |")
    lines += ["", "Class results use the evaluated subset and may have small denominators. Rows can appear in several different class groups.", "",
              "## Data checks", "", f"- Prepared data SHA256: `{analysis['data_sha256']}`.",
              f"- Split manifest SHA256: `{analysis['split_sha256']}`.",
              f"- Invalid candidate entries: {len(analysis['invalid_candidates'])}; invalid first choices: {overall['invalid_top1_rows']}.",
              f"- Candidate formula annotations corrected from parsed structures: {analysis['candidate_formula_annotations_corrected']}.",
              "- Row identities and split membership were checked; duplicate rows, mismatched references and hashes are rejected.",
              "- Checkpoint hash is preserved from the report; this analysis does not load or verify model weights."]
    lines += [f"- {note}" for note in analysis["provenance_notes"]]
    lines += ["", "## Interpretation limits", ""] + [f"- {note}" for note in analysis["limitations"]]
    lines += ["", "Use `errors.csv` to inspect individual mistakes and `rows.csv` for all evaluated cases. Choose future training examples from the training split only; do not train on these held-out rows.", ""]
    return "\n".join(lines)


def write_outputs(output_dir, analysis, rows, protected_paths=()):
    output_dir = Path(output_dir)
    targets = [output_dir / name for name in OUTPUT_NAMES]
    protected = {Path(path).resolve() for path in protected_paths if path}
    if any(target.resolve() in protected for target in targets):
        raise ValueError("An output path would overwrite an input")
    if any(target.exists() for target in targets):
        raise ValueError("Analysis outputs already exist; choose a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    for name, subset in (("rows.csv", rows), ("errors.csv", [row for row in rows if not row["top1_exact"]])):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    (output_dir / "analysis.md").write_text(render_markdown(analysis), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Full evaluate_spectrum.py JSON, including rows")
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--chemical_rows", help="Optional chemical_rows.csv from the dataset audit")
    parser.add_argument("--chemical_audit", help="Optional cache provenance JSON; defaults to sibling chemical_split_audit.json")
    parser.add_argument("--output_dir", required=True, help="New directory; existing analysis files will not be overwritten")
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        analysis, rows = analyze_report(report, args.data, args.splits, args.chemical_rows, args.chemical_audit)
        analysis["report_path"] = str(args.report)
        analysis["report_sha256"] = file_sha256(args.report)
        write_outputs(args.output_dir, analysis, rows,
                      (args.report, args.data, args.splits, args.chemical_rows, args.chemical_audit))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": args.output_dir, "scope": analysis["scope"],
                      "overall": analysis["overall"]}, indent=2))


if __name__ == "__main__":
    main()
