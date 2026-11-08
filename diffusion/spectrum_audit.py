"""Pure selection, reporting, and resume checks for molecular evaluation.

These helpers do not load models or infer chemistry. Candidate lists must
already contain unique, chemically validated structures in prediction order.
"""

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import numpy as np


SCHEMA_VERSION = 1
METRICS = ("top1_identity", "top5_identity", "topk_identity", "top1_formula")


@contextmanager
def exclusive_report(path):
    """OS-held lock released even on a hard process exit; leave its inode intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise ValueError(f"Another evaluator is using report {path}; wait for it to exit") from exc
        try:
            yield
        finally:
            release()


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _rows(values, name="rows", allow_empty=False):
    if not isinstance(values, (list, tuple, np.ndarray)) or np.ndim(values) != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of integer row IDs.")
    rows = [_integer(value, name) for value in values]
    if (not rows and not allow_empty) or len(rows) != len(set(rows)):
        raise ValueError(f"{name} must contain unique row IDs and be nonempty.")
    return rows


def _rows_sha256(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def _same_json(left, right):
    # Python equality treats True == 1 and 5 == 5.0; provenance must not.
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def select_rows(split_rows, selection="first", limit=500, seed=20261105):
    """Select a fixed ordered subset, preserving the legacy first-N option.

    ``all`` ignores ``limit`` and uses a seeded random order. A random subset
    is the prefix of that same permutation, so larger samples extend it.
    """
    rows = _rows(split_rows, "split_rows")
    seed = _integer(seed, "seed")
    if selection not in ("first", "random", "all"):
        raise ValueError("selection must be first, random, or all.")
    if selection != "all":
        limit = _integer(limit, "limit", minimum=1)
    if selection != "first":
        rows = np.random.default_rng(seed).permutation(rows).tolist()
    return rows if selection == "all" else rows[:limit]


def _split_rows(manifest, split):
    if split not in ("val", "test"):
        raise ValueError("Evaluation selection must use the val or test split.")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("splits"), dict):
        raise ValueError("A split manifest with explicit split rows is required.")
    return _rows(manifest["splits"].get(split), f"{split} split")


def make_selection(manifest, split_hash, split="val", selection="random", limit=1000, seed=20261105):
    rows = select_rows(_split_rows(manifest, split), selection, limit, seed)
    if not isinstance(manifest.get("data_sha256"), str) or not manifest["data_sha256"]:
        raise ValueError("The manifest must declare a nonempty data hash.")
    if not isinstance(split_hash, str) or not split_hash:
        raise ValueError("A nonempty split hash is required.")
    return {"schema_version": SCHEMA_VERSION, "data_sha256": manifest["data_sha256"],
            "split_sha256": split_hash, "split": split, "policy": selection,
            "seed": int(seed), "rows": rows, "rows_sha256": _rows_sha256(rows)}


def validate_selection(selection, manifest, split_hash, split):
    """Validate identity, membership, row order, and the declared selection recipe."""
    split_rows = _split_rows(manifest, split)
    if not isinstance(selection, dict) or type(selection.get("schema_version")) is not int or selection["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported selection schema_version.")
    for key, expected in (("data_sha256", manifest.get("data_sha256")), ("split_sha256", split_hash), ("split", split)):
        if not isinstance(expected, str) or not expected or selection.get(key) != expected:
            raise ValueError(f"Selection {key} differs from the requested data/split.")
    rows = _rows(selection.get("rows"), "selected rows")
    if not set(rows).issubset(split_rows):
        raise ValueError("Selected rows must belong to the requested held-out split.")
    if selection.get("rows_sha256") != _rows_sha256(rows):
        raise ValueError("Selection ordered row hash differs.")
    expected = select_rows(split_rows, selection.get("policy"), len(rows), selection.get("seed"))
    if rows != expected:
        raise ValueError("Selection rows/order differ from the declared policy and seed.")
    return rows


def atomic_json(path, payload):
    """Replace one UTF-8 JSON file atomically, leaving the old file on failure."""
    # Serialize first, so invalid payloads cannot replace an existing report.
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def row_record(row, identity, formula, candidates):
    """Compute graph metrics from ranked, unique valid candidates only."""
    row = _integer(row, "row")
    if not isinstance(identity, str) or not identity or not isinstance(formula, str) or not formula:
        raise ValueError("Reference identity and formula must be nonempty strings.")
    if not isinstance(candidates, list):
        raise ValueError("Candidates must be a ranked list.")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or any(not isinstance(candidate.get(key), str) or not candidate[key]
                                                  for key in ("smiles", "formula")):
            raise ValueError("Each candidate must contain nonempty smiles and formula strings.")
        if candidate["smiles"] in seen:
            raise ValueError("Candidate structures must be deduplicated before computing ranks.")
        seen.add(candidate["smiles"])
    return {"row": row, "reference_smiles": identity, "reference_formula": formula,
            "candidates": candidates,
            "top1_identity": bool(candidates and candidates[0]["smiles"] == identity),
            "top5_identity": any(item["smiles"] == identity for item in candidates[:5]),
            "topk_identity": any(item["smiles"] == identity for item in candidates),
            "top1_formula": bool(candidates and candidates[0]["formula"] == formula)}


def summarize_records(records):
    if not isinstance(records, list):
        raise ValueError("Records must be a list.")
    size = len(records)
    result = {"attempted_rows": size,
              "no_valid_candidate_fraction": sum(not record["candidates"] for record in records) / size if size else None}
    for metric in METRICS:
        if any(type(record.get(metric)) is not bool for record in records):
            raise ValueError(f"Record metric {metric} must be boolean.")
        result[metric + "_fraction"] = sum(record[metric] for record in records) / size if size else None
    return result


def validate_resume(report, provenance, selection, molecule_ids):
    """Return checked prefix records, refusing to mix evaluation experiments.

    A full set of records with complete=False is allowed: an interruption can
    happen after saving the final row but before writing the completion flag.
    Reference formula chemistry must additionally be checked by the caller.
    """
    if not isinstance(provenance, dict) or not isinstance(report, dict) or not _same_json(report.get("provenance"), provenance):
        raise ValueError("Resume provenance differs from this evaluation.")
    if not isinstance(selection, dict) or not _same_json(report.get("selection"), selection):
        raise ValueError("Resume selection differs from this evaluation.")
    if type(report.get("complete")) is not bool:
        raise ValueError("Resume complete must be a boolean.")
    selected = _rows(selection.get("rows"), "selected rows")
    if selection.get("rows_sha256") != _rows_sha256(selected):
        raise ValueError("Selection ordered row hash differs.")
    records = report.get("rows")
    if not isinstance(records, list) or len(records) > len(selected):
        raise ValueError("Resume records must be an ordered prefix of the selected rows.")
    attempted = report.get("attempted_rows")
    if type(attempted) is not int or attempted != len(records):
        raise ValueError("Resume attempted_rows differs from its record count.")
    if report["complete"] and len(records) != len(selected):
        raise ValueError("A complete report must contain every selected row.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Each resume record must be a dictionary.")
        row = _integer(record.get("row"), "resume row")
        if row != selected[index]:
            raise ValueError("Resume records must be a unique ordered prefix of the selected rows.")
        if row >= len(molecule_ids) or record.get("reference_smiles") != str(molecule_ids[row]):
            raise ValueError("Resume reference identity differs from the current dataset.")
        rebuilt = row_record(row, record["reference_smiles"], record.get("reference_formula"), record.get("candidates"))
        if any(type(record.get(key)) is not bool or record[key] != rebuilt[key] for key in METRICS):
            raise ValueError("Resume candidate metrics are inconsistent with its predictions.")
    for key, expected in summarize_records(records).items():
        if key in report and report[key] != expected:
            raise ValueError(f"Resume summary {key} differs from its record metrics.")
    return records
