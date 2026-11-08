"""Fixed held-out sampling and restart integrity without model or GPU imports."""
import copy
import json

import numpy as np
import pytest

from diffusion.spectrum_audit import (atomic_json, make_selection, row_record,
    select_rows, summarize_records, validate_resume, validate_selection)


def fixture_data():
    manifest = {"data_sha256": "data-hash", "splits": {"train": [0, 2], "val": [1, 3, 4, 5], "test": [6, 7]}}
    ids = ["C", "CC", "CCC", "CO", "CN", "CF", "O", "N"]
    selection = make_selection(manifest, "split-hash", limit=3, seed=19)
    provenance = {"checkpoint_sha256": "checkpoint", "ema_weights": True,
                  "beam_size": 5, "decode_protocol": "legacy", "spectrum_control": "none"}
    records = [row_record(row, ids[row], "formula", [{"smiles": ids[row], "formula": "formula"}])
               for row in selection["rows"]]
    return manifest, ids, selection, provenance, records


def report_for(selection, provenance, records, complete=False):
    return {"selection": copy.deepcopy(selection), "provenance": copy.deepcopy(provenance),
            "rows": copy.deepcopy(records), "complete": complete, **summarize_records(records)}


def test_random_selection_is_repeatable_and_extends_to_full_order():
    rows = list(range(30))
    first = select_rows(rows, "random", 10, 7)
    assert first == select_rows(rows, "random", 10, 7)
    assert first != rows[:10]
    assert first == select_rows(rows, "all", None, 7)[:10]
    assert set(select_rows(rows, "all", 1, 7)) == set(rows)
    assert select_rows(rows, "first", 4) == [0, 1, 2, 3]
    assert select_rows(rows, "random", 100, 7) == select_rows(rows, "all", None, 7)


@pytest.mark.parametrize("rows,policy,limit,seed", [([], "first", 1, 0), ([1, 1], "random", 1, 0),
    ([True, 1], "first", 1, 0), ([1.0, 2], "first", 1, 0), ([[-1]], "first", 1, 0),
    ([1], "invalid", 1, 0), ([1], "first", 0, 0), ([1], "first", True, 0),
    ([1], "random", 1, -1), ([1], "random", 1, 1.5)])
def test_selection_rejects_invalid_inputs(rows, policy, limit, seed):
    with pytest.raises(ValueError):
        select_rows(rows, policy, limit, seed)


def test_selection_manifest_validates_membership_hash_policy_and_provenance():
    manifest, _, selection, _, _ = fixture_data()
    assert validate_selection(selection, manifest, "split-hash", "val") == selection["rows"]
    assert all(type(row) is int for row in selection["rows"])
    for key, value in [("schema_version", True), ("data_sha256", "other"), ("split_sha256", "other"),
                       ("split", "test"), ("rows_sha256", "wrong"), ("seed", 7), ("rows", [0, 3, 4]),
                       ("rows", [1, 1]), ("rows", [True])]:
        changed = {**selection, key: value}
        with pytest.raises(ValueError):
            validate_selection(changed, manifest, "split-hash", "val")
    with pytest.raises(ValueError, match="val or test"):
        make_selection(manifest, "split-hash", split="train")
    with pytest.raises(ValueError):
        validate_selection(selection, manifest, "split-hash", "test")


def test_all_policy_does_not_allow_silently_truncated_selection():
    manifest, _, _, _, _ = fixture_data()
    selection = make_selection(manifest, "split-hash", selection="all", limit=1)
    assert len(selection["rows"]) == 4
    subset = make_selection(manifest, "split-hash", selection="random", limit=2)
    subset["policy"] = "all"
    with pytest.raises(ValueError, match="policy"):
        validate_selection(subset, manifest, "split-hash", "val")


def test_top5_is_fixed_even_when_search_returns_six_candidates():
    candidates = [{"smiles": name, "formula": "same-formula"} for name in ["A", "B", "C", "D", "E", "target"]]
    record = row_record(8, "target", "same-formula", candidates)
    assert record["top1_identity"] is False and record["top5_identity"] is False
    assert record["topk_identity"] is True and record["top1_formula"] is True
    fifth = row_record(8, "E", "same-formula", candidates)
    assert fifth["top5_identity"] is True
    with pytest.raises(ValueError, match="deduplicated"):
        row_record(8, "target", "same-formula", candidates + [candidates[0]])


def test_empty_predictions_count_as_failures_but_empty_run_has_no_accuracy():
    failed = row_record(1, "CC", "C2H6", [])
    passed = row_record(2, "C", "CH4", [{"smiles": "C", "formula": "CH4"}])
    assert summarize_records([failed, passed]) == {"attempted_rows": 2,
        "no_valid_candidate_fraction": .5, "top1_identity_fraction": .5,
        "top5_identity_fraction": .5, "topk_identity_fraction": .5, "top1_formula_fraction": .5}
    empty = summarize_records([])
    assert empty["attempted_rows"] == 0
    assert all(value is None for key, value in empty.items() if key != "attempted_rows")


def test_resume_matches_uninterrupted_results_and_accepts_final_row_before_completion():
    _, ids, selection, provenance, records = fixture_data()
    partial = report_for(selection, provenance, records[:1])
    resumed = validate_resume(partial, provenance, selection, ids) + records[1:]
    assert resumed == records
    assert summarize_records(resumed) == summarize_records(records)
    for complete in (False, True):
        report = report_for(selection, provenance, records, complete)
        assert validate_resume(report, provenance, selection, ids) == records
    empty = report_for(selection, provenance, [])
    assert validate_resume(empty, provenance, selection, ids) == []


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "different"), ("ema_weights", False),
    ("ema_weights", 1), ("beam_size", 20), ("beam_size", 5.0),
    ("decode_protocol", "new"), ("spectrum_control", "shuffled")])
def test_resume_cannot_mix_experiments(field, value):
    _, ids, selection, provenance, records = fixture_data()
    report = report_for(selection, provenance, records[:1])
    report["provenance"][field] = value
    with pytest.raises(ValueError, match="provenance"):
        validate_resume(report, provenance, selection, ids)


@pytest.mark.parametrize("mutation", ["duplicates", "order", "identity", "metric", "metric_type",
    "count", "count_type", "early_complete", "complete_type", "aggregate", "candidate", "selection"])
def test_resume_rejects_corrupt_records(mutation):
    _, ids, selection, provenance, records = fixture_data()
    report = report_for(selection, provenance, records[:2])
    if mutation == "duplicates":
        report["rows"][1] = copy.deepcopy(report["rows"][0])
    elif mutation == "order":
        report["rows"].reverse()
    elif mutation == "identity":
        report["rows"][0]["reference_smiles"] = "incorrect"
    elif mutation == "metric":
        report["rows"][0]["top1_identity"] = False
    elif mutation == "metric_type":
        report["rows"][0]["top1_identity"] = 1
    elif mutation == "count":
        report["attempted_rows"] = 1
    elif mutation == "count_type":
        report["attempted_rows"] = 2.0
    elif mutation == "early_complete":
        report["complete"] = True
    elif mutation == "complete_type":
        report["complete"] = 0
    elif mutation == "aggregate":
        report["top5_identity_fraction"] = 0.0
    elif mutation == "candidate":
        del report["rows"][0]["candidates"][0]["formula"]
    elif mutation == "selection":
        report["selection"]["rows"].reverse()
    with pytest.raises(ValueError):
        validate_resume(report, provenance, selection, ids)


def test_atomic_json_preserves_prior_report_on_invalid_payload_and_cleans_temporary_files(tmp_path):
    path = tmp_path / "nested" / "report.json"
    atomic_json(path, {"state": "initial", "unicode": "Å"})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {"invalid": float("nan")})
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    atomic_json(path, {"state": "complete"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"state": "complete"}
    assert list(path.parent.iterdir()) == [path]


def test_numpy_row_ids_roundtrip_to_standard_json():
    selected = select_rows(np.array([2, 7, 9], dtype=np.int64), "random", 2)
    assert json.loads(json.dumps(selected)) == selected
