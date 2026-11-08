"""Saved predictions must be scored strictly, reproducibly, and without leakage."""
import copy
import csv
import json

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from scripts.analyze_spectrum_errors import (
    analyze_report, file_sha256, main, render_markdown, tautomer_key, write_outputs,
)


@pytest.fixture
def audit_fixture(tmp_path):
    identities = ["C", "CCN", "CCO", "CC=O", "COC", "C1CC1", "CF", "CO", "N", "FCF"]
    data = tmp_path / "data.npz"
    np.savez(data, molecule_ids=np.asarray(identities), source_rows=np.arange(10) + 100,
             atom_types=np.zeros((10, 29), dtype=np.int64))
    split = tmp_path / "splits.json"
    manifest = {"schema_version": 1, "data_sha256": file_sha256(data), "sample_count": 10,
                "splits": {"train": [0, 1], "val": list(range(2, 9)), "test": [9]}}
    split.write_text(json.dumps(manifest), encoding="utf-8")
    candidate_lists = [
        ["OCC"],                         # noncanonical spelling still exact
        ["C=CO", "CC=O"],               # alternate tautomer, strict miss
        ["CCO", "COC"],                 # same formula, different graph
        ["CC", "CCC", "CO", "N", "C", "C1CC1"],  # correct only at rank 6
        [],                              # no candidate
        ["this is not smiles", "CO"],    # invalid rank 1 must not promote rank 2
        ["C", "N"],                     # wrong formula, correct rank 2
    ]
    records = [{"row": row, "reference_smiles": identities[row],
                "candidates": [{"smiles": smiles} for smiles in candidate_lists[position]],
                "top1_identity": True, "topk_identity": True, "top1_formula": True}
               for position, row in enumerate(range(2, 9))]
    report = {"configuration": {"split": "val", "beam_size": 10}, "complete": True,
              "data_sha256": file_sha256(data), "split_sha256": file_sha256(split),
              "attempted_rows": len(records), "rows": records, "selected_rows": list(range(2, 9))}
    return data, split, report, identities


def test_recomputes_fixed_top5_and_disjoint_errors_without_relaxing_tautomers(audit_fixture):
    data, split, report, _ = audit_fixture
    report["rows"][0]["candidates"][0]["formula"] = "WRONG"
    analysis, rows = analyze_report(report, data, split)
    overall = analysis["overall"]
    assert overall["top1_correct"] == 1
    assert overall["top5_correct"] == 5
    assert overall["coverage"] == {"rank1": 1, "rank2_to_5": 4, "absent_from_top5": 2}
    assert overall["error_categories"] == {
        "exact": 1, "same_formula_alternate_tautomer": 1,
        "same_formula_other_graph": 1, "wrong_formula": 2, "no_valid_candidate": 2}
    assert sum(overall["error_categories"].values()) == 7
    assert rows[3]["first_exact_rank"] == 6 and rows[3]["top5_exact"] is False
    assert rows[5]["invalid_top1"] is True and rows[5]["top1_exact"] is False
    assert rows[5]["first_exact_rank"] == 2 and rows[5]["top5_exact"] is True
    assert analysis["invalid_candidates"][0]["rank"] == 1
    assert analysis["candidate_formula_annotations_corrected"] == 1
    assert analysis["scope"] == "full_split" and analysis["can_report_full_split_accuracy"]
    assert analysis["by_class"]["contains_F"]["true"]["rows"] == 1
    assert analysis["by_class"]["rings"]["1"]["top1_correct"] == 0
    assert all(row["train_tautomer_family_seen"] is None for row in rows)


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r.update(data_sha256="bad"), "data hash"),
    (lambda r: r.update(split_sha256="bad"), "split hash"),
    (lambda r: r["rows"][0].update(reference_smiles="CCN"), "Reference identity"),
    (lambda r: r["rows"][0].update(reference_formula="C99"), "Reference formula"),
    (lambda r: r["rows"][0].update(row=0), "outside"),
    (lambda r: r["rows"][1].update(row=2), "duplicate"),
    (lambda r: r.update(attempted_rows=99), "record count"),
    (lambda r: r.update(rows=[]), "nonempty"),
    (lambda r: r.pop("rows"), "nonempty"),
    (lambda r: r.update(selected_rows=[2, 3]), "cover selected"),
    (lambda r: r.update(selection={"data_sha256": "bad"}), "Selection data hash"),
    (lambda r: r.update(configuration=None), "configuration must"),
])
def test_rejects_unverifiable_or_misaligned_results(audit_fixture, mutation, message):
    data, split, original, _ = audit_fixture
    report = copy.deepcopy(original)
    mutation(report)
    with pytest.raises(ValueError, match=message):
        analyze_report(report, data, split)


def test_partial_and_legacy_subset_labels_do_not_claim_full_accuracy(audit_fixture):
    data, split, original, _ = audit_fixture
    partial = copy.deepcopy(original)
    partial.update(complete=False, rows=partial["rows"][:2], attempted_rows=2)
    analysis, _ = analyze_report(partial, data, split)
    assert analysis["partial"] and not analysis["can_report_full_split_accuracy"]
    assert analysis["selected_rows_count"] == 7
    assert "This report is partial" in render_markdown(analysis)
    partial["rows"].reverse()
    with pytest.raises(ValueError, match="ordered prefix"):
        analyze_report(partial, data, split)
    legacy = copy.deepcopy(original)
    for key in ("complete", "selected_rows", "split_sha256"):
        legacy.pop(key)
    legacy.update(rows=legacy["rows"][:2], attempted_rows=2)
    analysis, _ = analyze_report(legacy, data, split)
    assert analysis["scope"] == "evaluated_subset"
    assert not analysis["can_report_full_split_accuracy"]
    assert analysis["provenance_notes"]
    assert "subset metrics" in render_markdown(analysis)


def test_invalid_candidate_chemistry_is_not_counted_as_valid(audit_fixture):
    data, split, report, _ = audit_fixture
    report["rows"][0]["candidates"] = [{"smiles": smiles} for smiles in ("C.C", "[NH4+]", "Cl", "[CH3]")]
    analysis, rows = analyze_report(report, data, split)
    assert rows[0]["valid_candidate_count"] == 0
    assert rows[0]["invalid_candidate_count"] == 4
    assert analysis["overall"]["no_valid_candidate_rows"] == 2


def test_optional_chemical_cache_checks_hash_identity_and_family(audit_fixture, tmp_path):
    data, split, report, identities = audit_fixture
    cache = tmp_path / "chemical_rows.csv"
    metadata = tmp_path / "chemical_split_audit.json"
    metadata.write_text(json.dumps({"data_sha256": file_sha256(data), "splits_sha256": file_sha256(split),
                                    "rdkit_version": rdBase.rdkitVersion}), encoding="utf-8")
    enumerator = rdMolStandardize.TautomerEnumerator()
    items = [{"row": row, "split": "train" if row < 2 else "test" if row == 9 else "val",
              "smiles": smiles, "tautomer_family": tautomer_key(Chem.MolFromSmiles(smiles), enumerator),
              "tautomer_status": "Completed"} for row, smiles in enumerate(identities)]
    def save():
        with cache.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    save()
    analysis, rows = analyze_report(report, data, split, cache)
    assert analysis["chemical_cache"]["sha256"] == file_sha256(cache)
    assert "train_tautomer_family_seen" in analysis["by_class"]
    assert all(row["train_tautomer_family_seen"] is False for row in rows)
    items[2]["tautomer_family"] = "WRONG"
    save()
    with pytest.raises(ValueError, match="tautomer key mismatch"):
        analyze_report(report, data, split, cache)
    items[2]["smiles"] = "CCN"
    save()
    with pytest.raises(ValueError, match="identity/split mismatch"):
        analyze_report(report, data, split, cache)
    metadata.write_text(json.dumps({"data_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes"):
        analyze_report(report, data, split, cache)


def test_outputs_and_cli_preserve_inputs_and_reject_overwriting(audit_fixture, tmp_path):
    data, split, report, _ = audit_fixture
    source = tmp_path / "prediction.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    source_hash = file_sha256(source)
    output = tmp_path / "audit"
    main(["--report", str(source), "--data", str(data), "--splits", str(split), "--output_dir", str(output)])
    assert {path.name for path in output.iterdir()} == {"analysis.json", "analysis.md", "rows.csv", "errors.csv"}
    assert file_sha256(source) == source_hash
    with (output / "errors.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 6
    analysis, rows = analyze_report(report, data, split)
    with pytest.raises(ValueError, match="already exist"):
        write_outputs(output, analysis, rows)
    with pytest.raises(ValueError, match="overwrite an input"):
        write_outputs(tmp_path / "protected", analysis, rows, [tmp_path / "protected" / "rows.csv"])
