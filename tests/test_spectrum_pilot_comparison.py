"""Pilot comparisons retain failed rows and reject changes to paired evidence."""
import copy
import hashlib
import json

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from diffusion.spectrum_audit import row_record, summarize_records
from scripts.compare_spectrum_pilot import SOURCE_FILES, compare_pilot, main


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def candidate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {"smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
            "formula": rdMolDescriptors.CalcMolFormula(mol)}


@pytest.fixture
def reports():
    references = ["CCO", "COC", "CC=O", "C1CC1", "CF", "CO"]
    selected = list(range(100, 106))
    base = [["CCO"], ["CCO", "COC"], [], ["CCC"], ["C", "CF"], ["C", "N", "CC", "CCC", "COC", "CO"]]
    ce = copy.deepcopy(base)
    ce[1], ce[3] = ["COC"], ["C1CC1"]
    ranking = copy.deepcopy(ce)
    ranking[2], ranking[4] = ["CC=O"], ["CF"]
    selection = {"schema_version": 1, "data_sha256": sha("data"), "split_sha256": sha("split"),
                 "split": "val", "policy": "random", "seed": 20261105, "rows": selected,
                 "rows_sha256": sha(json.dumps(selected, separators=(",", ":")))}
    provenance = {"data_sha256": sha("data"), "split_sha256": sha("split"), "checkpoint_sha256": sha("checkpoint"),
        "ema_weights": True, "checkpoint_training_diagnostic": False, "preprocessing": {"frequency_scale": 4000},
        "vocabulary": ["<pad>", "<bos>", "<eos>", "[C]", "[O]"],
        "split": "val", "beam_size": 10, "length_penalty": 0.6,
        "decode_protocol": "legacy_generate_v1_valid_unique_candidates", "spectrum_control": "none", "control_seed": 0,
        "device": "cuda", "torch_num_threads": 4, "cuda_version": "12.6", "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False, "versions": {"torch": "2.8.0", "numpy": "2.2.6", "rdkit": "2025.09.1", "selfies": "2.2.0"},
        "source_sha256": {name: sha(name) for name in SOURCE_FILES}}
    output = []
    for name, predictions in (("baseline", base), ("ce", ce), ("ranking", ranking)):
        prov = copy.deepcopy(provenance)
        prov["checkpoint_sha256"] = sha(name)
        records = [row_record(row, reference, candidate(reference)["formula"], [candidate(smiles) for smiles in predicted])
                   for row, reference, predicted in zip(selected, references, predictions)]
        output.append({"schema_version": 2, "complete": True, "configuration": {
            "split": "val", "beam_size": 10, "spectrum_control": "none", "control_seed": 0, "device": "cuda",
            "selection": "random", "seed": 20261105, "limit": 6, "checkpoint": f"{name}.ckpt", "output": f"{name}.json"},
            "selection": copy.deepcopy(selection), "selected_rows": selected.copy(), "provenance": prov,
            "data_sha256": sha("data"), "split_sha256": sha("split"), "checkpoint_sha256": sha(name),
            "ema_weights": True, "checkpoint_training_diagnostic": False,
            "identity_metric": "canonical non-isomeric SMILES; no ground-truth formula supplied",
            "candidate_ranking": "Legacy decoder order after validity filtering and identity deduplication; no reference-based ranking",
            "rows": records, **summarize_records(records)})
    return output


def refresh(report):
    report.update(summarize_records(report["rows"]))


def mined_report(ranking):
    report = copy.deepcopy(ranking)
    report["checkpoint_sha256"] = sha("mined")
    report["provenance"]["checkpoint_sha256"] = sha("mined")
    report["configuration"].update(checkpoint="mined.ckpt", output="mined.json")
    report["rows"][5] = row_record(105, "CO", "CH4O", [candidate("CO")])
    refresh(report)
    return report


def test_four_arm_comparison_preserves_old_pairs_and_isolates_negative_control(reports):
    old = compare_pilot(*reports)
    mined = mined_report(reports[2])
    untouched = copy.deepcopy(mined)
    result = compare_pilot(*reports, mined=mined)
    assert mined == untouched
    assert result == compare_pilot(*reports, mined=mined)
    for name, comparison in old["comparisons"].items():
        assert result["comparisons"][name] == comparison
    assert len(result["arms"]) == 4 and len(result["comparisons"]) == 6
    assert result["arms"]["mined"]["top1_identity_fraction"] == 1.0
    assert result["comparisons"]["mined_vs_baseline"]["metrics"]["top1_identity"]["gained_rows"] == 5
    assert result["comparisons"]["mined_vs_ce"]["metrics"]["top1_identity"]["gained_rows"] == 3
    comparison = result["comparisons"]["mined_vs_ranking"]
    assert comparison["metrics"]["top1_identity"]["gained_row_ids"] == [105]
    assert comparison["metrics"]["top5_identity"]["gained_row_ids"] == [105]
    assert result["recommendation"]["status"] == "mined_ranking_promising"
    assert result["recommendation"]["provisional"] is True
    assert result["negative_selection_assessment"]["mined_meets_criterion_against_all_controls"] is True
    assert result["negative_selection_assessment"]["matched_training_settings_verified_by_this_comparator"] is False
    assert result["weights_automatically_promoted"] is False


def test_mined_tie_with_static_does_not_establish_additional_benefit(reports):
    mined = copy.deepcopy(reports[2])
    result = compare_pilot(*reports, mined=mined)
    assert result["recommendation"]["status"] == "ranking_promising"
    assert "not demonstrated an additional gain" in result["recommendation"]["text"]
    assert result["negative_selection_assessment"]["mined_vs_static_meets_provisional_criterion"] is False
    assert result["negative_selection_assessment"]["mined_vs_static_top1_delta_percentage_points"] == 0.0


def test_mined_top1_gain_with_top5_decline_fails_negative_selection_criterion(reports):
    # Static has three first-choice hits but contains every reference in its first five.
    ranking = reports[2]
    for index in (1, 4, 5):
        record = ranking["rows"][index]
        reference = record["reference_smiles"]
        ranking["rows"][index] = row_record(record["row"], reference, record["reference_formula"],
                                          [candidate("CCC"), candidate(reference)])
    refresh(ranking)
    mined = mined_report(ranking)
    mined["rows"][1] = row_record(101, "COC", "C2H6O", [candidate("COC")])
    mined["rows"][4] = row_record(104, "CF", "CH3F", [])
    refresh(mined)
    result = compare_pilot(*reports, mined=mined)
    comparison = result["comparisons"]["mined_vs_ranking"]
    assert comparison["metrics"]["top1_identity"]["delta_percentage_points"] > 0
    assert comparison["metrics"]["top5_identity"]["delta_percentage_points"] < 0
    assert comparison["meets_provisional_criterion"] is False
    assert result["recommendation"]["status"] != "mined_ranking_promising"


@pytest.mark.parametrize("arm", [0, 1, 2, 3])
def test_any_diagnostic_arm_prevents_production_recommendation(reports, arm):
    all_reports = reports + [mined_report(reports[2])]
    all_reports[arm]["checkpoint_training_diagnostic"] = True
    all_reports[arm]["provenance"]["checkpoint_training_diagnostic"] = True
    result = compare_pilot(*all_reports)
    assert result["recommendation"]["status"] == "diagnostic_only"
    assert result["weights_automatically_promoted"] is False


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r.update(complete=False), "must be complete"),
    (lambda r: r["rows"][5].update(top1_identity=False), "row flags"),
    (lambda r: r["provenance"].update(length_penalty=1.0), "inference_provenance"),
    (lambda r: r["configuration"].update(split="test"), "validation"),
])
def test_mined_arm_receives_same_strict_validation(reports, mutation, message):
    mined = mined_report(reports[2])
    mutation(mined)
    with pytest.raises(ValueError, match=message):
        compare_pilot(*reports, mined=mined)


def test_fixed_top5_paired_counts_failures_and_deterministic_intervals(reports):
    untouched = copy.deepcopy(reports)
    result = compare_pilot(*reports)
    assert reports == untouched
    assert result == compare_pilot(*reports)
    assert result["rows"] == 6
    assert result["arms"]["baseline"]["top1_identity_fraction"] == 1 / 6
    assert result["arms"]["baseline"]["top5_identity_fraction"] == 3 / 6
    assert result["arms"]["baseline"]["topk_identity_fraction"] == 4 / 6
    assert result["arms"]["baseline"]["no_valid_candidate_fraction"] == 1 / 6
    rb = result["comparisons"]["rank_vs_baseline"]
    assert rb["metrics"]["top1_identity"]["gained_rows"] == 4
    assert rb["metrics"]["top1_identity"]["lost_rows"] == 0
    assert rb["metrics"]["top1_identity"]["gained_row_ids"] == [101, 102, 103, 104]
    assert rb["metrics"]["top5_identity"]["gained_rows"] == 2
    assert rb["metrics"]["top5_identity"]["unchanged_incorrect_rows"] == 1  # target is only rank 6
    assert result["comparisons"]["rank_vs_ce"]["metrics"]["top1_identity"]["gained_rows"] == 2
    assert result["recommendation"]["status"] == "ranking_promising"
    assert result["recommendation"]["provisional"] is True
    assert result["weights_automatically_promoted"] is False
    assert result["bootstrap"]["replicas"] == 2000
    assert result["comparisons"]["rank_vs_ce"]["metrics"]["top1_identity"]["ci_includes_zero"]
    assert "uncertain" in result["comparisons"]["rank_vs_ce"]["uncertainty"]


def test_losses_are_retained_and_top5_decline_disqualifies(reports):
    baseline, ce, ranking = reports
    ranking["rows"][0] = row_record(100, "CCO", "C2H6O", [])
    refresh(ranking)
    result = compare_pilot(*reports)
    metric = result["comparisons"]["rank_vs_baseline"]["metrics"]["top1_identity"]
    assert metric["lost_rows"] == 1 and metric["lost_row_ids"] == [100]
    assert metric["gained_rows"] == 4 and metric["delta_percentage_points"] == 50.0
    # Isolate a gain in first-choice ordering while deleting two previous Top5 hits.
    ranking["rows"] = copy.deepcopy(baseline["rows"])
    ranking["rows"][3] = row_record(103, "C1CC1", "C3H6", [candidate("C1CC1")])
    ranking["rows"][1] = row_record(101, "COC", "C2H6O", [])
    ranking["rows"][4] = row_record(104, "CF", "CH3F", [])
    refresh(ranking)
    result = compare_pilot(*reports)
    comparison = result["comparisons"]["rank_vs_baseline"]
    assert comparison["metrics"]["top1_identity"]["delta_percentage_points"] > 0
    assert comparison["metrics"]["top5_identity"]["delta_percentage_points"] < 0
    assert comparison["meets_provisional_criterion"] is False
    assert result["recommendation"]["status"] == "ce_only_promising"


@pytest.mark.parametrize("key,new_value", [
    ("length_penalty", 1.0), ("decode_protocol", "different_decoder"),
    ("preprocessing", {"frequency_scale": 2000}), ("vocabulary", ["<pad>", "different"]),
    ("versions", {"torch": "different", "numpy": "2.2.6", "rdkit": "2025.09.1", "selfies": "2.2.0"}),
    ("float32_matmul_precision", "high"), ("torch_num_threads", 8), ("cuda_version", "different"),
])
def test_rejects_inference_protocol_changes(reports, key, new_value):
    reports[2]["provenance"][key] = new_value
    with pytest.raises(ValueError, match="inference_provenance"):
        compare_pilot(*reports)


def test_rejects_source_hash_changes_and_missing_provenance(reports):
    reports[2]["provenance"]["source_sha256"]["diffusion/spectrum.py"] = sha("changed code")
    with pytest.raises(ValueError, match="inference_provenance"):
        compare_pilot(*reports)
    reports[2]["provenance"].pop("length_penalty")
    with pytest.raises(ValueError, match="incomplete inference provenance"):
        compare_pilot(*reports)


def test_rejects_selection_order_reference_and_partial_results(reports):
    original = copy.deepcopy(reports)
    report = reports[2]
    report["selected_rows"].reverse()
    report["selection"]["rows"] = report["selected_rows"].copy()
    report["selection"]["rows_sha256"] = sha(json.dumps(report["selected_rows"], separators=(",", ":")))
    report["rows"].reverse()
    with pytest.raises(ValueError, match="paired selection"):
        compare_pilot(*reports)
    reports = copy.deepcopy(original)
    reports[2]["complete"] = False
    with pytest.raises(ValueError, match="must be complete"):
        compare_pilot(*reports)
    reports = copy.deepcopy(original)
    reports[2]["rows"][0] = row_record(100, "COC", "C2H6O", [candidate("COC")])
    refresh(reports[2])
    with pytest.raises(ValueError, match="reference identity"):
        compare_pilot(*reports)


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r["rows"][0].update(top1_identity=False), "row flags"),
    (lambda r: r["rows"][5].update(top5_identity=True), "row flags"),
    (lambda r: r["rows"][0]["candidates"][0].update(formula="C99"), "candidate formula"),
    (lambda r: r["rows"][0]["candidates"][0].update(smiles="bad smiles"), "valid connected"),
    (lambda r: r.update(top1_identity_fraction=0.0), "reported summary"),
    (lambda r: r["configuration"].update(beam_size=20), "configuration and provenance beam_size"),
    (lambda r: r.update(data_sha256=sha("different data")), "data_sha256 differ"),
    (lambda r: r["selection"].update(split_sha256=sha("different split")), "selection split_sha256"),
    (lambda r: r.update(ema_weights=False), "EMA"),
    (lambda r: r["configuration"].update(split="test"), "validation"),
    (lambda r: r.update(attempted_rows=6.0), "attempted_rows"),
])
def test_rejects_tampered_flags_chemistry_hashes_and_configuration(reports, mutation, message):
    mutation(reports[2])
    with pytest.raises(ValueError, match=message):
        compare_pilot(*reports)


def test_matched_spectral_controls_are_accepted_as_diagnostic_only(reports):
    for report in reports:
        report["configuration"].update(spectrum_control="shuffled", control_seed=87)
        report["provenance"].update(spectrum_control="shuffled", control_seed=87)
    result = compare_pilot(*reports)
    assert result["recommendation"]["status"] == "diagnostic_only"
    reports[2]["configuration"]["control_seed"] = 88
    reports[2]["provenance"]["control_seed"] = 88
    with pytest.raises(ValueError, match="inference_provenance"):
        compare_pilot(*reports)


def test_cli_refuses_input_overwrite_and_only_reuses_identical_sources(reports, tmp_path, capsys):
    paths = [tmp_path / f"{name}.json" for name in ("baseline", "ce", "ranking")]
    for path, report in zip(paths, reports):
        path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "comparison.json"
    args = ["--baseline", str(paths[0]), "--ce", str(paths[1]), "--ranking", str(paths[2]), "--output", str(output)]
    main(args)
    original_bytes = output.read_bytes()
    main(args)
    assert output.read_bytes() == original_bytes
    assert "Reusing verified comparison" in capsys.readouterr().out
    paths[2].write_text(paths[2].read_text() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == original_bytes
    assert "different source hashes" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(args[:-1] + [str(paths[0])])
    assert "overwrite an input" in capsys.readouterr().err


def test_cli_mined_report_is_in_source_hashes_and_protected_from_overwrite(reports, tmp_path, capsys):
    paths = {name: tmp_path / f"{name}.json" for name in ("baseline", "ce", "ranking", "mined")}
    for (name, path), report in zip(paths.items(), reports + [mined_report(reports[2])]):
        path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "comparison.json"
    args = [part for name, path in paths.items() for part in (f"--{name}", str(path))]
    result = main(args + ["--output", str(output)])
    assert result["recommendation"]["status"] == "mined_ranking_promising"
    assert result["source_report_sha256"]["mined"] == hashlib.sha256(paths["mined"].read_bytes()).hexdigest()
    before = output.read_bytes()
    main(args + ["--output", str(output)])
    assert output.read_bytes() == before
    paths["mined"].write_text(paths["mined"].read_text() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(args + ["--output", str(output)])
    assert output.read_bytes() == before
    with pytest.raises(SystemExit):
        main(args + ["--output", str(paths["mined"])])
    assert "overwrite an input" in capsys.readouterr().err
