"""Saved-coordinate fidelity and failure handling for the offline viewer."""
import json
import re

import numpy as np
import pytest
from rdkit import Chem

from make_viewer import export_smiles_viewer, export_viewer, prediction_payload, smiles_payload


def saved_prediction(tmp_path):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    numbers = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    bonds = np.zeros((len(numbers), len(numbers)), dtype=np.int64)
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[left, right] = bonds[right, left] = int(bond.GetBondTypeAsDouble())
    # Deliberately arbitrary saved test coordinates: never presented as predictions.
    raw = np.arange(len(numbers) * 3, dtype=float).reshape(-1, 3) / 7
    final = raw + np.array([.123456789, -.345678912, .987654321])
    report = {"status": "SYNTHETIC TEST ONLY", "configuration": {"data": "MUST_NOT_READ_TARGET.npz"},
        "graph_candidates": [{"graph_index": 0, "smiles": "CCO", "conformers": [
            {"conformer_index": 0, "geometry_screen_passed": True, "refinement_status": "test_only"},
            {"conformer_index": 1, "sampling_status": "failed", "error": "diagnostic sampling failure"}]}]}
    (tmp_path / "prediction.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "input_ir.json").write_text(json.dumps({"frequencies": [1000., 3000.],
        "intensities": [1., 2.]}), encoding="utf-8")
    arrays = {"graph0_atom_types": numbers, "graph0_bond_types": bonds,
        "graph0_conformer0_raw_coords": raw, "graph0_conformer0_coords": final}
    np.savez(tmp_path / "coordinates.npz", **arrays)
    return report, arrays


def test_payload_preserves_saved_coordinates_and_explicit_failure(tmp_path):
    _, arrays = saved_prediction(tmp_path)
    payload = prediction_payload(tmp_path)
    samples = payload["candidates"][0]["conformers"]
    np.testing.assert_array_equal(samples[0]["raw"]["coords"], arrays["graph0_conformer0_raw_coords"])
    np.testing.assert_array_equal(samples[0]["final"]["coords"], arrays["graph0_conformer0_coords"])
    xyz = samples[0]["final"]["xyz"].splitlines()
    np.testing.assert_allclose(np.array([line.split()[1:] for line in xyz[2:]], float),
        arrays["graph0_conformer0_coords"], rtol=0, atol=5e-10)
    assert samples[1]["raw"] is None and samples[1]["final"] is None
    assert samples[1]["error"] == "diagnostic sampling failure"


def test_offline_html_contains_data_and_safely_escaped_title(tmp_path):
    report, _ = saved_prediction(tmp_path)
    report["status"] = "</script><script>alert('test')</script> __TITLE__"
    (tmp_path / "prediction.json").write_text(json.dumps(report), encoding="utf-8")
    output = export_viewer(tmp_path, title="SYNTHETIC <script>test</script>")
    document = output.read_text(encoding="utf-8")
    assert "<title>SYNTHETIC &lt;script&gt;test&lt;/script&gt;</title>" in document
    assert not re.search(r"<script[^>]+src=", document, re.I)
    embedded = re.search(r'<script type="application/json" id="prediction-data">(.*?)</script>',
                         document, re.S).group(1)
    assert "<" not in embedded
    assert json.loads(embedded)["status"] == report["status"]
    assert "__ENGINE__" not in document and "__APP_JS__" not in document


@pytest.mark.parametrize("mutation", ["nonfinite", "shape", "mismatched_graph", "duplicate_index"])
def test_export_rejects_ambiguous_or_corrupt_saved_data(tmp_path, mutation):
    report, arrays = saved_prediction(tmp_path)
    if mutation == "nonfinite":
        arrays["graph0_conformer0_coords"][0, 0] = np.nan
    elif mutation == "shape":
        arrays["graph0_conformer0_coords"] = np.zeros((2, 3))
    elif mutation == "mismatched_graph":
        report["graph_candidates"][0]["smiles"] = "CCN"
    else:
        report["graph_candidates"][0]["conformers"][1]["conformer_index"] = 0
    (tmp_path / "prediction.json").write_text(json.dumps(report), encoding="utf-8")
    np.savez(tmp_path / "coordinates.npz", **arrays)
    with pytest.raises(ValueError):
        prediction_payload(tmp_path)


def test_failure_report_does_not_need_coordinates_or_invent_a_molecule(tmp_path):
    saved_prediction(tmp_path)
    (tmp_path / "prediction.json").write_text(json.dumps({"status": "no_valid_completed_graph",
        "graph_candidates": []}), encoding="utf-8")
    (tmp_path / "coordinates.npz").unlink()
    payload = prediction_payload(tmp_path)
    assert payload["candidates"] == []
    assert payload["status"] == "no_valid_completed_graph"


def test_smiles_mode_generates_reproducible_nonplanar_conformers_without_ir(tmp_path):
    payload = smiles_payload("CCCCO")
    repeated = smiles_payload("CCCCO")
    assert payload["provenance"]["kind"] == "generated_from_2d"
    assert payload["provenance"]["method"] == "RDKit ETKDGv3"
    assert payload["input_ir"] is None
    candidate = payload["candidates"][0]
    assert candidate["formula"] == "C4H10O"
    assert candidate["decoder_log_probability"] is None
    assert len(candidate["conformers"]) == 3
    for sample, again in zip(candidate["conformers"], repeated["candidates"][0]["conformers"]):
        coords = np.array(sample["final"]["coords"])
        assert coords.shape == (15, 3) and np.isfinite(coords).all()
        assert np.linalg.matrix_rank(coords - coords.mean(axis=0)) == 3
        np.testing.assert_array_equal(coords, again["final"]["coords"])
        assert sample["geometry_screen_passed"] is None
        assert sample["refinement_status"] == "converged"
        molecule = Chem.MolFromMolBlock(sample["final"]["molblock"], removeHs=False)
        assert molecule.GetConformer().Is3D()
        assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == candidate["smiles"]
    output = export_smiles_viewer("CCCCO", tmp_path / "generated.html")
    embedded = re.search(r'<script type="application/json" id="prediction-data">(.*?)</script>',
        output.read_text(encoding="utf-8"), re.S).group(1)
    assert json.loads(embedded)["provenance"]["kind"] == "generated_from_2d"
    assert not (tmp_path / "coordinates.npz").exists()


def test_generated_charged_graph_keeps_its_identity_and_saved_export_still_uses_saved_data(tmp_path):
    generated = smiles_payload("C[NH3+]", num_conformers=1)
    mol = Chem.MolFromMolBlock(generated["candidates"][0]["conformers"][0]["final"]["molblock"], removeHs=False)
    assert Chem.GetFormalCharge(mol) == 1
    _, arrays = saved_prediction(tmp_path)
    saved = prediction_payload(tmp_path)
    assert saved["provenance"]["kind"] == "saved_prediction"
    np.testing.assert_array_equal(saved["candidates"][0]["conformers"][0]["final"]["coords"],
        arrays["graph0_conformer0_coords"])


@pytest.mark.parametrize("smiles,count", [("", 3), ("not a molecule", 3), ("CC.O", 3), ("*CC", 3), ("CC", 0)])
def test_generated_mode_rejects_invalid_or_ambiguous_input(smiles, count):
    with pytest.raises(ValueError):
        smiles_payload(smiles, num_conformers=count)
