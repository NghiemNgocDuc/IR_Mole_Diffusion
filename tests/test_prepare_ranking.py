"""Atomic static negatives recover sidecars and refuse ambiguous or altered artifacts."""
import json

import numpy as np
import pytest
from rdkit import Chem

from diffusion.data import file_sha256
import prepare_ranking


@pytest.fixture
def preparation(tmp_path):
    data, splits, output = [tmp_path / name for name in ("data.npz", "splits.json", "negatives.npz")]
    identities = ["C", "CC", "O", "N"]
    atoms = np.zeros((4, 8), dtype=np.int64)
    counts = []
    for index, identity in enumerate(identities):
        molecule = Chem.AddHs(Chem.MolFromSmiles(identity))
        counts.append(molecule.GetNumAtoms())
        atoms[index, :counts[-1]] = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    np.savez(data, coords=np.zeros((4, 8, 3), dtype=np.float32), atom_types=atoms,
             num_atoms=counts, molecule_ids=identities, spectra=np.ones((4, 16), dtype=np.float32))
    splits.write_text(json.dumps({"schema_version": 1, "sample_count": 4, "data_sha256": file_sha256(data),
        "splits": {"train": [0, 1], "val": [2], "test": [3]}}), encoding="utf-8")
    args = ["--data", str(data), "--splits", str(splits), "--output", str(output), "--seed", "10", "--max_pool", "4"]
    return args, output


def test_completed_reuse_and_metadata_recovery_preserve_archive(preparation):
    args, output = preparation
    first = prepare_ranking.main(args)
    archive_bytes, metadata_bytes = output.read_bytes(), output.with_suffix(".json").read_bytes()
    with np.load(output) as archive:
        assert "schema_version" not in archive and "configuration_json" in archive
    with pytest.raises(SystemExit):
        prepare_ranking.main(args)
    assert prepare_ranking.main(args + ["--resume"]) == first
    assert output.read_bytes() == archive_bytes and output.with_suffix(".json").read_bytes() == metadata_bytes
    output.with_suffix(".json").unlink()
    assert prepare_ranking.main(args + ["--resume"]) == first
    assert output.read_bytes() == archive_bytes and output.with_suffix(".json").read_bytes() == metadata_bytes


def test_interrupted_metadata_commit_recovers_without_recomputing(preparation, monkeypatch):
    args, output = preparation
    original = prepare_ranking.atomic_json
    def interrupted(*arguments, **keywords):
        raise OSError("simulated metadata write interruption")
    monkeypatch.setattr(prepare_ranking, "atomic_json", interrupted)
    with pytest.raises(SystemExit):
        prepare_ranking.main(args)
    assert output.is_file() and not output.with_suffix(".json").exists()
    original_bytes = output.read_bytes()
    monkeypatch.setattr(prepare_ranking, "atomic_json", original)
    monkeypatch.setattr(prepare_ranking, "build_negative_rows", lambda *a, **k: pytest.fail("must reuse verified archive"))
    prepare_ranking.main(args + ["--resume"])
    assert output.read_bytes() == original_bytes and output.with_suffix(".json").is_file()


def test_interrupted_archive_write_leaves_no_partial_target(preparation, monkeypatch):
    args, output = preparation
    original = prepare_ranking.np.savez_compressed
    def interrupted(handle, **arrays):
        handle.write(b"partial archive")
        raise OSError("simulated archive write interruption")
    monkeypatch.setattr(prepare_ranking.np, "savez_compressed", interrupted)
    with pytest.raises(SystemExit):
        prepare_ranking.main(args)
    assert not output.exists() and not output.with_suffix(".json").exists()
    assert not list(output.parent.glob(".negatives.npz.*.tmp"))
    monkeypatch.setattr(prepare_ranking.np, "savez_compressed", original)
    prepare_ranking.main(args + ["--resume"])
    assert output.exists()


@pytest.mark.parametrize("name,value", [("--seed", "11"), ("--max_pool", "5")])
def test_resume_rejects_changed_settings_without_overwriting(preparation, name, value):
    args, output = preparation
    prepare_ranking.main(args)
    before = output.read_bytes(), output.with_suffix(".json").read_bytes()
    with pytest.raises(SystemExit):
        prepare_ranking.main(args + ["--resume", name, value])
    assert before == (output.read_bytes(), output.with_suffix(".json").read_bytes())


def test_legacy_complete_archive_is_reusable_but_missing_sidecar_is_ambiguous(preparation):
    args, output = preparation
    report = prepare_ranking.main(args)
    with np.load(output) as archive:
        legacy_arrays = {key: archive[key].copy() for key in archive.files if key != "configuration_json"}
    np.savez_compressed(output, **legacy_arrays)
    report["output_sha256"] = file_sha256(output)
    output.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
    before = output.read_bytes(), output.with_suffix(".json").read_bytes()
    assert prepare_ranking.main(args + ["--resume"]) == report
    assert before == (output.read_bytes(), output.with_suffix(".json").read_bytes())
    output.with_suffix(".json").unlink()
    with pytest.raises(SystemExit):
        prepare_ranking.main(args + ["--resume"])
    assert output.read_bytes() == before[0] and not output.with_suffix(".json").exists()


def test_corrupt_archive_and_tampered_metadata_are_not_silently_replaced(preparation):
    args, output = preparation
    prepare_ranking.main(args)
    original_archive = output.read_bytes()
    report = json.loads(output.with_suffix(".json").read_text())
    report["held_out_labels_used"] = True
    output.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit):
        prepare_ranking.main(args + ["--resume"])
    assert output.read_bytes() == original_archive
    output.write_bytes(b"invalid partial archive")
    with pytest.raises(SystemExit):
        prepare_ranking.main(args + ["--resume"])
    assert output.read_bytes() == b"invalid partial archive"
