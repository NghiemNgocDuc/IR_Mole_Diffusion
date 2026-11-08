"""Training-only mining, generated-token semantics, and resumable provenance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diffusion.data import NativeNPZDataset, file_sha256
from diffusion.ranking import (RankingDataset, canonical_identity, encode_negative,
                              load_negative_tokens, validate_token_sequences)
from diffusion.spectrum_audit import select_rows
import mine_ranking


def data_fixture():
    # An invalid held-out label proves candidate construction/validation never
    # attempts to read or parse held-out identities.
    return {"molecule_ids": np.asarray(["CCO", "COC", "N", "C", "DO-NOT-PARSE-HELDOUT"]),
        "vocabulary": ["<pad>", "<bos>", "<eos>", "[C]", "[O]", "[N]"],
        "selfies_tokens": np.asarray([[1, 3, 3, 4, 2, 0], [1, 3, 4, 3, 2, 0],
                                     [1, 5, 2, 0, 0, 0], [1, 3, 2, 0, 0, 0], [1, 4, 2, 0, 0, 0]]),
        "coords": np.zeros((5, 29, 3), dtype=np.float32), "atom_types": np.zeros((5, 29), dtype=np.int64),
        "num_atoms": np.ones(5, dtype=np.int64), "mask": np.zeros((5, 29), dtype=bool),
        "spectra": np.asarray([[[i/10, 1.], [.8, .2]] for i in range(5)], dtype=np.float32),
        "spectra_mask": np.ones((5, 2), dtype=bool), "packed": True,
        "bond_types": None, "graph": None, "preprocessing": {"normalization": "fixture"}}


def archive_fixture(path, data, **overrides):
    sequence, identity = encode_negative("[O]", data["vocabulary"], 6, "CCO")
    provenance = {"data_sha256": "data", "split_sha256": "split", "checkpoint_sha256": "checkpoint",
        "vocabulary": data["vocabulary"], "preprocessing": data["preprocessing"],
        "split": "train", "held_out_labels_used": False, "limit": 4, "seed": 7}
    payload = {"schema_version": np.asarray(2), "training_rows": np.arange(4),
        "negative_rows": np.asarray([1, 0, 0, 2]), "data_sha256": np.asarray("data"),
        "split_sha256": np.asarray("split"), "provenance_json": np.asarray(json.dumps(provenance)),
        "mined_rows": np.asarray([0]), "mined_tokens": sequence[None, :],
        "mined_smiles": np.asarray([identity])}
    payload.update(overrides)
    np.savez(path, **payload)
    return payload


def test_generated_negative_outside_dataset_is_used_without_reading_heldout(tmp_path):
    data = data_fixture()
    path = tmp_path / "mined.npz"
    archive_fixture(path, data)
    negatives = load_negative_tokens(path, np.arange(4), data, "data", "split", "checkpoint")
    assert negatives[0].tolist() == [1, 4, 2, 0, 0, 0]
    np.testing.assert_array_equal(negatives[1:], data["selfies_tokens"][[0, 0, 2]])
    dataset = RankingDataset(NativeNPZDataset(data, np.arange(4)), negative_tokens=negatives)
    np.testing.assert_array_equal(dataset[0]["negative_selfies_tokens"], negatives[0])
    assert "negative_selfies_tokens" not in NativeNPZDataset(data, np.arange(4))[0]


def test_old_negative_archive_works_unchanged(tmp_path):
    data = data_fixture()
    path = tmp_path / "legacy.npz"
    np.savez(path, training_rows=np.arange(4), negative_rows=[1, 0, 0, 2],
             data_sha256=np.asarray("data"), split_sha256=np.asarray("split"))
    np.testing.assert_array_equal(load_negative_tokens(path, np.arange(4), data, "data", "split"),
                                  data["selfies_tokens"][[1, 0, 0, 2]])


@pytest.mark.parametrize("change,message", [
    ({"mined_rows": np.asarray([4])}, "training split"),
    ({"mined_rows": np.asarray([0, 0])}, "unique"),
    ({"negative_rows": np.asarray([4, 0, 0, 2])}, "training split"),
    ({"mined_tokens": np.asarray([[1, 3, 3, 4, 2, 0]]), "mined_smiles": np.asarray(["CCO"])}, "target identity"),
    ({"mined_tokens": np.asarray([[1, 4, 2, 0, 0, 0]]), "mined_smiles": np.asarray(["N"])}, "identity differ"),
    ({"mined_tokens": np.asarray([[1, 99, 2, 0, 0, 0]])}, "unknown vocabulary"),
    ({"mined_tokens": np.asarray([[1, 4, 2, 3, 0, 0]])}, "then PAD"),
    ({"mined_tokens": np.asarray([[1, 4, 0, 0, 0, 0]])}, "exactly one EOS"),
])
def test_archive_rejects_leakage_and_generated_token_corruption(tmp_path, change, message):
    data = data_fixture()
    path = tmp_path / "mined.npz"
    archive_fixture(path, data, **change)
    with pytest.raises(ValueError, match=message):
        load_negative_tokens(path, np.arange(4), data, "data", "split", "checkpoint")


def test_archive_requires_data_split_checkpoint_vocabulary_and_preprocessing(tmp_path):
    data = data_fixture()
    path = tmp_path / "mined.npz"
    payload = archive_fixture(path, data)
    for hash_args in [("different", "split", "checkpoint"), ("data", "different", "checkpoint"),
                      ("data", "split", "different")]:
        with pytest.raises(ValueError):
            load_negative_tokens(path, np.arange(4), data, *hash_args)
    for key, value in [("vocabulary", ["wrong"]), ("preprocessing", {}),
                       ("held_out_labels_used", True), ("split", "val")]:
        provenance = json.loads(str(payload["provenance_json"].item()))
        provenance[key] = value
        archive_fixture(path, data, provenance_json=np.asarray(json.dumps(provenance)))
        with pytest.raises(ValueError):
            load_negative_tokens(path, np.arange(4), data, "data", "split")


def test_encoding_preserves_actual_selfies_path_and_rejects_oov_overlength_same_graph():
    vocabulary = data_fixture()["vocabulary"]
    tokens, identity = encode_negative("[O][C][C]", vocabulary, 6, "COC", "CCO")
    assert tokens.tolist() == [1, 4, 3, 3, 2, 0] and identity == "CCO"
    with pytest.raises(ValueError, match="overlength"):
        encode_negative("[C]" * 5, vocabulary, 6, "N")
    with pytest.raises(ValueError, match="unknown vocabulary"):
        encode_negative("[F]", vocabulary, 6, "N")
    with pytest.raises(ValueError, match="target identity"):
        encode_negative("[O][C][C]", vocabulary, 6, "CCO")
    with pytest.raises(ValueError, match="identity differ"):
        encode_negative("[C]", vocabulary, 6, "N", "O")


def test_choose_uses_first_valid_wrong_candidate_not_label_formula_or_highest_raw_logprob():
    candidates = [{"smiles": "CCO", "selfies": "[C][C][O]", "logprob": -1},
        {"smiles": "N", "selfies": "[N]", "logprob": -9},
        {"smiles": "COC", "selfies": "[C][O][C]", "logprob": -2}]
    choice = mine_ranking.choose_negative(candidates, "CCO", data_fixture()["vocabulary"], 6)
    assert choice["smiles"] == "N"  # Keep the existing decoder's normalized order.
    assert mine_ranking.choose_negative(candidates[:1], "CCO", data_fixture()["vocabulary"], 6) is None


def resume_fixture():
    data = data_fixture()
    selected = select_rows(np.arange(4), "random", 3, 14)
    row = selected[0]
    tokens, identity = encode_negative("[O]", data["vocabulary"], 6, data["molecule_ids"][row])
    provenance = {"seed": 14}
    record = {"row": row, "reference_smiles": str(data["molecule_ids"][row]),
              "negative": {"smiles": identity, "selfies": "[O]", "tokens": tokens.tolist()}}
    previous = {"schema_version": 1, "provenance": provenance, "selected_rows": selected,
                "complete": False, "rows": [record]}
    return data, selected, provenance, previous


def test_resume_rejects_changed_provenance_order_reference_tokens_or_completion():
    data, selected, provenance, previous = resume_fixture()
    assert len(mine_ranking.validate_mining_resume(previous, provenance, selected, data)) == 1
    changes = [lambda p: p["provenance"].update(seed=15), lambda p: p.update(complete=True),
        lambda p: p["rows"][0].update(row=4), lambda p: p["rows"][0].update(reference_smiles="N"),
        lambda p: p["rows"][0]["negative"].update(tokens=[1, 5, 2, 0, 0, 0]),
        lambda p: p["rows"][0]["negative"].update(smiles="CC"),
        lambda p: p["rows"][0].update(row=True)]
    for change in changes:
        altered = copy.deepcopy(previous)
        change(altered)
        with pytest.raises(ValueError):
            mine_ranking.validate_mining_resume(altered, provenance, selected, data)


def fake_setup(tmp_path, monkeypatch, fail_on=None):
    data = data_fixture()
    data_path, split_path, checkpoint, fallback = [tmp_path / name for name in ("data.npz", "splits.json", "model.ckpt", "static.npz")]
    for path in (data_path, split_path, checkpoint):
        path.write_text(path.name)
    data_hash, split_hash = file_sha256(data_path), file_sha256(split_path)
    np.savez(fallback, training_rows=np.arange(4), negative_rows=[1, 0, 0, 2],
             data_sha256=np.asarray(data_hash), split_sha256=np.asarray(split_hash))
    manifest = {"data_sha256": data_hash, "splits": {"train": [0, 1, 2, 3], "val": [4], "test": []}}
    # parseable JSON contents; schema is deliberately intercepted below.
    split_path.write_text(json.dumps(manifest))
    split_hash = file_sha256(split_path)
    np.savez(fallback, training_rows=np.arange(4), negative_rows=[1, 0, 0, 2],
             data_sha256=np.asarray(data_hash), split_sha256=np.asarray(split_hash))
    class FakeModel:
        def __init__(self):
            self.hparams = SimpleNamespace(vocabulary=data["vocabulary"], max_length=6)
            self.run_metadata = {"data_sha256": data_hash, "split_sha256": split_hash,
                "preprocessing": data["preprocessing"], "diagnostic": True}
            self.calls = 0
        def use_ema_weights(self):
            return True
        def to(self, *args):
            return self
        def eval(self):
            return self
        def generate(self, *args):
            self.calls += 1
            if self.calls == fail_on:
                raise RuntimeError("simulated preemption")
            return [{"selfies": "[O]", "logprob": -2.}]
    monkeypatch.setattr(mine_ranking, "load_native_npz", lambda path: data)
    monkeypatch.setattr(mine_ranking, "validate_split_manifest", lambda *args: manifest)
    monkeypatch.setattr(mine_ranking.SpectrumStructureModel, "load_from_checkpoint", lambda path: FakeModel())
    args = SimpleNamespace(checkpoint=str(checkpoint), data=str(data_path), splits=str(split_path),
        fallback=str(fallback), output=str(tmp_path / "mined.npz"), limit=3, seed=19, beam_size=5,
        device="cpu", save_every=1, progress_interval=0., resume=False, allow_diagnostic=True)
    return args, data, manifest


def test_mining_interrupt_resume_completed_reuse_and_no_silent_overwrite(tmp_path, monkeypatch):
    args, data, manifest = fake_setup(tmp_path, monkeypatch, fail_on=2)
    with pytest.raises(RuntimeError, match="preemption"):
        mine_ranking.mine(args)
    records_path = Path(args.output).with_suffix(".mining.json")
    previous = json.loads(records_path.read_text())
    assert not previous["complete"] and len(previous["rows"]) == 1
    assert not Path(args.output).exists()
    # Each replacement is O, absent from every training identity; held-out label
    # remains unparsable throughout. Resume saves only the remaining two rows.
    args, data, manifest = fake_setup(tmp_path, monkeypatch)
    args.resume = True
    result = mine_ranking.mine(args)
    assert result["model_generated_pairs"] == 3 and result["static_fallback_pairs"] == 1
    assert json.loads(records_path.read_text())["rows"][0] == previous["rows"][0]
    saved_hash = file_sha256(args.output)
    assert mine_ranking.mine(args) == result
    assert file_sha256(args.output) == saved_hash
    args.resume = False
    with pytest.raises(ValueError, match="output exists"):
        mine_ranking.mine(args)
    args.resume = True
    args.beam_size = 10
    with pytest.raises(ValueError, match="provenance"):
        mine_ranking.mine(args)


def test_incomplete_summary_cannot_be_silently_replaced(tmp_path, monkeypatch):
    args, _, _ = fake_setup(tmp_path, monkeypatch)
    mine_ranking.mine(args)
    summary_path = Path(args.output).with_suffix(".json")
    summary = json.loads(summary_path.read_text())
    summary["model_generated_pairs"] = 999
    summary_path.write_text(json.dumps(summary))
    args.resume = True
    with pytest.raises(ValueError, match="summary differs"):
        mine_ranking.mine(args)


def test_invalid_token_boundaries_are_rejected():
    for tokens in ([[1, 3, 2, 2]], [[1, 0, 2, 0]], [[1, 3, 3, 0]], [[1., 3., 2., 0.]]):
        with pytest.raises(ValueError):
            validate_token_sequences(np.asarray(tokens), 6)
    assert canonical_identity("OCC") == canonical_identity("CCO")
