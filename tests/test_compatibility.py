"""Compatibility remains a separate trained scorer with a fixed, reference-blind pool."""
import copy
import json
import math

import numpy as np
import pytest
import selfies as sf
import torch
from rdkit import Chem

from diffusion.compatibility import (CompatibilityHead, cache_spectrum_features, fit_head,
    identity_contrastive_loss, molecular_fingerprints, rerank_candidates, training_fingerprints)
from diffusion.data import file_sha256
from diffusion.encoder import SpectrumTokenEncoder
from diffusion.spectrum import SpectrumStructureModel
from evaluate_spectrum import main as evaluate
from train_compatibility import main


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_duplicate_identities_are_positive_not_false_negatives():
    scores = torch.zeros((3, 3), requires_grad=True)
    loss = identity_contrastive_loss(scores, ["CCO", "CCO", "N"])
    assert float(loss.detach()) == pytest.approx((2 * math.log(1.5) + math.log(3)) / 3)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert float(identity_contrastive_loss(scores.detach(), ["CCO"] * 3)) == pytest.approx(0.)
    order = torch.tensor([2, 0, 1])
    assert identity_contrastive_loss(scores.detach()[order][:, order], ["N", "CCO", "CCO"]) == loss.detach()


def test_training_fingerprints_read_only_training_ids_and_canonicalize_duplicates():
    class GuardedLabels:
        def __len__(self):
            return 4
        def __getitem__(self, row):
            assert row in (0, 2), "held-out molecular label was accessed"
            return {0: "CCO", 2: "OCC"}[row]
    data = {"molecule_ids": GuardedLabels()}
    fingerprints, ids = training_fingerprints(data, [0, 2], [0, 2])
    assert ids == ["CCO", "CCO"]
    assert torch.equal(fingerprints[0], fingerprints[1])
    for wrong in ([0, 1], [2, 0], [0, 0]):
        with pytest.raises(ValueError, match="training split"):
            training_fingerprints(data, wrong, [0, 2])


def test_frozen_features_are_masked_and_preserve_encoder_weights():
    torch.manual_seed(5)
    encoder = SpectrumTokenEncoder(hidden_dim=8, n_layers=1, n_heads=2, dropout=0.)
    original = {key: value.clone() for key, value in encoder.state_dict().items()}
    data = {"packed": True, "spectra": np.array([[[.1, .2], [.3, .4], [123., 456.]]], dtype=np.float32),
            "spectra_mask": np.array([[True, True, False]])}
    first = cache_spectrum_features(encoder, data, [0], "cpu")
    data["spectra"][0, 2] = [-999., 222.]
    second = cache_spectrum_features(encoder, data, [0], "cpu")
    assert torch.equal(first, second)
    assert not encoder.training and all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(torch.equal(original[key], value) for key, value in encoder.state_dict().items())


def test_real_head_training_improves_toy_pair_retrieval_without_changing_features():
    identities = ["C", "N", "CO", "CCO", "CCN", "CC=O"]
    fingerprints, identities = molecular_fingerprints(identities)
    features = torch.eye(6)
    original = features.clone()
    head, history = fit_head(features, fingerprints, identities, epochs=50, batch_size=6,
                             lr=.005, projection_dim=16, seed=19)
    assert history[-1]["training_loss"] < history[0]["training_loss"] * .2
    assert torch.equal(head(features, fingerprints).argmax(1), torch.arange(6))
    assert torch.equal(features, original)
    assert all(row["training_rows"] == 6 and row["optimizer_steps"] == 1 for row in history)
    assert any(float(parameter.detach().abs().sum()) > 0 for parameter in head.parameters())


def test_singleton_remainder_is_merged_instead_of_dropping_training_pairs():
    fingerprints, ids = molecular_fingerprints(["C", "N", "CO"])
    _, history = fit_head(torch.eye(3), fingerprints, ids, epochs=1, batch_size=2, projection_dim=8)
    assert history[0]["training_rows"] == 3 and history[0]["optimizer_steps"] == 1


def test_reranking_is_reference_blind_stable_and_keeps_all_candidates():
    torch.manual_seed(6)
    head = CompatibilityHead(3, 8)
    candidates = [{"smiles": value, "formula": "unused-for-scoring"} for value in ("CCO", "CO", "N")]
    first = rerank_candidates(head, torch.tensor([1., .4, .2]), candidates)
    altered = copy.deepcopy(candidates)
    for candidate in altered:
        candidate.update(reference_smiles="F", is_correct=True, formula="another wrong label")
    second = rerank_candidates(head, torch.tensor([1., .4, .2]), altered)
    assert [(r["smiles"], r["compatibility_score"]) for r in first] == [(r["smiles"], r["compatibility_score"]) for r in second]
    assert sorted(r["smiles"] for r in first) == sorted(r["smiles"] for r in candidates)
    assert all("compatibility_score" not in candidate for candidate in candidates)
    for parameter in head.parameters():
        parameter.data.zero_()
    assert [row["smiles"] for row in rerank_candidates(head, torch.ones(3), candidates)] == [r["smiles"] for r in candidates]
    assert rerank_candidates(head, torch.ones(3), []) == []


@pytest.fixture
def tiny_experiment(tmp_path):
    torch.manual_seed(4)
    identities = ["C", "N", "O", "CC", "CN", "CO", "CCC", "CCO", "COC", "CF"]
    encodings = [list(sf.split_selfies(sf.encoder(value))) for value in identities]
    vocab = ["<pad>", "<bos>", "<eos>", *sorted({token for tokens in encodings for token in tokens})]
    tokens, atoms, counts = np.zeros((10, 5), dtype=np.int64), np.zeros((10, 29), dtype=np.int64), []
    for row, (identity, encoding) in enumerate(zip(identities, encodings)):
        values = [1, *[vocab.index(token) for token in encoding], 2]
        tokens[row, :len(values)] = values
        mol = Chem.AddHs(Chem.MolFromSmiles(identity))
        counts.append(mol.GetNumAtoms())
        atoms[row, :counts[-1]] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    data, splits, checkpoint, report = [tmp_path / name for name in ("data.npz", "splits.json", "baseline.ckpt", "baseline.json")]
    np.savez(data, coords=np.zeros((10, 29, 3), dtype=np.float32), atom_types=atoms, num_atoms=np.array(counts),
        molecule_ids=np.array(identities), selfies_tokens=tokens, vocabulary=np.array(vocab),
        spectra_packed=np.random.default_rng(5).uniform(.01, .8, (10, 2, 2)).astype(np.float32),
        spectra_mask=np.ones((10, 2), dtype=bool))
    splits.write_text(json.dumps({"schema_version": 1, "sample_count": 10, "data_sha256": file_sha256(data),
        "splits": {"train": list(range(6)), "val": [6, 7, 8], "test": [9]}}))
    model = SpectrumStructureModel(vocabulary=vocab,
        encoder_spec={"hidden_dim": 8, "n_layers": 1, "n_heads": 2, "dropout": 0.},
        decoder_layers=1, heads=2, max_length=5, dropout=0.)
    model.run_metadata = {"data_sha256": file_sha256(data), "split_sha256": file_sha256(splits),
        "preprocessing": None, "spectrum_control": "none", "diagnostic": True}
    model._ema = {name: value.detach().clone() for name, value in model.named_parameters()}
    payload = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}
    model.on_save_checkpoint(payload)
    torch.save(payload, checkpoint)
    evaluate(["--checkpoint", str(checkpoint), "--data", str(data), "--splits", str(splits), "--output", str(report),
              "--selection", "random", "--limit", "3", "--beam_size", "5", "--seed", "20261105"])
    output = tmp_path / "compatibility"
    args = ["--checkpoint", str(checkpoint), "--data", str(data), "--splits", str(splits),
            "--baseline_report", str(report), "--output_dir", str(output), "--device", "cpu",
            "--epochs", "1", "--batch_size", "4", "--projection_dim", "8", "--diagnostic"]
    return args, output, checkpoint, report


def test_real_cpu_training_reranking_report_hashes_and_protected_output(tiny_experiment):
    args, output, checkpoint, report = tiny_experiment
    before = {"checkpoint": file_sha256(checkpoint), "report": file_sha256(report)}
    result = main(args)
    assert result["candidate_pool_unchanged"] is True and result["rows"] == 3
    assert result["recommendation"] == "diagnostic_only"
    assert result["weights_automatically_promoted"] is False
    assert result["baseline_report_sha256"] == before["report"]
    config = json.loads((output / "configuration.json").read_text())
    assert config["training_rows"] == list(range(6)) and config["held_out_labels_used_for_training"] is False
    payload = torch.load(output / "compatibility.pt", weights_only=True)
    assert payload["training_history"][0]["training_rows"] == 6
    head = CompatibilityHead(**payload["spec"])
    head.load_state_dict(payload["state_dict"], strict=True)
    assert (output / "RESULTS.md").is_file()
    assert json.loads((output / "progress/progress.json").read_text())["status"] == "completed"
    assert file_sha256(checkpoint) == before["checkpoint"] and file_sha256(report) == before["report"]
    trained_hash = file_sha256(output / "compatibility.pt")
    with pytest.raises(SystemExit):
        main(args)
    assert file_sha256(output / "compatibility.pt") == trained_hash


def test_refuses_diagnostic_without_opt_in_and_modified_feature_source(tiny_experiment):
    args, output, _, report = tiny_experiment
    with pytest.raises(SystemExit):
        main(args[:-1])
    assert not output.exists()
    baseline = json.loads(report.read_text())
    baseline["provenance"]["source_sha256"]["diffusion/encoder.py"] = "a" * 64
    report.write_text(json.dumps(baseline))
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()


def test_rejects_incomplete_baseline_and_changed_checkpoint_hash(tiny_experiment):
    args, output, _, report = tiny_experiment
    original = json.loads(report.read_text())
    altered = copy.deepcopy(original)
    altered["complete"] = False
    report.write_text(json.dumps(altered))
    with pytest.raises(SystemExit):
        main(args)
    altered = copy.deepcopy(original)
    altered["checkpoint_sha256"] = altered["provenance"]["checkpoint_sha256"] = "b" * 64
    report.write_text(json.dumps(altered))
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()
