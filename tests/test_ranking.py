"""Leakage, objective-gradient, and paired-validation checks for ranking tuning."""
import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from diffusion.data import NativeNPZDataset, collate_native
from diffusion.ranking import RankingDataset, build_negative_rows, load_negative_rows
from diffusion.spectrum import SpectrumStructureModel
from scripts.compare_rank_results import compare_reports


def training_data():
    # Deliberately invalid held-out identity: negative preparation must never read it.
    return {"molecule_ids": np.array(["CCO", "COC", "CCO", "N", "HELD-OUT-DO-NOT-PARSE"]),
        "atom_types": np.array([[6, 6, 8, 1, 1, 1, 1, 1, 1], [6, 8, 6, 1, 1, 1, 1, 1, 1],
            [6, 6, 8, 1, 1, 1, 1, 1, 1], [7, 1, 1, 1, 0, 0, 0, 0, 0], [0]*9]),
        "coords": np.zeros((5, 9, 3)), "num_atoms": np.array([9, 9, 9, 4, 0]),
        "mask": np.array([[True]*9]*3 + [[True]*4+[False]*5, [False]*9]),
        "spectra": np.ones((5, 2, 2), dtype=np.float32), "packed": True,
        "spectra_mask": np.ones((5, 2), dtype=bool), "bond_types": None, "graph": None,
        "selfies_tokens": np.array([[1, 3, 3, 4, 2], [1, 3, 4, 3, 2], [1, 3, 3, 4, 2],
            [1, 5, 2, 0, 0], [1, 5, 5, 2, 0]])}


def test_negative_builder_uses_training_identities_only_and_prefers_formula():
    data, rows = training_data(), np.array([0, 1, 2, 3])
    negatives, kinds = build_negative_rows(data, rows, seed=10)
    assert set(negatives).issubset(set(rows))
    assert np.all(data["molecule_ids"][negatives] != data["molecule_ids"][rows])
    assert negatives[0] == negatives[2] == 1 and negatives[1] in (0, 2)
    np.testing.assert_array_equal(kinds[:3], [1, 1, 1])
    np.testing.assert_array_equal(negatives, build_negative_rows(data, rows, seed=10)[0])
    assert kinds[3] == 3
    with pytest.raises(ValueError):
        build_negative_rows(data, [0, 0])
    with pytest.raises(ValueError):
        build_negative_rows(data, [-1, 0])


def test_ranking_dataset_rejects_held_out_or_same_identity_and_preserves_schema():
    data = training_data()
    standard = NativeNPZDataset(data, [0, 1, 2, 3])
    for negatives in ([4, 0, 1, 0], [2, 0, 1, 0]):
        with pytest.raises(ValueError, match="different identity within the training split"):
            RankingDataset(standard, negatives)
    ranking = RankingDataset(standard, [1, 0, 1, 0])
    batch = collate_native([ranking[0], ranking[1]])
    assert torch.equal(batch["negative_selfies_tokens"], torch.tensor(data["selfies_tokens"][[1, 0]]))
    assert torch.equal(batch["selfies_tokens"], torch.tensor(data["selfies_tokens"][[0, 1]]))
    assert "negative_selfies_tokens" not in standard[0]


def test_negative_archive_hash_order_and_split_checks(tmp_path):
    path = tmp_path / "negatives.npz"
    def save(rows=(0, 1), negatives=(1, 0)):
        np.savez(path, training_rows=rows, negative_rows=negatives,
            data_sha256=np.asarray("data"), split_sha256=np.asarray("split"))
    save()
    np.testing.assert_array_equal(load_negative_rows(path, [0, 1], "data", "split"), [1, 0])
    with pytest.raises(ValueError, match="hashes"):
        load_negative_rows(path, [0, 1], "different", "split")
    with pytest.raises(ValueError, match="order"):
        load_negative_rows(path, [1, 0], "data", "split")
    save(negatives=(2, 0))
    with pytest.raises(ValueError, match="training split"):
        load_negative_rows(path, [0, 1], "data", "split")


def tiny_model(**kwargs):
    model = SpectrumStructureModel(vocabulary=["<pad>", "<bos>", "<eos>", "[C]", "[O]", "[N]"],
        encoder_spec={"hidden_dim": 16, "n_layers": 1, "n_heads": 4, "dropout": 0.},
        decoder_layers=1, heads=4, max_length=5, dropout=0., label_smoothing=0., **kwargs)
    model.log = lambda *args, **kwargs: None
    return model


def test_ranking_loss_matches_pairwise_scores_and_backpropagates_through_both_branches():
    torch.manual_seed(19)
    model = tiny_model(ranking_weight=.2, ranking_margin=100.)
    data = training_data()
    # Distinct spectra keep the two reciprocal pairs from cancelling as an
    # algebraic identity when the hinge is active for both examples.
    data["spectra"][1] = [[.1, .8], [.6, .2]]
    batch = collate_native([RankingDataset(NativeNPZDataset(data, [0, 1, 2, 3]), [1, 0, 1, 0])[i] for i in (0, 1)])
    logits = model(batch["spectra"], batch["selfies_tokens"][:, :-1])
    negative_logits = model(batch["spectra"], batch["negative_selfies_tokens"][:, :-1])
    labels, negatives = batch["selfies_tokens"][:, 1:], batch["negative_selfies_tokens"][:, 1:]
    nll = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=0, reduction="none")
    negative_nll = F.cross_entropy(negative_logits.transpose(1, 2), negatives, ignore_index=0, reduction="none")
    score = lambda losses, tokens: -losses.sum(1) / (tokens.ne(0).sum(1)+1).float().pow(.6)
    hinge = F.relu(100. - score(nll, labels) + score(negative_nll, negatives)).mean()
    expected = (nll.sum(1)/labels.ne(0).sum(1)).mean() + .2*hinge
    actual = model.step(batch, "train")
    torch.testing.assert_close(actual, expected)
    baseline = copy.deepcopy(model)
    baseline.hparams.ranking_weight = 0.
    baseline.step(batch, "train").backward()
    actual.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # Nonzero changes in encoder and decoder gradients show ranking affects the
    # spectrum-conditioned model, rather than merely logging an auxiliary scalar.
    encoder_differences = [(left.grad-right.grad).abs().sum()
        for left, right in zip(model.encoder.parameters(), baseline.encoder.parameters())
        if left.grad is not None and right.grad is not None]
    assert encoder_differences and sum(encoder_differences) > 1e-5
    assert (model.output.weight.grad-baseline.output.weight.grad).abs().sum() > 1e-5
    negative_only_gradient, = torch.autograd.grad(
        score(negative_nll, negatives).mean(), negative_logits, retain_graph=True)
    assert negative_only_gradient.abs().sum() > 0
    without_negatives = {key: value for key, value in batch.items() if key != "negative_selfies_tokens"}
    torch.testing.assert_close(model.step(without_negatives, "val"), baseline.step(without_negatives, "val"))
    with pytest.raises(ValueError, match="requires verified"):
        model.step(without_negatives, "train")


def test_paired_comparison_keeps_baseline_when_top1_fails_and_rejects_test_selection():
    base = {"data_sha256": "data", "identity_metric": "same", "configuration": {
        "split": "val", "limit": 2, "beam_size": 5, "spectrum_control": "none"},
        "rows": [{"row": 1, "reference_smiles": "CC", "top1_identity": True,
            "topk_identity": True, "top1_formula": True}]}
    candidate = copy.deepcopy(base)
    candidate["rows"][0]["top1_identity"] = False
    result = compare_reports(base, candidate)
    assert result["paired_metrics"]["top1_identity"]["lost_rows"] == 1
    assert result["recommendation"].startswith("Keep the baseline")
    base["configuration"]["split"] = candidate["configuration"]["split"] = "test"
    with pytest.raises(ValueError, match="not the test"):
        compare_reports(base, candidate)
