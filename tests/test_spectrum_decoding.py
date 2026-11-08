"""Consistent completed-beam scoring and controlled decoder comparisons."""
import copy
from types import SimpleNamespace

import pytest
import torch

from diffusion.spectrum import SpectrumStructureModel
from scripts.compare_decoding import compare_decoding
from tests.test_spectrum_pilot_comparison import reports  # shared valid report fixture


class ScriptedModel:
    """Prefix-dependent deterministic scores, no training/encoder uncertainty."""
    generate = SpectrumStructureModel.generate

    def __init__(self, seed):
        self.seed = seed
        self.hparams = SimpleNamespace(max_length=7, vocabulary=["<pad>", "<bos>", "<eos>", "[C]", "[N]"])
        self.encoder = SimpleNamespace(encode_spectra=lambda _: (torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool)))

    def decode(self, tokens, memory, mask):
        index = sum(int(token) * 7**position for position, token in enumerate(tokens[0]))
        logits = torch.randn(5, generator=torch.Generator().manual_seed(100000*self.seed + index)) * 2
        return logits[None, None, :].expand(1, tokens.shape[1], -1)


def unpruned_completed(model, width=2, penalty=.6):
    active, completed = [([1], 0.)], []
    for _ in range(model.hparams.max_length - 1):
        next_active = []
        for seq, score in active:
            probabilities = model.decode(torch.tensor([seq]), None, None)[0, -1].log_softmax(-1)
            probabilities[:2] = -torch.inf
            if len(seq) == 1:
                probabilities[2] = -torch.inf
            values, indices = probabilities.topk(width)
            for value, token in zip(values.tolist(), indices.tolist()):
                item = (seq + [token], score + value)
                (completed if token == 2 else next_active).append(item)
        # Active hypotheses have equal lengths at each step, so raw probability
        # order is equivalent under both normalization choices.
        active = sorted(next_active, key=lambda item: item[1], reverse=True)[:width]
    best = sorted(completed, key=lambda item: item[1]/len(item[0])**penalty, reverse=True)[:width]
    return [{"selfies": "".join(model.hparams.vocabulary[i] for i in seq[1:-1]), "log_probability": score} for seq, score in best]


def test_consistent_pruning_matches_unbounded_completed_oracle_and_legacy_default():
    different = 0
    for seed in range(40):
        model = ScriptedModel(seed)
        consistent = model.generate(None, 2, score_mode="consistent")
        legacy = model.generate(None, 2, score_mode="legacy")
        assert consistent == unpruned_completed(model)
        assert model.generate(None, 2) == legacy
        different += consistent != legacy
    assert different > 0  # The fixture exercises the actual normalization difference.


@pytest.mark.parametrize("kwargs", [{"score_mode": "bad"}, {"length_penalty": -1}, {"length_penalty": float("nan")}])
def test_invalid_scoring_rejected(kwargs):
    with pytest.raises(ValueError):
        ScriptedModel(1).generate(None, **kwargs)


def test_decoder_comparison_requires_frozen_checkpoint_and_rows(reports):
    baseline = reports[0]
    baseline["provenance"]["beam_size"] = baseline["configuration"]["beam_size"] = 5
    variant = copy.deepcopy(baseline)
    variant["provenance"]["decode_protocol"] = "consistent_generate_v1_valid_unique_candidates"
    variant["candidate_ranking"] = "Consistent decoder order"
    result = compare_decoding(baseline, {"consistent": variant})
    assert result["variants"]["consistent"]["summary"]["top5_identity_fraction"] == .5
    variant["checkpoint_sha256"] = variant["provenance"]["checkpoint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="same checkpoint"):
        compare_decoding(baseline, {"consistent": variant})


def test_decoder_comparison_labels_diagnostic_and_rejects_nonstandard_baseline(reports):
    baseline = reports[0]
    with pytest.raises(ValueError, match="legacy beam-5"):
        compare_decoding(baseline, {"copy": copy.deepcopy(baseline)})
    baseline["provenance"]["beam_size"] = baseline["configuration"]["beam_size"] = 5
    baseline["checkpoint_training_diagnostic"] = baseline["provenance"]["checkpoint_training_diagnostic"] = True
    assert compare_decoding(baseline, {"copy": copy.deepcopy(baseline)})["diagnostic_only"] is True
