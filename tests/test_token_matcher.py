import copy

import pytest
import torch

from diffusion.encoder import SpectrumTokenEncoder
from diffusion.token_matcher import TokenMatcher, identity_contrastive_loss


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def example(train_encoder=False, dropout=0.):
    torch.manual_seed(17)
    encoder = SpectrumTokenEncoder(hidden_dim=16, n_layers=1, n_heads=2, dropout=dropout)
    model = TokenMatcher(encoder, 12, 8, n_heads=2, n_layers=1,
                         train_encoder=train_encoder, dropout=dropout)
    spectra = {"packed": torch.tensor([[[.2, .4], [.7, .6], [0., 0.]],
                                        [[.1, .8], [.5, .3], [.9, .1]]]),
               "mask": torch.tensor([[True, True, False], [True, True, True]])}
    tokens = torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2]])
    return encoder, model, spectra, tokens


def test_matches_use_spectrum_and_molecule_and_are_batch_equivariant():
    _, model, spectra, tokens = example()
    model.eval()
    original = model(spectra, tokens)
    changed = {key: value.flip(0) for key, value in spectra.items()}
    assert not torch.allclose(original, model(changed, tokens), atol=1e-7)
    assert not torch.allclose(original, model(spectra, tokens.flip(0)), atol=1e-7)
    assert torch.allclose(original.flip(0), model(changed, tokens.flip(0)), atol=1e-6)
    hidden, mask = model.encode_molecule(tokens)
    perturbed = tokens.clone()
    perturbed[:, -2] = 8
    hidden_changed, _ = model.encode_molecule(perturbed)
    # The first token aggregates later molecular tokens: no causal masking.
    assert not torch.allclose(hidden[:, 0], hidden_changed[:, 0])
    assert mask.all()


def test_padding_does_not_change_matching_or_alignment():
    _, model, spectra, tokens = example()
    model.eval()
    memory, mask = model.encode_spectrum(spectra)
    hidden, mol_mask = model.encode_molecule(tokens)
    score = model.match(memory, mask, hidden, mol_mask)
    alignment = model.alignment(memory, mask, hidden, mol_mask)
    padded = {"packed": torch.cat((spectra["packed"], torch.full((2, 2, 2), float("nan"))), 1),
              "mask": torch.cat((spectra["mask"], torch.zeros(2, 2, dtype=torch.bool)), 1)}
    padded["packed"][0, 2] = float("nan")
    extra_tokens = torch.cat((tokens, torch.zeros(2, 2, dtype=torch.long)), 1)
    other_memory, other_mask = model.encode_spectrum(padded)
    other_hidden, other_mol_mask = model.encode_molecule(extra_tokens)
    assert torch.allclose(score, model.match(other_memory, other_mask, other_hidden, other_mol_mask), atol=1e-6)
    assert torch.allclose(alignment, model.alignment(other_memory, other_mask, other_hidden, other_mol_mask), atol=1e-6)
    other_memory[~other_mask] = float("nan")
    other_hidden[~other_mol_mask] = float("nan")
    assert torch.allclose(score, model.match(other_memory, other_mask, other_hidden, other_mol_mask), atol=1e-6)


@pytest.mark.parametrize("train_encoder", [False, True])
def test_gradient_flow_and_original_generator_are_independent(train_encoder):
    source, model, spectra, tokens = example(train_encoder)
    original = copy.deepcopy(source.state_dict())
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    memory, mask = model.encode_spectrum(spectra)
    hidden, mol_mask = model.encode_molecule(tokens)
    loss = model.match(memory, mask, hidden, mol_mask).square().mean()
    loss = loss + identity_contrastive_loss(model.alignment(memory, mask, hidden, mol_mask), ["C", "N"])
    loss.backward()
    encoder_grad = [p.grad for p in model.encoder.parameters()]
    assert any(g is not None and torch.count_nonzero(g) for g in encoder_grad) == train_encoder
    assert model.cross_attention.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(torch.equal(original[key], value) for key, value in source.state_dict().items())
    assert all(p.grad is None for p in source.parameters())
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(source.parameters(), model.encoder.parameters()))


def test_frozen_encoder_dropout_and_warmup_mode_transition():
    _, model, spectra, _ = example(dropout=.5)
    model.train()
    assert model.training and not model.encoder.training and model.molecule_encoder.training
    a, _ = model.encode_spectrum(spectra)
    b, _ = model.encode_spectrum(spectra)
    assert torch.equal(a, b) and not a.requires_grad
    model.set_encoder_trainable(True)
    assert model.encoder.training and all(p.requires_grad for p in model.encoder.parameters())
    model.eval()
    assert not model.encoder.training
    model.train()
    assert model.encoder.training
    model.set_encoder_trainable(False)
    assert not model.encoder.training and not any(p.requires_grad for p in model.encoder.parameters())


def test_duplicate_identity_alignment_has_no_false_negative_penalty():
    scores = torch.tensor([[.7, .7], [.7, .7]], requires_grad=True)
    assert identity_contrastive_loss(scores, ["C", "C"]).item() == pytest.approx(0.)
    assert identity_contrastive_loss(scores, ["C", "N"]).item() > 0
    loss = identity_contrastive_loss(torch.tensor([[.8, .1], [.1, .8]], requires_grad=True), ["C", "N"])
    assert torch.isfinite(loss)
    loss.backward()


def test_invalid_inputs_are_rejected_before_attention():
    _, model, spectra, tokens = example()
    for bad in (torch.zeros_like(tokens), torch.tensor([[1, 0, 3, 2]]), tokens.float(), tokens + 12):
        with pytest.raises(ValueError):
            model.encode_molecule(bad)
    bad_spectra = copy.deepcopy(spectra)
    bad_spectra["mask"][0] = False
    with pytest.raises(ValueError):
        model.encode_spectrum(bad_spectra)
    memory, mask = model.encode_spectrum(spectra)
    hidden, mol_mask = model.encode_molecule(tokens)
    with pytest.raises(ValueError):
        model.match(memory[:1], mask[:1], hidden, mol_mask)
    with pytest.raises(ValueError):
        identity_contrastive_loss(torch.ones(2, 2), ["C", "N"], temperature=0)


def test_hidden_adapter_and_rectangular_alignment():
    source, _, spectra, tokens = example()
    model = TokenMatcher(source, 12, 8, hidden_dim=8, n_heads=2, n_layers=1, dropout=0.)
    memory, mask = model.encode_spectrum(spectra)
    hidden, mol_mask = model.encode_molecule(tokens[:1])
    assert memory.shape[-1] == 8
    assert model.alignment(memory, mask, hidden, mol_mask).shape == (2, 1)
