import copy
import json
import numpy as np
import pytest
import torch

from test_compatibility import tiny_experiment, single_thread
from diffusion.data import file_sha256
from diffusion.residual_selector import (ResidualSelector, CandidateBatch, decoder_scores, targets_for,
    partition_validation, corrections, measure, ordered_candidates, fit_with_development)
import diffusion.residual_selector as residual
import train_residual_selector as runner
from train_hard_selector import main as run_mining


def candidate(smiles='C', logp=-1.):
    import selfies as sf
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    return {'smiles': smiles, 'selfies': sf.encoder(smiles), 'log_probability': logp,
            'formula': rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))}


def test_partition_is_deterministic_disjoint_complete():
    rows = list(range(30))
    dev, assessment = partition_validation(rows, 7, 123)
    assert (dev, assessment) == partition_validation(rows, 7, 123)
    assert len(dev) == 7 and len(assessment) == 23
    assert not set(dev) & set(assessment) and set(dev + assessment) == set(rows)
    for size in (0, 30):
        with pytest.raises(ValueError):
            partition_validation(rows, size, 1)


def test_reconstruct_decoder_scores_and_reject_missing_or_reordered_scores():
    c = candidate('CC', -2.)
    assert decoder_scores([c])[0] == pytest.approx(-2. / 4**.6)
    with pytest.raises(ValueError):
        decoder_scores([candidate('C', -3.), candidate('N', -1.)])
    for invalid in ({'smiles': 'C'}, candidate('C', float('nan')), candidate('C', 1.)):
        with pytest.raises(ValueError):
            decoder_scores([invalid])


def test_zero_initialization_bounds_permutation_and_empty_pool(single_thread):
    pools = [[candidate('C', -1.), candidate('N', -1.1)], []]
    batch = CandidateBatch(torch.randn(2, 8), pools)
    original = copy.deepcopy(pools)
    head = ResidualSelector(8)
    delta = corrections(head, batch, 'cpu')
    assert torch.count_nonzero(delta) == 0
    assert [c['smiles'] for c in ordered_candidates(batch, delta, 0)] == ['C', 'N']
    assert ordered_candidates(batch, delta, 1) == [] and pools == original
    result = measure(batch, delta, torch.tensor([0, -1]))
    assert result['rows'] == 2 and result['correct'] == 1 and result['top1'] == .5
    with torch.no_grad():
        head.score[-1].weight.fill_(100.)
        head.score[-1].bias.fill_(100.)
    delta = corrections(head, batch, 'cpu')
    assert delta.abs().max() <= .25
    features, fps = batch.inputs(torch.tensor([0]), 'cpu')
    head.eval()
    assert torch.allclose(head(features, fps.flip(1)), head(features, fps).flip(1))


def test_large_decoder_margin_cannot_be_overturned():
    batch = CandidateBatch(torch.ones(1, 2), [[candidate('C', -.1), candidate('N', -3.)]])
    delta = torch.tensor([[-.25, .25]])
    assert ordered_candidates(batch, delta, 0)[0]['smiles'] == 'C'


def test_missing_target_is_not_injected_or_removed_from_dev_denominator():
    records = [{'reference_smiles': 'N', 'candidates': [candidate('C')]},
               {'reference_smiles': 'C', 'candidates': []}]
    assert targets_for(records).tolist() == [-1, -1]
    batch = CandidateBatch(torch.ones(2, 3), [r['candidates'] for r in records])
    measured = measure(batch, torch.zeros_like(batch.base), targets_for(records))
    assert measured == {'rows': 2, 'correct': 0, 'top1': 0., 'target_present': 0, 'conditional_loss': None}


def test_development_selects_best_epoch_and_restores_it(monkeypatch, single_thread):
    pools = [[candidate('C', -1.), candidate('N', -1.01)]] * 3
    batch = CandidateBatch(torch.eye(3), pools)
    targets = torch.ones(3, dtype=torch.long)
    saved, calls = {}, []
    def fake_corrections(head, dev, device, batch_size):
        calls.append(1)
        return torch.tensor([[0., .1 if len(calls) == 1 else -.1]] * dev.rows)
    monkeypatch.setattr(residual, 'corrections', fake_corrections)
    def capture(history, head, selected):
        saved[history[-1]['epoch']] = copy.deepcopy(head.state_dict())
    head, history, selection = fit_with_development(batch, targets, batch, targets, device='cpu',
        epochs=8, patience=2, batch_size=2, on_epoch=capture)
    assert selection['epoch'] == 1 and not selection['baseline_retained']
    assert history[-1]['epoch'] == 3
    assert all(torch.equal(v, saved[1][k]) for k, v in head.state_dict().items())


def test_development_can_retain_exact_baseline(single_thread):
    pools = [[candidate('C', -.1), candidate('N', -3.)]] * 2
    batch = CandidateBatch(torch.eye(2), pools)
    target = torch.zeros(2, dtype=torch.long)
    head, history, selection = fit_with_development(batch, target, batch, target, device='cpu', epochs=5, patience=2)
    assert selection['epoch'] == 0 and selection['baseline_retained']
    assert history[-1]['epoch'] == 2
    assert torch.count_nonzero(corrections(head, batch, 'cpu')) == 0


def test_real_pipeline_freezes_selection_before_assessment_and_protects_inputs(tiny_experiment, single_thread, monkeypatch):
    args, mining_output, checkpoint, report = tiny_experiment
    run_mining(args + ['--limit', '6'])
    output = mining_output.parent / 'residual'
    protected = {p: file_sha256(p) for p in [checkpoint, report, mining_output / 'training_candidates.json']}
    cache = runner.cache_spectrum_features
    def observe_cache(*pos, **kw):
        label = kw.get('label', pos[6] if len(pos) > 6 else '')
        if label.startswith('Assessing'):
            assert (output / 'selection.json').is_file()
        return cache(*pos, **kw)
    monkeypatch.setattr(runner, 'cache_spectrum_features', observe_cache)
    arguments = ['--checkpoint', str(checkpoint), '--baseline_report', str(report),
                 '--mining_dir', str(mining_output), '--output_dir', str(output),
                 '--data', str(mining_output.parent / 'data.npz'), '--splits', str(mining_output.parent / 'splits.json'),
                 '--device', 'cpu', '--dev_size', '1', '--epochs', '2', '--patience', '1', '--batch_size', '4', '--diagnostic']
    result = runner.main(arguments)
    assert result['diagnostic'] and not result['weights_promoted']
    assert result['development']['baseline']['attempted_rows'] == 1
    assert result['assessment']['baseline']['attempted_rows'] == 2
    config = json.loads((output / 'configuration.json').read_text())
    assert set(config['gradient_training_rows']) == set(range(6))
    assert set(config['development_rows'] + config['assessment_rows']) == {6, 7, 8}
    assert not set(config['development_rows']) & set(config['assessment_rows'])
    assert 9 not in config['gradient_training_rows'] + config['development_rows'] + config['assessment_rows']
    assert all(file_sha256(p) == sha for p, sha in protected.items())
    assert (output / 'residual_selector_results.zip').is_file()
    with pytest.raises(SystemExit):
        runner.main(arguments + ['--resume'])
    (output / 'completion.json').unlink()
    # A restart is deterministic and only uses development for selection.
    repeated = runner.main(arguments + ['--resume'])
    for key in ('development', 'assessment'):
        assert repeated[key] == result[key]
    (output / 'completion.json').unlink()
    with pytest.raises(SystemExit):
        runner.main(arguments + ['--resume', '--partition_seed', '99'])
