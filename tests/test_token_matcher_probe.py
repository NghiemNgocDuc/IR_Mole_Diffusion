import copy
import json
import numpy as np
import pytest
from test_compatibility import tiny_experiment, single_thread
from test_residual_selector import candidate
from diffusion.data import file_sha256
from train_token_matcher import main as train, prediction_records
import probe_token_matcher as probe


def test_derangements_are_repeatable_grouped_and_singletons_excluded():
    donors, eligible = probe.donor_indices(6, 123)
    assert eligible.all() and sorted(donors.tolist()) == list(range(6))
    assert np.all(donors != np.arange(6))
    assert np.array_equal(donors, probe.donor_indices(6, 123)[0])
    groups = ['A', 'B', 'A', 'B', 'B', 'C']
    donors, eligible = probe.donor_indices(6, 321, groups)
    assert eligible.tolist() == [True, True, True, True, True, False]
    assert all(groups[i] == groups[j] for i, j in enumerate(donors))
    assert donors[-1] == 5
    with pytest.raises(ValueError):
        probe.donor_indices(1, 4)


def test_centered_score_change_distinguishes_ir_offset_from_ranking_effect():
    result = probe.score_change([[1., 2.], []], [[11., 12.], []], [True, True])
    assert result['mean_absolute_logit_change'] == 10
    assert result['mean_absolute_within_pool_centered_change'] == 0
    assert result['matcher_first_choice_changed'] == 0
    result = probe.score_change([[1., 2.]], [[2., 1.]], [True])
    assert result['matcher_first_choice_changed_fraction'] == 1


def test_real_replay_rejects_wrong_weights_and_changed_top_choice():
    records = [{'row': 2, 'reference_smiles': 'C', 'reference_formula': 'CH4',
                'candidates': [candidate('C', -1.), candidate('N', -2.)]}]
    scores = [[1., 2.]]
    saved = prediction_records(records, scores, .1)
    assert probe.validate_replay(records, scores, saved, .1) == 0
    with pytest.raises(ValueError, match='score mismatch'):
        probe.validate_replay(records, [[2., 1.]], saved, .1)
    bad = copy.deepcopy(saved)
    bad[0]['candidates'].reverse()
    with pytest.raises(ValueError, match='changed Top1'):
        probe.validate_replay(records, scores, bad, .1)


def test_probe_replays_all_selected_weights_without_mutating_training_run(tiny_experiment, single_thread, monkeypatch):
    _, base, checkpoint, report = tiny_experiment
    run_dir, output = base.parent / 'matcher', base.parent / 'probe'
    train(['--checkpoint', str(checkpoint), '--baseline_report', str(report), '--output_dir', str(run_dir),
           '--data', str(base.parent / 'data.npz'), '--splits', str(base.parent / 'splits.json'), '--device', 'cpu',
           '--epochs', '2', '--warmup_epochs', '1', '--batch_size', '4', '--dev_size', '1', '--matcher_layers', '1',
           '--progress_interval', '999', '--diagnostic'])
    protected = {str(p): file_sha256(p) for p in run_dir.rglob('*') if p.is_file()}
    # The probe must not call any fitting path.
    import train_token_matcher
    monkeypatch.setattr(train_token_matcher, 'train_epoch', lambda *a, **kw: pytest.fail('Probe attempted training'))
    result = probe.main(['--run_dir', str(run_dir), '--output_dir', str(output), '--device', 'cpu', '--diagnostic'])
    assert any(v.get('real_input_replayed') for v in result.values())
    for finding in result.values():
        if finding.get('baseline_fallback'):
            continue
        assert finding['maximum_replay_score_error'] < .001
        for label, control in finding['controls'].items():
            if not control.get('skipped'):
                assert control['eligible_rows'] == 2
                assert control['control_matcher_summary']['top5_identity_fraction'] == control['real_matcher_summary']['top5_identity_fraction']
    protocol = json.loads((output / 'protocol.json').read_text())
    assert not protocol['training_performed'] and not protocol['checkpoint_selection_performed']
    assert all(file_sha256(p) == digest for p, digest in protected.items())
    assert (output / 'token_matcher_probe_results.zip').exists()
    with pytest.raises(SystemExit):
        probe.main(['--run_dir', str(run_dir), '--output_dir', str(output), '--device', 'cpu', '--diagnostic'])
