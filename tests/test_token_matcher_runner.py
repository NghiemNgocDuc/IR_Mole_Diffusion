import copy
import json
import numpy as np
import pytest
from test_compatibility import tiny_experiment, single_thread
from test_residual_selector import candidate
from diffusion.data import file_sha256
import training.train_token_matcher as runner


def test_shuffle_keeps_every_training_row_and_merges_singletons():
    for count in (4, 5, 9, 10):
        batches = runner.batch_indices(count, 4, 42, 1)
        assert sorted(np.concatenate(batches)) == list(range(count))
        assert all(len(b) >= 2 for b in batches)
        assert np.array_equal(np.concatenate(batches), np.concatenate(runner.batch_indices(count, 4, 42, 1)))


def test_selection_counts_missing_targets_and_prefers_baseline_on_ties():
    records = [{'candidates': [candidate('C', -1.), candidate('N', -2.)], 'reference_smiles': 'N'},
               {'candidates': [], 'reference_smiles': 'C'},
               {'candidates': [candidate('C')], 'reference_smiles': 'O'}]
    scores = [[0., 10.], [], [2.]]
    choice, options = runner.select_weight(records, scores)
    assert choice['correct'] == 1 and choice['weight'] != 0
    assert len(options) == len(runner.MIX_WEIGHTS)
    assert runner.orders_for([r['candidates'] for r in records], scores, 0.) == [[0, 1], [], [0]]
    records[0]['reference_smiles'] = 'C'
    choice, _ = runner.select_weight(records, [[0., 0.], [], [0.]])
    assert choice == {'weight': 0., 'correct': 1}
    altered = copy.deepcopy(records)
    for r in altered:
        r['reference_smiles'] = 'F'
    assert runner.orders_for([r['candidates'] for r in records], scores, .3) == runner.orders_for([r['candidates'] for r in altered], scores, .3)
    with pytest.raises(ValueError, match='nonfinite'):
        runner.orders_for([[candidate('C')]], [[float('nan')]], .1)


def test_real_pipeline_seals_both_selections_and_preserves_inputs(tiny_experiment, single_thread, monkeypatch):
    _, prior_output, checkpoint, report = tiny_experiment
    output = prior_output.parent / 'token_matcher'
    protected = {p: file_sha256(p) for p in (checkpoint, report)}
    observed_assessment = []
    score = runner.score_pools
    def observe(*pos, **kw):
        label = kw.get('label', pos[-1])
        if label.startswith('assessment:'):
            selection = json.loads((output / 'selection.json').read_text())
            assert set(selection['arms']) == {'frozen', 'finetuned'}
            assert not selection['assessment_used_for_selection']
            for arm in selection['arms'].values():
                for mode in ('combined', 'matcher'):
                    choice = arm[mode]
                    if choice['epoch']:
                        assert file_sha256(output / choice['checkpoint']) == choice['checkpoint_sha256']
            observed_assessment.append(label)
        return score(*pos, **kw)
    monkeypatch.setattr(runner, 'score_pools', observe)
    args = ['--checkpoint', str(checkpoint), '--baseline_report', str(report), '--output_dir', str(output),
            '--data', str(prior_output.parent / 'data.npz'), '--splits', str(prior_output.parent / 'splits.json'),
            '--device', 'cpu', '--epochs', '2', '--warmup_epochs', '1', '--batch_size', '4', '--dev_size', '1',
            '--matcher_layers', '1', '--progress_interval', '999', '--diagnostic']
    result = runner.main(args)
    assert result['diagnostic'] and not result['weights_promoted'] and result['assessment_rows'] == 2
    assert len(observed_assessment) >= 2
    cfg = json.loads((output / 'configuration.json').read_text())
    assert cfg['gradient_training_rows'] == list(range(6))
    assert set(cfg['development_rows'] + cfg['assessment_rows']) == {6, 7, 8}
    assert not set(cfg['development_rows']) & set(cfg['assessment_rows']) and not cfg['test_used']
    arms = result['selection']['arms']
    assert arms['frozen']['initial_state_sha256'] == arms['finetuned']['initial_state_sha256']
    assert arms['frozen']['final_encoder_sha256'] == cfg['encoder_start_sha256']
    assert arms['finetuned']['final_encoder_sha256'] != cfg['encoder_start_sha256']
    histories = [json.loads((output / arm / 'history.json').read_text()) for arm in runner.ARMS]
    assert [r['negative_plan_sha256'] for r in histories[0]] == [r['negative_plan_sha256'] for r in histories[1]]
    assert all(e['training_rows'] == 6 for h in histories for e in h)
    assert [e['encoder_trainable'] for e in histories[0]] == [False, False]
    assert [e['encoder_trainable'] for e in histories[1]] == [False, True]
    assert all(file_sha256(p) == digest for p, digest in protected.items())
    assert (output / 'token_matcher_results.zip').is_file()
    for arm in result['arms'].values():
        for mode in arm.values():
            assert mode['summary']['attempted_rows'] == 2
            assert mode['summary']['top5_identity_fraction'] == result['baseline']['top5_identity_fraction']
    with pytest.raises(SystemExit):
        runner.main(args)


def test_changed_provenance_rejected_before_training(tiny_experiment, single_thread):
    _, output, checkpoint, report = tiny_experiment
    baseline = json.loads(report.read_text())
    baseline['provenance']['source_sha256']['diffusion/encoder.py'] = '0' * 64
    report.write_text(json.dumps(baseline))
    with pytest.raises(SystemExit):
        runner.main(['--checkpoint', str(checkpoint), '--baseline_report', str(report), '--output_dir', str(output),
            '--data', str(output.parent / 'data.npz'), '--splits', str(output.parent / 'splits.json'), '--device', 'cpu',
            '--epochs', '2', '--dev_size', '1', '--diagnostic'])
    assert not output.exists()
