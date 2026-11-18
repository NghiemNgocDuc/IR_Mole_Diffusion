import copy
import json
import math
import numpy as np
import pytest
from test_compatibility import tiny_experiment, single_thread
from test_residual_selector import candidate
from diffusion.data import file_sha256
from diffusion.matcher_task_audit import (select_training_indices, validate_negative_plan,
    matched_generated_groups, pool_statistics, paired_task_gap)
from training.train_token_matcher import main as train
import audit_matcher_training_tasks as audit


def record(reference, pool, row=0):
    return {'row': row, 'reference_smiles': reference, 'reference_formula': candidate(reference)['formula'], 'candidates': pool}


def test_training_selection_and_plans_reject_leakage_self_pairs_and_duplicates():
    rows = [1, 3, 4, 8]
    chosen = select_training_indices(rows, rows, 3, 52)
    assert len(set(chosen)) == 3 and chosen == select_training_indices(rows, rows, 3, 52)
    with pytest.raises(ValueError, match='training split'):
        select_training_indices([1, 3, 4, 9], rows, 3, 52)
    plan = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    assert np.array_equal(validate_negative_plan(plan, 4), plan)
    for value in (0, 2, 7):
        invalid = plan.copy()
        invalid[0, 0] = value
        with pytest.raises(ValueError):
            validate_negative_plan(invalid, 4)


def test_matched_questions_inject_training_positive_but_leave_natural_coverage_unchanged():
    original = [record('C', [candidate('N'), candidate('O'), candidate('CC')]),
                record('C', [candidate('C'), candidate('N')], 1)]
    untouched = copy.deepcopy(original)
    matched, indices = matched_generated_groups(original, [candidate('C'), candidate('C')])
    assert original == untouched and indices == [0]
    assert [c['smiles'] for c in matched[0]['candidates']] == ['C', 'N', 'O', 'CC']
    assert pool_statistics(original, [[1, 2, 3], [2, 1]])['coverage'] == .5
    assert pool_statistics(matched, [[4, 1, 2, 3]])['coverage'] == 1
    with pytest.raises(ValueError, match='reference'):
        matched_generated_groups(original, [candidate('N'), candidate('C')])


def test_metrics_count_missing_targets_and_ties_and_match_paired_rows():
    records = [record('CCO', [candidate('CCO'), candidate('COC')]), record('C', []),
               record('N', [candidate('N'), candidate('O')])]
    stats = pool_statistics(records, [[0, 0], [], [1, 0]])
    assert stats['rows'] == 3 and stats['target_present'] == 2
    assert stats['top1_correct'] == 2 and stats['strict_top1_correct'] == 1
    assert stats['target_tied_at_top'] == 1
    assert stats['same_formula_pairs'] == 1 and stats['same_formula_pair_ties'] == 1
    assert stats['same_formula_pair_accuracy'] == 0
    assert stats['mean_cross_entropy_target_present'] == pytest.approx((math.log(2)+math.log1p(math.exp(-1)))/2)
    gap = paired_task_gap(stats, pool_statistics(records, [[2, 0], [], [0, 1]]))
    assert gap['original_only_correct'] == 1 and gap['generated_only_correct'] == 1
    assert gap['generated_minus_original_percentage_points'] == 0
    empty = pool_statistics([], [])
    assert paired_task_gap(empty, empty)['generated_minus_original_percentage_points'] is None
    with pytest.raises(ValueError, match='nonfinite'):
        pool_statistics(records[:1], [[float('nan'), 0]])


def test_saved_plan_hash_and_training_label_guard(tmp_path):
    plan = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    path = tmp_path / 'negative_plan_01.npy'
    np.save(path, plan)
    metadata = {'sha256': [file_sha256(path)]}
    protected = {}
    audit.load_original_plan(tmp_path, metadata, 1, 4, protected)
    assert protected[str(path)] == metadata['sha256'][0]
    plan[0, 0] = 2
    np.save(path, plan)
    with pytest.raises(ValueError, match='hash'):
        audit.load_original_plan(tmp_path, metadata, 1, 4, {})
    with pytest.raises(ValueError, match='held-out'):
        audit.stored_candidate({}, 9, {0, 1}, 5)


def test_real_tiny_train_then_audit_is_read_only_and_uses_training_rows(tiny_experiment, single_thread, monkeypatch):
    _, base, checkpoint, report = tiny_experiment
    run_dir, output = base.parent / 'matcher_tasks_source', base.parent / 'task_audit'
    train(['--checkpoint', str(checkpoint), '--baseline_report', str(report), '--output_dir', str(run_dir),
        '--data', str(base.parent / 'data.npz'), '--splits', str(base.parent / 'splits.json'), '--device', 'cpu',
        '--epochs', '2', '--warmup_epochs', '1', '--batch_size', '4', '--dev_size', '1', '--matcher_layers', '1',
        '--progress_interval', '999', '--diagnostic'])
    protected = {str(p): file_sha256(p) for p in run_dir.rglob('*') if p.is_file()}
    protected.update({str(p): file_sha256(p) for p in (checkpoint, report, base.parent/'data.npz', base.parent/'splits.json')})
    import training.train_token_matcher as train_token_matcher
    monkeypatch.setattr(train_token_matcher, 'train_epoch', lambda *a, **kw: pytest.fail('Audit attempted training'))
    args = ['--run_dir', str(run_dir), '--output_dir', str(output), '--device', 'cpu', '--limit', '4', '--diagnostic']
    results = audit.main(args)
    assert results['training_rows'] == 4 and len(results['arms']) >= 2
    protocol = json.loads((output/'protocol.json').read_text())
    assert set(protocol['sampled_training_rows']).issubset(set(range(6)))
    assert not protocol['training_performed'] and not protocol['test_used']
    assert (output/'matcher_task_audit_results.zip').is_file()
    for summary in results['arms'].values():
        assert summary['maximum_replay_score_error'] <= .001
        assert summary['original_training_four_choices']['target_present'] == 4
        assert summary['original_training_four_choices']['mean_pool_size'] == 4
        assert summary['paired_task_gap']['rows'] == results['matched_rows']
    assert all(file_sha256(p) == digest for p,digest in protected.items())
    with pytest.raises(SystemExit):
        audit.main(args)
