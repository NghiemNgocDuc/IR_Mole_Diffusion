import copy
import json
import numpy as np
import pytest
import torch

from test_compatibility import tiny_experiment, single_thread
from diffusion.data import file_sha256
from diffusion.hard_selector import make_groups, fit_selector, candidate_scores
from diffusion.compatibility import molecular_fingerprints
from training.train_hard_selector import main, validate_mined


def test_exact_identity_exclusion_matched_counts_and_training_membership():
    records = [{'row': 0, 'reference_smiles': 'CCO',
                'candidates': [{'smiles': s} for s in ('OCC', 'CCN', 'CCN', 'CCC')]},
               {'row': 1, 'reference_smiles': 'N', 'candidates': [{'smiles': 'N'}]}]
    rows, groups = make_groups(records, [0, 1], ['CCO', 'N', 'O', 'C'], 5)
    assert rows == [0]
    assert groups['hard'] == [['CCO', 'CCN', 'CCC']]
    assert len(groups['random'][0]) == 3
    assert len(set(groups['random'][0])) == 3
    with pytest.raises(ValueError, match='training split'):
        make_groups(records, [0], ['C', 'O', 'N'], 5)
    with pytest.raises(ValueError, match='Duplicate'):
        make_groups(records * 2, [0, 1], ['C', 'O', 'N'], 5)


def test_selector_learns_candidate_discrimination_with_variable_groups(single_thread):
    features = torch.eye(3)
    groups = [['C', 'N'], ['N', 'C', 'O'], ['O', 'C', 'N']]
    head, history = fit_selector(features, groups, device='cpu', epochs=60, batch_size=3,
                                seed=4, lr=.01, projection_dim=16)
    assert history[-1]['loss'] < history[0]['loss']
    for row, group in enumerate(groups):
        fps, _ = molecular_fingerprints(group)
        assert candidate_scores(head, features[row:row+1], fps[None]).argmax(-1).item() == 0


def test_end_to_end_and_resume_checks(tiny_experiment, single_thread):
    args, output, checkpoint, report = tiny_experiment
    args += ['--limit', '6']
    protected = [file_sha256(checkpoint), file_sha256(report)]
    result = main(args)
    assert result['validation_rows'] == 3 and result['candidate_pool_unchanged']
    assert result['diagnostic'] and not result['weights_promoted']
    assert protected == [file_sha256(checkpoint), file_sha256(report)]
    assert (output / 'hard_selector_results.zip').is_file()
    config = json.loads((output / 'configuration.json').read_text())
    assert set(config['training_rows']) == set(range(6))
    groups = json.loads((output / 'training_groups.json').read_text())
    assert set(groups['rows']) <= set(range(6))
    assert [len(g) for g in groups['groups']['random']] == [len(g) for g in groups['groups']['hard']]
    random = json.loads((output / 'random_history.json').read_text())
    hard = json.loads((output / 'hard_history.json').read_text())
    assert random[0]['optimizer_steps'] == hard[0]['optimizer_steps']
    with pytest.raises(SystemExit):
        main(args)
    with pytest.raises(SystemExit):
        main(args + ['--resume'])
    # Simulate interruption before completion: reuse mined rows, retrain both heads deterministically.
    (output / 'RESULTS.md').unlink()
    rerun = main(args + ['--resume'])
    assert rerun == result
    (output / 'RESULTS.md').unlink()
    config['source_sha256']['diffusion/hard_selector.py'] = '0' * 64
    (output / 'configuration.json').write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        main(args + ['--resume'])


def test_cache_rejects_heldout_or_changed_references():
    data = {'molecule_ids': np.array(['C', 'N']), 'coords': np.zeros((2, 29, 3))}
    with pytest.raises(ValueError, match='prefix'):
        validate_mined([{'row': 1}], [0], data)
    with pytest.raises(ValueError, match='reference'):
        validate_mined([{'row': 0, 'reference_smiles': 'N', 'candidates': []}], [0], data)
