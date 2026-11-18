import copy
import numpy as np
import pytest
import selfies as sf
import torch
from diffusion.token_matcher_data import TrainingPairs, tokenize_selfies, spectra_batch


def training_data():
    identities = ['CCO', 'COC', 'CCC', 'CCN', 'CNC', 'CC=O', 'C=CO', 'CCCO']
    texts = [sf.encoder(s) for s in identities]
    vocabulary = ['<pad>', '<bos>', '<eos>', *sorted({t for s in texts for t in sf.split_selfies(s)})]
    return {'molecule_ids': identities, 'vocabulary': vocabulary,
            'selfies_tokens': tokenize_selfies(texts, vocabulary, 30).numpy()}


def test_plans_are_training_only_reproducible_distinct_and_refreshed():
    data = training_data()
    original_ids = data['molecule_ids']
    original_tokens = data['selfies_tokens']
    class Guard:
        def __init__(self, values):
            self.values = values
        def __getitem__(self, index):
            assert index in (0, 1, 2, 3, 4, 5, 6), 'Held-out label read during pair construction'
            return self.values[index]
    data['molecule_ids'], data['selfies_tokens'] = Guard(original_ids), Guard(original_tokens)
    pairs = TrainingPairs(data, list(range(7)), list(range(7)), max_len=30)
    first = pairs.negative_plan(1, 23)
    torch.manual_seed(999)
    assert np.array_equal(first, pairs.negative_plan(1, 23))
    assert not np.array_equal(first, pairs.negative_plan(2, 23))
    for i, negatives in enumerate(first):
        assert len(set(negatives)) == 3 and i not in negatives and np.all(negatives < 7)
        eligible = [j for j in range(7) if pairs.formulas[j] == pairs.formulas[i] and j not in (i, negatives[0])]
        if eligible:
            assert negatives[1] in eligible
    with pytest.raises(ValueError, match='training split'):
        TrainingPairs(data, [0, 1, 2, 7], list(range(7)))


def test_duplicate_identity_bad_sequence_and_short_training_set_rejected():
    data = training_data()
    rows = list(range(8))
    duplicate = copy.deepcopy(data)
    duplicate['molecule_ids'][1] = 'OCC'
    duplicate['selfies_tokens'][1] = duplicate['selfies_tokens'][0]
    with pytest.raises(ValueError, match='Duplicate canonical'):
        TrainingPairs(duplicate, rows, rows)
    wrong = copy.deepcopy(data)
    wrong['selfies_tokens'][1] = wrong['selfies_tokens'][0]
    with pytest.raises(ValueError, match='differs from'):
        TrainingPairs(wrong, rows, rows)
    with pytest.raises(ValueError, match='four training'):
        TrainingPairs(data, [0, 1, 2], [0, 1, 2])
    wrong = copy.deepcopy(data)
    wrong['selfies_tokens'][0, 1] = 0
    with pytest.raises(ValueError, match='right padding'):
        TrainingPairs(wrong, rows, rows)


def test_tokenization_does_not_truncate_or_silently_replace_unknowns():
    vocab = ['<pad>', '<bos>', '<eos>', '[C]', '[O]']
    assert tokenize_selfies(['[C]', '[C][O]'], vocab, 128).tolist() == [[1, 3, 2, 0], [1, 3, 4, 2]]
    for text in ('[N]', '[C]garbage', ''):
        with pytest.raises(ValueError):
            tokenize_selfies([text], vocab, 6)
    with pytest.raises(ValueError, match='no truncation'):
        tokenize_selfies(['[C][O]'], vocab, 3)
    assert tokenize_selfies([], vocab, 5).shape == (0, 0)


def test_spectra_batch_supports_both_explicit_schema_modes():
    data = {'packed': True, 'spectra': np.ones((4, 3, 2)), 'spectra_mask': np.ones((4, 3), dtype=bool)}
    batch = spectra_batch(data, [2, 0], 'cpu')
    assert batch['packed'].shape == (2, 3, 2) and batch['mask'].dtype == torch.bool
    data = {'packed': False, 'spectra': np.ones((4, 40))}
    assert spectra_batch(data, [3], 'cpu').shape == (1, 40)
