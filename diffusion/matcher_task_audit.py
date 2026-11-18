"""Training-task diagnostics; no gradients, model selection or data split changes."""
import copy
import numpy as np


def select_training_indices(training_rows, allowed_rows, limit, seed):
    if (list(training_rows) != list(allowed_rows) or not training_rows
            or len(set(training_rows)) != len(training_rows)):
        raise ValueError('Use the exact original training split.')
    if type(limit) is not int or limit < 1 or type(seed) is not int or seed < 0:
        raise ValueError('Positive sample limit and nonnegative seed required.')
    return np.random.default_rng(seed).permutation(len(training_rows))[:limit].tolist()


def validate_negative_plan(plan, count):
    if plan.shape != (count, 3) or plan.dtype.kind not in 'iu':
        raise ValueError('Original negative plan must have three integer indices per training row.')
    if np.any(plan < 0) or np.any(plan >= count):
        raise ValueError('Negative plan contains an index outside the training split.')
    choices = np.sort(np.column_stack((np.arange(count), plan)), axis=1)
    if np.any(choices[:, 1:] == choices[:, :-1]):
        raise ValueError('Negative plan contains self-negatives or duplicate choices.')
    return plan


def matched_generated_groups(records, positives):
    """Diagnostic-only positive + first three wrong generated identities.

    Positive serialization comes from the training dataset in both tasks. Natural
    decoder pools are never mutated. Exclude rows with fewer than three negatives.
    """
    if len(records) != len(positives):
        raise ValueError('Training reference and pool counts differ.')
    matched, indices = [], []
    for index, (record, positive) in enumerate(zip(records, positives)):
        if positive['smiles'] != record['reference_smiles']:
            raise ValueError('Diagnostic positive differs from training reference.')
        negatives, seen = [], {positive['smiles']}
        for c in record['candidates']:
            if c['smiles'] not in seen:
                negatives.append(c)
                seen.add(c['smiles'])
        if len(negatives) >= 3:
            matched.append({**copy.deepcopy(record), 'candidates': copy.deepcopy([positive, *negatives[:3]])})
            indices.append(index)
    return matched, indices


def pool_statistics(records, scores):
    """Explicit missing-target denominators and strict tie-aware choice metrics."""
    if len(records) != len(scores):
        raise ValueError('Row/score count mismatch.')
    present = correct = strict = ties = same_total = same_wins = same_ties = 0
    losses, widths, flags = [], [], []
    for record, logits in zip(records, scores):
        pool = record['candidates']
        values = np.asarray(logits, dtype=float)
        if values.shape != (len(pool),) or not np.isfinite(values).all():
            raise ValueError('Missing or nonfinite candidate scores.')
        if len({c['smiles'] for c in pool}) != len(pool):
            raise ValueError('Candidate identities must be distinct.')
        widths.append(len(pool))
        target = next((i for i,c in enumerate(pool) if c['smiles'] == record['reference_smiles']), None)
        win = False
        if target is not None:
            present += 1
            best = values.max()
            correct += int(int(values.argmax()) == target)
            other = np.delete(values, target)
            win = bool(not len(other) or values[target] > other.max())
            strict += int(win)
            ties += int(values[target] == best and np.count_nonzero(values == best) > 1)
            losses.append(float(best + np.log(np.exp(values - best).sum()) - values[target]))
            for i,c in enumerate(pool):
                if i != target and c['formula'] == record['reference_formula']:
                    same_total += 1
                    same_wins += int(values[target] > values[i])
                    same_ties += int(values[target] == values[i])
        flags.append(win)
    n = len(records)
    return {'rows': n, 'target_present': present, 'missing_target': n-present,
            'coverage': present/n if n else None, 'top1_correct': correct,
            'top1_fraction': correct/n if n else None, 'strict_top1_correct': strict,
            'strict_top1_fraction': strict/n if n else None,
            'conditional_top1_given_present': correct/present if present else None,
            'target_tied_at_top': ties, 'mean_cross_entropy_target_present': float(np.mean(losses)) if losses else None,
            'mean_pool_size': float(np.mean(widths)) if widths else None,
            'same_formula_pairs': same_total, 'same_formula_pair_wins': same_wins,
            'same_formula_pair_ties': same_ties,
            'same_formula_pair_accuracy': same_wins/same_total if same_total else None,
            'strict_correct_by_row': flags}


def paired_task_gap(original, generated):
    if original['rows'] != generated['rows']:
        raise ValueError('Task comparison requires the same paired rows.')
    left, right = np.array(original['strict_correct_by_row'], dtype=bool), np.array(generated['strict_correct_by_row'], dtype=bool)
    return {'rows': len(left), 'original_only_correct': int((left & ~right).sum()),
            'generated_only_correct': int((~left & right).sum()),
            'both_correct': int((left & right).sum()), 'both_wrong': int((~left & ~right).sum()),
            'generated_minus_original_percentage_points': float(100*(right.mean()-left.mean())) if len(left) else None}
