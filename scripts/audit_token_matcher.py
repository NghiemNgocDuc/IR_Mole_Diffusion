"""Recompute matcher gains/losses and molecular error groups from a results ZIP."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.data import file_sha256
from diffusion.residual_selector import decoder_scores
from diffusion.spectrum_audit import atomic_json, row_record, summarize_records
from scripts.compare_spectrum_pilot import paired_metrics, BOOTSTRAP_SEED


def original_pool(record):
    if all('decoder_rank' not in c for c in record['candidates']):
        # A decoder-only fallback stores the untouched original ordering.
        decoder_scores(record['candidates'])
        return record['candidates']
    pool = sorted(record['candidates'], key=lambda c: c['decoder_rank'])
    if [c['decoder_rank'] for c in pool] != list(range(1, len(pool) + 1)):
        raise ValueError('Missing or duplicate original decoder ranks.')
    decoder_scores(pool)
    return pool


def audit(archive, data_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh output directory.')
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Damaged results ZIP.')
        cfg = json.loads(z.read('configuration.json'))
        comparison = json.loads(z.read('comparison.json'))
        selection = json.loads(z.read('selection.json'))
        if selection != comparison['selection']:
            raise ValueError('Selection differs between reports.')
        expected_data = [v for k, v in cfg['protected_inputs'].items() if k.endswith('/' + cfg['arguments']['data'].replace('\\', '/')) or k == cfg['arguments']['data']]
        # Original paths may be absolute, or the data argument may have been relative.
        if not expected_data:
            expected_data = [v for k, v in cfg['protected_inputs'].items() if k.endswith('/' + Path(cfg['arguments']['data']).name)]
        if len(expected_data) != 1 or file_sha256(data_path) != expected_data[0]:
            raise ValueError('Dataset hash does not match original run.')
        with np.load(data_path, allow_pickle=False) as data:
            vocabulary = data['vocabulary'].tolist()
            tokens = data['selfies_tokens']
        expected_rows = cfg['assessment_rows']
        if len(set(expected_rows)) != len(expected_rows):
            raise ValueError('Duplicate assessment rows.')
        all_details, results = [], {}
        original_signatures = None
        for arm in ('frozen', 'finetuned'):
            for mode in ('combined', 'matcher'):
                label = arm + '/' + mode
                supplied = json.loads(z.read(f'assessment_{arm}_{mode}.json'))
                records = supplied['rows']
                if supplied['selection'] != selection['arms'][arm][mode] or [r['row'] for r in records] != expected_rows:
                    raise ValueError('Prediction selection or row ordering mismatch.')
                before, after, details = [], [], []
                signatures = []
                for r in records:
                    pool = original_pool(r)
                    signatures.append((r['row'], r['reference_smiles'], [(c['smiles'], c['selfies'], c['log_probability']) for c in pool]))
                    baseline = row_record(r['row'], r['reference_smiles'], r['reference_formula'], pool)
                    rebuilt = row_record(r['row'], r['reference_smiles'], r['reference_formula'], r['candidates'])
                    for metric in ('top1_identity', 'top5_identity', 'topk_identity', 'top1_formula'):
                        if r[metric] != rebuilt[metric]:
                            raise ValueError('Reported row metric differs from predictions.')
                    before.append(baseline)
                    after.append(rebuilt)
                    mol = Chem.MolFromSmiles(r['reference_smiles'])
                    if mol is None or rdMolDescriptors.CalcMolFormula(mol) != r['reference_formula']:
                        raise ValueError('Invalid reference structure or formula.')
                    old, new = (pool[0] if pool else None), (r['candidates'][0] if pool else None)
                    target = next((c for c in pool if c['smiles'] == r['reference_smiles']), None)
                    sequence = ''.join(vocabulary[int(i)] for i in tokens[r['row']] if i >= 3)
                    outcome = ('fixed' if rebuilt['top1_identity'] else 'both_wrong') if not baseline['top1_identity'] else ('both_correct' if rebuilt['top1_identity'] else 'broken')
                    scores = decoder_scores(pool)
                    details.append({'method': label, 'row': r['row'], 'outcome': outcome,
                        'changed_top1': bool(old and new and old['smiles'] != new['smiles']),
                        'reference': r['reference_smiles'], 'formula': r['reference_formula'],
                        'baseline_smiles': old['smiles'] if old else '', 'selected_smiles': new['smiles'] if new else '',
                        'baseline_correct': baseline['top1_identity'], 'selected_correct': rebuilt['top1_identity'],
                        'same_formula_switch': bool(old and new and old['formula'] == new['formula']),
                        'selected_formula_correct': rebuilt['top1_formula'],
                        'target_present': target is not None, 'target_decoder_rank': pool.index(target) + 1 if target else 0,
                        'target_serialization_matches_dataset': target['selfies'] == sequence if target else None,
                        'reference_rings': mol.GetRingInfo().NumRings(), 'reference_heavy_atoms': mol.GetNumHeavyAtoms(),
                        'reference_contains_N': any(a.GetAtomicNum() == 7 for a in mol.GetAtoms()),
                        'reference_contains_O': any(a.GetAtomicNum() == 8 for a in mol.GetAtoms()),
                        'decoder_margin_1_2': scores[0] - scores[1] if len(scores) > 1 else None,
                        'old_selfies_length': len(list(sf.split_selfies(old['selfies']))) if old else None,
                        'new_selfies_length': len(list(sf.split_selfies(new['selfies']))) if new else None})
                if original_signatures is not None and original_signatures != signatures:
                    raise ValueError('Original candidate pools differ across arms.')
                original_signatures = signatures
                recomputed = summarize_records(after)
                if any(recomputed[k] != comparison['arms'][arm][mode]['summary'][k] for k in recomputed):
                    raise ValueError('Summary accuracy differs from recomputed predictions.')
                groups = {}
                for name in ('reference_rings', 'reference_contains_N', 'reference_contains_O', 'target_serialization_matches_dataset'):
                    groups[name] = []
                    for value in sorted({d[name] for d in details}, key=str):
                        subset = [d for d in details if d[name] == value]
                        groups[name].append({'value': value, 'rows': len(subset),
                            'baseline_correct': sum(d['baseline_correct'] for d in subset),
                            'selected_correct': sum(d['selected_correct'] for d in subset),
                            'fixed': sum(d['outcome'] == 'fixed' for d in subset),
                            'broken': sum(d['outcome'] == 'broken' for d in subset)})
                outcome_details = {}
                for outcome in ('fixed', 'broken'):
                    subset = [d for d in details if d['outcome'] == outcome]
                    margins = [d['decoder_margin_1_2'] for d in subset if d['decoder_margin_1_2'] is not None]
                    outcome_details[outcome] = {'rows': len(subset),
                        'same_formula_switch': sum(d['same_formula_switch'] for d in subset),
                        'selected_formula_wrong': sum(not d['selected_formula_correct'] for d in subset),
                        'median_decoder_margin': float(np.median(margins)) if margins else None,
                        'shorter_selected_selfies': sum(d['new_selfies_length'] < d['old_selfies_length'] for d in subset),
                        'longer_selected_selfies': sum(d['new_selfies_length'] > d['old_selfies_length'] for d in subset)}
                results[label] = {'outcomes': dict(Counter(d['outcome'] for d in details)),
                    'changed_top1': sum(d['changed_top1'] for d in details), 'summary': recomputed,
                    'paired': paired_metrics(before, after, BOOTSTRAP_SEED), 'groups': groups,
                    'fixed_broken': outcome_details}
                all_details.extend(details)
        history = {a: json.loads(z.read(a + '/history.json')) for a in ('frozen', 'finetuned')}
    output.mkdir(parents=True)
    result = {'results_zip_sha256': file_sha256(archive), 'rows': len(expected_rows),
        'selection': selection, 'methods': results, 'history': history,
        'causal_ir_dependence_measured': False,
        'limits': 'Descriptive analysis of previously inspected assessment data. Saved scores cannot establish reliance on IR; controlled rescoring requires saved matcher weights.'}
    atomic_json(output / 'analysis.json', result)
    with (output / 'rows.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_details[0]))
        writer.writeheader()
        writer.writerows(all_details)
    lines = ['# Token matcher error audit', '', f'Paired assessment rows: {len(expected_rows)}.', '',
        '| Method | Top1 | Fixed | Broken | Changed first choice |', '|---|---:|---:|---:|---:|']
    for label, entry in results.items():
        counts = entry['outcomes']
        lines.append(f"| {label} | {100*entry['summary']['top1_identity_fraction']:.2f}% | {counts.get('fixed', 0)} | {counts.get('broken', 0)} | {entry['changed_top1']} |")
    selected = selection['selected']['arm'] + '/combined'
    entry = results[selected]
    lines += ['', '## Selected method: fixed versus broken', '', json.dumps(entry['fixed_broken'], indent=2), '',
        '## Reference ring count (overlapping diagnostics must not select a new model)', '',
        '| Rings | Rows | Baseline correct | Selected correct | Fixed | Broken |', '|---|---:|---:|---:|---:|---:|']
    for g in entry['groups']['reference_rings']:
        lines.append('| ' + ' | '.join(str(g[k]) for k in ('value', 'rows', 'baseline_correct', 'selected_correct', 'fixed', 'broken')) + ' |')
    lines += ['', result['limits'], '', 'Do not change checkpoints or mixing weights based on these assessment subgroups.', '']
    (output / 'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='token_matcher_results.zip')
    parser.add_argument('--data', default='data/processed/squirl.npz')
    parser.add_argument('--output', default='outputs/token_matcher_audit_1426578')
    args = parser.parse_args()
    audit(args.results, args.data, args.output)
    print(Path(args.output) / 'ANALYSIS.md')
