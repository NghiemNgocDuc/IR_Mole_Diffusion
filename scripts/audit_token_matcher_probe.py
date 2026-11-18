"""Verify saved IR controls against the original prediction archive, without weights."""
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_token_matcher import original_pool
from diffusion.residual_selector import decoder_scores


def main():
    output = ROOT / 'outputs/token_matcher_probe_audit_1444546'
    result = {'checks': [], 'arms': {}}
    with ZipFile(ROOT / 'token_matcher_results.zip') as original, ZipFile(ROOT / 'token_matcher_probe_results.zip') as probe:
        assert original.testzip() is None and probe.testzip() is None
        protocol = json.loads(probe.read('protocol.json'))
        findings = json.loads(probe.read('findings.json'))
        cfg = json.loads(original.read('configuration.json'))
        selection = json.loads(original.read('selection.json'))
        for filename, key in [('configuration.json', 'original_configuration_sha256'), ('selection.json', 'original_selection_sha256')]:
            assert hashlib.sha256(original.read(filename)).hexdigest() == protocol[key]
        assert hashlib.sha256((ROOT / 'probe_token_matcher.py').read_bytes()).hexdigest() == protocol['probe_source_sha256']
        assert cfg['assessment_rows'] == protocol['assessment_rows']
        ids = protocol['assessment_rows']
        assert len(ids) == len(set(ids))
        result['rows'] = len(ids)
        result['archive_sha256'] = hashlib.sha256((ROOT / 'token_matcher_probe_results.zip').read_bytes()).hexdigest()
        result['checks'] += ['Both archive CRCs', 'Original configuration and selection hashes', 'Probe source hash', 'Assessment identities and order']
        for label, finding in findings.items():
            arm, mode = label.split('/')
            saved = json.loads(original.read(f'assessment_{arm}_{mode}.json'))
            assert finding['selection'] == selection['arms'][arm][mode] == saved['selection']
            records = saved['rows']
            assert [r['row'] for r in records] == ids
            pools = [original_pool(r) for r in records]
            targets = [next((i for i,c in enumerate(pool) if c['smiles'] == r['reference_smiles']), -1) for r,pool in zip(records,pools)]
            base = [np.asarray(decoder_scores(pool)) for pool in pools]
            formulas = {r['row']: r['reference_formula'] for r in records}
            out = {'selection': finding['selection'], 'controls': {}}
            for control, stats in finding['controls'].items():
                rows = json.loads(probe.read(f'{arm}_{mode}_{control}.json'))['rows']
                mapping = protocol['controls'][control]
                assert [r['row'] for r in rows] == ids
                assert [r['donor_row'] for r in rows] == mapping['spectrum_donor_rows']
                assert sorted(r['donor_row'] for r in rows) == sorted(ids)
                assert [r['row'] for r in rows if r['eligible']] == mapping['eligible_rows']
                counts = {k: [] for k in ('real_selected', 'control_selected', 'real_matcher', 'control_matcher')}
                flips, raw_changes, centered_changes = [], [], []
                pairwise = {'decoder_correct_rows': 0, 'decoder_correct_matcher_prefers_target': 0,
                            'recoverable_rows': 0, 'recoverable_matcher_prefers_target_over_decoder': 0}
                for record, pool, target, b, row in zip(records, pools, targets, base, rows):
                    assert row['candidate_smiles'] == [c['smiles'] for c in pool]
                    real, altered = np.array(row['real_scores']), np.array(row['control_scores'])
                    assert real.shape == altered.shape == b.shape and np.isfinite(real).all() and np.isfinite(altered).all()
                    lookup = {c['smiles']: c['token_match_score'] for c in record['candidates']}
                    assert np.max(np.abs(real - [lookup[c['smiles']] for c in pool]), initial=0) <= protocol['score_tolerance']
                    if not row['eligible']:
                        assert row['donor_row'] == row['row']
                        continue
                    assert row['row'] != row['donor_row']
                    if control.startswith('same_formula'):
                        assert formulas[row['row']] == formulas[row['donor_row']]
                    w = finding['selection']['weight']
                    selected = real if w is None else b + w * real
                    changed = altered if w is None else b + w * altered
                    for name, scores in [('real_selected', selected), ('control_selected', changed), ('real_matcher', real), ('control_matcher', altered)]:
                        counts[name].append(bool(len(scores)) and int(np.argmax(scores)) == target)
                    assert (pool[int(np.argmax(selected))]['smiles'] if len(pool) else None) == (record['candidates'][0]['smiles'] if pool else None)
                    if len(real):
                        flips.append(int(np.argmax(real)) != int(np.argmax(altered)))
                        raw_changes.extend(np.abs(real-altered))
                        centered_changes.extend(np.abs(real-real.mean()-altered+altered.mean()))
                    if target == 0 and len(real)>1:
                        pairwise['decoder_correct_rows'] += 1
                        pairwise['decoder_correct_matcher_prefers_target'] += int(real[0] > max(real[1:]))
                    elif target > 0:
                        pairwise['recoverable_rows'] += 1
                        pairwise['recoverable_matcher_prefers_target_over_decoder'] += int(real[target] > real[0])
                assert len(counts['real_selected']) == stats['eligible_rows']
                for name, flags in counts.items():
                    assert np.isclose(np.mean(flags), stats[name + '_summary']['top1_identity_fraction'])
                for mode_name in ('selected', 'matcher'):
                    a, b = np.array(counts['real_'+mode_name]), np.array(counts['control_'+mode_name])
                    reported = stats[mode_name+'_paired_control_minus_real']['metrics']['top1_identity']
                    assert int((~a & b).sum()) == reported['gained_rows']
                    assert int((a & ~b).sum()) == reported['lost_rows']
                assert sum(flips) == stats['score_changes']['matcher_first_choice_changed']
                assert np.isclose(np.mean(raw_changes), stats['score_changes']['mean_absolute_logit_change'])
                assert np.isclose(np.mean(centered_changes), stats['score_changes']['mean_absolute_within_pool_centered_change'])
                out['controls'][control] = {**{k: {'correct': sum(v), 'rows': len(v), 'percent': 100*np.mean(v)} for k,v in counts.items()},
                    'pairwise_diagnostic': pairwise, 'first_choice_changes': sum(flips),
                    'within_pool_centered_change': float(np.mean(centered_changes))}
            result['arms'][label] = out
        result['checks'] += ['All 16 donor permutations and eligibility lists', 'Same-formula donor composition',
            'All raw candidate pools and replay logits', 'All Top1 counts and paired gains/losses', 'All reported rank and logit changes']
        result['limits'] = ['No model weights loaded: this verifies archived scores, not a new model inference.',
                            'No new checkpoint or mixture selected. Previously inspected validation rows only.']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'verified_analysis.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('Verified:', '; '.join(result['checks']))
    for arm, finding in result['arms'].items():
        print(arm, finding['controls']['shuffled_20261111']['pairwise_diagnostic'])
    print(output / 'verified_analysis.json')


if __name__ == '__main__':
    main()
