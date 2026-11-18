"""Audit a saved OPLS benchmark ZIP and produce a plain-language comparison.

Reads JSON/NumPy artifacts only; never executes exported Python or reruns engines.
Requires NumPy, but not Torch, RDKit, OpenMM, a GPU, or access to Unity.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np

ARMS = ('unguided', 'opls_guided')


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_read(content):
    def invalid(value):
        raise ValueError('Nonfinite JSON number: '+value)
    value = json.loads(content, parse_constant=invalid)
    def finite(item):
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError('Nonfinite JSON number')
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def stats(values):
    return {'n': len(values), 'median': float(np.median(values)) if values else None,
            'mean': float(np.mean(values)) if values else None}


def analyze(path):
    result = {'archive_sha256': digest(Path(path).read_bytes()), 'errors': [], 'warnings': [],
              'scope': 'Saved artifacts and report consistency only; engines not rerun.',
              'physical_correctness_validated': False}
    errors, warnings = result['errors'], result['warnings']
    def check(condition, message):
        if not condition:
            errors.append(message)
        return condition
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if (len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts
                or '\\' in n or ':' in n for n in names)):
            raise ValueError('Ambiguous or unsafe archive names')
        if sum(i.file_size for i in z.infolist()) > 1024**3 or any(i.file_size > 256*1024**2 for i in z.infolist()):
            raise ValueError('Archive exceeds analyzer size limit')
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
        report = json_read(z.read('report.json'))
        config = report['configuration']
        expected = 3*config['per_bin']*config['graphs']*config['conformers']*2
        check(report['planned_attempts'] == expected, 'Planned attempt count disagrees with configuration')
        result.update(run_status=report['status'], planned_attempts=expected,
                      checkpoints=report.get('checkpoints', {}), run_error=report.get('error'))
        if report['status'] != 'completed':
            warnings.append('Run is not fully completed; planned denominators include failures and unattempted slots.')
        inputs = report.get('inputs', [])
        if not inputs:
            warnings.append('No inputs were selected; inspect the run error before resubmitting.')
        else:
            check(len(inputs) == 3*config['per_bin'], 'Selected input count differs from frozen configuration')
            check(len({r['molecule_id'] for r in inputs}) == len(inputs), 'Repeated training identity in input selection')
            check([r['input_index'] for r in inputs] == list(range(len(inputs))), 'Input indices are missing or reordered')
            check(report['selection']['split'] == 'train', 'Unexpected data split')
        for filename, field in [('selection.json', 'selection_sha256'), ('input_ir.npz', 'input_ir_sha256'),
                                ('candidate_plan.json', 'candidate_plan_sha256')]:
            if field in report:
                check(filename in names and digest(z.read(filename)) == report[field], filename+' hash mismatch or absent')
        if 'selection.json' in names:
            selected = json_read(z.read('selection.json'))
            check(selected['selection'] == report.get('selection'), 'Selection metadata changed')
            check(selected['inputs'] == [{k: r[k] for k in ('row', 'molecule_id', 'selection_atom_count', 'selection_bin')}
                                         for r in inputs], 'Selected rows/identities changed')
        if 'input_ir.npz' in names:
            with np.load(io.BytesIO(z.read('input_ir.npz')), allow_pickle=False) as data:
                check(data['rows'].tolist() == [r['row'] for r in inputs], 'Saved spectra rows disagree with selection')
                check(np.isfinite(data['packed']).all(), 'Nonfinite input spectra')
        plan = json_read(z.read('candidate_plan.json')) if 'candidate_plan.json' in names else None
        if plan is None:
            warnings.append('Candidate plan not yet available; no complete comparison can be made.')
        else:
            check(len(plan) == len(inputs), 'Candidate plan input count changed')
        attempts, pairs, failures, setup_seconds, predicted_sizes = [], [], [], [], []
        seen = set()
        for i, row in enumerate(inputs):
            for g in row.get('graphs', []):
                gi = g['graph_index']
                folder = f'input_{i:03d}/graph_{gi}'
                key = (i, gi)
                check(key not in seen, f'{folder}: duplicate graph slot')
                seen.add(key)
                if plan is not None:
                    frozen = plan[i]['graphs'][gi]
                    check(g.get('smiles') == frozen.get('smiles'), folder+': candidate changed since frozen plan')
                    check([{k: a[k] for k in ('arm', 'conformer', 'seed', 'order_in_pair')} for a in g['attempts']] ==
                          [{k: a[k] for k in ('arm', 'conformer', 'seed', 'order_in_pair')} for a in frozen['attempts']],
                          folder+': attempt plan changed')
                parameters = g.get('parameters')
                if parameters:
                    check(parameters.get('ordered_smiles') == g['smiles'], folder+': parameter graph differs from candidate')
                    numbers = parameters['atom_numbers']
                    check(sorted(parameters['model_to_engine']) == list(range(len(numbers))), folder+': invalid atom mapping')
                    manifest_path = folder+'/parameters/manifest.json'
                    if check(manifest_path in names, folder+': missing parameter manifest'):
                        manifest = json_read(z.read(manifest_path))
                        for k, v in manifest.items():
                            check(parameters.get(k) == v, folder+': manifest field differs: '+k)
                    for name, sha in parameters['file_sha256'].items():
                        member = folder+'/parameters/'+name
                        check(member in names and digest(z.read(member)) == sha, folder+': parameter hash differs: '+name)
                if 'setup_seconds' in g:
                    setup_seconds.append(g['setup_seconds'])
                if 'predicted_atom_count' in g:
                    predicted_sizes.append(g['predicted_atom_count'])
                grouped = defaultdict(dict)
                check(len(g['attempts']) == config['conformers']*2, folder+': missing attempt slots')
                for a in g['attempts']:
                    arm, c = a['arm'], a['conformer']
                    label = folder+f'/{arm}_{c}'
                    check(arm in ARMS and 0 <= c < config['conformers'], label+': unknown arm or conformer')
                    check(arm not in grouped[c], label+': duplicate arm')
                    grouped[c][arm] = a
                    check(a['seed'] == config['seed']+(i*config['graphs']+gi)*config['conformers']+c,
                          label+': seed differs from predetermined schedule')
                    attempts.append(a)
                    if a['status'] not in ('completed', 'pending'):
                        failures.append({'input_index': i, 'graph_index': gi, 'smiles': g.get('smiles'),
                                         'arm': arm, 'seed': a['seed'], 'status': a['status'],
                                         'error': a.get('error', g.get('error', ''))})
                    for section in ('raw',):
                        if section in a:
                            check(all(np.isfinite(a[section][k]) for k in ('energy_kj_mol', 'force_rms_kj_mol_nm')),
                                  label+': nonfinite raw metric')
                    check(all(np.isfinite(v) and v >= 0 for v in a.get('timing', {}).values()), label+': invalid timing')
                    if a.get('raw') or a['status'] == 'completed':
                        raw_member = label+'_raw.npy'
                        if check(raw_member in names and parameters is not None, label+': missing raw array or parameters'):
                            raw = np.load(io.BytesIO(z.read(raw_member)), allow_pickle=False)
                            check(raw.shape == (len(parameters['atom_numbers']), 3) and np.isfinite(raw).all(), label+': invalid raw array')
                    if a['status'] == 'completed':
                        if check(label+'.npz' in names and parameters is not None, label+': completed attempt lacks arrays/parameters'):
                            with np.load(io.BytesIO(z.read(label+'.npz')), allow_pickle=False) as data:
                                check(np.array_equal(data['atom_numbers'], parameters['atom_numbers']), label+': atom order differs')
                                check(np.array_equal(data['bond_types'], parameters['bond_types']), label+': topology differs')
                                check(raw_member in names and np.array_equal(data['raw_angstrom'], raw), label+': raw arrays disagree')
                                after = data['relaxed_angstrom']
                                check(after.shape == data['raw_angstrom'].shape and np.isfinite(after).all(), label+': invalid relaxed array')
                                displacement = float(np.sqrt(np.mean(np.sum((after-data['raw_angstrom'])**2, axis=1))))
                                check(np.isclose(displacement, a['repair_rms_angstrom_unaligned'], atol=1e-7, rtol=1e-6), label+': repair displacement disagrees')
                                if not a['relaxation']['accepted']:
                                    check(np.array_equal(after, data['raw_angstrom']), label+': rejected relaxation changed returned coordinates')
                    if 'relaxation' in a:
                        relax = a['relaxation']
                        check(np.isclose(relax['before']['energy_kj_mol'], a['raw']['energy_kj_mol'], atol=1e-7, rtol=1e-9), label+': raw/relaxation initial energy differs')
                        accepted = relax['after']['energy_kj_mol'] <= relax['before']['energy_kj_mol']+1e-7
                        check(relax['accepted'] == accepted, label+': relaxation acceptance disagrees with energy')
                        check(relax['force_converged'] == bool(accepted and relax['after']['force_rms_kj_mol_nm'] <= 1.), label+': convergence flag disagrees')
                    trace = a.get('guidance_trace', [])
                    check(arm != 'unguided' or not trace, label+': unguided arm has corrections')
                    for step in trace:
                        check(step['alpha'] >= config['min_alpha'], label+': correction before noise gate')
                        check(step.get('max_displacement_angstrom', 0) <= config['max_step']+1e-7, label+': correction exceeds cap')
                        check(step['energy_after'] < step['energy_before'] if step['accepted'] else
                              step['energy_after'] == step['energy_before'], label+': correction acceptance/energy mismatch')
                    if arm == 'opls_guided' and a['status'] == 'completed' and not trace:
                        warnings.append(label+': no recorded guidance corrections; check gate and step settings.')
                for c, pair in grouped.items():
                    if not check(set(pair) == set(ARMS), folder+': unpaired conformer'):
                        continue
                    u, v = pair['unguided'], pair['opls_guided']
                    check(u['seed'] == v['seed'], folder+': paired seeds differ')
                    if 'raw' not in u or 'raw' not in v:
                        continue
                    item = {'input_index': i, 'graph_index': gi, 'conformer': c, 'smiles': g.get('smiles'),
                            'predicted_atom_count': g.get('predicted_atom_count'),
                            'raw_energy_delta_kj_mol': v['raw']['energy_kj_mol']-u['raw']['energy_kj_mol'],
                            'raw_force_delta_kj_mol_nm': v['raw']['force_rms_kj_mol_nm']-u['raw']['force_rms_kj_mol_nm']}
                    if all(a.get('relaxation', {}).get('accepted') for a in (u, v)):
                        item['relaxed_energy_delta_kj_mol'] = v['relaxation']['after']['energy_kj_mol']-u['relaxation']['after']['energy_kj_mol']
                    if all('repair_rms_angstrom_unaligned' in a for a in (u, v)):
                        item['repair_delta_angstrom'] = v['repair_rms_angstrom_unaligned']-u['repair_rms_angstrom_unaligned']
                    if all('generation' in a.get('timing', {}) for a in (u, v)) and u['timing']['generation'] > 0:
                        item['generation_time_ratio_guided_over_unguided'] = v['timing']['generation']/u['timing']['generation']
                    pairs.append(item)
        counts = dict(Counter(a['status'] for a in attempts))
        if plan is not None:
            check(len(attempts) == expected, 'Recorded slots do not cover the full frozen plan')
        if report['status'] == 'completed':
            check(counts == {'completed': expected}, 'Completed run contains unfinished/failed slots')
        if 'summary' in report:
            check(report['summary']['status_counts'] == counts, 'Reported status totals disagree with attempts')
            check(report['summary']['raw_pairs_assessed'] == len(pairs), 'Reported paired count disagrees')
        result.update(status_counts=counts, unrecorded_slots=expected-len(attempts), failures=failures, pairs=pairs,
                      raw_pairs_assessed=len(pairs), setup_seconds=stats(setup_seconds), predicted_atom_counts=dict(Counter(predicted_sizes)))
        result['arms'] = {}
        for arm in ARMS:
            group = [a for a in attempts if a['arm'] == arm]
            raw = [a['raw'] for a in group if 'raw' in a]
            relaxed = [a['relaxation'] for a in group if 'relaxation' in a]
            result['arms'][arm] = {'planned': expected//2, 'completed': sum(a['status'] == 'completed' for a in group),
                'raw_assessed': len(raw), 'raw_graph_matches': sum(bool(a.get('coordinate_inferred_graph_matches')) for a in raw),
                'relaxed_assessed': len(relaxed), 'relaxed_converged_graph_matches': sum(bool(a['accepted'] and a['force_converged']
                    and a['after'].get('coordinate_inferred_graph_matches')) for a in relaxed)}
        result['paired_statistics'] = {field: stats([p[field] for p in pairs if field in p]) for field in (
            'raw_energy_delta_kj_mol', 'raw_force_delta_kj_mol_nm', 'relaxed_energy_delta_kj_mol',
            'repair_delta_angstrom', 'generation_time_ratio_guided_over_unguided')}
        result['per_input'] = []
        for i in sorted({p['input_index'] for p in pairs}):
            values = [p for p in pairs if p['input_index'] == i]
            result['per_input'].append({'input_index': i, 'paired_count': len(values),
                'mean_raw_energy_delta_kj_mol': float(np.mean([p['raw_energy_delta_kj_mol'] for p in values])),
                'mean_raw_force_delta_kj_mol_nm': float(np.mean([p['raw_force_delta_kj_mol_nm'] for p in values]))})
        result['inputs_with_lower_mean_raw_energy'] = sum(r['mean_raw_energy_delta_kj_mol'] < 0 for r in result['per_input'])
        result['distinct_predicted_graphs'] = len({g['smiles'] for r in inputs for g in r.get('graphs', []) if 'smiles' in g})
    result['audit_status'] = 'issues_found' if errors else 'saved_artifacts_consistent'
    return result


def markdown(result):
    lines = ['# OPLS benchmark analysis', '', 'Archive audit: '+result['audit_status'],
             'Run status: '+result.get('run_status', 'unreadable'), '', result['scope'], '']
    if result['errors']:
        lines += ['**Do not interpret the comparison until these issues are resolved.**', '',
                  *['- '+e for e in result['errors']]]
    elif result.get('raw_pairs_assessed', 0):
        n = result['raw_pairs_assessed']
        lines += [f"Compared {n} raw pairs out of {result['planned_attempts']//2} planned pairs.",
                  f"Guidance lowered mean raw energy on {result['inputs_with_lower_mean_raw_energy']} / {len(result['per_input'])} assessed inputs.",
                  'This is a descriptive training-data result; seeds from one input are not independent molecules.', '',
                  '| Paired measure | Pairs | Median |', '|---|---:|---:|']
        for field, value in result['paired_statistics'].items():
            median = 'unavailable' if value['median'] is None else f"{value['median']:.6g}"
            lines.append(f"| {field} | {value['n']} | {median} |")
        lines += ['', 'Energy, force, and repair deltas are guided minus unguided; negative means lower.',
                  'Generation time ratio above 1 means guidance took longer. Setup time is separate.', '',
                  '| Arm | Completed / planned | Raw graph matches / assessed | Relaxed converged graph matches / assessed |',
                  '|---|---:|---:|---:|']
        for arm, a in result['arms'].items():
            lines.append(f"| {arm} | {a['completed']} / {a['planned']} | {a['raw_graph_matches']} / {a['raw_assessed']} | "
                         f"{a['relaxed_converged_graph_matches']} / {a['relaxed_assessed']} |")
        lines += ['', f"Distinct predicted graphs: {result['distinct_predicted_graphs']}.",
                  f"Attempt statuses: {result['status_counts']}.",
                  'Compare assessed rates alongside planned denominators; failures are not discarded.']
    else:
        lines += ['No usable paired comparison yet. Read the run error and completion status.']
    if result.get('run_error'):
        lines += ['', 'Run error: '+result['run_error']]
    if result['warnings']:
        lines += ['', 'Notes:', '', *['- '+w for w in result['warnings']]]
    lines += ['', 'Low force-field energy does not establish physical correctness, correct IR identity,',
              'stereochemistry, a global minimum, or Boltzmann populations. No model is promoted by this analysis.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = analyze(args.archive)
    except Exception as exc:
        result = {'audit_status': 'unreadable_or_invalid', 'errors': [f'{type(exc).__name__}: {exc}'],
                  'warnings': [], 'scope': 'No trustworthy comparison produced.', 'physical_correctness_validated': False}
    (args.output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.output/'ANALYSIS.md').write_text(markdown(result), encoding='utf-8')
    print(args.output/'ANALYSIS.md')
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
