"""Independent analyzer tests using explicit synthetic numeric artifacts."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from scripts.analyze_opls_benchmark import analyze, main, markdown


def fixture_members():
    members = {}
    def put(name, value):
        members[name] = json.dumps(value).encode()
        return hashlib.sha256(members[name]).hexdigest()
    def array(name, **arrays):
        data = io.BytesIO()
        np.savez_compressed(data, **arrays)
        members[name] = data.getvalue()
        return hashlib.sha256(members[name]).hexdigest()
    selection = {'split': 'train'}
    config = {'per_bin': 1, 'graphs': 1, 'conformers': 1, 'seed': 100, 'min_alpha': .8, 'max_step': .02}
    report = {'configuration': config, 'planned_attempts': 6, 'status': 'completed', 'selection': selection, 'inputs': []}
    plan, selected = [], []
    for i in range(3):
        identity = {'row': i, 'molecule_id': str(i), 'selection_atom_count': 5+10*i,
                    'selection_bin': ('3-10', '11-18', '19-29')[i]}
        selected.append(identity)
        graph = {'graph_index': 0, 'smiles': 'C', 'predicted_atom_count': 5, 'attempts': [], 'setup_seconds': .5}
        frozen = {'graph_index': 0, 'smiles': 'C', 'attempts': []}
        folder = f'input_{i:03d}/graph_0'
        parameter_content = b'synthetic parameters, not a real OPLS export'
        members[folder+'/parameters/template.mol'] = parameter_content
        parameters = {'ordered_smiles': 'C', 'atom_numbers': [6, 1, 1, 1, 1], 'model_to_engine': list(range(5)),
                      'bond_types': np.zeros((5, 5), dtype=int).tolist(),
                      'file_sha256': {'template.mol': hashlib.sha256(parameter_content).hexdigest()}}
        graph['parameters'] = parameters
        put(folder+'/parameters/manifest.json', parameters)
        for order, arm in enumerate(('unguided', 'opls_guided')):
            attempt = {'arm': arm, 'conformer': 0, 'seed': 100+i, 'order_in_pair': order, 'status': 'completed'}
            frozen['attempts'].append({**attempt, 'status': 'pending'})
            raw = np.arange(15, dtype=float).reshape(5, 3)*(.02 if order else .03)
            energy = 8. if order else 10.
            state = {'energy_kj_mol': energy, 'force_rms_kj_mol_nm': 4. if order else 6.,
                     'coordinate_inferred_graph_matches': True}
            attempt.update(raw=state, relaxation={'before': state,
                'after': {'energy_kj_mol': 1., 'force_rms_kj_mol_nm': .1, 'coordinate_inferred_graph_matches': True},
                'accepted': True, 'force_converged': True},
                repair_rms_angstrom_unaligned=float(np.sqrt(np.mean(np.sum((raw*.5)**2, axis=1)))),
                timing={'generation': 2. if order else 1., 'raw_assessment': .1, 'relaxation': .2})
            if order:
                attempt['guidance_trace'] = [{'alpha': .9, 'accepted': True, 'energy_before': 12., 'energy_after': 11.,
                                             'max_displacement_angstrom': .01}]
            stem = folder+'/'+arm+'_0'
            content = io.BytesIO()
            np.save(content, raw)
            members[stem+'_raw.npy'] = content.getvalue()
            array(stem+'.npz', raw_angstrom=raw, relaxed_angstrom=raw*.5,
                  atom_numbers=parameters['atom_numbers'], bond_types=parameters['bond_types'])
            graph['attempts'].append(attempt)
        report['inputs'].append({**identity, 'input_index': i, 'graphs': [graph]})
        plan.append({**identity, 'input_index': i, 'graphs': [frozen]})
    report['selection_sha256'] = put('selection.json', {'selection': selection, 'inputs': selected})
    report['candidate_plan_sha256'] = put('candidate_plan.json', plan)
    report['input_ir_sha256'] = array('input_ir.npz', rows=[0, 1, 2], packed=np.ones((3, 2, 2)), mask=np.ones((3, 2), dtype=bool))
    report['summary'] = {'status_counts': {'completed': 6}, 'raw_pairs_assessed': 3}
    put('report.json', report)
    return members


def write_zip(tmp_path, members):
    path = tmp_path/'results.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in members.items():
            z.writestr(name, content)
    return path


def mutate_report(members, change):
    report = json.loads(members['report.json'])
    change(report)
    members['report.json'] = json.dumps(report).encode()


def test_complete_comparison_recomputes_deltas_and_timing(tmp_path):
    result = analyze(write_zip(tmp_path, fixture_members()))
    assert result['errors'] == [] and result['warnings'] == []
    assert result['audit_status'] == 'saved_artifacts_consistent'
    assert result['raw_pairs_assessed'] == result['inputs_with_lower_mean_raw_energy'] == 3
    assert result['paired_statistics']['raw_energy_delta_kj_mol']['median'] == -2
    assert result['paired_statistics']['generation_time_ratio_guided_over_unguided']['median'] == 2
    assert result['paired_statistics']['relaxed_energy_delta_kj_mol']['median'] == 0
    assert 'descriptive training-data' in markdown(result)
    assert result['physical_correctness_validated'] is False


@pytest.mark.parametrize('problem,expected', [
    ('seed', 'seed'), ('hash', 'parameter hash'), ('array', 'raw arrays disagree'),
    ('gate', 'before noise gate'), ('cap', 'exceeds cap'), ('convergence', 'convergence flag'),
    ('summary', 'status totals'), ('candidate', 'candidate changed')])
def test_detects_corruption_and_withholds_interpretation(tmp_path, problem, expected):
    members = fixture_members()
    if problem == 'hash':
        members['input_000/graph_0/parameters/template.mol'] = b'changed'
    elif problem == 'array':
        data = io.BytesIO()
        np.save(data, np.ones((5, 3)))
        members['input_000/graph_0/unguided_0_raw.npy'] = data.getvalue()
    else:
        def change(r):
            g = r['inputs'][0]['graphs'][0]
            a = g['attempts'][1]
            if problem == 'seed':
                a['seed'] += 1
            elif problem == 'gate':
                a['guidance_trace'][0]['alpha'] = .5
            elif problem == 'cap':
                a['guidance_trace'][0]['max_displacement_angstrom'] = .1
            elif problem == 'convergence':
                a['relaxation']['force_converged'] = False
            elif problem == 'summary':
                r['summary']['status_counts'] = {'completed': 5}
            elif problem == 'candidate':
                g['smiles'] = 'CO'
        mutate_report(members, change)
    result = analyze(write_zip(tmp_path, members))
    assert result['audit_status'] == 'issues_found'
    assert any(expected in e for e in result['errors'])
    text = markdown(result)
    assert 'Do not interpret' in text and 'Guidance lowered' not in text


def test_partial_run_retains_planned_denominator(tmp_path):
    members = fixture_members()
    def change(r):
        r['status'] = 'partial'
        r['summary'] = {'status_counts': {'completed': 5, 'pending': 1}, 'raw_pairs_assessed': 2}
        a = r['inputs'][2]['graphs'][0]['attempts'][1]
        for k in list(a):
            if k not in ('arm', 'conformer', 'seed', 'order_in_pair'):
                del a[k]
        a['status'] = 'pending'
    mutate_report(members, change)
    result = analyze(write_zip(tmp_path, members))
    assert result['errors'] == [] and result['warnings']
    assert result['planned_attempts'] == 6 and result['raw_pairs_assessed'] == 2
    assert result['arms']['opls_guided']['planned'] == 3
    assert result['arms']['opls_guided']['completed'] == 2


def test_bad_archive_and_nonfinite_json_produce_failure_artifact(tmp_path):
    members = fixture_members()
    report = members['report.json'].decode().replace('"setup_seconds": 0.5', '"setup_seconds": 1e999')
    members['report.json'] = report.encode()
    path = write_zip(tmp_path, members)
    assert main([str(path), '--output', str(tmp_path/'analysis')]) == 1
    result = json.loads((tmp_path/'analysis/analysis.json').read_text())
    assert result['audit_status'] == 'unreadable_or_invalid'
    assert 'Nonfinite' in result['errors'][0]
    members = fixture_members()
    members['../unsafe'] = b'no extraction permitted'
    with pytest.raises(ValueError, match='unsafe'):
        analyze(write_zip(tmp_path, members))


def test_blocked_before_selection_is_readable_but_has_no_comparison(tmp_path):
    members = fixture_members()
    r = json.loads(members['report.json'])
    r = {k: r[k] for k in ('configuration', 'planned_attempts')}
    r.update(status='blocked', inputs=[], error='checkpoint unavailable')
    path = write_zip(tmp_path, {'report.json': json.dumps(r)})
    result = analyze(path)
    assert result['errors'] == [] and result['raw_pairs_assessed'] == 0
    assert result['unrecorded_slots'] == 6
    assert 'No usable paired comparison' in markdown(result)
