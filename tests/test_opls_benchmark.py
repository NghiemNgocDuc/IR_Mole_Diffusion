"""Fixed selection, paired attempts, failure denominators and report delivery."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
import benchmark_opls as b


def test_selection_is_unique_training_only_stratified_and_reproducible():
    ids = np.array([f'm{i//2}' for i in range(60)])
    counts = np.repeat([5]*10+[15]*10+[25]*10, 2)
    train = list(range(0, 56))
    selected, available = b.select_inputs(ids, counts, train, 3, 31)
    assert selected == b.select_inputs(ids, counts, train[::-1], 3, 31)[0]
    assert len(selected) == len({r['molecule_id'] for r in selected}) == 9
    assert all(r['row'] in train for r in selected)
    assert [r['selection_bin'] for r in selected] == ['3-10', '11-18', '19-29']*3
    assert available == {'3-10': 10, '11-18': 10, '19-29': 8}
    assert selected != b.select_inputs(ids, counts, train, 3, 32)[0]
    with pytest.raises(ValueError, match='only 8 available'):
        b.select_inputs(ids, counts, train, 9, 31)
    counts[1] = 7
    with pytest.raises(ValueError, match='Inconsistent'):
        b.select_inputs(ids, counts, train, 3, 31)


def report_with_plan():
    inputs = [{'input_index': 0, 'candidates': [{'smiles': 'C'}]},
              {'input_index': 1, 'candidates': [{'smiles': 'CC'}, {'smiles': 'CO'}]}]
    b.plan_graphs(inputs, 2, 3, 100)
    return {'status': 'running', 'planned_attempts': 24, 'inputs': inputs, 'limitations': []}


def test_plan_preserves_missing_slots_and_pairs_seeds_without_collisions():
    report = report_with_plan()
    seeds = []
    for row in report['inputs']:
        for graph in row['graphs']:
            for c in range(3):
                pair = [a for a in graph['attempts'] if a['conformer'] == c]
                assert len(pair) == 2 and pair[0]['seed'] == pair[1]['seed']
                assert {a['arm'] for a in pair} == set(b.ARMS)
                assert {a['order_in_pair'] for a in pair} == {0, 1}
                seeds.append(pair[0]['seed'])
    assert len(seeds) == len(set(seeds)) == 12
    summary = b.summarize(report)
    assert summary['status_counts'] == {'pending': 18, 'missing_candidate': 6}
    assert summary['planned_attempts'] == 24
    assert all(a['planned'] == 12 for a in summary['arms'].values())


def test_partial_pairs_and_rejected_relaxations_do_not_inflate_success():
    report = report_with_plan()
    pair = report['inputs'][0]['graphs'][0]['attempts'][:2]
    for a in pair:
        guided = a['arm'] == 'opls_guided'
        a.update(status='completed' if guided else 'failed',
                 raw={'energy_kj_mol': 8 if guided else 10, 'force_rms_kj_mol_nm': 2,
                      'coordinate_inferred_graph_matches': guided},
                 relaxation={'accepted': guided, 'force_converged': guided,
                             'after': {'energy_kj_mol': 1, 'coordinate_inferred_graph_matches': True}})
    s = b.summarize(report)
    assert s['raw_pairs_assessed'] == s['raw_pairs_lower_guided_energy'] == 1
    assert s['pairs'][0]['raw_energy_delta_kj_mol'] == -2
    assert 'relaxed_energy_delta_kj_mol' not in s['pairs'][0]
    assert s['arms']['unguided']['completed'] == 0
    assert s['arms']['unguided']['relaxed_converged_graph_matches'] == 0
    assert s['per_input_raw_energy_delta_kj_mol'] == [{'input_index': 0, 'paired_count': 1, 'mean_delta': -2.}]


class FakeModel:
    n_timesteps = 8
    def sample(self, spectra, atoms, count, **kwargs):
        assert kwargs['physics_guidance'] == 0
        xyz = torch.randn(1, int(count[0]), 3, generator=kwargs['generator'])
        xyz -= xyz.mean(1, keepdim=True)
        return xyz, torch.ones(xyz.shape[:2], dtype=torch.bool)


def fake_client(tmp_path, fail_relax=False):
    cache = tmp_path/'cache'
    cache.mkdir(exist_ok=True)
    for name in ('manifest.json', 'parameterization.log'):
        (cache/name).write_text('{}')
    class Client:
        def __init__(self, *args):
            self.metadata = {'cache_directory': str(cache), 'file_sha256': {}}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, op, coordinates):
            xyz = np.asarray(coordinates)
            state = {'energy_kj_mol': float((xyz**2).sum()), 'force_rms_kj_mol_nm': 2.,
                     'coordinate_inferred_graph_matches': False}
            if op == 'evaluate':
                return state
            assert op == 'relax'
            if fail_relax:
                raise RuntimeError('synthetic minimization failure')
            return {'coordinates': (xyz*.5).tolist(), 'before': state,
                    'after': {**state, 'energy_kj_mol': state['energy_kj_mol']*.25},
                    'accepted': True, 'force_converged': False}
    return Client


def graph_args():
    return SimpleNamespace(opls_python='unused', appimage='unused', cache='unused',
                           min_alpha=.8, max_step=.02, steps=4)


@pytest.mark.parametrize('fail_relax', [False, True])
def test_graph_run_keeps_raw_artifacts_and_paired_coordinates_on_relax_failure(tmp_path, monkeypatch, fail_relax):
    monkeypatch.setattr(b, 'OPLSClient', fake_client(tmp_path, fail_relax))
    report = report_with_plan()
    graph = report['inputs'][0]['graphs'][0]
    checkpoints = []
    folder = tmp_path/'graph'
    b.run_graph(graph_args(), FakeModel(), {}, graph, folder,
                lambda: checkpoints.append(json.loads(json.dumps(graph))), lambda: False)
    assert len(checkpoints) >= 12
    assert len(list(folder.glob('*_raw.npy'))) == 6
    for c in range(3):
        np.testing.assert_array_equal(np.load(folder/f'unguided_{c}_raw.npy'), np.load(folder/f'opls_guided_{c}_raw.npy'))
    assert all(a['status'] == ('failed' if fail_relax else 'completed') for a in graph['attempts'])
    assert all('raw' in a for a in graph['attempts'])
    if fail_relax:
        assert not list(folder.glob('*.npz'))
        assert all('synthetic minimization' in a['error'] for a in graph['attempts'])
    else:
        assert len(list(folder.glob('*.sdf'))) == 6
        assert all(a['repair_rms_angstrom_unaligned'] > 0 for a in graph['attempts'])


def test_parameter_failure_marks_all_planned_attempts(tmp_path, monkeypatch):
    class Fails:
        def __init__(self, *args):
            raise RuntimeError('parameterization unavailable')
    monkeypatch.setattr(b, 'OPLSClient', Fails)
    report = report_with_plan()
    graph = report['inputs'][0]['graphs'][0]
    b.run_graph(graph_args(), FakeModel(), {}, graph, tmp_path/'graph', lambda: None, lambda: False)
    assert all(a['status'] == 'graph_failed' for a in graph['attempts'])
    assert b.summarize(report)['status_counts']['graph_failed'] == 6


def test_stopping_preserves_pending_slots_and_partial_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'OPLSClient', fake_client(tmp_path))
    report = report_with_plan()
    graph = report['inputs'][0]['graphs'][0]
    b.run_graph(graph_args(), FakeModel(), {}, graph, tmp_path/'graph', lambda: None, lambda: True)
    report['status'] = 'partial'
    b.write_report(tmp_path, report, archive=True)
    assert all(a['status'] == 'pending' for a in graph['attempts'])
    with zipfile.ZipFile(tmp_path/'benchmark_results.zip') as z:
        assert z.testzip() is None
        saved = json.loads(z.read('report.json'))
        assert saved['summary']['planned_attempts'] == 24 and saved['status'] == 'partial'
        assert not any(n.endswith('.zip') for n in z.namelist())


def test_missing_checkpoints_deliver_actionable_archive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert b.main(['--experimental-opls', '--output', str(tmp_path/'result')]) == 1
    with zipfile.ZipFile(tmp_path/'result/benchmark_results.zip') as z:
        report = json.loads(z.read('report.json'))
        assert report['status'] == 'blocked' and 'checkpoint' in report['error']
        assert report['planned_attempts'] == 288
        assert report['physical_correctness_validated'] is False


def test_full_driver_freezes_plan_before_energy_and_loads_models_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metadata = {'data_sha256': 'datahash', 'split_sha256': 'splithash', 'preprocessing': {'scale': 1}}
    inventory = [{'path': str(tmp_path/name), 'model_class': kind, 'eligible': True, 'run_metadata': metadata}
                 for name, kind in [('xyz.ckpt', 'EquivariantDiffusion3D_v4'), ('ir.ckpt', 'SpectrumStructureModel')]]
    monkeypatch.setattr(b, 'checkpoint_inventory', lambda roots: inventory)
    original_hash = b.file_sha256
    monkeypatch.setattr(b, 'file_sha256', lambda p: 'splithash' if str(p) == 'splits.json' else
                        'ckpthash' if str(p).endswith('.ckpt') else original_hash(p))
    (tmp_path/'splits.json').write_text('{}')
    data = {'molecule_ids': np.array(['small', 'medium', 'large', 'heldout']),
            'num_atoms': np.array([5, 15, 25, 5]), 'packed': True, 'preprocessing': metadata['preprocessing'],
            'spectra': np.ones((4, 2, 2), dtype=np.float32), 'spectra_mask': np.ones((4, 2), dtype=bool)}
    monkeypatch.setattr(b, 'load_native_npz', lambda path: data)
    monkeypatch.setattr(b, 'validate_split_manifest', lambda *a: {'data_sha256': 'datahash', 'splits': {'train': [0, 1, 2]}})
    loads = []
    class Coordinate(FakeModel):
        @classmethod
        def load_from_checkpoint(cls, path):
            loads.append('coordinate')
            return cls()
        def cpu(self):
            return self
        def eval(self):
            return self
        def use_ema_weights(self):
            return True
    class Spectrum(Coordinate):
        @classmethod
        def load_from_checkpoint(cls, path):
            loads.append('spectrum')
            return cls()
        def generate(self, spectra, beam_size):
            assert set(spectra) == {'packed', 'mask'} and beam_size == 5
            return [{'selfies': '[C]', 'score': -1.}, {'selfies': '[C][O]', 'score': -2.}]
    monkeypatch.setattr(b, 'EquivariantDiffusion3D_v4', Coordinate)
    monkeypatch.setattr(b, 'SpectrumStructureModel', Spectrum)
    client = fake_client(tmp_path)
    class CheckedClient(client):
        def __init__(self, *args):
            assert (tmp_path/'result/selection.json').exists()
            plan = json.loads((tmp_path/'result/candidate_plan.json').read_text())
            assert len(plan) == 3 and all(len(r['graphs']) == 2 for r in plan)
            assert all(a['status'] == 'pending' for r in plan for g in r['graphs'] for a in g['attempts'])
            super().__init__(*args)
    monkeypatch.setattr(b, 'OPLSClient', CheckedClient)
    assert b.main(['--experimental-opls', '--output', str(tmp_path/'result'), '--coordinate-checkpoint', 'xyz.ckpt',
                  '--spectrum-checkpoint', 'ir.ckpt', '--splits', 'splits.json',
                  '--per-bin', '1', '--conformers', '2', '--steps', '4']) == 0
    assert loads == ['spectrum', 'coordinate']
    with zipfile.ZipFile(tmp_path/'result/benchmark_results.zip') as z:
        report = json.loads(z.read('report.json'))
        assert report['status'] == 'completed'
        assert report['summary']['status_counts'] == {'completed': 24}
        assert report['summary']['raw_pairs_assessed'] == 12
        assert {r['row'] for r in report['inputs']} == {0, 1, 2}
        assert all(p['raw_energy_delta_kj_mol'] == 0 for p in report['summary']['pairs'])
        assert len([n for n in z.namelist() if n.endswith('_raw.npy')]) == 24


def test_packaged_update_installs_over_pilot_and_preserves_unknown_edits(tmp_path):
    root = Path(__file__).resolve().parents[1]
    # Reproduce the user's deployed pilot, then install the new delivery.
    with zipfile.ZipFile(root/'outputs/unity_opls_pipeline.zip') as z:
        z.extractall(tmp_path)
    install = [sys.executable, str(tmp_path/'install_opls_pipeline.py')]
    assert subprocess.run(install, capture_output=True).returncode == 0
    model_before = (tmp_path/'diffusion/model.py').read_bytes()
    with zipfile.ZipFile(root/'outputs/unity_opls_benchmark.zip') as z:
        z.extractall(tmp_path)
    result = subprocess.run(install, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path/'diffusion/model.py').read_bytes() == model_before
    assert (tmp_path/'benchmark_opls.py').exists()
    assert subprocess.run(install, capture_output=True).returncode == 0
    target = tmp_path/'benchmark_opls.py'
    target.write_text('user edits')
    result = subprocess.run(install, capture_output=True, text=True)
    assert result.returncode != 0 and 'Stopped before changes' in result.stderr
    assert target.read_text() == 'user edits'
