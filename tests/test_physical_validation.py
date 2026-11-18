import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pytest

from diffusion import opls_backend
from scripts.validate_opls_physics import (aligned_rmsd, parse_orca,
    spectrum_comparison, xyz_text, prepare_pilot, collect, run_task)
from scripts import validate_opls_physics as validation


def test_coordinate_rejection_retries_but_other_errors_do_not(tmp_path, monkeypatch):
    model = opls_backend.declared_graph('CCO')
    calls = []
    def fake(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            kwargs['stdout'].write('Not run: 1 pair of unbonded atoms closer than a bond length')
        else:
            export = Path(args[args.index('-p')+1])
            export.mkdir()
            (export/'MOL.openmm.pdb').write_text('test fixture')
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(opls_backend.subprocess, 'run', fake)
    monkeypatch.setattr(opls_backend, 'map_export', lambda *a: list(range(9)))
    mapping, attempts = opls_backend.parameterize_templates(model, Path('fixture'), tmp_path)
    assert len(calls) == 2 and mapping == list(range(9))
    assert attempts[0]['status'] == 'coordinate_precheck_rejected'
    assert attempts[1]['status'] == 'validated_export'
    assert (tmp_path/'template_attempt_0/parameterization.log').exists()
    other = tmp_path/'other'; other.mkdir()
    def fail(args, **kwargs):
        return subprocess.CompletedProcess(args, 7)
    monkeypatch.setattr(opls_backend.subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='LigParGen failed'):
        opls_backend.parameterize_templates(model, Path('fixture'), other)
    assert len(list(other.glob('template_attempt_*'))) == 1


def test_template_retry_budget_is_bounded(tmp_path, monkeypatch):
    def reject(args, **kwargs):
        kwargs['stdout'].write('unbonded atoms closer than a bond length')
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(opls_backend.subprocess, 'run', reject)
    with pytest.raises(RuntimeError, match='All bounded'):
        opls_backend.parameterize_templates(opls_backend.declared_graph('CCO'), Path('fixture'), tmp_path)
    assert len(json.loads((tmp_path/'template_attempts.json').read_text())) == 3


def orca_text(frequency=100.):
    return ('THE OPTIMIZATION HAS CONVERGED\nVIBRATIONAL FREQUENCIES\n' +
        ''.join(f'{i}: {0 if i < 6 else frequency:.2f} cm**-1\n' for i in range(9)) +
        'NORMAL MODES\nIR SPECTRUM\nMode freq eps Int T**2 TX TY TZ\n'
        '6: 1000.00 0.002 10.13 0.000511 (0 0 0)\n\nORCA TERMINATED NORMALLY\n')


def test_qm_parser_does_not_promote_incomplete_or_imaginary_results():
    good = parse_orca(orca_text(), 3)
    assert good['status'] == 'no_imaginary_modes_at_reported_precision'
    assert good['ir_peaks_cm_1_km_mol'] == [[1000., 10.13]]
    assert not good['physical_correctness_validated']
    assert parse_orca(orca_text(-5), 3)['status'] == 'small_imaginary_modes_require_recheck'
    assert parse_orca(orca_text(-100), 3)['status'] == 'imaginary_modes_require_investigation'
    assert parse_orca(orca_text().replace('ORCA TERMINATED NORMALLY', ''), 3)['status'] == 'incomplete_or_unconverged'
    assert parse_orca(orca_text(), 4)['status'] == 'incomplete_or_unconverged'
    assert parse_orca(orca_text().replace('THE OPTIMIZATION HAS CONVERGED', ''), 3)['status'] == 'incomplete_or_unconverged'


def test_alignment_excludes_mirrors_and_ir_comparison_is_scale_invariant():
    xyz = np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]])
    rotation = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    assert aligned_rmsd(xyz, xyz@rotation+5) < 1e-12
    assert aligned_rmsd(xyz, xyz*np.array([-1,1,1])) > .1
    target = [[1000., 10.], [2000., 5.]]
    cosine, curve = spectrum_comparison(target, [[1000.,20.], [2000.,10.]])
    assert cosine == pytest.approx(1.) and curve.shape[1] == 3
    assert spectrum_comparison(target, [[1200.,10.]])[0] < .01
    with pytest.raises(ValueError):
        spectrum_comparison(target, [[1000.,-1.]])
    with pytest.raises(ValueError):
        xyz_text([6], [[float('nan'),0,0]])


def test_actual_archive_offline_plan_and_missing_result_report(tmp_path):
    source = Path('outputs/opls_diagnostic_source_1820024')
    if not source.exists():
        pytest.skip('Archived benchmark not extracted locally')
    plan = prepare_pilot(source.resolve(), tmp_path, engines=False)
    assert len(plan['tasks']) == 12
    assert set(t['stage'] for t in plan['tasks']) == {'raw', 'relaxed'}
    assert {t['input_index'] for t in plan['tasks']} == {1, 7, 13}
    assert all((tmp_path/t['directory']/'job.inp').read_text().startswith('! B3LYP') for t in plan['tasks'])
    collect(tmp_path)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert all(t['status'] == 'missing_result' for t in summary['tasks'])
    assert not summary['physical_correctness_validated']
    with zipfile.ZipFile(tmp_path/'physical_validation_results.zip') as z:
        assert z.testzip() is None
        assert 'qm_00/job.inp' in z.namelist()
    # Changed inputs cannot be executed or accepted.
    (tmp_path/'qm_00/job.inp').write_text('changed')
    run_task(tmp_path, 0, 'not-an-installed-program')
    result = json.loads((tmp_path/'qm_00/result.json').read_text())
    assert result['status'] == 'failed' and 'changed' in result['error']


def test_dft_result_checks_final_identity_and_repair(tmp_path, monkeypatch):
    folder = tmp_path/'qm_00'; folder.mkdir()
    # A water geometry with explicit hydrogens; run the actual coordinate
    # perception and assessment code against a controlled external-engine log.
    initial = xyz_text([8,1,1], [[0,0,0],[.96,0,0],[-.24,.93,0]])
    (folder/'start.xyz').write_text(initial)
    (folder/'job.inp').write_text('fixture input')
    task = {'id': 0, 'directory': 'qm_00', 'atom_numbers': [8,1,1],
            'smiles': 'O', 'input_index': 1,
            'input_sha256': opls_backend.sha256(folder/'job.inp'),
            'xyz_sha256': opls_backend.sha256(folder/'start.xyz')}
    (tmp_path/'plan.json').write_text(json.dumps({'tasks':[task]}))
    monkeypatch.setattr(validation.shutil, 'which', lambda p: 'fixture-orca')
    def fake_run(args, **kwargs):
        kwargs['stdout'].write(orca_text())
        (folder/'job.xyz').write_text(initial)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(validation.subprocess, 'run', fake_run)
    run_task(tmp_path, 0, 'fixture-orca')
    result = json.loads((folder/'result.json').read_text())
    assert result['local_minimum_supported_for_intended_graph']
    assert result['aligned_repair_rmsd_angstrom'] < 1e-12
    assert not result['physical_correctness_validated']


def test_packaged_update_installs_over_shipped_files_and_guards_edits(tmp_path):
    import sys
    package = Path('outputs/unity_physical_validation.zip')
    if not package.exists():
        pytest.skip('Run packager first')
    with zipfile.ZipFile(package) as z:
        z.extractall(tmp_path)  # Locally generated package only.
        manifest = json.loads(z.read('opls_pipeline_update/manifest.json'))
    # Reproduce the user's currently installed diagnostic versions where known.
    with zipfile.ZipFile('outputs/unity_opls_diagnostic.zip') as z:
        for name in manifest['files']:
            member = 'opls_pipeline_update/'+name
            if member in z.namelist():
                target = tmp_path/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
    command = [sys.executable, str(tmp_path/'install_opls_pipeline.py')]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    for name, item in manifest['files'].items():
        assert opls_backend.sha256(tmp_path/name) == item['new_sha256']
    (tmp_path/'diffusion/opls_backend.py').write_text('unknown local modification')
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode != 0 and 'unexpected existing edits' in process.stderr
