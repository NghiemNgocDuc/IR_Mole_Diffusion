"""Checkpoint discovery, report preservation, cache integrity and worker protocol."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch
from diffusion import opls_backend
from diffusion.opls_client import OPLSClient
from predict_opls import checkpoint_inventory, choose_checkpoint, main


def checkpoint(path, **updates):
    data = {'format_version': 2, 'model_class': 'EquivariantDiffusion3D_v4', 'global_step': 10,
            'run_metadata': {'diagnostic': False, 'spectrum_control': 'none'}, 'state_dict': {'dummy': torch.ones(1)}}
    data.update(updates)
    torch.save(data, path)


def test_checkpoint_discovery_rejects_legacy_and_ambiguity(tmp_path):
    checkpoint(tmp_path/'legacy.ckpt', format_version=None)
    checkpoint(tmp_path/'current.ckpt')
    checkpoint(tmp_path/'smoke.ckpt', run_metadata={'diagnostic': True})
    items = checkpoint_inventory([tmp_path])
    chosen = choose_checkpoint(items, 'EquivariantDiffusion3D_v4')
    assert chosen['path'] == str(tmp_path/'current.ckpt')
    checkpoint(tmp_path/'another.ckpt')
    with pytest.raises(ValueError, match='found 2'):
        choose_checkpoint(checkpoint_inventory([tmp_path]), 'EquivariantDiffusion3D_v4')
    assert choose_checkpoint(checkpoint_inventory([tmp_path]), 'EquivariantDiffusion3D_v4', tmp_path/'current.ckpt') == chosen


def test_missing_checkpoint_still_produces_downloadable_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(['--output', str(tmp_path/'result')]) == 1
    with zipfile.ZipFile(tmp_path/'result/pipeline_results.zip') as archive:
        report = json.loads(archive.read('report.json'))
        assert report['status'] == 'blocked' and 'checkpoint' in report['error']
        assert json.loads(archive.read('checkpoint_inventory.json')) == []
        assert report['physical_correctness_validated'] is False


def test_parameter_cache_reuse_and_tamper_detection(tmp_path, monkeypatch):
    fixture = Path('opls_ethanol_review.zip').resolve()
    if not fixture.exists():
        pytest.skip('Archived fixture unavailable')
    app = tmp_path/'app'
    app.write_text('fake executable for cache control-flow test')
    monkeypatch.setattr(opls_backend, 'APPIMAGE_SHA256', opls_backend.sha256(app))
    calls = []
    def fake_parameterizer(args, **kwargs):
        calls.append(args)
        output = Path(args[args.index('-p')+1])
        output.mkdir()
        with zipfile.ZipFile(fixture) as z:
            for suffix in ('py', 'xml', 'pdb'):
                member = next(n for n in z.namelist() if n.endswith('ETH.openmm.'+suffix))
                (output/('MOL.openmm.'+suffix)).write_bytes(z.read(member))
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(opls_backend.subprocess, 'run', fake_parameterizer)
    folder, meta = opls_backend.prepare('CCO', app, tmp_path/'cache')
    assert len(calls) == 1 and len(meta['model_to_engine']) == 9
    assert opls_backend.prepare('CCO', app, tmp_path/'cache') == (folder, meta)
    assert len(calls) == 1
    (folder/'export/MOL.openmm.xml').write_text('corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        opls_backend.prepare('CCO', app, tmp_path/'cache')


def test_persistent_worker_protocol_and_cleanup(tmp_path, monkeypatch):
    fake = tmp_path/'fake_worker.py'
    fake.write_text('import sys,json\nfor line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({"ok": True, "result": {"op":r["op"]}}),flush=True)\n')
    original = subprocess.Popen
    def start(args, **kwargs):
        return original([sys.executable, '-u', str(fake)], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', start)
    with OPLSClient('CCO', sys.executable, 'unused', tmp_path/'cache', tmp_path/'worker.log', timeout=5) as client:
        assert client.metadata['op'] == 'init'
        pid = client.process.pid
        assert client.request('evaluate', coordinates=[])['op'] == 'evaluate'
        assert client.process.pid == pid
    assert client.process.poll() is not None


def test_upload_installer_backs_up_and_refuses_unknown_edits(tmp_path):
    package = Path('outputs/unity_opls_pipeline.zip')
    if not package.exists():
        pytest.skip('Run the packager before checking the upload artifact')
    with zipfile.ZipFile(package) as z:
        z.extractall(tmp_path)  # This test uses our locally generated package only.
    target = tmp_path/'diffusion/model.py'
    target.parent.mkdir()
    original = subprocess.check_output(['git', 'show', 'HEAD:diffusion/model.py'])
    target.write_bytes(original)
    command = [sys.executable, str(tmp_path/'install_opls_pipeline.py')]
    installed = subprocess.run(command, capture_output=True, text=True)
    assert installed.returncode == 0, installed.stderr
    backups = list((tmp_path/'outputs').glob('opls_pipeline_install_*/diffusion/model.py'))
    if target.read_bytes() != original:
        assert len(backups) == 1 and backups[0].read_bytes() == original
    assert subprocess.run(command, capture_output=True).returncode == 0
    target.write_text('user change that must survive')
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'Stopped before changes' in rejected.stderr
    assert target.read_text() == 'user change that must survive'
