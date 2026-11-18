import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile
import pytest
from scripts import package_matcher_task_audit as package


def test_bash_routes_fixed_protocol_and_rejects_missing_or_existing_paths(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash unavailable')
    directory = tmp_path/'runs/token_matcher_1426578'
    directory.mkdir(parents=True)
    for name in ('configuration.json', 'selection.json', 'comparison.json', 'negative_plans.json'):
        (directory/name).write_text('{}')
    capture = tmp_path/'arguments.txt'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('MATCHER_', 'TASK_AUDIT_'))}
    env.update(SLURM_SUBMIT_DIR=tmp_path.as_posix(), SLURM_JOB_ID='99004', CONDA_ENV='base',
        CAPTURE=capture.as_posix(), LAUNCHER=(package.ROOT/'audit_matcher_training_tasks.sh').as_posix())
    script = 'module() { :; }; conda() { printf "%s\\n" "$@" > "$CAPTURE"; }; export -f module conda; source "$LAUNCHER"'
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert 'audit_matcher_training_tasks.py' in args
    assert args[args.index('--output_dir')+1] == 'runs/matcher_task_audit_99004'
    assert args[args.index('--limit')+1] == '2000' and args[args.index('--seed')+1] == '20261112'
    env['TASK_AUDIT_OUTPUT'] = 'runs/other audit'
    env['MATCHER_RUN'] = 'runs/other matcher'
    shutil.copytree(directory, tmp_path/env['MATCHER_RUN'])
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index('--run_dir')+1] == env['MATCHER_RUN']
    assert args[args.index('--output_dir')+1] == env['TASK_AUDIT_OUTPUT']
    (tmp_path/env['TASK_AUDIT_OUTPUT']).mkdir()
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Output exists' in result.stderr
    (tmp_path/env['MATCHER_RUN']/'negative_plans.json').unlink()
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Missing input' in result.stderr


def test_package_additions_hashes_and_lf(tmp_path, monkeypatch):
    for name in package.FILES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((package.ROOT/name).read_bytes())
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    artifact = package.main()
    with ZipFile(artifact) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(package.FILES) | {'MATCHER_TASK_AUDIT_MANIFEST.json'}
        manifest = json.loads(z.read('MATCHER_TASK_AUDIT_MANIFEST.json'))
        assert not manifest['weights_included'] and not manifest['experiment_model_sources_replaced']
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        assert b'\r' not in z.read('audit_matcher_training_tasks.sh')
        assert not any(n.endswith(('.pt', '.ckpt', '.npz', '.npy')) for n in z.namelist())
        assert 'train_token_matcher.py' not in z.namelist()
