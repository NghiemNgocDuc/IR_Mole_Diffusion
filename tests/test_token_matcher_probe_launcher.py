"""Check the actual Bash entry point and the additive probe distribution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile
import pytest

from scripts import package_token_matcher_probe as package


def test_launcher_routes_inputs_and_preserves_existing_output(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash not available')
    root = Path(__file__).resolve().parents[1]
    run = tmp_path / 'runs/token_matcher_1426578'
    run.mkdir(parents=True)
    for name in ('configuration.json', 'selection.json', 'comparison.json'):
        (run / name).write_text('{}')
    capture = tmp_path / 'arguments.txt'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MATCHER_', 'PROBE_'))}
    env.update(SLURM_SUBMIT_DIR=tmp_path.as_posix(), SLURM_JOB_ID='99003', CONDA_ENV='base',
               CAPTURE=capture.as_posix(), LAUNCHER=(root / 'probe_token_matcher.sh').as_posix())
    script = 'module() { :; }; conda() { printf "%s\\n" "$@" > "$CAPTURE"; }; export -f module conda; source "$LAUNCHER"'
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert 'probe_token_matcher.py' in args
    assert args[args.index('--run_dir') + 1] == 'runs/token_matcher_1426578'
    assert args[args.index('--output_dir') + 1] == 'runs/token_matcher_probe_99003'
    assert args[args.index('--device') + 1] == 'cuda'
    override = 'runs/other probe'
    env['PROBE_OUTPUT'] = override
    env['MATCHER_RUN'] = 'runs/other matcher'
    shutil.copytree(run, tmp_path / env['MATCHER_RUN'])
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index('--output_dir') + 1] == override
    assert args[args.index('--run_dir') + 1] == env['MATCHER_RUN']
    (tmp_path / override).mkdir()
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Output exists' in result.stderr
    (tmp_path / env['MATCHER_RUN'] / 'comparison.json').unlink()
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Missing input:' in result.stderr


def test_package_is_additive_and_hash_verified(tmp_path, monkeypatch):
    for name in package.FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package.ROOT / name).read_bytes())
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    artifact = package.main()
    with ZipFile(artifact) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(package.FILES) | {'TOKEN_MATCHER_PROBE_MANIFEST.json'}
        manifest = json.loads(archive.read('TOKEN_MATCHER_PROBE_MANIFEST.json'))
        assert not manifest['training_performed']
        assert not manifest['weights_included']
        assert not manifest['original_experiment_sources_replaced']
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert b'\r' not in archive.read('probe_token_matcher.sh')
        assert 'training/train_token_matcher.py' not in archive.namelist()
        assert not any(n.endswith(('.pt', '.ckpt')) for n in archive.namelist())
