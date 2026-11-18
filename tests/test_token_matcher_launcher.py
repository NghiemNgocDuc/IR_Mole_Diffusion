"""Exercise the real Bash launcher and inspect the distributable allowlist."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile
import pytest

from scripts import package_token_matcher as package


def test_launcher_routes_inputs_and_refuses_missing_or_existing_output(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash not available')
    root = Path(__file__).resolve().parents[1]
    names = ('runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt',
             'evaluation/full_validation_66855524/ranking.json', 'data/processed/squirl.npz', 'data/processed/splits.json')
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    capture = tmp_path / 'arguments.txt'
    env = {k: v for k, v in os.environ.items() if not k.startswith('MATCHER_')}
    env.update(SLURM_SUBMIT_DIR=tmp_path.as_posix(), SLURM_JOB_ID='99002', CONDA_ENV='base',
               CAPTURE=capture.as_posix(), LAUNCHER=(root / 'train_token_matcher.sh').as_posix())
    script = 'module() { :; }; conda() { printf "%s\\n" "$@" > "$CAPTURE"; }; export -f module conda; source "$LAUNCHER"'
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index('--output_dir') + 1] == 'runs/token_matcher_99002'
    assert args[args.index('--epochs') + 1] == '10'
    assert args[args.index('--warmup_epochs') + 1] == '1'
    assert args[args.index('--encoder_lr') + 1] == '0.00001'
    override = 'runs/other matcher'
    env['MATCHER_OUTPUT'] = override
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index('--output_dir') + 1] == override
    (tmp_path / override).mkdir()
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Output exists' in result.stderr
    (tmp_path / names[1]).unlink()
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Missing input:' in result.stderr


def test_package_contains_only_allowed_additions_with_valid_hashes(tmp_path, monkeypatch):
    actual_root = package.ROOT
    for name in package.FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((actual_root / name).read_bytes())
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    artifact = package.main()
    with ZipFile(artifact) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(package.FILES) | {'TOKEN_MATCHER_MANIFEST.json'}
        manifest = json.loads(archive.read('TOKEN_MATCHER_MANIFEST.json'))
        assert not manifest['model_weights_included'] and not manifest['scientific_accuracy_measured_locally']
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert b'\r' not in archive.read('train_token_matcher.sh')
        assert 'diffusion/spectrum.py' not in archive.namelist()
