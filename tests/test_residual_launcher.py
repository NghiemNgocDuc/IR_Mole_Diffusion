import os
from pathlib import Path
import shutil
import subprocess
import pytest


def test_launcher_routes_existing_inputs_and_refuses_missing_input(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash not available')
    root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / 'runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt'
    report = tmp_path / 'evaluation/full_validation_66855524/ranking.json'
    mining = tmp_path / 'runs/hard_selector_1361586'
    for path in (checkpoint, report, mining / 'configuration.json', mining / 'training_candidates.json'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    capture = tmp_path / 'arguments.txt'
    env = {k: v for k, v in os.environ.items() if not k.startswith('RESIDUAL_')}
    env.update(SLURM_SUBMIT_DIR=tmp_path.as_posix(), SLURM_JOB_ID='99001', CONDA_ENV='base',
               CAPTURE=capture.as_posix(), LAUNCHER=(root / 'training/train_residual_selector.sh').as_posix())
    script = 'module() { :; }; conda() { printf "%s\\n" "$@" > "$CAPTURE"; }; export -f module conda; source "$LAUNCHER"'
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert args[args.index('--output_dir') + 1] == 'runs/residual_selector_99001'
    assert args[args.index('--mining_dir') + 1] == 'runs/hard_selector_1361586'
    assert args[args.index('--dev_size') + 1] == '2000'
    assert '--resume' not in args
    env.update(RESIDUAL_OUTPUT='runs/residual_selector_99000', RESIDUAL_RESUME='1')
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert '--resume' in args and args[args.index('--output_dir') + 1] == 'runs/residual_selector_99000'
    report.unlink()
    result = subprocess.run([str(bash), '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Missing input:' in result.stderr
