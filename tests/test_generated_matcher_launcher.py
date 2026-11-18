import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile
import pytest
from scripts import package_generated_matcher as package


def test_launcher_protocol_and_preflight(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash unavailable')
    paths=['runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt',
           'evaluation/full_validation_66855524/ranking.json','data/processed/squirl.npz','data/processed/splits.json']
    for name in paths:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fixture')
    capture=tmp_path/'arguments.txt'
    env={k:v for k,v in os.environ.items() if not k.startswith(('MATCHER_','GENERATED_MATCHER_'))}
    env.update(SLURM_SUBMIT_DIR=tmp_path.as_posix(),SLURM_JOB_ID='99005',CONDA_ENV='base',
               CAPTURE=capture.as_posix(),LAUNCHER=(package.ROOT/'train_generated_matcher.sh').as_posix())
    script='module() { :; }; conda() { printf "%s\\n" "$@" > "$CAPTURE"; }; export -f module conda; source "$LAUNCHER"'
    def run():
        return subprocess.run([str(bash),'-c',script],env=env,capture_output=True,text=True)
    result=run(); assert result.returncode==0,result.stderr
    args=capture.read_text().splitlines()
    assert 'train_generated_matcher.py' in args
    for key,value in [('training_limit','5000'),('epochs','10'),('mining_seed','20261112'),('output_dir','runs/generated_matcher_99005')]:
        assert args[args.index('--'+key)+1]==value
    env['GENERATED_MATCHER_OUTPUT']='runs/space output'
    assert run().returncode==0
    (tmp_path/env['GENERATED_MATCHER_OUTPUT']).mkdir()
    assert 'Output exists' in run().stderr
    (tmp_path/paths[0]).unlink()
    assert 'Missing input' in run().stderr


def test_package_integrity_and_preserves_old_sources(tmp_path,monkeypatch):
    for name in package.FILES:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((package.ROOT/name).read_bytes())
    monkeypatch.setattr(package,'ROOT',tmp_path)
    with ZipFile(package.main()) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(package.FILES)|{'GENERATED_MATCHER_MANIFEST.json'}
        manifest=json.loads(z.read('GENERATED_MATCHER_MANIFEST.json'))
        assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in manifest['sha256'].items())
        assert 'train_token_matcher.py' not in z.namelist()
        assert 'diffusion/token_matcher.py' not in z.namelist()
        assert b'\r' not in z.read('train_generated_matcher.sh')
