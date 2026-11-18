"""Package an additive token matcher update without data or model weights."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = ('diffusion/token_matcher.py', 'diffusion/token_matcher_data.py', 'training/train_token_matcher.py',
         'training/train_token_matcher.sh', 'docs/experiments/TOKEN_MATCHER_EXPERIMENT.md', 'docs/TOP1_RESEARCH_AND_NEXT_EXPERIMENT.md',
         'tests/test_token_matcher.py', 'tests/test_token_matcher_data.py', 'tests/test_token_matcher_runner.py',
         'tests/test_token_matcher_launcher.py', 'scripts/package_token_matcher.py')


def main():
    payload = {name: (ROOT / name).read_bytes() for name in FILES}
    if b'\r' in payload['training/train_token_matcher.sh']:
        raise ValueError('Slurm launcher must use Unix LF line endings.')
    manifest = {'schema_version': 1, 'purpose': 'Full-token IR/molecule matching: frozen versus adapted private encoder',
        'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
        'requires': 'Existing accuracy-v2 and residual-selector releases, ranking checkpoint and full-validation report',
        'gradient_training_split': 'train only', 'selection_split': '2000 previously inspected validation rows',
        'assessment_split': 'remaining previously inspected validation rows', 'test_used': False,
        'model_weights_included': False, 'scientific_accuracy_measured_locally': False, 'weights_promoted': False}
    path = ROOT / 'outputs/unity_token_matcher_update.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        for name, contents in payload.items():
            archive.writestr(name, contents)
        archive.writestr('TOKEN_MATCHER_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        for name, sha in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
    print(path)
    print('Bytes:', path.stat().st_size)
    print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
    return path


if __name__ == '__main__':
    main()
