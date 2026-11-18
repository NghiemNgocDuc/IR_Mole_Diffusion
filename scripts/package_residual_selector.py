"""Package additive development/residual-selector files without scientific inputs."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = ('diffusion/residual_selector.py', 'training/train_residual_selector.py', 'training/train_residual_selector.sh',
         'docs/experiments/RESIDUAL_SELECTOR_EXPERIMENT.md', 'tests/test_residual_selector.py',
         'tests/test_residual_launcher.py', 'scripts/package_residual_selector.py')


def main():
    payload = {name: (ROOT / name).read_bytes() for name in FILES}
    if b'\r' in payload['training/train_residual_selector.sh']:
        raise ValueError('Launcher requires Unix LF line endings.')
    manifest = {'schema_version': 1, 'purpose': 'Development-selected bounded decoder-score corrections',
        'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
        'requires': 'Existing hard-selector release, completed mining run 1361586, ranking checkpoint and full-validation report',
        'gradient_training_split': 'train', 'checkpoint_selection_split': '2000 rows from previously reviewed val',
        'assessment_split': 'remaining 10478 previously reviewed val rows', 'test_used': False,
        'model_weights_included': False, 'scientific_improvement_measured_locally': False}
    path = ROOT / 'outputs/unity_residual_selector_update.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        for name, contents in payload.items():
            archive.writestr(name, contents)
        archive.writestr('RESIDUAL_SELECTOR_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        for name, sha in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
    print(path)
    print('Bytes:', path.stat().st_size)
    print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
