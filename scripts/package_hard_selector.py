"""Package additive hard-selector files; never include datasets or model weights."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = ('diffusion/hard_selector.py', 'training/train_hard_selector.py', 'training/train_hard_selector.sh',
         'docs/experiments/HARD_SELECTOR_EXPERIMENT.md', 'tests/test_hard_selector.py', 'tests/test_compatibility.py',
         'scripts/package_hard_selector.py')


def main():
    payload = {name: (ROOT / name).read_bytes() for name in FILES}
    if b'\r' in payload['training/train_hard_selector.sh']:
        raise ValueError('Launcher must have Unix LF line endings.')
    manifest = {'schema_version': 1, 'purpose': 'Matched random vs generated-hard-negative selector pilot',
                'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
                'requires': 'Existing accuracy-v2/compatibility code, ranking checkpoint, full-validation report and processed data',
                'default_checkpoint': 'runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt',
                'default_report': 'evaluation/full_validation_66855524/ranking.json',
                'trained_weights_included': False, 'scientific_accuracy_measured_locally': False}
    destination = ROOT / 'outputs/unity_hard_selector_update.zip'
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr('HARD_SELECTOR_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(destination)
    print('SHA256:', hashlib.sha256(destination.read_bytes()).hexdigest())
    print('Bytes:', destination.stat().st_size)


if __name__ == '__main__':
    main()
