"""Ship a training-task diagnostic without changing experiment model sources."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = ('audit_matcher_training_tasks.py', 'audit_matcher_training_tasks.sh',
    'diffusion/matcher_task_audit.py', 'docs/experiments/MATCHER_TRAINING_TASK_AUDIT.md',
    'tests/test_matcher_task_audit.py', 'tests/test_matcher_task_audit_launcher.py',
    'scripts/package_matcher_task_audit.py', 'docs/results/token_matcher_probe_1444546.md')


def main():
    contents = {name: (ROOT/name).read_bytes() for name in FILES}
    if b'\r' in contents['audit_matcher_training_tasks.sh']:
        raise ValueError('Bash launcher must have LF line endings.')
    manifest = {'schema_version': 1, 'weights_included': False, 'training_performed': False,
        'experiment_model_sources_replaced': False,
        'requires_existing_run': 'runs/token_matcher_1426578',
        'sha256': {n: hashlib.sha256(b).hexdigest() for n,b in contents.items()}}
    output = ROOT/'outputs/unity_matcher_task_audit_update.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as z:
        for name, value in contents.items():
            z.writestr(name, value)
        z.writestr('MATCHER_TASK_AUDIT_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with ZipFile(output) as z:
        assert z.testzip() is None
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    print(output)
    print('SHA256:', hashlib.sha256(output.read_bytes()).hexdigest())
    return output


if __name__ == '__main__':
    main()
