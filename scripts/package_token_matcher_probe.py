"""Package read-only diagnostics without replacing original experiment sources."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = ('probe_token_matcher.py', 'probe_token_matcher.sh', 'docs/experiments/TOKEN_MATCHER_PROBE.md',
         'scripts/audit_token_matcher.py', 'scripts/package_token_matcher_probe.py',
         'tests/test_token_matcher_probe.py', 'tests/test_token_matcher_probe_launcher.py',
         'docs/results/token_matcher_1426578.md')


def main():
    contents = {name: (ROOT / name).read_bytes() for name in FILES}
    if b'\r' in contents['probe_token_matcher.sh']:
        raise ValueError('Launcher must use LF line endings.')
    manifest = {'schema_version': 1, 'training_performed': False, 'weights_included': False,
        'original_experiment_sources_replaced': False, 'purpose': 'IR dependence controls and saved-prediction audit',
        'sha256': {n: hashlib.sha256(b).hexdigest() for n, b in contents.items()}}
    output = ROOT / 'outputs/unity_token_matcher_probe_update.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
        archive.writestr('TOKEN_MATCHER_PROBE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(output)
    print('SHA256:', hashlib.sha256(output.read_bytes()).hexdigest())
    return output


if __name__ == '__main__':
    main()
