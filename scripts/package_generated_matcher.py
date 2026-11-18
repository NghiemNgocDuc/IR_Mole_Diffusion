"""Package only the new controlled experiment and its tests/documentation."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
FILES = ('train_generated_matcher.py', 'train_generated_matcher.sh',
         'diffusion/generated_matcher_data.py', 'GENERATED_MATCHER_EXPERIMENT.md',
         'tests/test_generated_matcher.py', 'tests/test_generated_matcher_launcher.py',
         'scripts/package_generated_matcher.py')


def main():
    contents = {n: (ROOT/n).read_bytes() for n in FILES}
    assert b'\r' not in contents['train_generated_matcher.sh']
    manifest = {'schema_version': 1, 'weights_included': False,
        'requires_existing_token_matcher_project': True,
        'sha256': {n: hashlib.sha256(b).hexdigest() for n,b in contents.items()}}
    path = ROOT/'outputs/unity_generated_matcher_update.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w', ZIP_DEFLATED) as z:
        for name, value in contents.items():
            z.writestr(name, value)
        z.writestr('GENERATED_MATCHER_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with ZipFile(path) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in manifest['sha256'].items())
    print(path)
    print(hashlib.sha256(path.read_bytes()).hexdigest())
    return path


if __name__ == '__main__':
    main()
