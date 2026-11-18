"""Package a guarded update; accept only known previously shipped versions."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('diffusion/opls_backend.py', 'diffusion/graph_identity.py',
         'scripts/diagnose_opls_benchmark.py', 'scripts/compare_opls_engines.py',
         'scripts/validate_opls_physics.py', 'run_physical_validation.sh',
         'run_physics_dft.sh', 'collect_physics_validation.sh',
         'tests/test_physical_validation.py', 'PHYSICAL_VALIDATION_PILOT.md')


def main():
    known = {name: set() for name in FILES}
    for archive in (ROOT/'outputs').glob('unity_*.zip'):
        if archive.name == 'unity_physical_validation.zip':
            continue
        with zipfile.ZipFile(archive) as z:
            for name in FILES:
                for member in (name, 'opls_pipeline_update/'+name):
                    if member in z.namelist():
                        known[name].add(hashlib.sha256(z.read(member)).hexdigest())
    manifest = {'files': {}}
    destination = ROOT/'outputs/unity_physical_validation.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            content = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
            manifest['files'][name] = {'new_sha256': hashlib.sha256(content).hexdigest(),
                                     'previous_sha256': sorted(known[name])}
            z.writestr('opls_pipeline_update/'+name, content)
        z.writestr('opls_pipeline_update/manifest.json', json.dumps(manifest, indent=2)+'\n')
        z.writestr('install_opls_pipeline.py', (ROOT/'install_opls_pipeline.py').read_bytes().replace(b'\r\n', b'\n'))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
    print(destination)


if __name__ == '__main__':
    main()
