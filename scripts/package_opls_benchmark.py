"""Package benchmark plus the exact dependencies used in the successful pilot."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NEW_FILES = ('benchmark_opls.py', 'run_opls_benchmark.sh', 'OPLS_BENCHMARK.md',
             'tests/test_opls_benchmark.py', 'docs/results/opls_pipeline_1807809.md')
DEPENDENCIES = ('predict_opls.py', 'diffusion/model.py', 'diffusion/opls_bridge.py',
                'diffusion/opls_backend.py', 'diffusion/opls_client.py',
                'diffusion/opls_guidance.py', 'scripts/opls_worker.py')


def main():
    destination = ROOT/'outputs/unity_opls_benchmark.zip'
    manifest, contents = {'files': {}}, {}
    with zipfile.ZipFile(ROOT/'outputs/unity_opls_pipeline.zip') as old:
        previous = json.loads(old.read('opls_pipeline_update/manifest.json'))['files']
        for name in DEPENDENCIES:
            content = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
            digest = hashlib.sha256(content).hexdigest()
            if digest != previous[name]['new_sha256']:
                raise ValueError('Pilot dependency changed; review before packaging: '+name)
            contents[name] = content
            manifest['files'][name] = {'new_sha256': digest, 'previous_sha256': previous[name]['previous_sha256']}
    for name in NEW_FILES:
        content = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
        contents[name] = content
        manifest['files'][name] = {'new_sha256': hashlib.sha256(content).hexdigest(), 'previous_sha256': []}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in contents.items():
            z.writestr('opls_pipeline_update/'+name, content)
        z.writestr('opls_pipeline_update/manifest.json', json.dumps(manifest, indent=2)+'\n')
        z.writestr('install_opls_pipeline.py', (ROOT/'install_opls_pipeline.py').read_bytes().replace(b'\r\n', b'\n'))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
    print(destination)
    print('Bytes:', destination.stat().st_size)


if __name__ == '__main__':
    main()
