"""Stage only the authorized OPLS integration files, with guarded installation."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['diffusion/model.py', 'diffusion/opls_bridge.py', 'diffusion/opls_backend.py',
         'diffusion/opls_client.py', 'diffusion/opls_guidance.py', 'scripts/opls_worker.py',
         'predict_opls.py', 'run_opls_pipeline.sh', 'docs/experiments/OPLS_PIPELINE_PILOT.md',
         'tests/test_opls_sampler.py', 'tests/test_opls_pipeline.py',
         'docs/results/opls_readiness_1807422.md']


def main():
    destination = ROOT/'outputs/unity_opls_pipeline.zip'
    destination.parent.mkdir(exist_ok=True)
    original_model = subprocess.check_output(['git', 'show', 'HEAD:diffusion/model.py'], cwd=ROOT)
    previous = {'diffusion/model.py': [hashlib.sha256(original_model).hexdigest(),
                  hashlib.sha256(original_model.replace(b'\n', b'\r\n')).hexdigest()],
                'diffusion/opls_bridge.py': ['4738804c42cc56c6b2bfbf50370bda8a699d8d4c2fb78bee58d6828042263293']}
    manifest = {'files': {}}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            content = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
            manifest['files'][name] = {'new_sha256': hashlib.sha256(content).hexdigest(),
                                       'previous_sha256': previous.get(name, [])}
            z.writestr('opls_pipeline_update/'+name, content)
        z.writestr('opls_pipeline_update/manifest.json', json.dumps(manifest, indent=2)+'\n')
        z.writestr('install_opls_pipeline.py', (ROOT/'install_opls_pipeline.py').read_bytes().replace(b'\r\n', b'\n'))
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:
            raise RuntimeError('Package CRC failure')
    print(destination)
    print('Bytes:', destination.stat().st_size)


if __name__ == '__main__':
    main()
