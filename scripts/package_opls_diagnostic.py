"""Guarded update for graph comparison and a focused saved-geometry diagnostic."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=('diffusion/graph_identity.py','diffusion/opls_backend.py',
       'scripts/diagnose_opls_benchmark.py','diagnose_opls_benchmark.sh',
       'tests/test_opls_diagnostics.py','OPLS_DIAGNOSTIC.md')


def main():
    with zipfile.ZipFile(ROOT/'outputs/unity_opls_benchmark.zip') as z:
        old=json.loads(z.read('opls_pipeline_update/manifest.json'))['files']
    manifest={'files':{}}
    destination=ROOT/'outputs/unity_opls_diagnostic.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            content=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            manifest['files'][name]={'new_sha256':hashlib.sha256(content).hexdigest(),
                'previous_sha256':[old[name]['new_sha256']] if name in old else []}
            z.writestr('opls_pipeline_update/'+name,content)
        z.writestr('opls_pipeline_update/manifest.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('install_opls_pipeline.py',(ROOT/'install_opls_pipeline.py').read_bytes().replace(b'\r\n',b'\n'))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
    print(destination)


if __name__=='__main__':
    main()
