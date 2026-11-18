"""Ship only the raw-default entry points and the bounded precision diagnostic."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=('predict_opls.py','benchmark_opls.py','scripts/compare_opls_engines.py',
       'scripts/probe_opls_precision.py','probe_opls_precision.sh',
       'tests/test_pipeline_raw_default.py','tests/test_opls_benchmark.py',
       'docs/experiments/PIPELINE_RAW_DEFAULT.md','docs/results/physical_validation_1849289_review.md')


def main():
    destination=ROOT/'outputs/unity_pipeline_raw_default.zip'
    known={name:set() for name in FILES}
    for path in (ROOT/'outputs').glob('unity_*.zip'):
        if path==destination: continue
        with zipfile.ZipFile(path) as archive:
            for name in FILES:
                for member in (name,'opls_pipeline_update/'+name):
                    if member in archive.namelist():
                        known[name].add(hashlib.sha256(archive.read(member)).hexdigest())
    manifest={'files':{}}
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            content=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
            manifest['files'][name]={'new_sha256':hashlib.sha256(content).hexdigest(),
                                     'previous_sha256':sorted(known[name])}
            archive.writestr('opls_pipeline_update/'+name,content)
        archive.writestr('opls_pipeline_update/manifest.json',json.dumps(manifest,indent=2)+'\n')
        archive.writestr('install_opls_pipeline.py',(ROOT/'install_opls_pipeline.py').read_bytes().replace(b'\r\n',b'\n'))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    print(destination)


if __name__=='__main__': main()
