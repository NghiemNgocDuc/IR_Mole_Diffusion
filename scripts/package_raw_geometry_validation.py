"""Additive package: contains only the isolated raw_geometry_validation directory."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=('check.py','submit.sh','run.sh','collect.sh','README.md','test_check.py')


def main():
    destination=ROOT/'outputs/unity_raw_geometry_validation.zip'
    hashes={}
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            content=(ROOT/'raw_geometry_validation'/name).read_bytes().replace(b'\r\n',b'\n')
            hashes[name]=hashlib.sha256(content).hexdigest()
            archive.writestr('raw_geometry_validation/'+name,content)
        archive.writestr('raw_geometry_validation/package_manifest.json',json.dumps({'sha256':hashes},indent=2)+'\n')
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(n.startswith('raw_geometry_validation/') for n in archive.namelist())
    print(destination)


if __name__=='__main__': main()
