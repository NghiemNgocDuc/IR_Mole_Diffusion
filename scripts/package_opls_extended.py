"""Confirm improper coverage, collect hashes, and archive the extended diagnostics."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def improper_count(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'IMPROPERS' for t in node.targets):
            return len(ast.literal_eval(node.value))
    raise ValueError('Loader has no explicit IMPROPERS table')


def main(folder):
    result = {'independent_engine_parity_verified': False, 'physical_correctness_validated': False, 'molecules': {}}
    for name in ('ACE', 'BEN'):
        loader = folder / name / f'{name}.openmm.py'
        count = improper_count(loader.read_text())
        report = json.loads((folder / f'{name}-verification.json').read_text())
        result['molecules'][name] = {'improper_count': count,
            'numerical_checks_passed': report['status'] == 'numerical_checks_passed',
            'loader_sha256': hashlib.sha256(loader.read_bytes()).hexdigest()}
    result['passed'] = all(r['improper_count'] > 0 and r['numerical_checks_passed'] for r in result['molecules'].values())
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with zipfile.ZipFile(folder / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.name != 'results.zip' and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(folder))
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]).resolve()))
