"""Install the staged pipeline update after checking all existing file hashes."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    stage = root/'opls_pipeline_update'
    manifest = json.loads((stage/'manifest.json').read_text())
    conflicts = []
    entries = []
    for name, info in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid update path')
        source, target = stage/relative, root/relative
        if not source.resolve().is_relative_to(stage.resolve()) or not target.resolve().is_relative_to(root):
            raise ValueError('Update path escapes workspace')
        if digest(source) != info['new_sha256']:
            raise ValueError('Corrupt staged update: ' + name)
        if target.exists() and digest(target) not in [info['new_sha256'], *info['previous_sha256']]:
            conflicts.append(name)
        entries.append((source, target, info))
    if conflicts:
        raise SystemExit('Stopped before changes: unexpected existing edits in ' + ', '.join(conflicts))
    backup = root/'outputs'/('opls_pipeline_install_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for source, target, info in entries:
        if target.exists() and digest(target) == info['new_sha256']:
            continue
        if target.exists():
            saved = backup/target.relative_to(root)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print('Pipeline update installed. Existing versions backed up when changed:', backup)


if __name__ == '__main__':
    main()
