"""JSON-lines physics worker; run with .runtime/opls-tools/bin/python."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.opls_backend import OPLSBackend


def main():
    backend = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                op = request['op']
                if op == 'init':
                    if backend is not None:
                        raise ValueError('Worker already initialized')
                    backend = OPLSBackend(request['smiles'], request['appimage'], request['cache'])
                    result = backend.metadata
                elif backend is None:
                    raise ValueError('Worker must be initialized first')
                elif op == 'correct':
                    result = backend.correct(request['coordinates'], request['max_step'])
                elif op == 'evaluate':
                    result = backend.assess(request['coordinates'])
                elif op == 'relax':
                    result = backend.relax(request['coordinates'])
                else:
                    raise ValueError('Unknown worker operation')
            response = {'ok': True, 'result': result}
        except Exception as exc:
            response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
