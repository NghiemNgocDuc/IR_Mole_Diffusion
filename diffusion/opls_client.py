"""One persistent, timeout-bounded physics subprocess per active candidate graph."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


class OPLSClient:
    def __init__(self, smiles, python, appimage, cache, log, timeout=360):
        root = Path(__file__).resolve().parents[1]
        self.timeout = timeout
        self.log = open(log, 'w', encoding='utf-8')
        try:
            self.process = subprocess.Popen([str(Path(python).resolve()), '-u', str(root/'scripts/opls_worker.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENMM_CPU_THREADS': '1'}, cwd=root, bufsize=1)
        except Exception:
            self.log.close()
            raise
        self.responses = queue.Queue()
        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            self.metadata = self.request('init', smiles=smiles, appimage=str(Path(appimage).resolve()), cache=str(Path(cache).resolve()))
        except Exception:
            self.close()
            raise

    def request(self, op, **kwargs):
        try:
            self.process.stdin.write(json.dumps({'op': op, **kwargs}, allow_nan=False) + '\n')
            self.process.stdin.flush()
            line = self.responses.get(timeout=self.timeout)
            if line is None:
                raise RuntimeError('OPLS worker exited; inspect worker log')
            response = json.loads(line)
            if not response['ok']:
                raise RuntimeError(response['error'])
            return response['result']
        except (queue.Empty, BrokenPipeError) as exc:
            self.close()
            raise RuntimeError('OPLS worker timed out or stopped; inspect worker log') from exc

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin:
            self.process.stdin.close()
        self.reader.join(timeout=2)
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
