"""Independently check saved readiness arrays and report arithmetic (no engines)."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_opls_engines import parse_vectors, parse_energy


def audit(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, 'Archive CRC error'
        report = json.loads(z.read('report.json'))
        checks = {}
        for name, molecule in report['molecules'].items():
            bridge = molecule['coordinate_bridge']
            with np.load(io.BytesIO(z.read(name + '_bridge_coordinates.npz')), allow_pickle=False) as arrays:
                count = len(arrays['model_to_engine'])
                assert sorted(arrays['model_to_engine'].tolist()) == list(range(count))
                assert arrays['model_to_engine'].tolist() == bridge['model_to_engine']
                assert np.isfinite(arrays['after_angstrom']).all()
                assert np.max(np.linalg.norm(arrays['after_angstrom'] - arrays['before_angstrom'], axis=1)) <= .200001
            for geometry, g in molecule['engine_comparison']['geometries'].items():
                folder = f'{name}_{geometry}/'
                energy = parse_energy(z.read(folder + 'potential.xvg').decode())
                assert energy == g['gromacs_kj_mol']
                delta = abs(energy - g['openmm_kj_mol'])
                assert np.isclose(delta, g['absolute_difference_kj_mol'], atol=1e-12, rtol=0) and delta <= .05
                gf = parse_vectors(z.read(folder + 'forces.xvg').decode(), count)
                of = np.load(io.BytesIO(z.read(folder + 'openmm_forces_kj_mol_nm.npy')), allow_pickle=False)
                assert np.allclose(of, gf, atol=.2, rtol=.001)
                force_delta = float(np.max(np.abs(of-gf)))
                assert np.isclose(force_delta, g['force_max_abs_difference_kj_mol_nm'], atol=1e-12, rtol=0)
                checks[name + '_' + geometry] = {'energy_difference_kj_mol': delta, 'force_max_difference_kj_mol_nm': force_delta}
            last = bridge['initial_energy_kj_mol']
            for step in bridge['trace']:
                assert np.isclose(last, step['energy_before'])
                assert step['energy_after'] <= step['energy_before']
                assert step.get('max_displacement_angstrom', 0) <= .020000001
                last = step['energy_after']
            assert np.isclose(last, bridge['final_energy_kj_mol']) and bridge['passed']
        assert report['passed']
    return {'archive_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'saved_outputs_consistent': True, 'checks': checks,
            'scope': 'Saved numerical outputs and arithmetic independently checked; engines not rerun.'}


if __name__ == '__main__':
    result = audit(sys.argv[1])
    out = Path('outputs/opls_readiness_review_1807422/audit.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
