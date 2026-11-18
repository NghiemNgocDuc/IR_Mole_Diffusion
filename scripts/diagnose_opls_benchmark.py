"""Rescore graph notation and diagnose saved high-force geometries; no sampling."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.graph_identity import same_achiral_graph

REPORT_SHA256 = '7eb6f5d2bba6fa3c013e67a9f29902d811349b826a34bf99ed2719d37f2c6439'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plane_sine(x, i, j, k):
    a, b = x[i]-x[j], x[k]-x[j]
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.linalg.norm(np.cross(a, b))/denominator) if denominator else 0.


def geometry(x, bonds):
    x, bonds = np.asarray(x, dtype=float), np.asarray(bonds)
    if x.ndim != 2 or x.shape[1] != 3 or bonds.shape != (len(x), len(x)) or not np.isfinite(x).all():
        raise ValueError('Invalid geometry/topology')
    angles, distances, contacts = [], [], []
    neighbors = [set(np.flatnonzero(bonds[i]).tolist()) for i in range(len(x))]
    for j in range(len(x)):
        for i, k in itertools.combinations(sorted(neighbors[j]), 2):
            a, b = x[i]-x[j], x[k]-x[j]
            product = np.linalg.norm(a)*np.linalg.norm(b)
            cosine = float(np.dot(a, b)/product) if product else 1.
            angles.append({'atoms': [i, j, k], 'sine': plane_sine(x, i, j, k),
                           'degrees': float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
                           'bond_orders': [int(bonds[i,j]), int(bonds[j,k])]})
        for i in range(j):
            item = {'atoms': [i, j], 'distance_angstrom': float(np.linalg.norm(x[i]-x[j]))}
            if bonds[i,j]:
                distances.append(item)
            elif not neighbors[i].intersection(neighbors[j]):
                contacts.append(item)  # Exclude bonded and 1-3 pairs.
    return {'most_collinear_bond_angles': sorted(angles, key=lambda a:a['sine'])[:6],
            'shortest_bonds': sorted(distances, key=lambda a:a['distance_angstrom'])[:5],
            'longest_bonds': sorted(distances, key=lambda a:-a['distance_angstrom'])[:5],
            'closest_pairs_beyond_two_bonds': sorted(contacts, key=lambda a:a['distance_angstrom'])[:5],
            'note': 'Descriptive distances/angles, not rejection thresholds; linear bond angles can be chemically expected.'}


def rescore(report):
    changes, counts = [], {}
    for arm in ('unguided', 'opls_guided'):
        count = Counter()
        for row in report['inputs']:
            for g in row['graphs']:
                for a in g['attempts']:
                    if a['arm'] != arm:
                        continue
                    for stage, state in [('raw', a.get('raw')), ('relaxed', a.get('relaxation', {}).get('after'))]:
                        if not state:
                            continue
                        count[stage+'_assessed'] += 1
                        old = bool(state.get('coordinate_inferred_graph_matches'))
                        inferred = state.get('coordinate_inferred_smiles')
                        new = same_achiral_graph(inferred, g['smiles']) if inferred else False
                        count[stage+'_original_matches'] += int(old)
                        count[stage+'_corrected_matches'] += int(new)
                        if stage == 'relaxed' and a['relaxation']['force_converged'] and a['relaxation']['accepted']:
                            count['original_converged_matches'] += int(old)
                            count['corrected_converged_matches'] += int(new)
                        if old != new:
                            changes.append({'input_index': row['input_index'], 'graph_index': g['graph_index'],
                                'arm': arm, 'conformer': a['conformer'], 'stage': stage,
                                'declared': g['smiles'], 'inferred': inferred, 'original_match': old, 'corrected_match': new})
        counts[arm] = dict(count)
    return {'counts': counts, 'changes': changes, 'method': 'explicit_h_achiral_v1',
            'note': 'Rechecks recorded inferred graphs; does not rerun bond perception or overwrite original results.'}


class Engine:
    def __init__(self, folder, metadata):
        import openmm
        from openmm import unit
        self.mm, self.unit, self.mapping = openmm, unit, np.array(metadata['model_to_engine'])
        if sorted(self.mapping.tolist()) != list(range(len(self.mapping))):
            raise ValueError('Invalid mapping')
        for relative, expected in metadata['file_sha256'].items():
            path = (folder/relative).resolve()
            if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
                raise ValueError('Parameter path/hash mismatch')
        loader = folder/'export/MOL.openmm.py'
        spec = importlib.util.spec_from_file_location('saved_ligpargen_export', loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pdb, system = module.build()
        if system.getNumParticles() != len(self.mapping) or system.getNumForces() > 32:
            raise ValueError('Unexpected system size')
        if system.usesPeriodicBoundaryConditions() or system.getNumConstraints():
            raise ValueError('Expected unconstrained isolated system')
        self.system = system
        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i)
        self.integrator = openmm.VerletIntegrator(.001)
        self.context = openmm.Context(system, self.integrator, openmm.Platform.getPlatformByName('Reference'))
        self.template = np.array(pdb.positions.value_in_unit(unit.nanometer))[self.mapping]*10

    def state(self, x_nm, group=-1):
        self.context.setPositions(x_nm*self.unit.nanometer)
        state = self.context.getState(getEnergy=True, getForces=True, groups=group)
        return float(state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole)), np.array(
            state.getForces(asNumpy=True).value_in_unit(self.unit.kilojoule_per_mole/self.unit.nanometer))

    def diagnose(self, model_xyz, reference=None):
        engine_xyz = np.empty_like(model_xyz, dtype=float)
        engine_xyz[self.mapping] = np.asarray(model_xyz)/10
        energy, forces = self.state(engine_xyz)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError('Nonfinite recomputed energy/force')
        rows, summed_force, summed_energy = [], np.zeros_like(forces), 0.
        for i, force in enumerate(self.system.getForces()):
            e, f = self.state(engine_xyz, 1 << i)
            if not np.isfinite(e) or not np.isfinite(f).all():
                raise ValueError('Nonfinite force-group result')
            summed_energy += e
            summed_force += f
            atom, axis = np.unravel_index(np.abs(f).argmax(), f.shape)
            row = {'force_index': i, 'type': type(force).__name__, 'energy_kj_mol': e,
                   'force_rms_kj_mol_nm': float(np.sqrt(np.mean(f*f))),
                   'max_component_kj_mol_nm': float(np.abs(f).max()),
                   'worst_engine_atom': int(atom), 'worst_model_atom': int(np.flatnonzero(self.mapping == atom)[0]),
                   'worst_axis': int(axis)}
            if isinstance(force, self.mm.PeriodicTorsionForce):
                torsions = []
                for t in range(force.getNumTorsions()):
                    a,b,c,d,period,phase,k = force.getTorsionParameters(t)
                    strength = float(k.value_in_unit(self.unit.kilojoule_per_mole))
                    if strength == 0:
                        continue
                    torsions.append({'term': t, 'engine_atoms': [a,b,c,d], 'periodicity': period,
                        'k_kj_mol': strength, 'min_plane_sine': min(plane_sine(engine_xyz,a,b,c),plane_sine(engine_xyz,b,c,d))})
                row['most_collinear_nonzero_torsions'] = sorted(torsions, key=lambda t:t['min_plane_sine'])[:8]
            rows.append(row)
        dominant = max(rows, key=lambda r:r['max_component_kj_mol_nm'])
        atom, axis = dominant['worst_engine_atom'], dominant['worst_axis']
        finite_differences = []
        for h in (1e-4, 1e-6, 1e-8):
            plus, minus = engine_xyz.copy(), engine_xyz.copy()
            plus[atom,axis] += h
            minus[atom,axis] -= h
            numeric = -(self.state(plus,1<<dominant['force_index'])[0]-self.state(minus,1<<dominant['force_index'])[0])/(2*h)
            if not np.isfinite(numeric):
                raise ValueError('Nonfinite finite-difference probe')
            finite_differences.append({'step_nm': h, 'numeric_force_kj_mol_nm': numeric})
        _, dominant_forces = self.state(engine_xyz, 1<<dominant['force_index'])
        result = {'energy_kj_mol': energy, 'force_rms_kj_mol_nm': float(np.sqrt(np.mean(forces*forces))),
            'groups': rows, 'force_group_sum_matches': bool(np.allclose(summed_force, forces, atol=1e-5, rtol=1e-8)),
            'energy_group_sum_matches': bool(np.isclose(summed_energy, energy, atol=1e-6, rtol=1e-8)),
            'dominant_group': dominant['type'], 'dominant_force_index': dominant['force_index'],
            'finite_difference_probe': {'engine_atom': atom, 'axis': axis,
                'analytic_force_kj_mol_nm': float(dominant_forces[atom,axis]), 'values': finite_differences,
                'note': 'Step sensitivity near collinear geometry is diagnostic, not a pass/fail physical test.'}}
        if reference:
            result['replays_saved_energy'] = bool(np.isclose(energy, reference['energy_kj_mol'], atol=.001, rtol=1e-8))
            result['replays_saved_force_rms'] = bool(np.isclose(result['force_rms_kj_mol_nm'], reference['force_rms_kj_mol_nm'], atol=.05, rtol=.001))
        return result


def run(source, output, engine=False):
    report_path = source/'report.json'
    if sha(report_path) != REPORT_SHA256:
        raise ValueError('Expected the unchanged report from benchmark 1820024')
    report = json.loads(report_path.read_text())
    result = {'status': 'running', 'source_report_sha256': REPORT_SHA256, 'engine_enabled': engine,
              'graph_rescoring': rescore(report), 'graphs': [], 'physical_correctness_validated': False,
              'limitations': 'No resampling, minimization, tuning or model promotion. Diagnostics use saved coordinates.'}
    def persist():
        (output/'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    selected, control_added = [], False
    for row in report['inputs']:
        for g in row['graphs']:
            failed_force = any('relaxation' in a and not a['relaxation']['force_converged'] for a in g['attempts'])
            control = (not failed_force and not control_added and g['status'] == 'completed' and
                       all(a['relaxation']['force_converged'] for a in g['attempts']))
            if failed_force or control:
                selected.append((row, g, 'nonconverged' if failed_force else 'first_converged_control'))
                control_added |= control
    for row,g,reason in selected:
        prefix = source/f"input_{row['input_index']:03d}"/f"graph_{g['graph_index']}"
        entry = {'input_index': row['input_index'], 'graph_index': g['graph_index'], 'smiles': g['smiles'],
                 'selection_reason': reason, 'geometries': []}
        result['graphs'].append(entry)
        try:
            backend = Engine(prefix/'parameters', g['parameters']) if engine else None
            if backend:
                entry['template'] = backend.diagnose(backend.template)
                entry['template_geometry'] = geometry(backend.template, g['parameters']['bond_types'])
            for a in g['attempts']:
                if 'relaxation' not in a:
                    continue
                with np.load(prefix/f"{a['arm']}_{a['conformer']}.npz", allow_pickle=False) as data:
                    if not np.array_equal(data['atom_numbers'],g['parameters']['atom_numbers']) or not np.array_equal(data['bond_types'],g['parameters']['bond_types']):
                        raise ValueError('Saved topology differs from parameter mapping')
                    for stage in ('raw', 'relaxed'):
                        x = data[stage+'_angstrom']
                        state = a['raw'] if stage == 'raw' else a['relaxation']['after']
                        item = {'arm': a['arm'], 'conformer': a['conformer'], 'stage': stage,
                                'saved_assessment': state, 'geometry': geometry(x,data['bond_types'])}
                        if backend:
                            item['force_diagnosis'] = backend.diagnose(x,state)
                        entry['geometries'].append(item)
                persist()
            entry['status'] = 'completed'
            if backend:
                del backend
        except Exception as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print('Diagnosed', row['input_index'], g['graph_index'], entry['status'], flush=True)
        persist()
    # Collect the retained parameterization failure log, without retrying LigParGen.
    log = source.parents[1]/'outputs/opls_parameter_cache/40b428daf7e5414c172b5340c2a0077dc8c87284f809b2adb68166c3ef39e1c9.work-4c4cj3_o/parameterization.log'
    result['parameterization_failure_log_found'] = log.is_file()
    if log.is_file():
        (output/'failed_parameterization.log').write_bytes(log.read_bytes())
    result['status'] = 'completed' if all(g['status']=='completed' for g in result['graphs']) else 'completed_with_failures'
    persist()
    with zipfile.ZipFile(output/'diagnostic_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in output.iterdir():
            if p.is_file() and p.suffix != '.zip':
                z.write(p,p.name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/opls_benchmark_1820024'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result = run(args.source.resolve(), args.output.resolve(), args.engine)
    print(args.output/'diagnostic_results.zip')
    return 0 if result['status']=='completed' else 1


if __name__ == '__main__':
    sys.exit(main())
