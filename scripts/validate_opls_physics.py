"""Bounded diagnostic pilot: saved geometry parity, template retry, and DFT/IR.

This is a selected training-set diagnostic, not a population-level validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.opls_backend import prepare, sha256
from diffusion.graph_identity import same_achiral_graph
from scripts.diagnose_opls_benchmark import REPORT_SHA256, Engine
from scripts.compare_opls_engines import compare

SELECTION = ((1, 0, 'converged_control'), (7, 0, 'torsion_failure'),
             (13, 0, 'torsion_failure'))
METHOD = 'B3LYP D4 def2-SVP TightSCF TightOpt Opt Freq'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def xyz_text(numbers, xyz):
    from rdkit import Chem
    xyz = np.asarray(xyz, float)
    if xyz.shape != (len(numbers), 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid coordinates')
    table = Chem.GetPeriodicTable()
    return f'{len(numbers)}\nCoordinates in angstrom; declared atom order\n' + ''.join(
        f'{table.GetElementSymbol(int(z))} {x:.12f} {y:.12f} {v:.12f}\n'
        for z, (x, y, v) in zip(numbers, xyz))


def prepare_pilot(source, output, gmx=None, engines=True):
    if sha256(source/'report.json') != REPORT_SHA256:
        raise ValueError('Expected unchanged benchmark 1820024 report')
    report = json.loads((source/'report.json').read_text())
    plan = {'status': 'preparing', 'source_report_sha256': REPORT_SHA256,
            'method': METHOD, 'environment': 'isolated_vacuum', 'tasks': [],
            'physical_correctness_validated': False,
            'selection_scope': 'Three prespecified training-set diagnostic examples; conformer 0, both arms, raw and relaxed. Not a generalization estimate.'}
    # Never guess spectral units: recover and verify the original preprocessing.
    try:
        data_path = Path(report['configuration']['data'])
        if sha256(data_path) != report['selection']['data_sha256']:
            raise ValueError('Original dataset hash mismatch')
        with np.load(data_path, allow_pickle=False) as data:
            scales = json.loads(str(data['preprocessing_json'].item()))
        if sha256(source/'input_ir.npz') != report['input_ir_sha256']:
            raise ValueError('Input IR hash mismatch')
        with np.load(source/'input_ir.npz', allow_pickle=False) as ir:
            targets = {}
            for index, _, _ in SELECTION:
                if int(ir['rows'][index]) != report['inputs'][index]['row']:
                    raise ValueError('Input spectrum row mismatch')
                peaks = ir['packed'][index][ir['mask'][index]].astype(float)
                peaks[:, 0] *= scales['frequency_scale']
                peaks[:, 1] *= scales['intensity_scale']
                targets[str(index)] = peaks.tolist()
        plan['ir_targets'] = targets
        plan['ir_preprocessing'] = scales
        plan['ir_scope'] = 'Comparison to the supplied dataset spectrum, not a verified experimental spectrum. Fixed 20 cm^-1 Gaussian FWHM; 400-4000 cm^-1; frequency scale 1.0; descriptive cosine only.'
    except Exception as exc:
        plan['ir_unavailable'] = str(exc)
    for index, graph_index, reason in SELECTION:
        row = next(r for r in report['inputs'] if r['input_index'] == index)
        graph = next(g for g in row['graphs'] if g['graph_index'] == graph_index)
        prefix = source/f'input_{index:03d}'/f'graph_{graph_index}'
        meta = graph['parameters']
        for name, digest in meta['file_sha256'].items():
            path = (prefix/'parameters'/name).resolve()
            if not path.is_relative_to((prefix/'parameters').resolve()) or sha256(path) != digest:
                raise ValueError('Archived parameter hash/path mismatch')
        cases = []
        for arm in ('unguided', 'opls_guided'):
            npz = prefix/f'{arm}_0.npz'
            with np.load(npz, allow_pickle=False) as data:
                if (not np.array_equal(data['atom_numbers'], meta['atom_numbers']) or
                        not np.array_equal(data['bond_types'], meta['bond_types'])):
                    raise ValueError('Saved coordinate topology mismatch')
                for stage in ('raw', 'relaxed'):
                    xyz = data[stage+'_angstrom']
                    task_id = len(plan['tasks'])
                    task_dir = output/f'qm_{task_id:02d}'
                    task_dir.mkdir()
                    (task_dir/'start.xyz').write_text(xyz_text(meta['atom_numbers'], xyz))
                    (task_dir/'job.inp').write_text(
                        f'! {METHOD}\n%maxcore 2500\n%geom MaxIter 200 end\n* xyzfile 0 1 start.xyz\n')
                    plan['tasks'].append({'id': task_id, 'directory': task_dir.name,
                        'input_index': index, 'graph_index': graph_index, 'smiles': graph['smiles'],
                        'reason': reason, 'arm': arm, 'stage': stage, 'conformer': 0,
                        'atom_numbers': meta['atom_numbers'], 'source_npz_sha256': sha256(npz),
                        'input_sha256': sha256(task_dir/'job.inp'),
                        'xyz_sha256': sha256(task_dir/'start.xyz')})
                    engine_xyz = np.empty_like(xyz)
                    engine_xyz[np.array(meta['model_to_engine'])] = xyz/10
                    if arm == 'opls_guided':
                        cases.append((stage, engine_xyz))
        if engines:
            entry = {'input_index': index, 'graph_index': graph_index}
            plan.setdefault('torsion_checks', []).append(entry)
            try:
                backend = Engine(prefix/'parameters', meta)
                with np.load(prefix/'opls_guided_0.npz', allow_pickle=False) as data:
                    entry['original_geometry_diagnosis'] = backend.diagnose(data['relaxed_angstrom'])
                del backend
                if not gmx:
                    raise RuntimeError('GROMACS unavailable; parity remains unverified')
                # The complete original cache retains GROMACS exports; the
                # benchmark archive copied only OpenMM files. Verify these
                # OpenMM files still match before pairing exports from cache.
                cache = Path(meta['cache_directory'])
                for name, digest in meta['file_sha256'].items():
                    if sha256(cache/name) != digest:
                        raise ValueError('Original cache hash mismatch')
                destination = output/f'parity_{index}'
                destination.mkdir()
                entry['comparison'] = compare(cache/'export', 'MOL', destination, gmx,
                                              forces=True, coordinate_cases=cases)
                entry['status'] = 'completed'
            except Exception as exc:
                entry.update(status='unresolved', error=str(exc))
        write_json(output/'plan.json', plan)
    if engines:
        try:
            folder, metadata = prepare('CNC12COC1C2C#N',
                Path(report['configuration']['appimage']), output/'retry_cache')
            plan['template_retry'] = {'status': 'parameterized', 'metadata': metadata}
        except Exception as exc:
            plan['template_retry'] = {'status': 'failed', 'error': str(exc)}
    plan['status'] = 'prepared'
    write_json(output/'plan.json', plan)
    print(f'Prepared {len(plan["tasks"])} DFT tasks in {output}', flush=True)
    return plan


def parse_orca(text, atom_count):
    """Conservative ORCA 5/6 text reader; incomplete outputs cannot pass."""
    normal = 'ORCA TERMINATED NORMALLY' in text
    converged = 'THE OPTIMIZATION HAS CONVERGED' in text
    frequencies = []
    ir = []
    if 'VIBRATIONAL FREQUENCIES' in text:
        section = text.rsplit('VIBRATIONAL FREQUENCIES', 1)[1].split('NORMAL MODES')[0]
        frequencies = [float(v) for v in re.findall(r'^\s*\d+:\s*([-+\d.]+)\s+cm\*\*-1', section, re.M)]
    if 'IR SPECTRUM' in text:
        section = text.rsplit('IR SPECTRUM', 1)[1]
        if re.search(r'Mode\s+freq\s+eps\s+Int', section):
            for line in section.splitlines():
                m = re.match(r'^\s*\d+:\s+([-+\d.]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+', line)
                if m:
                    ir.append([float(m[1]), float(m[3])])
                elif ir:
                    break
    complete = len(frequencies) == 3*atom_count and np.isfinite(frequencies).all()
    energies = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-+\d.Ee]+)', text)
    versions = re.findall(r'Program Version\s+(\S+)', text)
    negatives = [f for f in frequencies if f < 0]
    if not normal or not converged or not complete:
        status = 'incomplete_or_unconverged'
    elif any(f < -20 for f in negatives):
        status = 'imaginary_modes_require_investigation'
    elif negatives:
        status = 'small_imaginary_modes_require_recheck'
    else:
        status = 'no_imaginary_modes_at_reported_precision'
    return {'status': status, 'normal_termination': normal, 'optimization_converged': converged,
            'orca_version': versions[-1] if versions else None,
            'final_electronic_energy_hartree': float(energies[-1]) if energies else None,
            'frequency_count_complete': bool(complete), 'frequencies_cm_1': frequencies,
            'negative_frequencies_cm_1': negatives, 'ir_peaks_cm_1_km_mol': ir,
            'physical_correctness_validated': False}


def aligned_rmsd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a-a.mean(0), b-b.mean(0)
    u, _, vt = np.linalg.svd(a.T@b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u@vt)
    return float(np.sqrt(np.mean(np.sum((a@u@correction@vt-b)**2, axis=1))))


def spectrum_comparison(target, predicted):
    grid = np.arange(400., 4001., 2.)
    curves = []
    for peaks in (target, predicted):
        peaks = np.asarray(peaks, float)
        if peaks.ndim != 2 or peaks.shape[1] != 2 or not np.isfinite(peaks).all() or np.any(peaks[:, 1] < 0):
            raise ValueError('Invalid IR peaks')
        peaks = peaks[(peaks[:, 0] >= 400) & (peaks[:, 0] <= 4000)]
        curve = (np.exp(-4*np.log(2)*((grid[:, None]-peaks[:, 0])/20)**2)*peaks[:, 1]).sum(1)
        if np.linalg.norm(curve) == 0:
            raise ValueError('No nonzero IR signal in comparison range')
        curves.append(curve/np.linalg.norm(curve))
    return float(curves[0]@curves[1]), np.column_stack((grid, *curves))


def run_task(output, index, orca):
    plan = json.loads((output/'plan.json').read_text())
    task = plan['tasks'][index]
    folder = output/task['directory']
    result = {'task': task, 'status': 'failed', 'physical_correctness_validated': False}
    try:
        if sha256(folder/'job.inp') != task['input_sha256'] or sha256(folder/'start.xyz') != task['xyz_sha256']:
            raise ValueError('DFT inputs changed after preparation')
        if (folder/'job.out').exists():
            raise ValueError('Existing DFT output; refusing overwrite')
        executable = shutil.which(orca)
        if not executable:
            raise ValueError('ORCA unavailable')
        with (folder/'job.out').open('w') as log:
            completed = subprocess.run([str(Path(executable).resolve()), 'job.inp'], cwd=folder,
                stdout=log, stderr=subprocess.STDOUT, timeout=13200)
        result.update(parse_orca((folder/'job.out').read_text(errors='replace'), len(task['atom_numbers'])))
        result['orca_output_sha256'] = sha256(folder/'job.out')
        result['orca_executable'] = str(Path(executable).resolve())
        result['returncode'] = completed.returncode
        if completed.returncode:
            result['status'] = 'engine_failed'
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        molecule = Chem.MolFromXYZBlock((folder/'job.xyz').read_text())
        if molecule is None or [a.GetAtomicNum() for a in molecule.GetAtoms()] != task['atom_numbers']:
            raise ValueError('Final DFT coordinate atom order/count mismatch')
        rdDetermineBonds.DetermineBonds(molecule, charge=0)
        result['achiral_graph_preserved'] = same_achiral_graph(molecule, task['smiles'])
        result['final_smiles'] = Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=False)
        result['final_xyz_sha256'] = sha256(folder/'job.xyz')
        start = Chem.MolFromXYZBlock((folder/'start.xyz').read_text())
        a, b = start.GetConformer().GetPositions(), molecule.GetConformer().GetPositions()
        result['aligned_repair_rmsd_angstrom'] = aligned_rmsd(a, b)
        result['max_pair_distance_change_angstrom'] = float(np.max(np.abs(
            np.linalg.norm(a[:, None]-a[None, :], axis=-1)-np.linalg.norm(b[:, None]-b[None, :], axis=-1))))
        result['local_minimum_supported_for_intended_graph'] = bool(
            result['status'] == 'no_imaginary_modes_at_reported_precision' and result['achiral_graph_preserved'])
        if result['ir_peaks_cm_1_km_mol'] and str(task['input_index']) in plan.get('ir_targets', {}):
            cosine, curves = spectrum_comparison(plan['ir_targets'][str(task['input_index'])], result['ir_peaks_cm_1_km_mol'])
            result['ir_cosine_descriptive'] = cosine
            np.savetxt(folder/'ir_comparison.csv', curves, delimiter=',',
                       header='wavenumber_cm_1,target_unit_norm,dft_unit_norm', comments='')
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', status='failed')
        result['local_minimum_supported_for_intended_graph'] = False
    write_json(folder/'result.json', result)
    print(task['id'], result['status'], flush=True)


def collect(output):
    plan = json.loads((output/'plan.json').read_text())
    tasks = []
    for task in plan['tasks']:
        path = output/task['directory']/'result.json'
        tasks.append(json.loads(path.read_text()) if path.exists() else
                     {'task': task, 'status': 'missing_result', 'local_minimum_supported_for_intended_graph': False})
    summary = {'plan': plan, 'tasks': tasks, 'physical_correctness_validated': False,
               'note': 'DFT screening method and vacuum assumptions limit conclusions. No global minimum, experimental identity, stereo validation, Boltzmann population or generalization claim.'}
    write_json(output/'summary.json', summary)
    lines = ['# Physical-validation pilot', '',
             'Selected training-set diagnostics; no claim of physical correctness.', '',
             'Template retry: '+plan.get('template_retry', {}).get('status', 'not run'),
             'IR preprocessing: '+('verified' if 'ir_targets' in plan else plan.get('ir_unavailable', 'unavailable')), '',
             'Engine comparison (rounded GROMACS coordinates; original singular-geometry validity remains separate):']
    for check in plan.get('torsion_checks', []):
        geometries = check.get('comparison', {}).get('geometries', {})
        outcome = ('passed' if all(g['passed'] for g in geometries.values()) else 'failed') if geometries else 'unresolved'
        lines.append(f"- Input {check['input_index']}: {outcome}. {check.get('error', '')}")
    lines += ['',
             '| Task | Input | Arm | Start | Result | Graph retained | DFT repair RMSD (A) | IR cosine |',
             '|---|---|---|---|---|---|---|---|']
    for r in tasks:
        t = r['task']
        lines.append(f"| {t['id']} | {t['input_index']} | {t['arm']} | {t['stage']} | {r['status']} | {r.get('achiral_graph_preserved', 'unknown')} | {r.get('aligned_repair_rmsd_angstrom', 'unknown')} | {r.get('ir_cosine_descriptive', 'unavailable')} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    destination = output/'physical_validation_results.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.md', '.log', '.out', '.inp', '.xyz', '.csv', '.xvg', '.itp', '.mdp', '.top'):
                z.write(p, p.relative_to(output))
    print('Download:', destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'collect'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', type=Path, default=Path('outputs/opls_benchmark_1820024'))
    parser.add_argument('--gmx')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--index', type=int)
    parser.add_argument('--orca', default='orca')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        output.mkdir(parents=True, exist_ok=False)
        prepare_pilot(args.source.resolve(), output, args.gmx, not args.offline)
        collect(output)
    elif args.action == 'run':
        run_task(output, args.index, args.orca)
    else:
        collect(output)


if __name__ == '__main__':
    main()
