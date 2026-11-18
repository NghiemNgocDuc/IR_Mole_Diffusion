"""Compare single-point OpenMM/GROMACS energies for three archived exports."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np

MDP = '''integrator = md
nsteps = 0
continuation = yes
dt = 0.001
constraints = none
cutoff-scheme = Verlet
nstlist = 1
verlet-buffer-tolerance = -1
rlist = 4.0
rcoulomb = 4.0
rvdw = 4.0
coulombtype = Cut-off
coulomb-modifier = None
vdwtype = Cut-off
vdw-modifier = None
DispCorr = no
pbc = xyz
tcoupl = no
pcoupl = no
comm-mode = None
gen-vel = no
nstcalcenergy = 1
nstenergy = 1
nstlog = 1
nstfout = 1
nstxout = 1
'''


def parse_energy(text):
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(('#', '@'))]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ValueError('Expected exactly one time/potential-energy row')
    time, energy = map(float, rows[0])
    if not np.isfinite(energy) or time != 0:
        raise ValueError('Invalid energy or unexpected time')
    return energy


def gro_text(names, residue, xyz):
    lines = ['Identical coordinates for engine comparison', str(len(names))]
    for i, (name, position) in enumerate(zip(names, xyz), 1):
        lines.append(f'{1:5d}{residue:<5s}{name:>5s}{i:5d}' + ''.join(f'{v:13.8f}' for v in position))
    return '\n'.join(lines + ['10.00000000 10.00000000 10.00000000']) + '\n'


def command(args, folder, log, input_text=None):
    process = subprocess.run(args, cwd=folder, input=input_text, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (folder / log).write_text(process.stdout)
    if process.returncode:
        raise RuntimeError(f'{args[1]} failed with exit {process.returncode}; see {folder / log}')
    return process.stdout


def parse_vectors(text, count):
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(('#', '@'))]
    if len(rows) != 1 or len(rows[0]) != 1 + 3 * count:
        raise ValueError('Expected exactly one time plus N xyz vectors row')
    values = np.asarray(rows[0], dtype=float)
    if values[0] != 0 or not np.isfinite(values).all():
        raise ValueError('Invalid vectors or unexpected time')
    return values[1:].reshape(count, 3)


def compare(source, name, output, gmx, forces=False, coordinate_cases=None, coordinate_offset_nm=5.0):
    import openmm
    from openmm import unit
    source = source.resolve()
    loader = source / f'{name}.openmm.py'
    spec = importlib.util.spec_from_file_location('export_' + name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pdb, system = module.build()
    names = [a.name for a in pdb.topology.atoms()]
    # Require the two topologies to describe exactly the same atom order.
    itp = (source / f'{name}.gmx.itp').read_text()
    section = ''; gmx_names = []
    for line in itp.splitlines():
        line = line.split(';')[0].strip()
        if line.startswith('['): section = line.strip('[] ').strip()
        elif line and section == 'atoms': gmx_names.append(line.split()[4])
    if names != gmx_names:
        raise ValueError('OpenMM/GROMACS atom-name order mismatch')
    if system.usesPeriodicBoundaryConditions() or system.getNumConstraints():
        raise ValueError('Expected unconstrained isolated OpenMM system')
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
    original = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    results = {}
    cases = coordinate_cases if coordinate_cases is not None else [
        ('template', original), ('perturbed', original + np.random.default_rng(20261115).normal(0, .002, original.shape))]
    for label, xyz in cases:
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != original.shape or not np.isfinite(xyz).all():
            raise ValueError('Invalid comparison coordinates')
        folder = output / f'{name}_{label}'; folder.mkdir()
        # Center and quantize once. Both engines get these same eight-decimal nm coordinates.
        if not np.isfinite(coordinate_offset_nm) or not 0.5 <= coordinate_offset_nm <= 9.5:
            raise ValueError('Diagnostic coordinate offset must be inside the box')
        xyz = np.round(xyz - xyz.mean(axis=0) + coordinate_offset_nm, 8)
        if np.max(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)) >= 1.0:
            raise ValueError('Molecule too large for this isolated-box diagnostic')
        (folder / 'input.gro').write_text(gro_text(names, name, xyz))
        (folder / 'molecule.itp').write_text(itp)
        (folder / 'system.top').write_text('#include "molecule.itp"\n[ system ]\nIsolated molecule\n[ molecules ]\n' + name + ' 1\n')
        (folder / 'singlepoint.mdp').write_text(MDP)
        command([gmx, 'grompp', '-f', 'singlepoint.mdp', '-c', 'input.gro', '-p', 'system.top', '-o', 'run.tpr'], folder, 'grompp.log')
        command([gmx, 'mdrun', '-s', 'run.tpr', '-deffnm', 'run', '-ntomp', '1', '-nb', 'cpu', '-bonded', 'cpu', '-pin', 'off'], folder, 'mdrun.log')
        command([gmx, 'energy', '-f', 'run.edr', '-o', 'potential.xvg', '-xvg', 'none'], folder, 'energy.log', 'Potential\n0\n')
        gmx_energy = parse_energy((folder / 'potential.xvg').read_text())
        context.setPositions(xyz * unit.nanometer)
        omm_energy = float(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        difference = abs(omm_energy - gmx_energy)
        results[label] = {'coordinate_offset_nm': coordinate_offset_nm,
                          'openmm_kj_mol': omm_energy, 'gromacs_kj_mol': gmx_energy,
                          'absolute_difference_kj_mol': difference, 'passed': bool(np.isfinite(difference) and difference <= .05)}
        if forces:
            command([gmx, 'traj', '-f', 'run.trr', '-s', 'run.tpr', '-of', 'forces.xvg', '-ox', 'positions.xvg',
                     '-xvg', 'none', '-fp', '-nocom', '-nolen'], folder, 'traj.log', 'System\n')
            gf = parse_vectors((folder / 'forces.xvg').read_text(), len(names))
            gx = parse_vectors((folder / 'positions.xvg').read_text(), len(names))
            # Check the coordinates that GROMACS actually evaluated, including float32 rounding.
            if not np.allclose(xyz, gx, atol=1e-6, rtol=0):
                raise ValueError('GROMACS altered input coordinates beyond precision tolerance')
            context.setPositions(gx * unit.nanometer)
            state = context.getState(getEnergy=True, getForces=True)
            of = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
            evaluated_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            force_pass = bool(np.allclose(of, gf, atol=.2, rtol=1e-3))
            np.save(folder / 'openmm_forces_kj_mol_nm.npy', of)
            results[label].update({'force_parity_passed': force_pass,
                'openmm_at_gromacs_coordinates_kj_mol': evaluated_energy,
                'energy_difference_at_gromacs_coordinates_kj_mol': abs(evaluated_energy-gmx_energy),
                'coordinate_note': 'GROMACS precision can change nearly collinear geometries; parity here does not validate the original singular geometry.',
                'force_max_abs_difference_kj_mol_nm': float(np.abs(of-gf).max()),
                'force_atol_kj_mol_nm': .2, 'force_rtol': 1e-3,
                'coordinate_roundoff_max_nm': float(np.abs(xyz-gx).max())})
            results[label]['passed'] = bool(abs(evaluated_energy-gmx_energy) <= .05 and force_pass)
    return {'geometries': results, 'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [loader, source / f'{name}.openmm.xml', source / f'{name}.openmm.pdb', source / f'{name}.gmx.itp']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gmx', required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    gmx = shutil.which(args.gmx)
    report = {'status': 'error', 'energy_tolerance_kj_mol': .05, 'molecules': {},
              'physical_correctness_validated': False, 'force_parity_verified': False}
    try:
        if not gmx: raise ValueError('GROMACS executable unavailable')
        command([gmx, '--version'], output, 'gromacs-version.txt')
        for name, source in [('ETH', 'outputs/opls_ethanol_1788228'), ('ACE', 'outputs/opls_extended_1791099/ACE'), ('BEN', 'outputs/opls_extended_1791099/BEN')]:
            report['molecules'][name] = compare(Path(source), name, output, gmx)
        passed = all(g['passed'] for r in report['molecules'].values() for g in r['geometries'].values())
        report['status'] = 'energy_parity_passed' if passed else 'energy_parity_failed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with zipfile.ZipFile(output / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'results.zip': archive.write(path, path.relative_to(output))
    print(json.dumps(report, indent=2, allow_nan=False))
    print('Download:', output / 'results.zip')
    return 0 if report['status'] == 'energy_parity_passed' else 1


if __name__ == '__main__':
    sys.exit(main())
