"""Bounded ethanol-only OPLS relaxation diagnostic; not a production validator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from verify_opls_export import run as verify_export


def force_metrics(force):
    force = np.asarray(force, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3 or not len(force) or not np.isfinite(force).all():
        raise ValueError('Invalid forces')
    return {'rms_force_kj_mol_nm': float(np.sqrt(np.mean(force**2))),
            'max_force_component_kj_mol_nm': float(np.abs(force).max())}


def inferred_identity(elements, positions_nm):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    lines = [str(len(elements)), 'coordinate-based ethanol diagnostic']
    for element, xyz in zip(elements, np.asarray(positions_nm) * 10):
        lines.append(element + ' ' + ' '.join(f'{v:.10f}' for v in xyz))
    molecule = Chem.MolFromXYZBlock('\n'.join(lines) + '\n')
    if molecule is None:
        raise ValueError('Could not read coordinates')
    rdDetermineBonds.DetermineBonds(molecule, charge=0)
    return Chem.MolToSmiles(Chem.RemoveHs(molecule))


def run(loader, output):
    import openmm
    from openmm import app, unit
    verification = verify_export(loader)
    if verification['status'] != 'numerical_checks_passed':
        raise ValueError('Export failed prerequisite numerical verification')
    spec = importlib.util.spec_from_file_location('ethanol_export', loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pdb, system = module.build()
    elements = [atom.element.symbol for atom in pdb.topology.atoms()]
    original = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    if sorted(elements) != sorted(['C']*2 + ['H']*6 + ['O']):
        raise ValueError('This diagnostic supports ethanol only')
    if inferred_identity(elements, original) != 'CCO':
        raise ValueError('Starting geometry does not infer as ethanol')
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
    force_unit = unit.kilojoule_per_mole / unit.nanometer

    def snapshot():
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        xyz = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        if not np.isfinite(energy) or not np.isfinite(xyz).all():
            raise ValueError('Nonfinite minimized state')
        return {'energy_kj_mol': energy, **force_metrics(state.getForces(asNumpy=True).value_in_unit(force_unit))}, xyz

    reports = {}
    variants = {'template': original,
                'perturbed': original + np.random.default_rng(20261115).normal(0, 0.002, original.shape)}
    for label, start in variants.items():
        context.setPositions(start * unit.nanometer)
        before, before_xyz = snapshot()
        openmm.LocalEnergyMinimizer.minimize(context, tolerance=1.0, maxIterations=1000)
        after, after_xyz = snapshot()
        identity = inferred_identity(elements, after_xyz)
        checks = {'energy_not_increased': after['energy_kj_mol'] <= before['energy_kj_mol'] + 1e-6,
                  'force_rms_at_most_1_kj_mol_nm': after['rms_force_kj_mol_nm'] <= 1.0,
                  'coordinate_inferred_identity_is_ethanol': identity == 'CCO'}
        for name, xyz in [('before', before_xyz), ('after', after_xyz)]:
            np.save(output / f'{label}_{name}_nm.npy', xyz)
            with (output / f'{label}_{name}.pdb').open('w') as stream:
                app.PDBFile.writeFile(pdb.topology, xyz * unit.nanometer, stream)
        reports[label] = {'before': before, 'after': after, 'checks': checks,
                          'coordinate_inferred_smiles_after': identity,
                          'passed': all(checks.values())}
    return {'status': 'relaxation_checks_passed' if all(r['passed'] for r in reports.values()) else 'relaxation_checks_failed',
            'geometries': reports, 'max_iterations': 1000, 'force_rms_target_kj_mol_nm': 1.0,
            'openmm_version': openmm.__version__, 'platform': 'Reference',
            'prerequisite_verification': verification,
            'loader_sha256': hashlib.sha256(loader.read_bytes()).hexdigest(),
            'physical_correctness_validated': False, 'global_minimum_established': False,
            'opls_reference_engine_parity_verified': False,
            'note': 'Local minimization and coordinate-based bond perception only; not proof of physical identity or a global minimum.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loader', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.loader.resolve(), output)
    except Exception as exc:
        report = {'status': 'relaxation_error', 'error': f'{type(exc).__name__}: {exc}',
                  'physical_correctness_validated': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'relaxation_checks_passed' else 1


if __name__ == '__main__':
    sys.exit(main())
