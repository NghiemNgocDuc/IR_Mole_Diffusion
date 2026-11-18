"""Numerical checks for a trusted, locally generated LigParGen OpenMM loader.

Passing these checks does NOT establish OPLS parameter parity or physical truth.
The loader is Python code: use only your own LigParGen output.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


def numerical_checks(evaluate, positions, step=1e-4, energy_atol=1e-3,
                     force_atol=0.05, force_rtol=1e-3):
    """evaluate accepts nm, returns kJ/mol and kJ/mol/nm (force, not gradient)."""
    x = np.asarray(positions, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite, nonempty N x 3 coordinates in nm')
    energy, force = evaluate(x)
    force = np.asarray(force, dtype=float)
    if force.shape != x.shape or not np.isfinite(force).all() or not np.isfinite(energy):
        raise ValueError('Nonfinite energy/forces or incorrect force shape')
    numeric = np.empty_like(x)
    for atom in range(len(x)):
        for axis in range(3):
            plus, minus = x.copy(), x.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numeric[atom, axis] = -(evaluate(plus)[0] - evaluate(minus)[0]) / (2 * step)
    translated_energy, translated_force = evaluate(x + [0.31, -0.27, 0.19])
    angle = 0.71
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated_energy, rotated_force = evaluate(x @ rotation.T)
    checks = {
        'force_matches_negative_energy_gradient': bool(np.allclose(force, numeric, atol=force_atol, rtol=force_rtol)),
        'translation_energy_invariant': bool(np.isclose(energy, translated_energy, atol=energy_atol, rtol=0)),
        'translation_force_invariant': bool(np.allclose(force, translated_force, atol=force_atol, rtol=force_rtol)),
        'rotation_energy_invariant': bool(np.isclose(energy, rotated_energy, atol=energy_atol, rtol=0)),
        'rotation_force_covariant': bool(np.allclose(force @ rotation.T, rotated_force, atol=force_atol, rtol=force_rtol)),
        'net_force_near_zero': bool(np.linalg.norm(force.sum(axis=0)) <= force_atol * len(x)),
    }
    return {'checks': checks, 'passed': all(checks.values()),
            'energy_kj_mol': float(energy),
            'max_force_component_kj_mol_nm': float(np.abs(force).max()),
            'max_force_gradient_error_kj_mol_nm': float(np.abs(force - numeric).max()),
            'finite_difference_step_nm': step,
            'energy_atol_kj_mol': energy_atol,
            'force_atol_kj_mol_nm': force_atol, 'force_rtol': force_rtol}


def run(loader):
    import openmm
    from openmm import unit

    loader = Path(loader).resolve()
    spec = importlib.util.spec_from_file_location('ligpargen_local_export', loader)
    module = importlib.util.module_from_spec(spec)
    previous = Path.cwd()
    try:
        os.chdir(loader.parent)
        spec.loader.exec_module(module)
        pdb, system = module.build()
    finally:
        os.chdir(previous)
    positions = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=float)
    if system.getNumParticles() != len(positions):
        raise ValueError('Particle count differs from exported coordinates')
    if system.usesPeriodicBoundaryConditions():
        raise ValueError('This first verifier requires an isolated, nonperiodic system')
    if system.getNumConstraints() or any(system.isVirtualSite(i) for i in range(len(positions))):
        raise ValueError('Constraints and virtual sites require a separate verification protocol')
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))

    def evaluate(x):
        context.setPositions(x * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))

    # A perturbed geometry tests nonzero forces even if the exported template is minimized.
    perturbed = positions + np.random.default_rng(20261115).normal(0, 0.002, positions.shape)
    reports = {'template': numerical_checks(evaluate, positions),
               'perturbed': numerical_checks(evaluate, perturbed)}
    return {'status': 'numerical_checks_passed' if all(r['passed'] for r in reports.values()) else 'numerical_checks_failed',
            'physical_correctness_validated': False,
            'opls_reference_engine_parity_verified': False,
            'atom_mapping_to_model_verified': False,
            'openmm_version': openmm.__version__, 'platform': 'Reference',
            'particle_count': len(positions),
            'forces': [type(f).__name__ for f in system.getForces()],
            'export_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(loader.parent.glob('*.openmm.*')) if p.is_file()},
            'serialized_system_sha256': hashlib.sha256(openmm.XmlSerializer.serialize(system).encode()).hexdigest(),
            'geometries': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loader', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.loader)
    except Exception as exc:
        result = {'status': 'verification_error', 'error': f'{type(exc).__name__}: {exc}',
                  'physical_correctness_validated': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'numerical_checks_passed' else 1


if __name__ == '__main__':
    sys.exit(main())
