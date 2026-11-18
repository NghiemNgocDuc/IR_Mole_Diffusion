"""Explicit atom-order/unit adapter and bounded coordinate correction.

No atom matching is inferred from noisy xyz. The optional sampler policy uses
this adapter on estimated clean coordinates, through an isolated worker.
"""
import numpy as np


class MappedEnergy:
    def __init__(self, evaluate_nm, model_to_engine):
        self.evaluate_nm = evaluate_nm
        raw = np.asarray(model_to_engine)
        if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Mapping must be a nonempty integer permutation')
        self.mapping = raw.astype(int)
        if sorted(self.mapping.tolist()) != list(range(len(raw))):
            raise ValueError('Mapping must contain each engine atom exactly once')

    def evaluate(self, positions_angstrom):
        x = np.asarray(positions_angstrom, dtype=float)
        if x.shape != (len(self.mapping), 3) or not np.isfinite(x).all():
            raise ValueError('Expected finite model-order N x 3 Angstrom positions')
        engine = np.empty_like(x)
        engine[self.mapping] = x / 10
        energy, forces = self.evaluate_nm(engine)
        forces = np.asarray(forces, dtype=float)
        if forces.shape != x.shape or not np.isfinite(forces).all() or not np.isfinite(energy):
            raise ValueError('Invalid energy/forces from backend')
        return float(energy), forces[self.mapping] / 10


def bounded_correction(evaluator, positions_angstrom, max_step_angstrom=.02, backtracks=12,
                       preserve_centroid=False):
    """One force-directed, energy-decreasing step with a per-atom displacement cap.

    Intended for predicted clean coordinates after a noise gate, not noisy xt.
    Retains input if no finite energy-decreasing trial is found. Not sampling a
    Boltzmann distribution and not an early-rejection policy.
    """
    if not np.isfinite(max_step_angstrom) or max_step_angstrom <= 0 or backtracks < 1:
        raise ValueError('Positive step cap and backtracking budget required')
    x = np.asarray(positions_angstrom, dtype=float).copy()
    energy, force = evaluator.evaluate(x)
    if preserve_centroid:
        force = force - force.mean(axis=0)
    peak = float(np.linalg.norm(force, axis=1).max())
    if peak == 0:
        return x, {'accepted': False, 'reason': 'zero_force', 'energy_before': energy, 'energy_after': energy}
    displacement = force * (max_step_angstrom / peak)
    for trial in range(backtracks):
        candidate = x + displacement * 0.5**trial
        try:
            new_energy, _ = evaluator.evaluate(candidate)
        except (ValueError, FloatingPointError):
            continue
        if new_energy < energy:
            return candidate, {'accepted': True, 'energy_before': energy, 'energy_after': new_energy,
                'backtracks': trial, 'max_displacement_angstrom': float(np.linalg.norm(candidate-x, axis=1).max())}
    return x, {'accepted': False, 'reason': 'no_decreasing_step', 'energy_before': energy, 'energy_after': energy}
