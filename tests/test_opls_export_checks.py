import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('verify_opls_export', Path(__file__).resolve().parents[1] / 'scripts/verify_opls_export.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def harmonic(x):
    delta = x[1] - x[0]
    distance = np.linalg.norm(delta)
    energy = 50 * (distance - 0.1) ** 2
    f = 100 * (distance - 0.1) * delta / distance
    return energy, np.array([f, -f])


def test_known_energy_and_force():
    report = verifier.numerical_checks(harmonic, [[0, 0, 0], [0.2, 0, 0]])
    assert report['passed']
    assert report['energy_kj_mol'] == pytest.approx(0.5)


@pytest.mark.parametrize('factor', [-1, 0.1, 10])
def test_wrong_sign_and_unit_conversion_fail(factor):
    def bad(x):
        e, f = harmonic(x)
        return e, factor * f
    report = verifier.numerical_checks(bad, [[0, 0, 0], [0.2, 0, 0]])
    assert not report['passed']
    assert not report['checks']['force_matches_negative_energy_gradient']


def test_nonfinite_coordinates_rejected():
    with pytest.raises(ValueError):
        verifier.numerical_checks(harmonic, [[float('nan'), 0, 0]])


def test_external_position_restraint_fails_invariance():
    def restrained(x):
        return float((x*x).sum()), -2*x
    report = verifier.numerical_checks(restrained, [[0.1, 0.2, 0.3]])
    assert report['checks']['force_matches_negative_energy_gradient']
    assert not report['checks']['translation_energy_invariant']
    assert not report['passed']
