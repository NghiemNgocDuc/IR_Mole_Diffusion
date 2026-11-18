import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diffusion.opls_bridge import MappedEnergy,bounded_correction
from compare_opls_engines import parse_vectors


def harmonic(x):
    weights=np.array([1,2])[:,None]
    return float((weights*x*x).sum()),-2*weights*x


def test_mapping_units_and_gradient():
    bridge=MappedEnergy(harmonic,[1,0]);x=np.array([[2.,0,0],[3.,0,0]])
    energy,force=bridge.evaluate(x)
    assert energy==pytest.approx(.17)
    assert force[:,0]==pytest.approx([-.08,-.06])
    corrected,record=bounded_correction(bridge,x)
    assert record['accepted'] and record['energy_after']<energy
    assert np.linalg.norm(corrected-x,axis=1).max()<=.020000001


@pytest.mark.parametrize('mapping',[[0,0],[1,2],[0.,1.],[]])
def test_bad_mappings_rejected(mapping):
    with pytest.raises(ValueError):MappedEnergy(harmonic,mapping)


def test_nonfinite_and_shape_rejected():
    bridge=MappedEnergy(harmonic,[0,1])
    for x in [np.zeros((3,3)),np.full((2,3),np.nan)]:
        with pytest.raises(ValueError):bridge.evaluate(x)


def test_bad_force_direction_retains_original():
    def wrong(x):
        e,f=harmonic(x);return e,-f
    x=np.ones((2,3));after,record=bounded_correction(MappedEnergy(wrong,[0,1]),x)
    assert not record['accepted'] and np.array_equal(x,after)


def test_vectors_require_exact_complete_frame():
    assert parse_vectors('0 1 2 3 4 5 6',2).tolist()==[[1,2,3],[4,5,6]]
    for text in ['0 1 2','0 1 2 nan','1 1 2 3','0 1 2 3\n0 1 2 3']:
        with pytest.raises(ValueError):parse_vectors(text,1)
