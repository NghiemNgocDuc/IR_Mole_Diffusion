import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from rdkit import Chem
from diffusion.graph_identity import same_achiral_graph
from scripts.diagnose_opls_benchmark import geometry, rescore, run


@pytest.mark.parametrize('left,right', [
    ('CCOC(=N)C(C)=NO','[H]N=C(OCC)C(C)=NO'),
    ('CCOC(=NO)C(C)=N','[H]N=C(C)C(=NO)OCC'),
    ('CCO','[H]OC([H])([H])C([H])([H])[H]'),
    ('c1ccccc1','C1=CC=CC=C1'), ('[CH3:1][OH:2]','CO'),
    ('N[C@H](C)O','N[C@@H](C)O')])
def test_hydrogen_representation_and_intended_achiral_equivalence(left,right):
    assert same_achiral_graph(left,right)
    assert same_achiral_graph(Chem.AddHs(Chem.MolFromSmiles(left)),right)


@pytest.mark.parametrize('left,right', [('CCO','COC'), ('C','[CH3]'), ('[NH4+]','N'),
                                     ('C','[13CH4]'), ('CC','C.C'), ('C=O','CO')])
def test_distinct_graphs_are_not_merged(left,right):
    assert not same_achiral_graph(left,right)


def test_geometry_uses_declared_bonds_and_is_rotation_translation_invariant():
    x=np.array([[0.,0,0],[1.,0,0],[2.,1e-8,0],[2.,1.,0]])
    bonds=np.zeros((4,4),int)
    for i,j in [(0,1),(1,2),(2,3)]:
        bonds[i,j]=bonds[j,i]=1
    result=geometry(x,bonds)
    assert result['most_collinear_bond_angles'][0]['atoms']==[0,1,2]
    assert result['most_collinear_bond_angles'][0]['sine'] < 1e-7
    assert [p['atoms'] for p in result['closest_pairs_beyond_two_bonds']]==[[0,3]]
    rotated=geometry(x[:,[1,2,0]]+[4.,5.,6.],bonds)
    assert np.isclose(rotated['shortest_bonds'][0]['distance_angstrom'],result['shortest_bonds'][0]['distance_angstrom'])
    with pytest.raises(ValueError):
        geometry(np.ones((4,2)),bonds)


def test_actual_archive_rescoring_preserves_original_and_corrects_notation():
    path=Path('benchmark_results.zip')
    if not path.exists():
        pytest.skip('Downloaded benchmark unavailable')
    with zipfile.ZipFile(path) as z:
        report=json.loads(z.read('report.json'))
    original=json.dumps(report,sort_keys=True)
    result=rescore(report)
    assert json.dumps(report,sort_keys=True)==original
    for arm in ('unguided','opls_guided'):
        assert result['counts'][arm]['original_converged_matches']==106
        assert result['counts'][arm]['corrected_converged_matches']==112
    assert all(c['input_index']==11 and c['corrected_match'] for c in result['changes'])


def test_diagnostic_refuses_unrecognized_report(tmp_path):
    source=tmp_path/'source';source.mkdir()
    output=tmp_path/'output';output.mkdir()
    (source/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='unchanged report'):
        run(source,output)
