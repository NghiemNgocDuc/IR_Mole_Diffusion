from types import SimpleNamespace
import numpy as np
import pytest
import torch
from rdkit import Chem
import predict_opls as p
import benchmark_opls as b
from scripts.probe_opls_precision import coordinate_cases
import scripts.probe_opls_precision as probe
import json


def args(experimental=False):
    return SimpleNamespace(conformers=2,seed=123,device='cpu',steps=3,experimental_opls=experimental,
        min_alpha=.8,max_step=.02,opls_python='missing',appimage='missing',cache='missing')


class Model:
    n_timesteps=10
    def __init__(self): self.policies=[]
    def sample(self,spectra,atoms,count,**kwargs):
        self.policies.append(kwargs['opls_guidance'])
        assert kwargs['physics_guidance']==0.
        return torch.randn((1,int(count[0]),3),generator=kwargs['generator']), None


def test_default_generates_without_parameterization_or_backend(tmp_path,monkeypatch):
    def forbidden(*a,**k): raise AssertionError('OPLS backend started in default mode')
    monkeypatch.setattr(p,'OPLSClient',forbidden)
    model=Model();entry={'smiles':'CCO','graph_index':0,'attempts':[]}
    p.run_candidate(args(),model,{},entry,tmp_path)
    assert model.policies==[None,None]
    assert len(entry['attempts'])==2
    for a in entry['attempts']:
        assert a['generation_status']=='completed' and a['assessment_status']=='not_requested'
        assert a['primary_geometry_stage']=='raw'
        assert p.file_sha256(tmp_path/a['raw_coordinates_file'])==a['raw_coordinates_sha256']
        with np.load(tmp_path/f"unguided_{a['conformer']}.npz") as data:
            assert 'raw_angstrom' in data and 'relaxed_angstrom' not in data
    assert len(list(tmp_path.glob('*_raw.sdf')))==2


def test_backend_failure_preserves_baseline_and_records_unavailable_guided(tmp_path,monkeypatch):
    def broken(*a,**k): raise RuntimeError('parameterization failed')
    monkeypatch.setattr(p,'OPLSClient',broken)
    entry={'smiles':'CCO','graph_index':0,'attempts':[]}; model=Model()
    p.run_candidate(args(True),model,{},entry,tmp_path)
    assert len(entry['attempts'])==4
    assert [a['generation_status'] for a in entry['attempts']]==['completed','completed','unavailable','unavailable']
    assert len(list(tmp_path.glob('*_raw.npy')))==2


@pytest.mark.parametrize('fail',[False,True])
def test_assessment_never_replaces_raw_and_records_failure_separately(tmp_path,fail):
    template=Chem.AddHs(Chem.MolFromSmiles('CCO'))
    attempt=p.generate_attempt(args(),Model(),{},template,tmp_path,'unguided',0,123)
    before=(tmp_path/'unguided_0_raw.npy').read_bytes()
    class Client:
        def request(self,op,coordinates):
            if fail: raise RuntimeError('force engine failed')
            if op=='evaluate': return {'energy_kj_mol':10}
            return {'coordinates':(np.asarray(coordinates)+.1).tolist(),'after':{'energy_kj_mol':1}}
    p.assess_attempt(attempt,tmp_path,template,Client())
    assert attempt['generation_status']=='completed'
    assert attempt['assessment_status']==('failed' if fail else 'completed')
    assert (tmp_path/'unguided_0_raw.npy').read_bytes()==before
    if not fail:
        with np.load(tmp_path/'unguided_0.npz') as data:
            np.testing.assert_array_equal(data['raw_angstrom'],np.load(tmp_path/'unguided_0_raw.npy'))
            np.testing.assert_allclose(data['relaxed_angstrom'],data['raw_angstrom']+.1)
        assert (tmp_path/'unguided_0_opls_relaxed.sdf').is_file()


def test_guidance_cannot_be_enabled_without_explicit_flag(tmp_path):
    template=Chem.AddHs(Chem.MolFromSmiles('CCO'));model=Model()
    result=p.generate_attempt(args(),model,{},template,tmp_path,'opls_guided',0,1,object())
    assert result['generation_status']=='failed' and model.policies==[]
    assert not list(tmp_path.glob('*_raw.npy'))


def test_benchmark_requires_opt_in_before_any_work(tmp_path):
    with pytest.raises(SystemExit) as exc:
        b.main(['--output',str(tmp_path/'no_run')])
    assert exc.value.code==2 and not (tmp_path/'no_run').exists()


def test_precision_probe_preserves_originals_and_uses_fixed_small_displacements():
    original=np.arange(15,dtype=float).reshape(5,3)/100
    backup=original.copy();cases=coordinate_cases(original,original)
    np.testing.assert_array_equal(original,backup)
    np.testing.assert_array_equal(cases[1][1],original)
    np.testing.assert_allclose(np.linalg.norm(cases[2][1]-original,axis=1),1e-4)
    np.testing.assert_allclose(np.linalg.norm(cases[3][1]-original,axis=1),1e-3)
    assert np.array_equal(cases[2][1],coordinate_cases(original,original)[2][1])


def test_precision_driver_maps_atoms_and_retains_failed_comparisons(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();output=tmp_path/'out';output.mkdir()
    cache=tmp_path/'cache';cache.mkdir();(cache/'export').mkdir()
    parameter=cache/'export/MOL.openmm.xml';parameter.write_text('fixture')
    metadata={'cache_directory':str(cache),'file_sha256':{'export/MOL.openmm.xml':probe.digest(parameter)},
              'model_to_engine':[2,0,1]}
    inputs=[{} for _ in range(14)]
    points=np.arange(9).reshape(3,3).astype(float)
    for i in (1,7,13):
        inputs[i]={'graphs':[{'graph_index':0,'parameters':metadata}]}
        folder=source/f'input_{i:03d}'/'graph_0';folder.mkdir(parents=True)
        np.savez(folder/'opls_guided_0.npz',raw_angstrom=points,relaxed_angstrom=points+.01)
    (source/'report.json').write_text(json.dumps({'inputs':inputs}))
    monkeypatch.setattr(probe,'REPORT_SHA256',probe.digest(source/'report.json'))
    monkeypatch.setattr(probe,'command',lambda *a:None)
    seen=[]
    def compare(*a,**kwargs):
        seen.append(kwargs)
        np.testing.assert_array_equal(kwargs['coordinate_cases'][0][1],points[[1,2,0]]/10.)
        return {'geometries':{'relaxed':{'passed':False},'raw':{'passed':True}}}
    monkeypatch.setattr(probe,'compare',compare)
    r=probe.run(source,output,'fake')
    assert len(seen)==6 and {x['coordinate_offset_nm'] for x in seen}=={.5,5.}
    assert all(not c['comparison']['geometries']['relaxed']['passed'] for c in r['cases'])
    assert r['guidance_approved'] is False
    parameter.write_text('tamper')
    with pytest.raises(ValueError,match='cache changed'):
        probe.run(source,output,'fake')
