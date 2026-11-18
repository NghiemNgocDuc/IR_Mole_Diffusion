import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from rdkit import Chem

from raw_geometry_validation import check


def engrad(numbers,xyz,gradient,energy=-76.):
    return ('# Number of atoms\n'+str(len(numbers))+'\n# Total energy in Eh\n'+str(energy)+
            '\n# Gradient in Eh/bohr\n'+ '\n'.join(f'{v:.14E}' for v in np.asarray(gradient).ravel())+
            '\n# Atomic numbers and coordinates in bohr\n'+ '\n'.join(
                str(z)+' '+' '.join(f'{v:.14f}' for v in row/check.BOHR_ANGSTROM)
                for z,row in zip(numbers,np.asarray(xyz)))+'\n')


NUMBERS=[8,1,1]
XYZ=np.array([[0.,0,0],[.96,0,0],[-.24,.93,0]])
BONDS=np.array([[0,1,1],[1,0,0],[1,0,0]])
GRADIENT=np.array([[.001,-.002,.003],[-.001,.001,0],[0,.001,-.003]])


def test_gradient_force_sign_units_and_coordinate_provenance():
    result=check.parse_engrad(engrad(NUMBERS,XYZ,GRADIENT),NUMBERS,XYZ)
    expected=-GRADIENT*check.HARTREE_KJ_MOL/(check.BOHR_ANGSTROM/10)
    np.testing.assert_allclose(result['force_kj_mol_nm'],expected)
    assert result['energy_hartree']==-76.
    assert result['force_component_rms_kj_mol_nm']==pytest.approx(np.sqrt(np.mean(expected**2)))
    assert result['force_max_atom_norm_kj_mol_nm']==pytest.approx(np.linalg.norm(expected,axis=1).max())
    assert result['net_force_norm_kj_mol_nm']<1e-10
    assert result['coordinates_unchanged']


@pytest.mark.parametrize('fault', ['truncated','nonfinite','atom_order','coordinates','count'])
def test_invalid_gradient_cannot_be_accepted(fault):
    text=engrad(NUMBERS,XYZ,GRADIENT)
    numbers=NUMBERS; xyz=XYZ
    if fault=='truncated': text='\n'.join(text.splitlines()[:-1])
    if fault=='nonfinite': text=text.replace('-76.0','nan')
    if fault=='atom_order': numbers=[1,8,1]
    if fault=='coordinates': xyz=XYZ+.01
    if fault=='count': numbers=[8,1]
    with pytest.raises(ValueError): check.parse_engrad(text,numbers,xyz)


def test_geometry_and_reference_mapping_do_not_depend_on_global_frame():
    r=check.geometry(NUMBERS,BONDS,XYZ,XYZ)
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    transformed=check.geometry(NUMBERS,BONDS,XYZ@rotation+8,XYZ@rotation+8)
    np.testing.assert_allclose([b['angstrom'] for b in r['bonds']], [b['angstrom'] for b in transformed['bonds']])
    assert r['coordinate_inferred_achiral_graph_matches']
    assert not r['exact_atom_overlap']
    assert not r['nonbonded_contacts_excluding_1_2_and_1_3']  # H-H is 1-3.
    permutation=np.array([2,0,1])
    m=check.reference_mapping(NUMBERS,BONDS,np.array(NUMBERS)[permutation],BONDS[np.ix_(permutation,permutation)])
    assert np.array_equal(np.array(NUMBERS)[permutation][m],NUMBERS)
    with pytest.raises(ValueError): check.reference_mapping(NUMBERS,BONDS,[7,1,1],BONDS)


def make_plan(tmp_path):
    folder=tmp_path/'task_00'; folder.mkdir()
    (folder/'start.xyz').write_text(check.xyz_text(NUMBERS,XYZ))
    (folder/'job.inp').write_text(f'! {check.METHOD}\n* xyzfile 0 1 start.xyz\n')
    task={'id':0,'directory':'task_00','input_index':1,'graph_index':0,'conformer':0,
          'arm':'unguided','stage':'raw','atom_numbers':NUMBERS,'source_npz_sha256':'archive',
          'input_sha256':check.sha(folder/'job.inp'),'xyz_sha256':check.sha(folder/'start.xyz')}
    check.save(tmp_path/'plan.json',{'tasks':[task]})
    return folder,task


def test_runner_measures_fixed_coordinates_and_refuses_duplicates(tmp_path,monkeypatch):
    folder,task=make_plan(tmp_path)
    monkeypatch.setattr(check.shutil,'which',lambda _: 'orca-fixture')
    def engine(args,**kwargs):
        kwargs['stdout'].write('Program Version 6.1.0\nORCA TERMINATED NORMALLY\n')
        (folder/'job.engrad').write_text(engrad(NUMBERS,XYZ,GRADIENT))
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(check.subprocess,'run',engine)
    result=check.run_task(tmp_path,0,'orca-fixture')
    assert result['status']=='evaluated' and result['coordinates_unchanged']
    assert check.sha(folder/'start.xyz')==task['xyz_sha256']
    assert not result['physical_correctness_validated']
    saved=(folder/'result.json').read_bytes()
    with pytest.raises(RuntimeError,match='Existing task output'):
        check.run_task(tmp_path,0,'orca-fixture')
    assert saved==(folder/'result.json').read_bytes()


def test_runner_rejects_engine_failure_even_with_gradient_file(tmp_path,monkeypatch):
    folder,_=make_plan(tmp_path)
    monkeypatch.setattr(check.shutil,'which',lambda _: 'orca-fixture')
    def engine(args,**kwargs):
        kwargs['stdout'].write('Program Version 6.1.0\nSCF NOT CONVERGED\nORCA TERMINATED NORMALLY\n')
        (folder/'job.engrad').write_text(engrad(NUMBERS,XYZ,GRADIENT))
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(check.subprocess,'run',engine)
    assert check.run_task(tmp_path,0,'orca-fixture')['status']=='failed'


def test_actual_saved_geometries_produce_15_engrad_only_jobs(tmp_path):
    source=Path('outputs/opls_diagnostic_source_1820024')
    if not source.exists(): pytest.skip('Archived source unavailable')
    plan=check.prepare(source.resolve(),Path('data/processed/squirl.npz'),tmp_path)
    assert len(plan['tasks'])==15
    assert sum(t['arm']=='reference' for t in plan['tasks'])==3
    for t in plan['tasks']:
        inp=(tmp_path/t['directory']/'job.inp').read_text()
        assert 'EnGrad' in inp and 'Opt' not in inp and 'Freq' not in inp
        assert check.sha(tmp_path/t['directory']/'start.xyz')==t['xyz_sha256']
        if t['stage']=='raw':
            with np.load(source/f"input_{t['input_index']:03d}"/'graph_0'/f"{t['arm']}_0.npz") as a:
                molecule=Chem.MolFromXYZBlock((tmp_path/t['directory']/'start.xyz').read_text())
                np.testing.assert_allclose(molecule.GetConformer().GetPositions(),a['raw_angstrom'],atol=1e-10,rtol=0)
    summary=check.collect(tmp_path)
    assert all(r['status']=='missing_result' for r in summary['results'])
    assert not summary['reference_comparisons']


def test_reference_energy_differences_are_within_molecule_and_same_engine_version():
    ref={'task':{'input_index':1,'arm':'reference','id':4},'status':'evaluated',
         'orca_version':'6.1.0','energy_hartree':-76.,'force_component_rms_kj_mol_nm':2.}
    raw=copy.deepcopy(ref); raw.update(energy_hartree=-75.99,force_component_rms_kj_mol_nm=10.)
    raw['task']={'input_index':1,'arm':'unguided','id':0}
    other=copy.deepcopy(raw); other['task']['input_index']=7
    comparisons=check.paired_summary([raw,ref,other])
    assert len(comparisons)==1
    assert comparisons[0]['energy_minus_dataset_reference_kj_mol']==pytest.approx(.01*check.HARTREE_KJ_MOL)
    ref['orca_version']='6.1.1'
    assert not check.paired_summary([raw,ref])


def test_optimization_join_checks_coordinates_outputs_and_method(tmp_path):
    folder,task=make_plan(tmp_path)
    optimized=tmp_path/'optimized'; oldfolder=optimized/'qm_00'; oldfolder.mkdir(parents=True)
    (oldfolder/'start.xyz').write_bytes((folder/'start.xyz').read_bytes())
    (oldfolder/'job.inp').write_text('! B3LYP D4 def2-SVP TightSCF TightOpt Opt Freq')
    (oldfolder/'job.out').write_text('completed optimization fixture')
    (oldfolder/'job.xyz').write_bytes((folder/'start.xyz').read_bytes())
    old={**task,'directory':'qm_00','input_sha256':check.sha(oldfolder/'job.inp')}
    previous={'task':old,'orca_version':'6.1.0','normal_termination':True,
              'optimization_converged':True,'achiral_graph_preserved':True,'returncode':0,
              'orca_output_sha256':check.sha(oldfolder/'job.out'),
              'final_xyz_sha256':check.sha(oldfolder/'job.xyz'),
              'final_electronic_energy_hartree':-76.01,'aligned_repair_rmsd_angstrom':.1,
              'local_minimum_supported_for_intended_graph':True}
    check.save(oldfolder/'result.json',previous)
    check.save(optimized/'plan.json',{'method':'B3LYP D4 def2-SVP TightSCF TightOpt Opt Freq',
                                     'source_report_sha256':check.REPORT_SHA,'tasks':[old]})
    results=[{'task':task,'status':'evaluated','orca_version':'6.1.0','energy_hartree':-76.}]
    joined=check.optimization_comparisons(tmp_path,results,optimized)
    assert joined['rows'][0]['status']=='compared'
    assert joined['rows'][0]['energy_drop_to_optimized_geometry_kj_mol']==pytest.approx(.01*check.HARTREE_KJ_MOL)
    (oldfolder/'job.xyz').write_text('changed')
    assert check.optimization_comparisons(tmp_path,results,optimized)['rows'][0]['status']!='compared'
