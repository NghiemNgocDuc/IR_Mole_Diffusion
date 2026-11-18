"""Independent, fixed-coordinate DFT checks; does not import the running pilot.

Each task is EnGrad only. No optimization, coordinate repair, or model changes.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

REPORT_SHA = '7eb6f5d2bba6fa3c013e67a9f29902d811349b826a34bf99ed2719d37f2c6439'
SELECTION = (1, 7, 13)
METHOD = 'B3LYP D4 def2-SVP TightSCF EnGrad'
BOHR_ANGSTROM = 0.529177210544
HARTREE_KJ_MOL = 2625.49963948


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def xyz_text(numbers, xyz):
    xyz = np.asarray(xyz, float)
    if xyz.shape != (len(numbers), 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid coordinates')
    table = Chem.GetPeriodicTable()
    return f'{len(numbers)}\nOriginal coordinates in angstrom; no relaxation\n'+''.join(
        f'{table.GetElementSymbol(int(z))} {x:.12f} {y:.12f} {v:.12f}\n'
        for z, (x, y, v) in zip(numbers, xyz))


def graph_key(mol):
    mol = Chem.AddHs(Chem.Mol(mol))
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True, allHsExplicit=True)


def topology(numbers, bonds):
    bonds = np.asarray(bonds)
    if bonds.shape != (len(numbers), len(numbers)) or not np.array_equal(bonds, bonds.T) or np.any(np.diag(bonds)):
        raise ValueError('Invalid declared topology')
    mol = Chem.RWMol()
    for number in numbers:
        atom = Chem.Atom(int(number)); atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    codes = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for i in range(len(numbers)):
        for j in range(i):
            if bonds[i, j]:
                if bonds[i, j] not in codes:
                    raise ValueError('Unsupported bond code')
                mol.AddBond(i, j, codes[bonds[i, j]])
    mol = mol.GetMol(); Chem.SanitizeMol(mol)
    return mol


def reference_mapping(numbers, bonds, ref_numbers, ref_bonds):
    target, reference = topology(numbers, bonds), topology(ref_numbers, ref_bonds)
    if graph_key(target) != graph_key(reference):
        raise ValueError('Reference graph differs; no same-molecule energy comparison')
    mapping = reference.GetSubstructMatch(target, useChirality=False)
    if len(mapping) != len(numbers) or len(set(mapping)) != len(numbers):
        raise ValueError('Reference atom mapping incomplete')
    return np.asarray(mapping, int)


def geometry(numbers, bonds, xyz, reference=None):
    xyz = np.asarray(xyz, float)
    if xyz.shape != (len(numbers), 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid geometry')
    bonds = np.asarray(bonds)
    topology(numbers, bonds)
    distances = np.linalg.norm(xyz[:, None]-xyz[None, :], axis=-1)
    neighbors = [set(np.flatnonzero(bonds[i]).tolist()) for i in range(len(numbers))]
    lengths, angles, contacts = [], [], []
    table = Chem.GetPeriodicTable()
    for i in range(len(numbers)):
        for j in range(i):
            if bonds[i,j]:
                entry = {'atoms': [j,i], 'order': int(bonds[i,j]), 'angstrom': float(distances[i,j])}
                if reference is not None:
                    entry['difference_from_reference_angstrom'] = float(distances[i,j]-np.linalg.norm(reference[i]-reference[j]))
                lengths.append(entry)
            elif not neighbors[i].intersection(neighbors[j]):
                radii = table.GetRvdw(int(numbers[i]))+table.GetRvdw(int(numbers[j]))
                contacts.append({'atoms': [j,i], 'angstrom': float(distances[i,j]),
                                 'distance_over_vdw_sum': float(distances[i,j]/radii)})
        for j,k in itertools.combinations(sorted(neighbors[i]), 2):
            def angle(x):
                a,b = x[j]-x[i], x[k]-x[i]
                denominator = np.linalg.norm(a)*np.linalg.norm(b)
                return None if denominator == 0 else float(np.degrees(np.arccos(np.clip(a@b/denominator,-1,1))))
            entry = {'atoms': [j,i,k], 'degrees': angle(xyz)}
            if reference is not None:
                entry['reference_degrees'] = angle(reference)
            angles.append(entry)
    result = {'bonds': lengths, 'angles': angles,
              'nonbonded_contacts_excluding_1_2_and_1_3': sorted(contacts, key=lambda c:c['distance_over_vdw_sum']),
              'exact_atom_overlap': bool(np.any(distances[np.triu_indices(len(numbers),1)] == 0)),
              'note': 'Descriptive geometry only. vdW overlap is not a universal failure criterion. Reference mapping is one graph isomorphism, not symmetry-minimized.'}
    try:
        inferred = Chem.MolFromXYZBlock(xyz_text(numbers, xyz))
        rdDetermineBonds.DetermineBonds(inferred, charge=0)
        result['coordinate_inferred_achiral_graph_matches'] = graph_key(inferred) == graph_key(topology(numbers,bonds))
    except Exception as exc:
        result.update(coordinate_inferred_achiral_graph_matches=False, bond_perception_error=str(exc))
    return result


def prepare(source, data_path, output):
    if sha(source/'report.json') != REPORT_SHA:
        raise ValueError('Unexpected benchmark report')
    report = json.loads((source/'report.json').read_text())
    if sha(data_path) != report['selection']['data_sha256']:
        raise ValueError('Reference dataset hash mismatch')
    with np.load(data_path, allow_pickle=False) as data:
        indices = [report['inputs'][i]['row'] for i in SELECTION]
        references = {k: data[k][indices] for k in ('coords','atom_types','bond_types','num_atoms','molecule_ids')}
    plan = {'status': 'preparing', 'method': METHOD, 'charge': 0, 'multiplicity': 1,
            'environment': 'isolated_vacuum', 'geometry_optimization': False,
            'source_report_sha256': REPORT_SHA, 'dataset_sha256': report['selection']['data_sha256'],
            'tasks': [], 'physical_correctness_validated': False,
            'scope': 'Three selected training-set examples, seed/conformer 0. Reference coordinates are dataset controls, not assumed DFT minima or experimental truth.'}
    for ref_index, index in enumerate(SELECTION):
        row = report['inputs'][index]
        graph = next(g for g in row['graphs'] if g['graph_index'] == 0)
        numbers, bonds = graph['parameters']['atom_numbers'], np.array(graph['parameters']['bond_types'])
        if graph_key(topology(numbers,bonds)) != graph_key(Chem.MolFromSmiles(graph['smiles'])):
            raise ValueError('Declared graph mismatch')
        count = int(references['num_atoms'][ref_index])
        ref_numbers = references['atom_types'][ref_index,:count]
        ref_bonds = references['bond_types'][ref_index,:count,:count]
        if str(references['molecule_ids'][ref_index]) != row['molecule_id']:
            raise ValueError('Reference identity/row mismatch')
        mapping = reference_mapping(numbers,bonds,ref_numbers,ref_bonds)
        ref_xyz = references['coords'][ref_index,:count][mapping].astype(float)
        prefix = source/f'input_{index:03d}'/'graph_0'
        cases = []
        for arm in ('unguided', 'opls_guided'):
            path = prefix/f'{arm}_0.npz'
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved['atom_numbers'],numbers) or not np.array_equal(saved['bond_types'],bonds):
                    raise ValueError('Saved topology differs from declared graph')
                for stage in ('raw','relaxed'):
                    cases.append((arm, stage, saved[stage+'_angstrom'].copy(), sha(path)))
        cases.append(('reference','dataset',ref_xyz,None))
        for arm,stage,xyz,source_hash in cases:
            task_id = len(plan['tasks']); folder = output/f'task_{task_id:02d}'; folder.mkdir()
            (folder/'start.xyz').write_text(xyz_text(numbers,xyz))
            (folder/'job.inp').write_text(f'! {METHOD}\n%maxcore 2500\n* xyzfile 0 1 start.xyz\n')
            save(folder/'geometry.json',geometry(numbers,bonds,xyz,ref_xyz))
            task = {'id':task_id, 'directory':folder.name,'input_index':index,'row':row['row'],
                    'smiles':graph['smiles'],'graph_index':0,'conformer':0,'arm':arm,'stage':stage,
                    'atom_numbers':numbers,'source_npz_sha256':source_hash,
                    'reference_model_to_dataset':mapping.tolist(),
                    'input_sha256':sha(folder/'job.inp'),'xyz_sha256':sha(folder/'start.xyz')}
            plan['tasks'].append(task)
        save(output/'plan.json',plan)
    plan['status']='prepared'; save(output/'plan.json',plan)
    print(f'Prepared {len(plan["tasks"])} fixed-coordinate DFT tasks',flush=True)
    return plan


def parse_engrad(text, expected_numbers, expected_xyz):
    lines = [s.split('#',1)[0].strip() for s in text.splitlines()]
    fields = ' '.join(s for s in lines if s).replace('D','E').replace('d','e').split()
    if not fields:
        raise ValueError('Empty gradient file')
    n = int(fields[0])
    # Native ORCA .engrad: N, energy, 3N gradient components, N rows Z/x/y/z.
    if n != len(expected_numbers) or len(fields) != 2+7*n:
        raise ValueError('Incomplete/unsupported gradient file or atom count mismatch')
    values = np.array([float(v) for v in fields[1:]])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite energy/gradient/coordinates')
    energy = float(values[0]); gradient = values[1:1+3*n].reshape(n,3)
    atoms_xyz = values[1+3*n:].reshape(n,4)
    if not np.array_equal(atoms_xyz[:,0],expected_numbers):
        raise ValueError('Gradient atom order differs from submitted geometry')
    evaluated = atoms_xyz[:,1:]*BOHR_ANGSTROM
    if not np.allclose(evaluated,expected_xyz,atol=1e-6,rtol=0):
        raise ValueError('Gradient evaluated at different coordinates')
    force = -gradient*HARTREE_KJ_MOL/(BOHR_ANGSTROM/10)
    return {'energy_hartree':energy, 'gradient_hartree_bohr':gradient.tolist(),
            'force_kj_mol_nm':force.tolist(),
            'force_component_rms_kj_mol_nm':float(np.sqrt(np.mean(force**2))),
            'force_max_atom_norm_kj_mol_nm':float(np.linalg.norm(force,axis=1).max()),
            'net_force_norm_kj_mol_nm':float(np.linalg.norm(force.sum(0))),
            'coordinate_max_error_angstrom':float(np.max(np.abs(evaluated-expected_xyz))),
            'coordinates_unchanged':True}


def run_task(output,index,orca):
    plan = json.loads((output/'plan.json').read_text()); task = plan['tasks'][index]
    folder = output/task['directory']
    # Existing results are never overwritten by an accidental duplicate submission.
    if (folder/'job.out').exists() or (folder/'result.json').exists():
        raise RuntimeError('Existing task output; refusing to rerun/overwrite')
    result = {'task':task,'status':'failed','physical_correctness_validated':False}
    try:
        if sha(folder/'job.inp') != task['input_sha256'] or sha(folder/'start.xyz') != task['xyz_sha256']:
            raise ValueError('Submitted input changed')
        executable = shutil.which(orca)
        if not executable:
            raise ValueError('ORCA executable unavailable')
        with (folder/'job.out').open('w') as log:
            process = subprocess.run([str(Path(executable).resolve()),'job.inp'],cwd=folder,
                stdout=log,stderr=subprocess.STDOUT,timeout=3300)
        text = (folder/'job.out').read_text(errors='replace')
        if process.returncode or 'ORCA TERMINATED NORMALLY' not in text:
            raise ValueError('ORCA did not terminate normally')
        if 'SCF NOT CONVERGED' in text.upper() or 'SCF CONVERGENCE FAILURE' in text.upper():
            raise ValueError('SCF failed to converge')
        versions = re.findall(r'Program Version\s+(\S+)',text)
        if not versions or not versions[-1].startswith('6.1.'):
            raise ValueError('Expected ORCA 6.1.x for this paired pilot')
        molecule = Chem.MolFromXYZBlock((folder/'start.xyz').read_text())
        if molecule is None:
            raise ValueError('Cannot read original XYZ')
        result.update(parse_engrad((folder/'job.engrad').read_text(),task['atom_numbers'],molecule.GetConformer().GetPositions()))
        if sha(folder/'start.xyz') != task['xyz_sha256']:
            raise ValueError('Original coordinates changed during calculation')
        result.update(status='evaluated',orca_version=versions[-1],orca_executable=str(Path(executable).resolve()),
                      output_sha256=sha(folder/'job.out'),gradient_sha256=sha(folder/'job.engrad'))
        vectors = np.column_stack((np.arange(len(task['atom_numbers'])),task['atom_numbers'],result['force_kj_mol_nm']))
        np.savetxt(folder/'forces.csv',vectors,delimiter=',',header='atom_index,atomic_number,Fx_kJ_mol_nm,Fy_kJ_mol_nm,Fz_kJ_mol_nm',comments='')
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',coordinates_unchanged=None)
    save(folder/'result.json',result)
    print(index,result['status'],flush=True)
    return result


def paired_summary(results):
    comparisons = []
    for index in SELECTION:
        rows = [r for r in results if r['task']['input_index']==index and r['status']=='evaluated']
        references = [r for r in rows if r['task']['arm']=='reference']
        for r in rows:
            if r['task']['arm']=='reference':
                continue
            if references and r['orca_version']==references[0]['orca_version']:
                reference=references[0]
                comparisons.append({'task_id':r['task']['id'],'reference_task_id':reference['task']['id'],
                    'energy_minus_dataset_reference_kj_mol':(r['energy_hartree']-reference['energy_hartree'])*HARTREE_KJ_MOL,
                    'force_rms_minus_reference_kj_mol_nm':r['force_component_rms_kj_mol_nm']-reference['force_component_rms_kj_mol_nm']})
    return comparisons


def optimization_comparisons(output, results, optimized):
    """Join completed optimization results only after method/input provenance checks."""
    if optimized is None or not (optimized/'plan.json').is_file():
        return {'status':'not_available','rows':[]}
    old_plan=json.loads((optimized/'plan.json').read_text())
    if old_plan.get('method') != 'B3LYP D4 def2-SVP TightSCF TightOpt Opt Freq' or old_plan.get('source_report_sha256') != REPORT_SHA:
        return {'status':'incompatible_method_or_source','rows':[]}
    joined=[]
    for result in results:
        task=result['task']
        if task['arm']=='reference' or result['status']!='evaluated':
            continue
        match=[t for t in old_plan['tasks'] if all(t[k]==task[k] for k in
               ('input_index','graph_index','conformer','arm','stage'))]
        entry={'task_id':task['id'],'status':'pending_or_unavailable'}
        joined.append(entry)
        if len(match)!=1:
            continue
        old=match[0]; folder=optimized/old['directory']
        try:
            previous=json.loads((folder/'result.json').read_text())
            if (previous['task'] != old or previous.get('orca_version') != result['orca_version'] or
                    not previous.get('normal_termination') or not previous.get('optimization_converged') or
                    not previous.get('achiral_graph_preserved') or previous.get('returncode') != 0):
                entry['status']='optimization_failed_or_not_comparable'; continue
            if old['source_npz_sha256'] != task['source_npz_sha256']:
                raise ValueError('Source coordinate archive mismatch')
            for name,key in [('job.inp','input_sha256'),('start.xyz','xyz_sha256')]:
                if sha(folder/name)!=old[key]:
                    raise ValueError('Optimization input changed')
            if sha(folder/'job.out')!=previous['orca_output_sha256'] or sha(folder/'job.xyz')!=previous['final_xyz_sha256']:
                raise ValueError('Optimization output changed')
            original=Chem.MolFromXYZBlock((folder/'start.xyz').read_text())
            current=Chem.MolFromXYZBlock((output/task['directory']/'start.xyz').read_text())
            if ([a.GetAtomicNum() for a in original.GetAtoms()] != task['atom_numbers'] or
                    not np.allclose(original.GetConformer().GetPositions(),current.GetConformer().GetPositions(),atol=1e-10,rtol=0)):
                raise ValueError('Optimization started from different coordinates')
            final_energy=previous['final_electronic_energy_hartree']
            if final_energy is None or not np.isfinite(final_energy):
                raise ValueError('Final energy unavailable')
            entry.update(status='compared',
                energy_drop_to_optimized_geometry_kj_mol=(result['energy_hartree']-final_energy)*HARTREE_KJ_MOL,
                aligned_repair_rmsd_angstrom=previous['aligned_repair_rmsd_angstrom'],
                local_minimum_supported=previous.get('local_minimum_supported_for_intended_graph',False),
                optimization_result_sha256=sha(folder/'result.json'))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            entry['note']=str(exc)
    return {'status':'joined_available_results','source_directory':str(optimized),'rows':joined}


def collect(output, optimized=None):
    plan=json.loads((output/'plan.json').read_text()); results=[]
    for task in plan['tasks']:
        folder=output/task['directory']; path=folder/'result.json'
        if sha(folder/'job.inp') != task['input_sha256'] or sha(folder/'start.xyz') != task['xyz_sha256']:
            raise ValueError('Submitted input changed after preparation')
        if path.exists():
            r=json.loads(path.read_text())
            if r['task'] != task:
                raise ValueError('Result belongs to a different task')
            if r['status']=='evaluated' and (sha(folder/'job.engrad')!=r['gradient_sha256'] or sha(folder/'job.out')!=r['output_sha256']):
                raise ValueError('Output changed after assessment')
        else:
            r={'task':task,'status':'missing_result'}
        results.append(r)
    comparison=paired_summary(results)
    repair=optimization_comparisons(output,results,optimized)
    summary={'plan':plan,'results':results,'reference_comparisons':comparison,
             'optimization_comparisons':repair,
             'physical_correctness_validated':False,
             'note':'Energy differences apply only within the same molecular graph and DFT setup. Dataset references are not guaranteed minima. Small forces alone do not distinguish minima from saddle points. Selected training examples do not establish generalization.'}
    save(output/'summary.json',summary)
    comparisons={r['task_id']:r for r in comparison}
    lines=['# DFT at unchanged generated coordinates','',
           'No coordinate optimization in these jobs. All forces in kJ/mol/nm.',
           'Energy differences are relative to the corresponding dataset geometry, not a proven minimum.','',
           '| Task | Input | Arm | Coordinates | Status | Force RMS (components) | Max atomic force | Energy minus reference (kJ/mol) |',
           '|---|---|---|---|---|---|---|---|']
    for r in results:
        t=r['task']; c=comparisons.get(t['id'],{})
        lines.append(f"| {t['id']} | {t['input_index']} | {t['arm']} | {t['stage']} | {r['status']} | {r.get('force_component_rms_kj_mol_nm','—')} | {r.get('force_max_atom_norm_kj_mol_nm','—')} | {c.get('energy_minus_dataset_reference_kj_mol','—')} |")
    lines += ['', '## Repair measured by the separate optimization jobs', '',
              '| Task | Comparison status | DFT energy drop (kJ/mol) | Repair RMSD (A) | Local minimum supported |',
              '|---|---|---|---|---|']
    for row in repair['rows']:
        lines.append(f"| {row['task_id']} | {row['status']} | {row.get('energy_drop_to_optimized_geometry_kj_mol','—')} | {row.get('aligned_repair_rmsd_angstrom','—')} | {row.get('local_minimum_supported','—')} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    destination=output/'raw_geometry_results.zip'
    # Only include relevant inputs/results; ORCA scratch/GBW files stay on Unity.
    with zipfile.ZipFile(destination.with_suffix('.zip.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.md','.inp','.xyz','.out','.engrad','.csv','.log'):
                z.write(path,path.relative_to(output))
    destination.with_suffix('.zip.tmp').replace(destination)
    print('Download:',destination,flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','collect'))
    parser.add_argument('--source',type=Path,default=Path('outputs/opls_benchmark_1820024'))
    parser.add_argument('--data',type=Path,default=Path('data/processed/squirl.npz'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int)
    parser.add_argument('--orca',default='orca')
    parser.add_argument('--optimized',type=Path,default=Path('outputs/physical_validation_1849289'))
    args=parser.parse_args(); output=args.output.resolve()
    if args.action=='prepare':
        output.mkdir(parents=True,exist_ok=False)
        prepare(args.source.resolve(),args.data.resolve(),output)
        collect(output,args.optimized.resolve())
    elif args.action=='run':
        result=run_task(output,args.index,args.orca)
        return 0 if result['status']=='evaluated' else 1
    else:
        collect(output,args.optimized.resolve())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
