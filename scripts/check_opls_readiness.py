"""One collected report for force parity, graph mapping, and the coordinate bridge."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.opls_bridge import MappedEnergy, bounded_correction
from compare_opls_engines import compare, command

MOLECULES = [('ETH', 'CCO', 'outputs/opls_ethanol_1788228'),
             ('ACE', 'CC(=O)C', 'outputs/opls_extended_1791099/ACE'),
             ('BEN', 'c1ccccc1', 'outputs/opls_extended_1791099/BEN')]


def map_graph(smiles, pdb_text):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    # Match the same AddHs(MolFromSmiles()) ordering used by predict_structure.py.
    graph = Chem.MolFromSmiles(smiles)
    if graph is None: raise ValueError('Invalid declared graph')
    model = Chem.AddHs(graph)
    exported = Chem.MolFromPDBBlock(pdb_text, removeHs=False, proximityBonding=False)
    if exported is None: raise ValueError('Invalid exported topology')
    # These three neutral closed-shell fixtures have explicit hydrogens. Recover
    # bond orders from their declared PDB connectivity, without proximity bonds.
    rdDetermineBonds.DetermineBondOrders(exported, charge=0)
    if model.GetNumAtoms() != exported.GetNumAtoms() or model.GetNumBonds() != exported.GetNumBonds():
        raise ValueError('Topology size differs')
    mapping = exported.GetSubstructMatch(model, useChirality=True)
    if len(mapping) != model.GetNumAtoms(): raise ValueError('No complete graph mapping')
    # Symmetric atoms can have multiple mappings: retain this explicit mapping for all uses.
    return list(mapping)


def bridge_check(source, name, smiles, output):
    import openmm
    from openmm import unit
    source = source.resolve()
    mapping = map_graph(smiles, (source / f'{name}.openmm.pdb').read_text())
    loader = source / f'{name}.openmm.py'
    spec = importlib.util.spec_from_file_location('bridge_' + name, loader)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pdb, system = module.build()
    integrator = openmm.VerletIntegrator(.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))

    def evaluate_nm(x):
        context.setPositions(x * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)), np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))

    xyz = np.asarray(pdb.positions.value_in_unit(unit.nanometer))
    xyz += np.random.default_rng(20261115).normal(0, .002, xyz.shape)
    adapter = MappedEnergy(evaluate_nm, mapping)
    model_xyz = xyz[mapping] * 10
    direct_e, direct_f = evaluate_nm(xyz)
    bridge_e, bridge_f = adapter.evaluate(model_xyz)
    roundtrip = bool(np.isclose(direct_e, bridge_e, atol=1e-7, rtol=0) and np.allclose(bridge_f, direct_f[mapping]/10, atol=1e-6, rtol=1e-6))
    # Central differences in MODEL Angstrom coordinates test mapping and the factor of 10.
    numeric = np.empty_like(model_xyz); h=1e-4
    for i in range(len(model_xyz)):
        for axis in range(3):
            plus=model_xyz.copy(); minus=model_xyz.copy()
            plus[i,axis]+=h; minus[i,axis]-=h
            numeric[i,axis]=-(adapter.evaluate(plus)[0]-adapter.evaluate(minus)[0])/(2*h)
    gradient_ok = bool(np.allclose(numeric,bridge_f,atol=.02,rtol=1e-3))
    current=model_xyz.copy(); trace=[]
    for _ in range(10):
        current, record=bounded_correction(adapter,current)
        trace.append(record)
    # A deliberately permuted model order must give the same physical energies and forces.
    permutation=np.random.default_rng(18).permutation(len(mapping))
    permuted=MappedEnergy(evaluate_nm,np.asarray(mapping)[permutation])
    pe,pf=permuted.evaluate(model_xyz[permutation])
    permutation_ok=bool(np.isclose(pe,bridge_e,atol=1e-7,rtol=0) and np.allclose(pf,bridge_f[permutation],atol=1e-6,rtol=1e-6))
    final_e,_=adapter.evaluate(current)
    correction_ok=bool(final_e < bridge_e and any(t['accepted'] for t in trace)
                       and all(t.get('max_displacement_angstrom',0)<=.020000001 for t in trace))
    np.savez(output / f'{name}_bridge_coordinates.npz', before_angstrom=model_xyz,after_angstrom=current,model_to_engine=mapping)
    return {'passed': roundtrip and gradient_ok and permutation_ok and correction_ok,
            'model_to_engine':mapping,'roundtrip_passed':roundtrip,'angstrom_gradient_passed':gradient_ok,
            'permuted_order_passed':permutation_ok,'bounded_correction_passed':correction_ok,
            'initial_energy_kj_mol':bridge_e,'final_energy_kj_mol':final_e,'trace':trace,
            'scope':'Known graph/template test in predictor atom convention; no checkpoint inference or diffusion sampling performed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gmx',required=True); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    report={'molecules':{},'physical_correctness_validated':False,'actual_generator_inference_tested':False,
            'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__).resolve(),root/'scripts/compare_opls_engines.py',root/'diffusion/opls_bridge.py']},
            'coordinate_checkpoint_inventory':[str(p.relative_to(root)) for p in (root/'ckpt').rglob('*')
                if p.is_file() and p.suffix in ('.ckpt','.pt','.pth')]}
    gmx=shutil.which(args.gmx)
    try:
        if not gmx: raise ValueError('No GROMACS executable')
        command([gmx,'--version'],out,'gromacs-version.txt')
    except Exception as exc: report['engine_setup_error']=str(exc)
    for name,smiles,source in MOLECULES:
        stages={};report['molecules'][name]=stages
        for stage,action in [('engine_comparison',lambda:compare(Path(source),name,out,gmx,forces=True)),
                             ('coordinate_bridge',lambda:bridge_check(Path(source),name,smiles,out))]:
            try:
                if stage=='engine_comparison' and not gmx: raise ValueError('No GROMACS executable')
                stages[stage]=action()
            except Exception as exc: stages[stage]={'error':f'{type(exc).__name__}: {exc}','passed':False}
    def passed(stages):
        engine=stages['engine_comparison']; bridge=stages['coordinate_bridge']
        return ('geometries' in engine and all(g['passed'] for g in engine['geometries'].values()) and bridge.get('passed',False))
    report['passed']=all(passed(s) for s in report['molecules'].values()) and 'engine_setup_error' not in report
    report['status']='ready_for_controlled_pipeline_integration' if report['passed'] else 'checks_need_review'
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with zipfile.ZipFile(out/'readiness_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='readiness_results.zip': archive.write(p,p.relative_to(out))
    print(json.dumps(report,indent=2,allow_nan=False));print('Download:',out/'readiness_results.zip')
    return 0 if report['passed'] else 1


if __name__=='__main__': sys.exit(main())
