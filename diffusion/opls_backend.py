"""Local, cached LigParGen parameterization and a persistent OpenMM context.

Initial scope: connected neutral closed-shell H/C/N/O/F graphs, <=29 atoms,
no isotope labels or explicitly specified stereo. Noisy xyz never sets topology.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np
from diffusion.opls_bridge import MappedEnergy, bounded_correction
from diffusion.preprocessing import bond_array

APPIMAGE_SHA256 = 'f9f2ae20f78bc6d3af720411f66e7d0f512c8f196ff702545b075f18fafa22ae'
PROTOCOL = 'neutral-achiral-opls-pilot-v1'
TEMPLATE_SEEDS = (20261115, 20261116, 20261117)


def parameterize_templates(model, appimage, work):
    """Retry only embedding/coordinate-precheck failures; never bypass checks.

    Three 90-second subprocess budgets keep preparation below the worker's
    360-second timeout for these small molecules. Failed attempts are retained.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom
    attempts = []
    for index, seed in enumerate(TEMPLATE_SEEDS):
        folder = work / f'template_attempt_{index}'
        folder.mkdir()
        template = Chem.Mol(model)
        template.RemoveAllConformers()
        options = rdDistGeom.ETKDGv3()
        options.randomSeed, options.numThreads = seed, 1
        options.useRandomCoords = index > 0
        record = {'seed': seed, 'random_coordinates': index > 0}
        attempts.append(record)
        if rdDistGeom.EmbedMolecule(template, options) != 0:
            record['status'] = 'embedding_failed'
            (work/'template_attempts.json').write_text(json.dumps(attempts, indent=2))
            continue
        Chem.MolToMolFile(template, str(folder/'template.mol'))
        command = [str(appimage), 'ligpargen', '-i', str(folder/'template.mol'),
                   '-n', 'MOL', '-r', 'MOL', '-c', '0', '-cgen', 'CM1A-LBCC',
                   '-p', str(folder/'export')]
        record['status'] = 'parameterization_started'
        (work/'template_attempts.json').write_text(json.dumps(attempts, indent=2))
        with (folder/'parameterization.log').open('w') as log:
            result = subprocess.run(command, cwd=folder,
                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                stdout=log, stderr=subprocess.STDOUT, timeout=90)
        message = (folder/'parameterization.log').read_text(errors='replace')
        # LigParGen may return zero even when its precheck rejects coordinates.
        coordinate_rejection = ('unbonded atoms closer than a bond length' in message)
        if coordinate_rejection:
            record['status'] = 'coordinate_precheck_rejected'
            (work/'template_attempts.json').write_text(json.dumps(attempts, indent=2))
            continue
        if result.returncode:
            record['status'] = 'engine_failed'
            record['returncode'] = result.returncode
            (work/'template_attempts.json').write_text(json.dumps(attempts, indent=2))
            raise RuntimeError(f'LigParGen failed; see {folder}/parameterization.log')
        mapping = map_export(model, folder/'export/MOL.openmm.pdb')
        for name in ('template.mol', 'parameterization.log'):
            shutil.copy2(folder/name, work/name)
        shutil.copytree(folder/'export', work/'export')
        record['status'] = 'validated_export'
        (work/'template_attempts.json').write_text(json.dumps(attempts, indent=2))
        return mapping, attempts
    raise RuntimeError(f'All bounded parameter-template attempts failed; see {work}/template_attempts.json')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def declared_graph(smiles):
    from rdkit import Chem
    graph = Chem.MolFromSmiles(smiles)
    if graph is None or len(Chem.GetMolFrags(graph)) != 1:
        raise ValueError('Unsupported OPLS graph: requires one connected molecule')
    for atom in graph.GetAtoms():
        if (atom.GetAtomicNum() not in (1, 6, 7, 8, 9) or atom.GetFormalCharge() or
                atom.GetIsotope() or atom.GetNumRadicalElectrons() or
                atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            raise ValueError('Unsupported OPLS graph: element/charge/isotope/radical/specified stereo')
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in graph.GetBonds()):
        raise ValueError('Specified bond stereo is outside this first OPLS pilot')
    model = Chem.AddHs(graph)
    if not 3 <= model.GetNumAtoms() <= 29:
        raise ValueError('OPLS pilot requires 3 to 29 explicit atoms')
    return model


def map_export(model, path):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    exported = Chem.MolFromPDBFile(str(path), removeHs=False, proximityBonding=False)
    if exported is None:
        raise ValueError('Cannot read exported PDB')
    rdDetermineBonds.DetermineBondOrders(exported, charge=0)
    Chem.RemoveStereochemistry(exported)  # This pilot explicitly measures achiral graphs.
    mapping = exported.GetSubstructMatch(model, useChirality=True)
    if (len(mapping) != model.GetNumAtoms() or exported.GetNumAtoms() != model.GetNumAtoms()
            or exported.GetNumBonds() != model.GetNumBonds()):
        raise ValueError('Exported graph does not match declared graph; no coordinates were sampled')
    return list(mapping)


def prepare(smiles, appimage, cache):
    from rdkit import Chem, rdBase
    model = declared_graph(smiles)
    appimage, cache = Path(appimage).resolve(), Path(cache).resolve()
    apphash = sha256(appimage)
    if apphash != APPIMAGE_SHA256:
        raise ValueError('LigParGen executable differs from the tested 3.0.1 AppImage')
    settings = {'protocol': PROTOCOL, 'ordered_smiles': smiles, 'charge_model': 'CM1A-LBCC',
                'template_method': 'RDKit ETKDGv3 bounded coordinate retries', 'template_seeds': list(TEMPLATE_SEEDS),
                'rdkit_version': rdBase.rdkitVersion, 'appimage_sha256': apphash,
                'backend_sha256': sha256(__file__)}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / key
    if not final.exists():
        lock = cache / (key + '.lock')
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise RuntimeError('OPLS cache entry is being prepared; retry later (stale locks require inspection)') from exc
        try:
            # Retain failed work directories for diagnosis. Only rename a complete entry.
            work = Path(tempfile.mkdtemp(prefix=key + '.work-', dir=cache))
            mapping, attempts = parameterize_templates(model, appimage, work)
            files = ['template.mol', 'export/MOL.openmm.py', 'export/MOL.openmm.xml', 'export/MOL.openmm.pdb']
            manifest = {**settings, 'template_attempts': attempts, 'model_to_engine': mapping,
                        'atom_numbers': [a.GetAtomicNum() for a in model.GetAtoms()],
                        'bond_types': bond_array(model).tolist(),
                        'file_sha256': {name: sha256(work / name) for name in files}}
            (work / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            work.rename(final)
        finally:
            lock.rmdir()
    manifest = json.loads((final / 'manifest.json').read_text())
    if any(manifest.get(k) != v for k, v in settings.items()):
        raise ValueError('Cached parameter settings mismatch')
    expected_files = {'template.mol', 'export/MOL.openmm.py', 'export/MOL.openmm.xml', 'export/MOL.openmm.pdb'}
    if set(manifest['file_sha256']) != expected_files:
        raise ValueError('Incomplete parameter-cache manifest')
    for name, digest in manifest['file_sha256'].items():
        if sha256(final / name) != digest:
            raise ValueError('Parameter-cache hash mismatch: ' + name)
    if map_export(model, final / 'export/MOL.openmm.pdb') != manifest['model_to_engine']:
        raise ValueError('Cached atom mapping mismatch')
    if (manifest['atom_numbers'] != [a.GetAtomicNum() for a in model.GetAtoms()] or
            manifest['bond_types'] != bond_array(model).tolist()):
        raise ValueError('Cached declared topology mismatch')
    return final, manifest


class OPLSBackend:
    def __init__(self, smiles, appimage, cache):
        import openmm
        from openmm import unit
        self.unit, self.openmm = unit, openmm
        folder, self.metadata = prepare(smiles, appimage, cache)
        self.metadata = {**self.metadata, 'cache_directory': str(folder), 'openmm_version': openmm.__version__,
                         'platform': 'Reference', 'environment': 'isolated_vacuum', 'physical_correctness_validated': False}
        # Execute only this locally generated, hash-checked LigParGen loader. XML
        # alone loses the geometric LJ mixing rule and BOSS improper conventions.
        spec = importlib.util.spec_from_file_location('opls_local_export', folder / 'export/MOL.openmm.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pdb, system = module.build()
        count = len(self.metadata['atom_numbers'])
        if (system.getNumParticles() != count or system.getNumConstraints() or
                system.usesPeriodicBoundaryConditions() or any(system.isVirtualSite(i) for i in range(count))):
            raise ValueError('Unexpected OpenMM system topology')
        charges = [sum(float(f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge))
                       for i in range(count)) for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if len(charges) != 1 or abs(charges[0]) > 1e-4:
            raise ValueError('Exported partial charges do not sum to declared neutral charge')
        self.integrator = openmm.VerletIntegrator(.001)
        self.context = openmm.Context(system, self.integrator, openmm.Platform.getPlatformByName('Reference'))
        self.adapter = MappedEnergy(self.evaluate_nm, self.metadata['model_to_engine'])
        self.adapter.evaluate(np.asarray(pdb.positions.value_in_unit(unit.nanometer))[self.adapter.mapping] * 10)

    def evaluate_nm(self, xyz):
        u = self.unit
        self.context.setPositions(xyz * u.nanometer)
        state = self.context.getState(getEnergy=True, getForces=True)
        return (float(state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)),
                np.asarray(state.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole/u.nanometer)))

    def evaluate(self, xyz):
        energy, force = self.adapter.evaluate(xyz)
        return {'energy_kj_mol': energy, 'force_rms_kj_mol_nm': float(np.sqrt(np.mean((force*10)**2)))}

    def assess(self, xyz):
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from diffusion.graph_identity import same_achiral_graph
        result = self.evaluate(xyz)
        numbers = self.metadata['atom_numbers']
        block = str(len(numbers)) + '\n\n' + '\n'.join(
            f'{Chem.GetPeriodicTable().GetElementSymbol(z)} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}'
            for z, p in zip(numbers, xyz))
        try:
            perceived = Chem.MolFromXYZBlock(block)
            rdDetermineBonds.DetermineBonds(perceived, charge=0)
            inferred = Chem.MolToSmiles(Chem.RemoveHs(perceived), isomericSmiles=False)
            result.update(coordinate_inferred_smiles=inferred,
                          coordinate_inferred_graph_matches=same_achiral_graph(perceived, self.metadata['ordered_smiles']),
                          graph_comparison='explicit_h_achiral_v1')
        except Exception as exc:
            result.update(coordinate_inferred_graph_matches=False, bond_perception_error=str(exc))
        return result

    def correct(self, xyz, max_step):
        corrected, record = bounded_correction(self.adapter, xyz, max_step_angstrom=max_step, preserve_centroid=True)
        return {'coordinates': corrected.tolist(), 'record': record}

    def relax(self, xyz):
        before = self.assess(xyz)  # Also sets the context using verified units/order.
        self.openmm.LocalEnergyMinimizer.minimize(self.context, tolerance=1., maxIterations=500)
        engine = np.asarray(self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(self.unit.nanometer))
        candidate = engine[self.adapter.mapping] * 10
        candidate += np.asarray(xyz).mean(0) - candidate.mean(0)
        after = self.assess(candidate)
        accepted = after['energy_kj_mol'] <= before['energy_kj_mol'] + 1e-7
        return {'coordinates': (candidate if accepted else np.asarray(xyz)).tolist(), 'before': before,
                'after': after, 'accepted': accepted, 'force_converged': bool(accepted and after['force_rms_kj_mol_nm'] <= 1.),
                'scope': 'Bounded local minimization; no physical identity or global minimum claim'}
