"""IR -> predicted graphs -> preserved raw coordinates; OPLS is opt-in research.

Uses a training spectrum for engineering diagnosis, never reference graph/xyz.
Missing/legacy/ambiguous checkpoints produce one actionable inventory archive.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.model import EquivariantDiffusion3D_v4
from diffusion.spectrum import SpectrumStructureModel
from diffusion.opls_client import OPLSClient
from diffusion.opls_guidance import OPLSGuidance
from diffusion.preprocessing import bond_array
from evaluate_spectrum import valid_candidates


def checkpoint_inventory(roots):
    inventory = []
    paths = sorted({p.resolve() for root in roots if Path(root).exists()
                    for p in Path(root).rglob('*') if p.suffix in ('.ckpt', '.pt', '.pth') and p.is_file()})
    for path in paths:
        item = {'path': str(path), 'bytes': path.stat().st_size}
        try:
            # No arbitrary pickle execution or architecture migration. mmap reads
            # metadata without materializing all parameter storage in RAM.
            checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            metadata = checkpoint.get('run_metadata', {})
            item.update(format_version=checkpoint.get('format_version'), model_class=checkpoint.get('model_class'),
                        global_step=checkpoint.get('global_step', 0), run_metadata=metadata)
            item['eligible'] = bool(item['format_version'] == 2 and item['global_step'] > 0 and
                metadata.get('diagnostic') is False and metadata.get('spectrum_control') == 'none' and
                item['model_class'] in ('SpectrumStructureModel', 'EquivariantDiffusion3D_v4'))
            item['reason'] = 'compatible metadata' if item['eligible'] else 'legacy, untrained, diagnostic, control, or different model'
            del checkpoint
        except Exception as exc:
            item.update(eligible=False, reason=f'{type(exc).__name__}: {exc}')
        inventory.append(item)
    gc.collect()
    return inventory


def choose_checkpoint(inventory, model_class, explicit=None, metadata=None):
    candidates = [i for i in inventory if i.get('eligible') and i.get('model_class') == model_class]
    if metadata:
        candidates = [i for i in candidates if all(i['run_metadata'].get(k) == metadata.get(k)
                     for k in ('data_sha256', 'split_sha256', 'preprocessing'))]
    if explicit:
        candidates = [i for i in candidates if Path(i['path']) == Path(explicit).resolve()]
    if len(candidates) != 1:
        raise ValueError(f'Need one eligible {model_class} checkpoint; found {len(candidates)}. '
                         'See checkpoint_inventory.json. Specify a compatible path explicitly if several exist. '
                         'Legacy and diagnostic weights are not substituted.')
    return candidates[0]


def save_geometry(folder, stem, template, points, arm, stage):
    from rdkit import Chem
    path = folder/(stem+'_'+stage+'.sdf')
    molecule = Chem.Mol(template)
    molecule.RemoveAllConformers()
    conf = Chem.Conformer(template.GetNumAtoms())
    for i, p in enumerate(points):
        conf.SetAtomPosition(i, tuple(map(float, p)))
    molecule.AddConformer(conf)
    molecule.SetProp('arm', arm)
    molecule.SetProp('geometry_stage', stage)
    molecule.SetProp('physical_correctness_validated', 'false')
    with Chem.SDWriter(str(path)) as writer:
        writer.write(molecule)


def generate_attempt(args, model, spectra, template, folder, arm, conformer, seed, client=None):
    """Save the sampler output before any optional assessment can fail."""
    numbers = [a.GetAtomicNum() for a in template.GetAtoms()]
    bonds = bond_array(template)
    stem = f'{arm}_{conformer}'
    attempt = {'arm': arm, 'conformer': conformer, 'seed': seed,
               'generation_status': 'failed', 'assessment_status': 'not_requested',
               'physical_correctness_validated': False}
    started = time.monotonic()
    policy = None
    try:
        if arm == 'opls_guided':
            if not args.experimental_opls or client is None:
                raise ValueError('OPLS guidance requires explicit experimental mode and a backend')
            policy = OPLSGuidance([client], args.min_alpha, args.max_step)
        xyz, mask = model.sample(spectra,
            torch.tensor([numbers], dtype=torch.long, device=args.device),
            torch.tensor([len(numbers)], device=args.device),
            n_steps=min(args.steps, model.n_timesteps),
            graph={'bond_types': torch.tensor(bonds[None], dtype=torch.long, device=args.device)},
            physics_guidance=0., generator=torch.Generator(device=args.device).manual_seed(seed),
            opls_guidance=policy)
        raw = xyz[0].detach().cpu().numpy()
        if raw.shape != (len(numbers), 3) or not np.isfinite(raw).all():
            raise ValueError('Sampler returned invalid coordinates')
        path = folder/(stem+'_raw.npy')
        np.save(path, raw)
        np.savez_compressed(folder/(stem+'.npz'), raw_angstrom=raw,
                            atom_numbers=numbers, bond_types=bonds)
        save_geometry(folder, stem, template, raw, arm, 'raw')
        attempt.update(status='completed', generation_status='completed',
                       raw_coordinates_file=path.name, raw_coordinates_sha256=file_sha256(path),
                       primary_geometry_stage='raw')
    except Exception as exc:
        attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        attempt['seconds'] = time.monotonic()-started
        if policy is not None:
            attempt['guidance_trace'] = policy.trace
    return attempt


def assess_attempt(attempt, folder, template, client):
    """OPLS results are annotations; they never replace or reject raw output."""
    if attempt['generation_status'] != 'completed':
        return
    path = folder/attempt['raw_coordinates_file']
    try:
        if file_sha256(path) != attempt['raw_coordinates_sha256']:
            raise ValueError('Raw coordinates changed before assessment')
        raw = np.load(path, allow_pickle=False)
        attempt['raw'] = client.request('evaluate', coordinates=raw.tolist())
        relaxed = client.request('relax', coordinates=raw.tolist())
        after = np.asarray(relaxed.pop('coordinates'))
        if after.shape != raw.shape or not np.isfinite(after).all():
            raise ValueError('Invalid relaxed geometry')
        stem = f"{attempt['arm']}_{attempt['conformer']}"
        save_geometry(folder, stem, template, after, attempt['arm'], 'opls_relaxed')
        np.savez_compressed(folder/(stem+'.npz'), raw_angstrom=raw, relaxed_angstrom=after,
            atom_numbers=[a.GetAtomicNum() for a in template.GetAtoms()], bond_types=bond_array(template))
        attempt.update(relaxation=relaxed, assessment_status='completed',
                       opls_assessment_scope='experimental; no physical acceptance or candidate rejection')
    except Exception as exc:
        attempt.update(assessment_status='failed', assessment_error=f'{type(exc).__name__}: {exc}')
    if file_sha256(path) != attempt['raw_coordinates_sha256']:
        raise ValueError('Raw coordinate preservation check failed')


def run_candidate(args, model, spectra, entry, folder):
    from diffusion.opls_backend import declared_graph
    template = declared_graph(entry['smiles'])
    for c in range(args.conformers):
        seed = args.seed + entry['graph_index']*1000 + c
        entry['attempts'].append(generate_attempt(args, model, spectra, template, folder,
                                                 'unguided', c, seed))
    if not args.experimental_opls:
        return
    try:
        with OPLSClient(entry['smiles'], args.opls_python, args.appimage, args.cache, folder/'worker.log') as client:
            entry['parameters'] = client.metadata
            for relative in [*client.metadata['file_sha256'], 'manifest.json', 'parameterization.log']:
                target = folder/'parameters'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(client.metadata['cache_directory'])/relative, target)
            for attempt in list(entry['attempts']):
                assess_attempt(attempt, folder, template, client)
                guided = generate_attempt(args, model, spectra, template, folder, 'opls_guided',
                                          attempt['conformer'], attempt['seed'], client)
                entry['attempts'].append(guided)
                assess_attempt(guided, folder, template, client)
    except Exception as exc:
        entry.update(experimental_backend_status='failed', experimental_backend_error=f'{type(exc).__name__}: {exc}')
        for a in entry['attempts']:
            if a.get('assessment_status') == 'not_requested':
                a['assessment_status'] = 'unavailable'
        for c in range(args.conformers):
            if not any(a['arm']=='opls_guided' and a['conformer']==c for a in entry['attempts']):
                entry['attempts'].append({'arm':'opls_guided','conformer':c,
                    'seed':args.seed+entry['graph_index']*1000+c,'status':'unavailable',
                    'generation_status':'unavailable','assessment_status':'unavailable'})


def pilot(args, output, report):
    from rdkit import Chem
    roots = [Path('ckpt'), Path('runs'), Path('lightning_logs')]
    roots.extend(Path(p).parent for p in (args.coordinate_checkpoint, args.spectrum_checkpoint) if p)
    inventory = checkpoint_inventory(roots)
    (output/'checkpoint_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    report['checkpoint_inventory'] = 'checkpoint_inventory.json'
    coordinate = choose_checkpoint(inventory, 'EquivariantDiffusion3D_v4', args.coordinate_checkpoint)
    # This is the previously evaluated graph checkpoint; use it if present and
    # compatible. Otherwise require an unambiguous metadata-compatible candidate.
    known = Path('runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt')
    spectrum_path = args.spectrum_checkpoint or (str(known) if known.exists() else None)
    spectrum = choose_checkpoint(inventory, 'SpectrumStructureModel', spectrum_path, coordinate['run_metadata'])
    report['checkpoints'] = {role: {'path': item['path'], 'sha256': file_sha256(item['path'])}
                             for role, item in [('coordinate', coordinate), ('spectrum', spectrum)]}
    data = load_native_npz(args.data)
    splits = validate_split_manifest(json.loads(Path(args.splits).read_text()), data['molecule_ids'], args.data)
    for item in (coordinate, spectrum):
        meta = item['run_metadata']
        if (meta.get('data_sha256') != splits['data_sha256'] or meta.get('split_sha256') != file_sha256(args.splits)
                or meta.get('preprocessing') != data.get('preprocessing') or not data['packed']):
            raise ValueError('Training spectrum provenance/preprocessing does not match both checkpoints')
    row = int(splits['splits']['train'][0])
    report['input'] = {'row': row, 'split': 'train', 'engineering_pilot_only': True,
                       'reference_graph_or_coordinates_used': False}
    spectra = {'packed': torch.tensor(data['spectra'][row:row+1], device=args.device),
               'mask': torch.tensor(data['spectra_mask'][row:row+1], device=args.device)}
    np.savez_compressed(output/'input_ir.npz', **{k: v.cpu().numpy() for k, v in spectra.items()})
    del data
    coordinate_model = EquivariantDiffusion3D_v4.load_from_checkpoint(coordinate['path']).to(args.device).eval()
    spectrum_model = SpectrumStructureModel.load_from_checkpoint(spectrum['path']).to(args.device).eval()
    report['ema_used'] = {'coordinate': coordinate_model.use_ema_weights(), 'spectrum': spectrum_model.use_ema_weights()}
    candidates = valid_candidates(spectrum_model.generate(spectra, beam_size=5))[:args.graphs]
    del spectrum_model
    if not candidates:
        raise ValueError('Decoder returned no supported complete candidate graphs')
    report['graphs'] = []
    for graph_index, candidate in enumerate(candidates):
        entry = {**candidate, 'graph_index': graph_index, 'attempts': []}
        report['graphs'].append(entry)
        folder = output/f'graph_{graph_index}'
        folder.mkdir()
        try:
            run_candidate(args, coordinate_model, spectra, entry, folder)
        except Exception as exc:
            entry.update(status='generation_failed', error=f'{type(exc).__name__}: {exc}')
    attempts = [a for g in report['graphs'] for a in g['attempts']]
    report['attempt_count'] = len(attempts)
    report['planned_attempt_count'] = len(candidates)*args.conformers*(2 if args.experimental_opls else 1)
    report['completed_attempt_count'] = sum(a.get('status') == 'completed' for a in attempts)
    report['completed_assessment_count'] = sum(a.get('assessment_status') == 'completed' for a in attempts)
    report['assessment_requested'] = args.experimental_opls
    report['status'] = ('pilot_completed' if report['completed_attempt_count'] == report['planned_attempt_count']
                        and (not args.experimental_opls or report['completed_assessment_count'] == report['planned_attempt_count'])
                        else 'pilot_completed_with_failures')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coordinate-checkpoint')
    parser.add_argument('--spectrum-checkpoint')
    parser.add_argument('--data', default='data/processed/squirl.npz')
    parser.add_argument('--splits', default='data/processed/splits.json')
    parser.add_argument('--opls-python', default='.runtime/opls-tools/bin/python')
    parser.add_argument('--appimage', default='tools/LigParGen-3.0.1.AppImage')
    parser.add_argument('--cache', default='outputs/opls_parameter_cache')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--graphs', type=int, default=2)
    parser.add_argument('--conformers', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--experimental-opls', action='store_true',
                        help='Enable paired experimental guidance/relaxation; never a physical acceptance rule')
    parser.add_argument('--min-alpha', type=float, default=.8)
    parser.add_argument('--max-step', type=float, default=.02)
    args = parser.parse_args(argv)
    if min(args.graphs, args.conformers, args.steps) < 1:
        parser.error('Graph, conformer, and step counts must be positive')
    OPLSGuidance([], args.min_alpha, args.max_step)  # Validate before expensive work.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    report = {'status': 'blocked', 'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'physical_correctness_validated': False, 'early_rejection_enabled': False,
              'primary_geometry_stage': 'raw', 'opls_experimental_enabled': args.experimental_opls,
              'limitations': ['Engineering pilot on one training spectrum, not an accuracy evaluation.',
                 'OPLS guidance is a bounded optimization heuristic, not Boltzmann sampling.',
                 'SDF bonds are declared topology; energy and convergence do not prove molecular identity.',
                 'No conformer probabilities or cross-graph energy ranking. No weights promoted.']}
    root = Path(__file__).resolve().parent
    report['source_sha256'] = {n: file_sha256(root/n) for n in ('predict_opls.py', 'diffusion/model.py',
        'diffusion/opls_backend.py', 'diffusion/opls_bridge.py', 'diffusion/opls_client.py', 'diffusion/opls_guidance.py', 'scripts/opls_worker.py')}
    try:
        pilot(args, output, report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    summary = ['# OPLS coordinate pipeline pilot', '', 'Status: '+report['status'], '']
    if 'error' in report:
        summary += [report['error'], '', 'Checkpoint discovery is in checkpoint_inventory.json.',
                    'No legacy weights were substituted and no new model was trained.']
    if 'graphs' in report:
        summary += ['Raw coordinates are the primary output. OPLS energies below are experimental annotations.', '',
                    '| Graph | Arm | Seed | Generation | OPLS assessment | Raw OPLS energy | Relaxed OPLS energy |',
                    '|---|---|---:|---|---|---:|---:|']
        for graph in report['graphs']:
            if not graph['attempts']:
                summary += [f"Graph {graph['graph_index']}: {graph.get('error', 'unassessed')}"]
            for a in graph['attempts']:
                raw = a.get('raw', {}).get('energy_kj_mol', 'unavailable')
                after = a.get('relaxation', {}).get('after', {}).get('energy_kj_mol', 'unavailable')
                summary.append(f"| {graph['graph_index']} | {a['arm']} | {a['seed']} | {a.get('generation_status', a['status'])} | {a.get('assessment_status', 'unavailable')} | {raw} | {after} |")
        summary += ['', f"Completed {report['completed_attempt_count']} / {report['planned_attempt_count']} planned attempts."]
    summary += ['', 'Compare energies only within the same graph/parameter set. This is an engineering pilot,',
                'not physical correctness, Top1 accuracy, or Boltzmann population validation.']
    (output/'RESULTS.md').write_text('\n'.join(summary)+'\n')
    with zipfile.ZipFile(output/'pipeline_results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.name != 'pipeline_results.zip':
                archive.write(p, p.relative_to(output))
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'download': str(output/'pipeline_results.zip')}, indent=2))
    return 0 if report['status'] == 'pilot_completed' else 1


if __name__ == '__main__':
    sys.exit(main())
