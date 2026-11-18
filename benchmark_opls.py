"""Fixed training-only paired OPLS experiment; no selection on energy outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import zipfile

import numpy as np
import torch
from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.model import EquivariantDiffusion3D_v4
from diffusion.spectrum import SpectrumStructureModel
from diffusion.opls_backend import declared_graph
from diffusion.opls_client import OPLSClient
from diffusion.opls_guidance import OPLSGuidance
from diffusion.preprocessing import bond_array
from evaluation.evaluate_spectrum import valid_candidates
from predict_opls import checkpoint_inventory, choose_checkpoint

ARMS = ('unguided', 'opls_guided')
BINS = ((3, 10), (11, 18), (19, 29))
SOURCES = ('benchmark_opls.py', 'predict_opls.py', 'diffusion/model.py',
           'diffusion/opls_backend.py', 'diffusion/opls_bridge.py',
           'diffusion/opls_client.py', 'diffusion/opls_guidance.py', 'scripts/opls_worker.py',
           'diffusion/data.py', 'diffusion/spectrum.py', 'evaluation/evaluate_spectrum.py')


def select_inputs(ids, counts, train, per_bin, seed):
    """Stratify by explicit atom count; one randomly chosen row per identity."""
    rng = np.random.default_rng(seed)
    groups = {}
    for row in sorted(map(int, train)):
        groups.setdefault(str(ids[row]), []).append(row)
    representatives = []
    for identity, rows in sorted(groups.items()):
        if len({int(counts[r]) for r in rows}) != 1:
            raise ValueError('Inconsistent atom counts within training identity ' + identity)
        row = int(rng.choice(rows))
        representatives.append({'row': row, 'molecule_id': identity, 'selection_atom_count': int(counts[row])})
    selected, available = [], {}
    for low, high in BINS:
        label = f'{low}-{high}'
        pool = [r for r in representatives if low <= r['selection_atom_count'] <= high]
        available[label] = len(pool)
        if len(pool) < per_bin:
            raise ValueError(f'Need {per_bin} unique training identities in {label} atoms; only {len(pool)} available')
        for index in rng.choice(len(pool), per_bin, replace=False):
            selected.append({**pool[int(index)], 'selection_bin': label})
    # Interleave size strata so a partial run does not cover only small molecules.
    selected = [selected[b*per_bin+i] for i in range(per_bin) for b in range(len(BINS))]
    return selected, available


def plan_graphs(inputs, graphs, conformers, seed):
    """Keep missing/unsupported candidates in the intended attempt denominator."""
    for i, entry in enumerate(inputs):
        entry['graphs'] = []
        for g in range(graphs):
            candidate = entry.get('candidates', [])[g:g+1]
            item = {'graph_index': g, 'status': 'pending', 'attempts': []}
            if candidate:
                item.update(candidate[0])
            else:
                item['status'] = 'missing_candidate'
            for c in range(conformers):
                paired_seed = seed + (i*graphs+g)*conformers+c
                order = ARMS if (i+g+c) % 2 == 0 else ARMS[::-1]
                for position, arm in enumerate(order):
                    item['attempts'].append({'arm': arm, 'conformer': c, 'seed': paired_seed,
                        'order_in_pair': position, 'status': 'pending' if candidate else 'missing_candidate'})
            entry['graphs'].append(item)


def describe(values):
    return {'n': len(values), 'median': float(np.median(values)) if values else None,
            'mean': float(np.mean(values)) if values else None}


def summarize(report):
    attempts = [a for r in report.get('inputs', []) for g in r.get('graphs', []) for a in g['attempts']]
    result = {'planned_attempts': report['planned_attempts'], 'recorded_attempts': len(attempts),
              'status_counts': dict(Counter(a['status'] for a in attempts)), 'arms': {}, 'pairs': []}
    for arm in ARMS:
        group = [a for a in attempts if a['arm'] == arm]
        raw = [a['raw'] for a in group if 'raw' in a]
        relaxed = [a['relaxation'] for a in group if 'relaxation' in a]
        result['arms'][arm] = {
            'planned': report['planned_attempts']//2,
            'completed': sum(a['status'] == 'completed' for a in group),
            'raw_assessed': len(raw),
            'raw_graph_matches': sum(bool(a.get('coordinate_inferred_graph_matches')) for a in raw),
            'raw_force_rms_kj_mol_nm': describe([a['force_rms_kj_mol_nm'] for a in raw]),
            'relaxation_assessed': len(relaxed),
            'relaxed_converged_graph_matches': sum(bool(a['accepted'] and a['force_converged'] and
                a['after'].get('coordinate_inferred_graph_matches')) for a in relaxed),
            'repair_rms_angstrom_unaligned': describe([a['repair_rms_angstrom_unaligned'] for a in group
                                                      if 'repair_rms_angstrom_unaligned' in a]),
            'phase_seconds': {phase: describe([a['timing'][phase] for a in group if phase in a.get('timing', {})])
                              for phase in ('generation', 'raw_assessment', 'relaxation')},
        }
    for r in report.get('inputs', []):
        for g in r.get('graphs', []):
            for c in sorted({a['conformer'] for a in g['attempts']}):
                pair = {a['arm']: a for a in g['attempts'] if a['conformer'] == c}
                if len(pair) != 2 or any('raw' not in a for a in pair.values()):
                    continue
                u, v = pair['unguided'], pair['opls_guided']
                if u['seed'] != v['seed']:
                    raise ValueError('Paired seeds differ')
                item = {'input_index': r['input_index'], 'graph_index': g['graph_index'], 'seed': u['seed'],
                        'raw_energy_delta_kj_mol': v['raw']['energy_kj_mol']-u['raw']['energy_kj_mol'],
                        'raw_force_delta_kj_mol_nm': v['raw']['force_rms_kj_mol_nm']-u['raw']['force_rms_kj_mol_nm']}
                if all(a.get('relaxation', {}).get('accepted') for a in (u, v)):
                    item['relaxed_energy_delta_kj_mol'] = (v['relaxation']['after']['energy_kj_mol']-
                                                          u['relaxation']['after']['energy_kj_mol'])
                result['pairs'].append(item)
    pairs = result['pairs']
    result['raw_pairs_assessed'] = len(pairs)
    result['raw_pairs_lower_guided_energy'] = sum(p['raw_energy_delta_kj_mol'] < 0 for p in pairs)
    result['raw_pairs_lower_guided_force'] = sum(p['raw_force_delta_kj_mol_nm'] < 0 for p in pairs)
    # Average seeds and candidates within each spectrum first; no seed-level confidence interval.
    result['per_input_raw_energy_delta_kj_mol'] = [
        {'input_index': i, 'paired_count': len(group), 'mean_delta': float(np.mean(group))}
        for i in sorted({p['input_index'] for p in pairs})
        if (group := [p['raw_energy_delta_kj_mol'] for p in pairs if p['input_index'] == i])]
    result['distinct_predicted_graphs'] = len({g['smiles'] for r in report.get('inputs', [])
        for g in r.get('graphs', []) if 'smiles' in g})
    return result


def write_report(output, report, archive=False):
    report['summary'] = summarize(report)
    temporary = output/'report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(output/'report.json')
    s = report['summary']
    lines = ['# Fixed OPLS training-spectrum experiment', '', 'Status: '+report['status'],
        f"Planned attempts: {s['planned_attempts']}; statuses: {s['status_counts']}",
        f"Raw paired assessments: {s['raw_pairs_assessed']} / {s['planned_attempts']//2}",
        f"Pairs with lower guided raw energy: {s['raw_pairs_lower_guided_energy']}",
        f"Pairs with lower guided raw force: {s['raw_pairs_lower_guided_force']}",
        f"Distinct predicted graphs: {s['distinct_predicted_graphs']}", '',
        '| Arm | Completed / planned | Raw graph matches / assessed | Relaxed converged graph matches / assessed |',
        '|---|---:|---:|---:|']
    for arm, a in s['arms'].items():
        lines.append(f"| {arm} | {a['completed']} / {a['planned']} | {a['raw_graph_matches']} / {a['raw_assessed']} | "
                     f"{a['relaxed_converged_graph_matches']} / {a['relaxation_assessed']} |")
    lines += ['', 'Failures and unattempted slots remain in the planned denominator. Conditional rates above',
              'must be read alongside that denominator. Negative paired deltas favor guidance.', '', *report['limitations']]
    if report.get('error'):
        lines += ['', report['error']]
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if archive:
        temporary_zip = output/'benchmark_results.zip.tmp'
        with zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.name not in ('benchmark_results.zip', 'benchmark_results.zip.tmp', 'report.json.tmp'):
                    z.write(p, p.relative_to(output))
        temporary_zip.replace(output/'benchmark_results.zip')


def run_graph(args, model, spectra, graph_entry, folder, checkpoint, should_stop):
    from rdkit import Chem
    folder.mkdir(parents=True)
    start = time.monotonic()
    try:
        template = declared_graph(graph_entry['smiles'])
        numbers = [a.GetAtomicNum() for a in template.GetAtoms()]
        graph_entry['predicted_atom_count'] = len(numbers)
        graph_entry['predicted_heavy_atom_count'] = template.GetNumHeavyAtoms()
        with OPLSClient(graph_entry['smiles'], args.opls_python, args.appimage, args.cache, folder/'worker.log') as client:
            graph_entry['setup_seconds'] = time.monotonic()-start
            graph_entry['parameters'] = client.metadata
            for relative in [*client.metadata['file_sha256'], 'manifest.json', 'parameterization.log']:
                target = folder/'parameters'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(client.metadata['cache_directory'])/relative, target)
            atoms = torch.tensor([numbers], dtype=torch.long)
            count = torch.tensor([len(numbers)])
            bonds = bond_array(template)
            graph = {'bond_types': torch.tensor(bonds[None], dtype=torch.long)}
            graph_entry['status'] = 'running'
            for attempt in graph_entry['attempts']:
                if should_stop():
                    graph_entry['status'] = 'partial'
                    return
                policy = OPLSGuidance([client], args.min_alpha, args.max_step) if attempt['arm'] == 'opls_guided' else None
                attempt.update(status='running', timing={})
                started = time.monotonic()
                checkpoint()
                try:
                    phase = time.monotonic()
                    xyz, mask = model.sample(spectra, atoms, count, n_steps=min(args.steps, model.n_timesteps),
                        graph=graph, physics_guidance=0., generator=torch.Generator().manual_seed(attempt['seed']),
                        opls_guidance=policy)
                    attempt['timing']['generation'] = time.monotonic()-phase
                    raw = xyz[0, mask[0]].detach().cpu().numpy()
                    if raw.shape != (len(numbers), 3) or not np.isfinite(raw).all():
                        raise ValueError('Invalid sampled coordinates')
                    stem = f"{attempt['arm']}_{attempt['conformer']}"
                    np.save(folder/(stem+'_raw.npy'), raw)
                    phase = time.monotonic()
                    attempt['raw'] = client.request('evaluate', coordinates=raw.tolist())
                    attempt['timing']['raw_assessment'] = time.monotonic()-phase
                    checkpoint()
                    phase = time.monotonic()
                    relaxed = client.request('relax', coordinates=raw.tolist())
                    after = np.asarray(relaxed.pop('coordinates'))
                    attempt['timing']['relaxation'] = time.monotonic()-phase
                    if after.shape != raw.shape or not np.isfinite(after).all():
                        raise ValueError('Invalid relaxed coordinates')
                    attempt['relaxation'] = relaxed
                    attempt['repair_rms_angstrom_unaligned'] = float(np.sqrt(np.mean(np.sum((after-raw)**2, axis=1))))
                    np.savez_compressed(folder/(stem+'.npz'), raw_angstrom=raw, relaxed_angstrom=after,
                                        atom_numbers=numbers, bond_types=bonds)
                    with Chem.SDWriter(str(folder/(stem+'.sdf'))) as writer:
                        for label, points in [('raw', raw), ('relaxed', after)]:
                            molecule = Chem.Mol(template)
                            conf = Chem.Conformer(len(numbers))
                            for i, p in enumerate(points):
                                conf.SetAtomPosition(i, tuple(map(float, p)))
                            molecule.AddConformer(conf)
                            molecule.SetProp('geometry_stage', label)
                            molecule.SetProp('arm', attempt['arm'])
                            molecule.SetProp('seed', str(attempt['seed']))
                            writer.write(molecule)
                    attempt['status'] = 'completed'
                except Exception as exc:
                    attempt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                finally:
                    attempt['seconds'] = time.monotonic()-started
                    if policy:
                        attempt['guidance_trace'] = policy.trace
                    checkpoint()
            graph_entry['status'] = 'completed' if all(a['status'] == 'completed' for a in graph_entry['attempts']) else 'completed_with_failures'
    except Exception as exc:
        graph_entry.update(status='graph_failed', error=f'{type(exc).__name__}: {exc}')
        for a in graph_entry['attempts']:
            if a['status'] in ('pending', 'running'):
                a.update(status='graph_failed', error=graph_entry['error'])
    finally:
        graph_entry['total_seconds'] = time.monotonic()-start
        checkpoint()


def experiment(args, output, report, should_stop):
    phase = time.monotonic()
    inventory = checkpoint_inventory([Path(args.coordinate_checkpoint).parent, Path(args.spectrum_checkpoint).parent])
    (output/'checkpoint_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    coordinate = choose_checkpoint(inventory, 'EquivariantDiffusion3D_v4', args.coordinate_checkpoint)
    spectrum = choose_checkpoint(inventory, 'SpectrumStructureModel', args.spectrum_checkpoint, coordinate['run_metadata'])
    report['checkpoints'] = {role: {'path': c['path'], 'sha256': file_sha256(c['path'])}
                             for role, c in [('coordinate', coordinate), ('spectrum', spectrum)]}
    data = load_native_npz(args.data)
    splits = validate_split_manifest(json.loads(Path(args.splits).read_text()), data['molecule_ids'], args.data)
    split_hash = file_sha256(args.splits)
    for c in (coordinate, spectrum):
        meta = c['run_metadata']
        if (meta.get('data_sha256') != splits['data_sha256'] or meta.get('split_sha256') != split_hash
                or meta.get('preprocessing') != data.get('preprocessing') or not data['packed']):
            raise ValueError('Data/preprocessing does not match both checkpoints')
    selected, available = select_inputs(data['molecule_ids'], data['num_atoms'], splits['splits']['train'], args.per_bin, args.selection_seed)
    report['selection'] = {'seed': args.selection_seed, 'available_unique_identities_per_bin': available,
        'data_sha256': splits['data_sha256'], 'split_sha256': split_hash,
        'reference_atom_count_used_only_for_stratification': True,
        'reference_bonds_coordinates_not_used_for_generation_or_scoring': True, 'split': 'train'}
    report['inputs'] = [{**r, 'input_index': i} for i, r in enumerate(selected)]
    rows = [r['row'] for r in selected]
    packed = data['spectra'][rows].copy()
    masks = data['spectra_mask'][rows].copy()
    del data
    (output/'selection.json').write_text(json.dumps({'selection': report['selection'], 'inputs': selected}, indent=2)+'\n')
    report['selection_sha256'] = file_sha256(output/'selection.json')
    np.savez_compressed(output/'input_ir.npz', packed=packed, mask=masks, rows=rows)
    report['input_ir_sha256'] = file_sha256(output/'input_ir.npz')
    spectrum_model = SpectrumStructureModel.load_from_checkpoint(spectrum['path']).cpu().eval()
    report['ema_used'] = {'spectrum': spectrum_model.use_ema_weights()}
    # Freeze all graphs before parameterization or energy queries; never backfill failures.
    for i, entry in enumerate(report['inputs']):
        if should_stop():
            report['status'] = 'partial'
            return
        try:
            with torch.inference_mode():
                entry['decoded_beams'] = spectrum_model.generate(
                    {'packed': torch.tensor(packed[i:i+1]), 'mask': torch.tensor(masks[i:i+1])}, beam_size=5)
                entry['candidates'] = valid_candidates(entry['decoded_beams'])[:args.graphs]
        except Exception as exc:
            entry.update(candidates=[], decoding_error=f'{type(exc).__name__}: {exc}')
        print(f'Decoded input {i+1}/{len(selected)}', flush=True)
    del spectrum_model
    plan_graphs(report['inputs'], args.graphs, args.conformers, args.seed)
    (output/'candidate_plan.json').write_text(json.dumps(report['inputs'], indent=2)+'\n')
    report['candidate_plan_sha256'] = file_sha256(output/'candidate_plan.json')
    model = EquivariantDiffusion3D_v4.load_from_checkpoint(coordinate['path']).cpu().eval()
    report['ema_used']['coordinate'] = model.use_ema_weights()
    report['preparation_seconds'] = time.monotonic()-phase
    write_report(output, report, archive=True)
    for i, entry in enumerate(report['inputs']):
        spectra = {'packed': torch.tensor(packed[i:i+1]), 'mask': torch.tensor(masks[i:i+1])}
        for g in entry['graphs']:
            if should_stop():
                report['status'] = 'partial'
                return
            if g['status'] == 'missing_candidate':
                continue
            print(f"Input {i+1}/{len(selected)}, graph {g['graph_index']+1}: {g['smiles']}", flush=True)
            run_graph(args, model, spectra, g, output/f"input_{i:03d}"/f"graph_{g['graph_index']}",
                      lambda: write_report(output, report), should_stop)
            write_report(output, report, archive=True)
    attempts = [a for r in report['inputs'] for g in r['graphs'] for a in g['attempts']]
    report['status'] = ('partial' if any(a['status'] == 'pending' for a in attempts) else
                       'completed' if all(a['status'] == 'completed' for a in attempts) else 'completed_with_failures')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coordinate-checkpoint', default='runs/diffusion/checkpoints/best.ckpt')
    parser.add_argument('--spectrum-checkpoint', default='runs/spectrum_pilot_66855524/ranking_attempt_1/checkpoints/last.ckpt')
    parser.add_argument('--data', default='data/processed/squirl.npz')
    parser.add_argument('--splits', default='data/processed/splits.json')
    parser.add_argument('--opls-python', default='.runtime/opls-tools/bin/python')
    parser.add_argument('--appimage', default='tools/LigParGen-3.0.1.AppImage')
    parser.add_argument('--cache', default='outputs/opls_parameter_cache')
    parser.add_argument('--per-bin', type=int, default=8)
    parser.add_argument('--graphs', type=int, default=2)
    parser.add_argument('--conformers', type=int, default=3)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--selection-seed', type=int, default=20261116)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--min-alpha', type=float, default=.8)
    parser.add_argument('--max-step', type=float, default=.02)
    parser.add_argument('--budget-minutes', type=float, default=18.)
    parser.add_argument('--experimental-opls', action='store_true')
    args = parser.parse_args(argv)
    if not args.experimental_opls:
        parser.error('Paired OPLS benchmark is experimental; use --experimental-opls explicitly. For raw generation use predict_opls.py.')
    if min(args.per_bin, args.graphs, args.conformers, args.steps) < 1 or args.graphs > 5:
        parser.error('Positive counts required, with at most five graphs')
    if not np.isfinite(args.budget_minutes) or args.budget_minutes <= 0:
        parser.error('Positive finite runtime budget required')
    OPLSGuidance([], args.min_alpha, args.max_step)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    start, stop = time.monotonic(), {'requested': False}
    def should_stop():
        return stop['requested'] or time.monotonic()-start >= args.budget_minutes*60
    def on_signal(signum, frame):
        stop['requested'] = True
    saved_handlers = {}
    for name in ('SIGUSR1', 'SIGTERM', 'SIGINT'):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            saved_handlers[sig] = signal.signal(sig, on_signal)
    root = Path(__file__).resolve().parent
    report = {'status': 'running', 'configuration': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'planned_attempts': 3*args.per_bin*args.graphs*args.conformers*2, 'inputs': [],
        'physical_correctness_validated': False, 'early_rejection_enabled': False,
        'decoding': {'beam_size': 5, 'candidate_rule': 'first valid unique graphs, same as successful pilot',
                     'score_mode': 'model default (legacy)', 'length_penalty': 'model default'},
        'source_sha256': {n: file_sha256(root/n) for n in SOURCES},
        'limitations': ['Training-only engineering experiment; no Top1 or generalization claim.',
            'Input sizes are stratified using reference atom counts; generated topology comes only from IR decoding.',
            'Unsupported and missing candidates are retained as failures; no replacements based on outcomes.',
            'Compare energy differences only within the same candidate graph and parameter set.',
            'Seeds and candidates from one spectrum are not independent molecules; summaries are descriptive.',
            'Coordinate-inferred graph agreement ignores stereochemistry and is not proof of physical identity.',
            'Guidance is bounded optimization, not Boltzmann sampling; no global minimum or population claim.',
            'Raw-to-relaxed displacement uses fixed atom order without alignment; rejected relaxation retains raw coordinates.',
            'Phase timings include worker communication; graph setup is reported separately. CPU only.']}
    try:
        write_report(output, report, archive=True)
        experiment(args, output, report, should_stop)
    except Exception as exc:
        report.update(status='blocked', error=f'{type(exc).__name__}: {exc}')
    finally:
        report['total_seconds'] = time.monotonic()-start
        if should_stop():
            report['stop_requested_or_budget_reached'] = True
        write_report(output, report, archive=True)
        for sig, handler in saved_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({'status': report['status'], 'summary': report['summary']['status_counts'],
                      'download': str(output/'benchmark_results.zip')}, indent=2), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
