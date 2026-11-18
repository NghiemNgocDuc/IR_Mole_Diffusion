"""Read-only IR dependence controls for sealed token-matcher checkpoints."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch

from diffusion.data import file_sha256
from diffusion.progress import EvaluationProgress
from diffusion.spectrum_audit import atomic_json, exclusive_report, summarize_records
from diffusion.token_matcher import TokenMatcher
from scripts.compare_spectrum_pilot import paired_metrics, BOOTSTRAP_SEED
from training.train_token_matcher import validate_inputs, score_pools, prediction_records, orders_for

ROOT = Path(__file__).resolve().parent


def donor_indices(count, seed, groups=None):
    """Random cyclic derangement within groups; singletons explicitly ineligible.

    Groups use reference formulas ONLY for a diagnostic control, never a prediction
    feature or model selection. A cyclic shift has no self-donors for size >= 2.
    """
    if count < 2 or (groups is not None and len(groups) != count):
        raise ValueError('IR controls require at least two correctly grouped rows.')
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i in range(count):
        buckets[groups[i] if groups is not None else 'all'].append(i)
    donors = np.arange(count, dtype=np.int64)
    eligible = np.zeros(count, dtype=bool)
    for indices in buckets.values():
        if len(indices) > 1:
            order = rng.permutation(indices)
            donors[order] = np.roll(order, 1)
            eligible[order] = True
    if np.any(donors[eligible] == np.arange(count)[eligible]) or len(set(donors)) != count:
        raise ValueError('Control mapping is not a valid derangement of eligible rows.')
    return donors, eligible


def score_change(real, control, eligible):
    absolute, centered = [], []
    flipped, counted = 0, 0
    for a, b, use in zip(real, control, eligible):
        if not use or not a:
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Control score arrays differ or are nonfinite.')
        absolute.extend(np.abs(a - b).tolist())
        centered.extend(np.abs((a - a.mean()) - (b - b.mean())).tolist())
        counted += 1
        flipped += int(a.argmax() != b.argmax())
    return {'nonempty_pools': counted, 'matcher_first_choice_changed': flipped,
            'matcher_first_choice_changed_fraction': flipped / counted if counted else None,
            'mean_absolute_logit_change': float(np.mean(absolute)) if absolute else None,
            'mean_absolute_within_pool_centered_change': float(np.mean(centered)) if centered else None}


def validate_replay(records, real, saved_records, weight, tolerance=1e-3):
    """A shuffled-input claim is invalid if the real-input result cannot replay."""
    if [r['row'] for r in records] != [r['row'] for r in saved_records] or len(real) != len(records):
        raise ValueError('Replay row order mismatch.')
    maximum = 0.
    for original, values, saved in zip(records, real, saved_records):
        lookup = {c['smiles']: c for c in saved['candidates']}
        if set(lookup) != {c['smiles'] for c in original['candidates']}:
            raise ValueError('Saved candidate pool differs from original generator report.')
        for c, value in zip(original['candidates'], values):
            other = lookup[c['smiles']]
            if any(c[k] != other[k] for k in ('selfies', 'formula', 'log_probability')):
                raise ValueError('Saved candidate content differs from original report.')
            difference = abs(value - other['token_match_score'])
            if not np.isfinite(difference) or difference > tolerance:
                raise ValueError('Real-input replay score mismatch; inspect checkpoint/code/device before interpreting controls.')
            maximum = max(maximum, difference)
    orders = orders_for([r['candidates'] for r in records], real, weight)
    for record, order, saved in zip(records, orders, saved_records):
        replay_top = record['candidates'][order[0]]['smiles'] if order else None
        saved_top = saved['candidates'][0]['smiles'] if saved['candidates'] else None
        if replay_top != saved_top:
            raise ValueError('Real-input replay changed Top1; do not continue to IR controls.')
    return maximum


def run(args):
    directory, output = Path(args.run_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError('Choose a fresh probe output directory; existing results are preserved.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; submit this diagnostic to a GPU node.')
    cfg_path, selection_path = directory / 'configuration.json', directory / 'selection.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    comparison = json.loads((directory / 'comparison.json').read_text(encoding='utf-8'))
    if selection != comparison['selection'] or selection.get('assessment_used_for_selection') is not False:
        raise ValueError('Require matching sealed selection and comparison reports.')
    if cfg.get('diagnostic') and not args.diagnostic:
        raise ValueError('Diagnostic run requires --diagnostic; no scientific accuracy claim.')
    for name, digest in cfg['source_sha256'].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f'Original inference/training code changed: {name}. Restore the exact run version before probing.')
    for name, digest in cfg['protected_inputs'].items():
        if file_sha256(name) != digest:
            raise ValueError(f'Original scientific input changed: {name}')
    initial_hashes = {str(p): file_sha256(p) for p in (cfg_path, selection_path, directory / 'comparison.json')}
    data, manifest, baseline, generator, diagnostic = validate_inputs(SimpleNamespace(**cfg['arguments']))
    train, dev, assess = (cfg[k] for k in ('gradient_training_rows', 'development_rows', 'assessment_rows'))
    if (train != manifest['splits']['train'] or len(set(assess)) != len(assess) or len(set(dev)) != len(dev)
            or set(dev) & set(assess) or set(train) & (set(dev) | set(assess))
            or set(dev) | set(assess) != set(manifest['splits']['val'])):
        raise ValueError('Original split partition mismatch.')
    by_row = {r['row']: r for r in baseline['rows']}
    records = [by_row[i] for i in assess]
    pools = [r['candidates'] for r in records]
    max_len = int(generator.hparams.max_length)
    controls = []
    for seed in (20261111, 20261112, 20261113):
        donors, eligible = donor_indices(len(assess), seed)
        controls.append((f'shuffled_{seed}', donors, eligible))
    donors, eligible = donor_indices(len(assess), 20261111, [r['reference_formula'] for r in records])
    controls.append(('same_formula_shuffled_20261111', donors, eligible))
    output.mkdir(parents=True)
    protocol = {'training_performed': False, 'weights_promoted': False, 'checkpoint_selection_performed': False,
        'original_configuration_sha256': initial_hashes[str(cfg_path)],
        'original_selection_sha256': initial_hashes[str(selection_path)],
        'probe_source_sha256': file_sha256(__file__), 'device': args.device,
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'score_tolerance': .001, 'assessment_rows': assess, 'diagnostic': diagnostic or args.diagnostic,
        'controls': {label: {'spectrum_donor_rows': [assess[i] for i in donors],
                     'eligible_rows': [r for r, use in zip(assess, eligible) if use]} for label, donors, eligible in controls},
        'limits': ['Previously inspected assessment data; diagnostic only, not a fresh accuracy evaluation.',
                   'Reference formulas are used only to construct the same-formula control; singletons are excluded from its paired comparison.',
                   'Decoder scores and candidate pools remain fixed; the probe isolates matcher dependence on IR.',
                   'Shuffling measures dependence on paired spectra, not physical correctness or proof of the cause of poor accuracy.']}
    atomic_json(output / 'protocol.json', protocol)
    findings = {}
    with EvaluationProgress(output / 'progress', interval=15) as progress:
        for arm in ('frozen', 'finetuned'):
            for mode in ('combined', 'matcher'):
                label = arm + '/' + mode
                choice = selection['arms'][arm][mode]
                if not choice['epoch']:
                    findings[label] = {'baseline_fallback': True, 'reason': 'No matcher checkpoint selected.'}
                    continue
                checkpoint = (directory / choice['checkpoint']).resolve()
                if not checkpoint.is_relative_to(directory.resolve()):
                    raise ValueError('Checkpoint path must remain inside the original run directory.')
                if file_sha256(checkpoint) != choice['checkpoint_sha256']:
                    raise ValueError('Selected matcher checkpoint hash mismatch.')
                initial_hashes[str(checkpoint)] = choice['checkpoint_sha256']
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if payload['configuration_sha256'] != initial_hashes[str(cfg_path)] or payload['epoch'] != choice['epoch']:
                    raise ValueError('Checkpoint configuration or epoch mismatch.')
                matcher = TokenMatcher(generator.encoder, len(data['vocabulary']), max_len,
                    n_heads=int(generator.encoder.spec['n_heads']), n_layers=cfg['arguments']['matcher_layers'],
                    dropout=.1, train_encoder=False).to(args.device)
                matcher.load_state_dict(payload['state_dict'], strict=True)
                matcher.eval().requires_grad_(False)
                saved_path = directory / f'assessment_{arm}_{mode}.json'
                initial_hashes[str(saved_path)] = file_sha256(saved_path)
                saved = json.loads(saved_path.read_text(encoding='utf-8'))
                if saved['selection'] != choice:
                    raise ValueError('Saved prediction selection mismatch.')
                real = score_pools(matcher, data, assess, pools, data['vocabulary'], max_len,
                    args.device, cfg['arguments']['batch_size'], progress, label + ': real-input replay')
                max_error = validate_replay(records, real, saved['rows'], choice['weight'])
                real_selected = prediction_records(records, real, choice['weight'])
                real_pure = prediction_records(records, real, None)
                finding = {'selection': choice, 'real_input_replayed': True, 'maximum_replay_score_error': max_error,
                    'real_selected_summary': summarize_records(real_selected), 'real_matcher_summary': summarize_records(real_pure),
                    'controls': {}}
                for control, donors, eligible in controls:
                    if not eligible.any():
                        finding['controls'][control] = {'eligible_rows': 0, 'skipped': True}
                        continue
                    altered = score_pools(matcher, data, [assess[i] for i in donors], pools, data['vocabulary'], max_len,
                        args.device, cfg['arguments']['batch_size'], progress, label + ': ' + control)
                    changed_selected = prediction_records(records, altered, choice['weight'])
                    changed_pure = prediction_records(records, altered, None)
                    use = lambda rows: [r for r, live in zip(rows, eligible) if live]
                    finding['controls'][control] = {'eligible_rows': int(eligible.sum()),
                        'real_selected_summary': summarize_records(use(real_selected)),
                        'real_matcher_summary': summarize_records(use(real_pure)),
                        'control_selected_summary': summarize_records(use(changed_selected)),
                        'control_matcher_summary': summarize_records(use(changed_pure)),
                        'selected_paired_control_minus_real': paired_metrics(use(real_selected), use(changed_selected), BOOTSTRAP_SEED),
                        'matcher_paired_control_minus_real': paired_metrics(use(real_pure), use(changed_pure), BOOTSTRAP_SEED),
                        'score_changes': score_change(real, altered, eligible)}
                    atomic_json(output / f'{arm}_{mode}_{control}.json', {
                        'rows': [{'row': r['row'], 'donor_row': assess[donors[i]], 'eligible': bool(eligible[i]),
                                  'candidate_smiles': [c['smiles'] for c in r['candidates']],
                                  'real_scores': real[i], 'control_scores': altered[i]} for i, r in enumerate(records)]})
                findings[label] = finding
                atomic_json(output / 'findings.json', findings)
                del matcher, payload
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
        for path, digest in {**cfg['protected_inputs'], **initial_hashes}.items():
            if file_sha256(path) != digest:
                raise ValueError(f'Input or checkpoint changed during read-only probe: {path}')
        for name, digest in cfg['source_sha256'].items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError(f'Original source changed during probe: {name}')
        if file_sha256(__file__) != protocol['probe_source_sha256']:
            raise ValueError('Probe source changed during execution.')
        lines = ['# Token matcher IR dependence probe', '',
            'No training, checkpoint selection or model promotion. Real-input replay checked before shuffling.', '',
            '| Checkpoint | Control | Eligible | True IR matcher Top1 | Control matcher Top1 | Matcher first-choice changes |',
            '|---|---|---:|---:|---:|---:|']
        for name, finding in findings.items():
            for control, result in finding.get('controls', {}).items():
                if result.get('skipped'):
                    continue
                fraction = result['score_changes']['matcher_first_choice_changed_fraction']
                changed = f'{100*fraction:.2f}%' if fraction is not None else 'n/a (no candidates)'
                lines.append(f"| {name} | {control} | {result['eligible_rows']} | {100*result['real_matcher_summary']['top1_identity_fraction']:.2f}% | {100*result['control_matcher_summary']['top1_identity_fraction']:.2f}% | {changed} |")
        lines += ['', 'Similar accuracy alone does not establish IR independence: also examine score changes and first-choice changes.',
            'A substantial drop with wrong IR supports use of paired spectral information, but does not establish adequate discrimination among close structures.',
            'Do not tune weights or select another checkpoint from these controls.', '', *protocol['limits'], '']
        (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        with ZipFile(output / 'token_matcher_probe_results.zip', 'w', ZIP_DEFLATED) as z:
            for path in sorted(output.glob('*.json')):
                z.write(path, path.name)
            z.write(output / 'RESULTS.md', 'RESULTS.md')
        atomic_json(output / 'completion.json', {'complete': True,
            'results_sha256': file_sha256(output / 'token_matcher_probe_results.zip')})
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', default='runs/token_matcher_1426578')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args(argv)
    try:
        with exclusive_report(Path(args.output_dir).with_suffix('.run')):
            result = run(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f'Complete: {args.output_dir}/RESULTS.md\nDownload: {args.output_dir}/token_matcher_probe_results.zip', flush=True)
    return result


if __name__ == '__main__':
    main()
