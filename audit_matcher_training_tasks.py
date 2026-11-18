"""Compare saved matchers on original training choices and generated alternatives."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from diffusion.data import file_sha256
from diffusion.matcher_task_audit import (select_training_indices, validate_negative_plan,
    matched_generated_groups, pool_statistics, paired_task_gap)
from diffusion.progress import EvaluationProgress
from diffusion.ranking import canonical_identity
from diffusion.residual_selector import decoder_scores
from diffusion.spectrum_audit import atomic_json, exclusive_report
from diffusion.token_matcher import TokenMatcher
from diffusion.token_matcher_data import spectra_batch, tokenize_selfies
from evaluate_spectrum import valid_candidates
from probe_token_matcher import validate_replay
from train_token_matcher import validate_inputs, score_pools

ROOT = Path(__file__).resolve().parent


def stored_candidate(data, row, train_set, max_len):
    if row not in train_set:
        raise ValueError('A diagnostic training choice accessed a held-out identity.')
    tokens = np.asarray(data['selfies_tokens'][row])
    live = tokens[:np.count_nonzero(tokens)]
    if (len(live) < 3 or live[0] != 1 or live[-1] != 2 or np.any(live[1:-1] < 3)
            or np.any(tokens[len(live):] != 0)):
        raise ValueError('Invalid training token serialization.')
    text = ''.join(data['vocabulary'][int(i)] for i in live[1:-1])
    tokenize_selfies([text], data['vocabulary'], max_len)
    identity = canonical_identity(data['molecule_ids'][row])
    if not identity or canonical_identity(sf.decoder(text)) != identity:
        raise ValueError('Stored training serialization differs from its reference identity.')
    return {'training_row': int(row), 'smiles': identity, 'selfies': text,
            'formula': rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(identity))}


def load_original_plan(directory, metadata, epoch, count, protected):
    path = directory / f'negative_plan_{epoch:02d}.npy'
    if not 1 <= epoch <= len(metadata['sha256']):
        raise ValueError('Selected epoch is absent from the original negative plans.')
    expected = metadata['sha256'][epoch-1]
    if file_sha256(path) != expected:
        raise ValueError('Original training negative plan hash differs.')
    protected[str(path)] = expected
    return validate_negative_plan(np.load(path, allow_pickle=False), count)


def percentage(value):
    return 'n/a' if value is None else f'{100*value:.2f}%'


def run(args):
    directory, output = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    if output.exists() or output.is_relative_to(directory):
        raise ValueError('Choose a fresh output directory outside the original matcher run.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; submit the audit to a GPU node.')
    protected = {}
    def read_json(path):
        protected[str(path)] = file_sha256(path)
        return json.loads(path.read_text(encoding='utf-8'))
    cfg = read_json(directory / 'configuration.json')
    selection = read_json(directory / 'selection.json')
    comparison = read_json(directory / 'comparison.json')
    plans = read_json(directory / 'negative_plans.json')
    if selection != comparison['selection'] or selection.get('assessment_used_for_selection') is not False:
        raise ValueError('Original model selection reports differ or used assessment labels.')
    if not plans.get('local_indices_into_gradient_training_rows') or not plans.get('shared_by_both_arms'):
        raise ValueError('Require original shared training-only negative plans.')
    if cfg.get('diagnostic') and not args.diagnostic:
        raise ValueError('Diagnostic checkpoint requires --diagnostic.')
    sources = {**cfg['source_sha256'], **{n: file_sha256(ROOT/n) for n in (
        'audit_matcher_training_tasks.py', 'diffusion/matcher_task_audit.py', 'probe_token_matcher.py')}}
    for name, digest in sources.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError(f'Original inference source changed: {name}')
    protected.update(cfg['protected_inputs'])
    for path, digest in protected.items():
        if file_sha256(path) != digest:
            raise ValueError(f'Original scientific input changed: {path}')
    data, manifest, baseline, generator, diagnostic = validate_inputs(SimpleNamespace(**cfg['arguments']))
    training = cfg['gradient_training_rows']
    indices = select_training_indices(training, manifest['splits']['train'], args.limit, args.seed)
    sampled = [training[i] for i in indices]
    train_set = set(training)
    max_len = int(generator.hparams.max_length)
    selections, saved_plans = {}, {}
    # Fail on missing or changed saved files before spending GPU time generating.
    for arm in ('frozen', 'finetuned'):
        for mode in ('combined', 'matcher'):
            choice = selection['arms'][arm][mode]
            if choice['epoch'] == 0:
                continue
            path = (directory / choice['checkpoint']).resolve()
            if not path.is_relative_to(directory) or file_sha256(path) != choice['checkpoint_sha256']:
                raise ValueError('Selected matcher checkpoint path or hash differs.')
            protected[str(path)] = choice['checkpoint_sha256']
            if choice['epoch'] not in saved_plans:
                saved_plans[choice['epoch']] = load_original_plan(directory, plans, choice['epoch'], len(training), protected)
            history = read_json(directory / arm / 'history.json')
            entry = next(e for e in history if e['epoch'] == choice['epoch'])
            if entry['negative_plan_sha256'] != plans['sha256'][choice['epoch']-1]:
                raise ValueError('Checkpoint epoch history and original negative plan disagree.')
            saved = read_json(directory / f'assessment_{arm}_{mode}.json')
            if saved['selection'] != choice or [r['row'] for r in saved['rows']] != cfg['assessment_rows']:
                raise ValueError('Original assessment replay rows or selection differ.')
            selections[arm+'/'+mode] = (choice, path, saved)
    if not selections:
        raise ValueError('No trained matcher checkpoint was selected in the original run.')
    output.mkdir(parents=True)
    protocol = {'schema_version': 1, 'training_performed': False, 'weights_promoted': False,
        'new_checkpoint_selection': False, 'test_used': False, 'diagnostic': diagnostic or args.diagnostic,
        'sample_seed': args.seed, 'requested_rows': args.limit, 'sampled_training_rows': sampled,
        'local_training_indices': indices, 'negative_plan_policy': 'saved plan from each checkpoint epoch',
        'matched_policy': 'dataset positive plus first three distinct wrong generated candidates; excluded if fewer than three',
        'generation': {'beam_size': 5, 'length_penalty': .6, 'ema_weights': True},
        'original_configuration_sha256': protected[str(directory / 'configuration.json')],
        'original_selection_sha256': protected[str(directory / 'selection.json')],
        'source_sha256': sources, 'protected_inputs': dict(protected), 'device': args.device,
        'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'limits': ['All newly scored task-comparison rows are original training rows seen by both generator and matcher.',
            'This is an in-sample diagnostic, not validation accuracy or an out-of-fold experiment.',
            'Matched four-choice groups inject the known training positive; these are diagnostic questions, not natural decoder predictions.',
            'Only a prefix of already saved assessment predictions is replayed as a software check; no new holdout model selection.',
            'A task gap is consistent with negative-distribution mismatch but does not separate it from memorization or other causes.']}
    atomic_json(output / 'protocol.json', protocol)
    cache = {}
    def candidate(row):
        if row not in cache:
            cache[row] = stored_candidate(data, row, train_set, max_len)
        return cache[row]
    positives = [candidate(row) for row in sampled]
    natural = []
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    generator.to(args.device).eval().requires_grad_(False)
    with EvaluationProgress(output / 'progress', interval=15) as progress:
        progress.start('Generating fixed candidates for sampled training molecules', len(sampled), 'molecules')
        with torch.inference_mode():
            for index, (row, positive) in enumerate(zip(sampled, positives)):
                pool = valid_candidates(generator.generate(spectra_batch(data, [row], args.device), 5, .6), data['coords'].shape[1])
                if pool:
                    tokenize_selfies([c['selfies'] for c in pool], data['vocabulary'], max_len)
                decoder_scores(pool)
                natural.append({'row': row, 'reference_smiles': positive['smiles'],
                    'reference_formula': positive['formula'], 'candidates': pool})
                if (index+1) % 100 == 0:
                    atomic_json(output / 'generated_training_candidates.json', {'complete': False, 'rows': natural})
                progress.update(index+1)
        atomic_json(output / 'generated_training_candidates.json', {'complete': True, 'rows': natural})
        matched, eligible = matched_generated_groups(natural, positives)
        matched_rows = [sampled[i] for i in eligible]
        baseline_stats = pool_statistics(natural, [decoder_scores(r['candidates']) for r in natural])
        results = {'training_rows': len(sampled), 'matched_rows': len(matched),
            'excluded_fewer_than_three_generated_negatives': len(sampled)-len(matched),
            'natural_decoder': baseline_stats, 'arms': {}, 'limits': protocol['limits']}
        by_row = {r['row']: r for r in baseline['rows']}
        for label, (choice, path, saved) in selections.items():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['configuration_sha256'] != protocol['original_configuration_sha256'] or payload['epoch'] != choice['epoch']:
                raise ValueError('Checkpoint payload differs from its sealed run configuration or epoch.')
            matcher = TokenMatcher(generator.encoder, len(data['vocabulary']), max_len,
                n_heads=int(generator.encoder.spec['n_heads']), n_layers=cfg['arguments']['matcher_layers'],
                dropout=.1, train_encoder=False).to(args.device)
            matcher.load_state_dict(payload['state_dict'], strict=True)
            matcher.eval().requires_grad_(False)
            replay_saved = saved['rows'][:64]
            replay_records = [by_row[r['row']] for r in replay_saved]
            replay_scores = score_pools(matcher, data, [r['row'] for r in replay_records],
                [r['candidates'] for r in replay_records], data['vocabulary'], max_len, args.device,
                cfg['arguments']['batch_size'], progress, label+': replay saved assessment prefix')
            error = validate_replay(replay_records, replay_scores, replay_saved, choice['weight'])
            plan = saved_plans[choice['epoch']]
            original = []
            for i, row, positive in zip(indices, sampled, positives):
                pool = [positive, *[candidate(training[int(j)]) for j in plan[i]]]
                if len({c['smiles'] for c in pool}) != 4:
                    raise ValueError('Original plan contains duplicate or false-negative identities.')
                original.append({'row': row, 'reference_smiles': positive['smiles'],
                    'reference_formula': positive['formula'], 'candidates': pool})
            scored = {}
            for task, records in [('original_training_four_choices', original),
                                  ('natural_generated', natural), ('matched_generated_four_choices', matched)]:
                scores = score_pools(matcher, data, [r['row'] for r in records], [r['candidates'] for r in records],
                    data['vocabulary'], max_len, args.device, cfg['arguments']['batch_size'], progress, label+': '+task) if records else []
                scored[task] = (records, scores, pool_statistics(records, scores))
            original_paired_stats = pool_statistics([original[i] for i in eligible],
                [scored['original_training_four_choices'][1][i] for i in eligible])
            raw_logits = scored['natural_generated'][1]
            combined = [v if choice['weight'] is None else (np.array(decoder_scores(r['candidates'])) + choice['weight']*np.array(v)).tolist()
                        for r,v in zip(natural, raw_logits)]
            summary = {'selection': choice, 'replay_rows': len(replay_saved), 'maximum_replay_score_error': error,
                **{name: value[2] for name,value in scored.items()},
                'original_four_choices_on_matched_rows': original_paired_stats,
                'natural_selected_combination': pool_statistics(natural, combined),
                'paired_task_gap': paired_task_gap(original_paired_stats, scored['matched_generated_four_choices'][2])}
            results['arms'][label] = summary
            atomic_json(output / (label.replace('/', '_')+'_scores.json'), {
                name: [{'record': r, 'logits': v} for r,v in zip(records, scores)]
                for name, (records, scores, stats) in scored.items()})
            atomic_json(output / 'findings.json', results)
            del matcher, payload
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        for path, digest in protected.items():
            if file_sha256(path) != digest:
                raise ValueError(f'Input changed during diagnostic: {path}')
        for name, digest in sources.items():
            if file_sha256(ROOT/name) != digest:
                raise ValueError(f'Source changed during diagnostic: {name}')
        lines = ['# Matcher training-task diagnostic', '',
            f'Sampled training molecules: {len(sampled)}. Matched four-choice rows: {len(matched)}. Excluded: {len(sampled)-len(matched)}.',
            'All these molecules were seen during training. No accuracy improvement is claimed.', '',
            '| Saved checkpoint | Original four choices, matched rows | Generated four choices, same rows | Natural generated matcher Top1 |',
            '|---|---:|---:|---:|']
        for label, value in results['arms'].items():
            lines.append(f"| {label} | {percentage(value['original_four_choices_on_matched_rows']['strict_top1_fraction'])} | {percentage(value['matched_generated_four_choices']['strict_top1_fraction'])} | {percentage(value['natural_generated']['top1_fraction'])} |")
        lines += ['', 'The first two columns require the target to strictly outrank all three alternatives; ties fail.',
            'Their positives use the same stored SELFIES and are inserted for diagnosis. They are not decoder prediction accuracy.',
            f"Natural decoder Top1: {percentage(baseline_stats['top1_fraction'])}; candidate coverage: {percentage(baseline_stats['coverage'])}.",
            'A larger four-choice task gap supports investigating training comparisons. Little gap does not rule out generalization failure on unseen molecules.',
            'Inspect findings.json for matching cross-entropy, same-formula pair comparisons, paired outcomes, coverage and fixed combined scores.', '',
            *protocol['limits'], '', 'No training or model promotion occurred. Return matcher_task_audit_results.zip for review.', '']
        (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        with ZipFile(output / 'matcher_task_audit_results.zip', 'w', ZIP_DEFLATED) as z:
            for p in sorted(output.glob('*.json')):
                z.write(p, p.name)
            z.write(output / 'RESULTS.md', 'RESULTS.md')
        atomic_json(output / 'completion.json', {'complete': True, 'results_sha256': file_sha256(output/'matcher_task_audit_results.zip')})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', default='runs/token_matcher_1426578')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args(argv)
    try:
        with exclusive_report(Path(args.output_dir).with_suffix('.run')):
            result = run(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f'Complete: {args.output_dir}/RESULTS.md\nDownload: {args.output_dir}/matcher_task_audit_results.zip', flush=True)
    return result


if __name__ == '__main__':
    main()
