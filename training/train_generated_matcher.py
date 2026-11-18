"""Controlled full-token IR/molecule matching with development-only selection.

The original generator and its validation candidate pools are immutable inputs.
Two frozen-encoder arms share initialization, anchors and budget; only negatives differ.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from torch.nn import functional as F
from rdkit import rdBase

from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.progress import EvaluationProgress
from diffusion.ranking import canonical_identity
from diffusion.residual_selector import decoder_scores, partition_validation
from diffusion.spectrum import SpectrumStructureModel
from diffusion.spectrum_audit import atomic_json, exclusive_report, row_record, summarize_records, validate_selection
from diffusion.token_matcher import TokenMatcher, identity_contrastive_loss
from diffusion.token_matcher_data import TrainingPairs, spectra_batch, tokenize_selfies
from diffusion.generated_matcher_data import prepare_pairs, generated_plan
from evaluation.evaluate_spectrum import valid_candidates
from scripts.compare_spectrum_pilot import BOOTSTRAP_SEED, paired_metrics, same_json, validate_report

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SOURCES = ("diffusion/encoder.py", "diffusion/model.py", "diffusion/data.py",
                   "diffusion/spectrum.py", "evaluation/evaluate_spectrum.py", "diffusion/spectrum_audit.py")
# Fixed before looking at this experiment's assessment outcomes. None = matcher only.
MIX_WEIGHTS = (0., .1, .3, 1., 3., None)
ARMS = ('original', 'generated')


def state_hash(module):
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def batch_indices(count, batch_size, seed, epoch):
    """Independent RNG; merge a singleton remainder for contrastive alignment."""
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1701])).permutation(count)
    batches = [order[i:i + batch_size] for i in range(0, count, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        remainder = batches.pop()
        batches[-1] = np.concatenate((batches[-1], remainder))
    return batches


def train_epoch(matcher, optimizer, data, pairs, negative_plan, args, epoch, arm, progress):
    matcher.set_encoder_trainable(False)
    matcher.train()
    progress.start(f'{arm}: training epoch {epoch}/{args.epochs}', len(pairs.rows), 'training pairs')
    losses = np.zeros(3, dtype=np.float64)
    seen = 0
    for indices in batch_indices(len(pairs.rows), args.batch_size, args.seed, epoch):
        choices = np.column_stack((pairs.positive_indices[indices], negative_plan[indices]))
        unique, inverse = np.unique(choices, return_inverse=True)
        inverse = torch.as_tensor(inverse.reshape(choices.shape), device=args.device, dtype=torch.long)
        memory, peak_mask = matcher.encode_spectrum(spectra_batch(data, [pairs.rows[i] for i in indices], args.device))
        molecule, mol_mask = matcher.encode_molecule(pairs.tokens[unique].to(args.device))
        positive = inverse[:, 0]
        alignment = matcher.alignment(memory, peak_mask, molecule[positive], mol_mask[positive])
        align_loss = identity_contrastive_loss(alignment, [pairs.identities[i] for i in indices], temperature=.1)
        batch, width = choices.shape
        owners = torch.arange(batch, device=args.device).repeat_interleave(width)
        selected = inverse.flatten()
        logits = matcher.match(memory[owners], peak_mask[owners], molecule[selected], mol_mask[selected]).reshape(batch, width)
        match_loss = F.cross_entropy(logits, torch.zeros(batch, dtype=torch.long, device=args.device))
        loss = match_loss + args.alignment_weight * align_loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss; no results may be promoted.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses += np.array([loss.item(), match_loss.item(), align_loss.item()]) * batch
        seen += batch
        progress.update(seen)
    return {'epoch': epoch, 'training_rows': seen, 'loss': float(losses[0] / seen),
            'matching_loss': float(losses[1] / seen), 'alignment_loss': float(losses[2] / seen),
            'encoder_trainable': matcher.encoder.training and any(p.requires_grad for p in matcher.encoder.parameters())}


@torch.inference_mode()
def score_pools(matcher, data, rows, pools, vocabulary, max_len, device, batch_size, progress, label):
    """Only spectra and predicted candidates enter the scorer; no references."""
    if len(rows) != len(pools):
        raise ValueError('Candidate pool/row count mismatch.')
    matcher.eval()
    result = []
    progress.start(label, len(rows), 'molecules')
    for start in range(0, len(rows), batch_size):
        selected_rows = rows[start:start + batch_size]
        selected_pools = pools[start:start + batch_size]
        memory, mask = matcher.encode_spectrum(spectra_batch(data, selected_rows, device))
        owners, candidates = [], []
        for i, pool in enumerate(selected_pools):
            owners.extend([i] * len(pool))
            candidates.extend(c['selfies'] for c in pool)
        values = []
        # Bound inference memory even if a future report contains larger pools.
        for chunk in range(0, len(candidates), batch_size):
            tokens = tokenize_selfies(candidates[chunk:chunk + batch_size], vocabulary, max_len).to(device)
            molecule, mol_mask = matcher.encode_molecule(tokens)
            owner = torch.tensor(owners[chunk:chunk + batch_size], dtype=torch.long, device=device)
            logits = matcher.match(memory[owner], mask[owner], molecule, mol_mask)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite candidate matching logits.')
            values.extend(logits.cpu().double().tolist())
        offset = 0
        for pool in selected_pools:
            result.append(values[offset:offset + len(pool)])
            offset += len(pool)
        progress.update(start + len(selected_rows))
    return result


def orders_for(pools, scores, weight):
    if len(pools) != len(scores):
        raise ValueError('Score/pool count mismatch.')
    orders = []
    for pool, logits in zip(pools, scores):
        if len(logits) != len(pool) or not np.isfinite(logits).all():
            raise ValueError('Missing or nonfinite matcher scores.')
        base = decoder_scores(pool)
        if weight == 0:
            orders.append(list(range(len(pool))))
            continue
        combined = logits if weight is None else [a + weight * b for a, b in zip(base, logits)]
        orders.append(sorted(range(len(pool)), key=lambda i: -combined[i]))
    return orders


def correct_count(records, orders):
    return sum(bool(order) and r['candidates'][order[0]]['smiles'] == r['reference_smiles']
               for r, order in zip(records, orders))


def summarize_with_coverage(records):
    summary = summarize_records(records)
    present = sum(r['top5_identity'] for r in records)
    correct = sum(r['top1_identity'] for r in records)
    summary.update(target_present_rows=present,
                   conditional_top1_given_present=correct / present if present else None)
    return summary


def select_weight(records, scores):
    """Stable ties favor baseline, then smaller weights. All denominators retained."""
    pools = [r['candidates'] for r in records]
    options = [{'weight': w, 'correct': correct_count(records, orders_for(pools, scores, w))} for w in MIX_WEIGHTS]
    best = max(options, key=lambda x: x['correct'])
    return dict(best), options


def prediction_records(records, scores, weight):
    orders = orders_for([r['candidates'] for r in records], scores, weight)
    result = []
    for record, logits, order in zip(records, scores, orders):
        candidates = [{**record['candidates'][i], 'decoder_rank': i + 1, 'token_match_score': logits[i]} for i in order]
        result.append(row_record(record['row'], record['reference_smiles'], record['reference_formula'], candidates))
    return result


def save_weights(path, matcher, epoch, configuration_hash):
    temporary = path.with_suffix('.tmp')
    torch.save({'schema_version': 1, 'spec': matcher.spec, 'epoch': epoch,
                'configuration_sha256': configuration_hash,
                'state_dict': {k: v.detach().cpu().clone() for k, v in matcher.state_dict().items()}}, temporary)
    temporary.replace(path)


def validate_inputs(args):
    print('Verifying original checkpoint, data, split, candidates and inference source hashes...', flush=True)
    baseline = json.loads(Path(args.baseline_report).read_text(encoding='utf-8'))
    checked = validate_report(baseline, 'baseline candidate pool')
    provenance = checked['provenance']
    for key, value in [('beam_size', 5), ('length_penalty', .6), ('ema_weights', True),
                       ('decode_protocol', 'legacy_generate_v1_valid_unique_candidates'), ('spectrum_control', 'none')]:
        if provenance.get(key) != value:
            raise ValueError(f'Unsupported baseline {key}; require {value!r}.')
    data = load_native_npz(args.data)
    manifest = validate_split_manifest(json.loads(Path(args.splits).read_text(encoding='utf-8')), data['molecule_ids'], args.data)
    for key, path in [('data_sha256', args.data), ('split_sha256', args.splits), ('checkpoint_sha256', args.checkpoint)]:
        if baseline[key] != file_sha256(path):
            raise ValueError(f'Baseline report {key} differs from supplied input.')
    rows = validate_selection(baseline['selection'], manifest, baseline['split_sha256'], 'val')
    if set(rows) != set(manifest['splits']['val']):
        raise ValueError('A completed full-validation candidate report is required.')
    for record, row in zip(baseline['rows'], rows):
        if record['reference_smiles'] != canonical_identity(data['molecule_ids'][row]):
            raise ValueError('Validation reference differs from dataset identity.')
        if valid_candidates(record['candidates'], data['coords'].shape[1]) != record['candidates']:
            raise ValueError('Candidate SELFIES, identity or formula is inconsistent.')
        decoder_scores(record['candidates'])
    for name in FEATURE_SOURCES:
        if provenance['source_sha256'][name] != file_sha256(ROOT / name):
            raise ValueError(f'Baseline inference source changed: {name}; regenerate report with matching code.')
    model = SpectrumStructureModel.load_from_checkpoint(args.checkpoint)
    if file_sha256(args.checkpoint) != baseline['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during loading.')
    for key, expected in [('data_sha256', baseline['data_sha256']), ('split_sha256', baseline['split_sha256']),
                          ('preprocessing', data.get('preprocessing'))]:
        if not same_json(model.run_metadata.get(key), expected) or not same_json(provenance[key], expected):
            raise ValueError(f'Checkpoint/report {key} differs from dataset.')
    if list(model.hparams.vocabulary) != data['vocabulary'] or provenance['vocabulary'] != data['vocabulary']:
        raise ValueError('Checkpoint/report/dataset vocabulary mismatch.')
    if model.run_metadata.get('spectrum_control', 'none') != 'none' or not model.use_ema_weights():
        raise ValueError('Uncontrolled EMA checkpoint required.')
    diagnostic = bool(model.run_metadata.get('diagnostic') or baseline.get('checkpoint_training_diagnostic'))
    if diagnostic and not args.diagnostic:
        raise ValueError('Diagnostic checkpoint requires --diagnostic; software checks only.')
    model.eval().requires_grad_(False)
    return data, manifest, baseline, model, diagnostic or args.diagnostic


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Output directory already exists. Keep prior artifacts and choose a fresh output directory.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; submit this experiment with one GPU.')
    data, manifest, baseline, generator, diagnostic = validate_inputs(args)
    train_rows = manifest['splits']['train']
    max_len = int(generator.hparams.max_length)
    print(f'Checking stored molecular tokens and preparing {len(train_rows):,} training-only identities...', flush=True)
    pairs = TrainingPairs(data, train_rows, train_rows, max_len=max_len,
        on_progress=lambda done, total: print(f'Training identity/token checks: {done:,}/{total:,}', flush=True))
    dev_rows, assessment_rows = partition_validation(baseline['selected_rows'], args.dev_size, args.partition_seed)
    by_row = {r['row']: r for r in baseline['rows']}
    dev = [by_row[i] for i in dev_rows]
    assessment = [by_row[i] for i in assessment_rows]
    # Preflight candidate encodings without training or scoring held-out examples.
    for record in baseline['rows']:
        if record['candidates']:
            tokenize_selfies([c['selfies'] for c in record['candidates']], data['vocabulary'], max_len)
    source_names = (*FEATURE_SOURCES, 'diffusion/ranking.py', 'diffusion/progress.py',
        'diffusion/residual_selector.py', 'diffusion/compatibility.py', 'diffusion/token_matcher.py',
        'diffusion/token_matcher_data.py', 'training/train_token_matcher.py', 'training/train_generated_matcher.py', 'diffusion/generated_matcher_data.py', 'scripts/compare_spectrum_pilot.py')
    protected = {str(Path(p).resolve()): file_sha256(p) for p in (args.checkpoint, args.data, args.splits, args.baseline_report)}
    output.mkdir(parents=True)
    full_pairs = pairs
    pairs, generated_indices = prepare_pairs(data, full_pairs, generator, args, output)
    original_encoder_hash = state_hash(generator.encoder)
    cfg = {'schema_version': 1, 'arguments': vars(args), 'diagnostic': diagnostic,
           'protected_inputs': protected, 'source_sha256': {n: file_sha256(ROOT / n) for n in source_names},
           'gradient_training_rows': pairs.rows, 'negative_identity_source_rows': train_rows, 'development_rows': dev_rows, 'assessment_rows': assessment_rows,
           'test_used': False, 'held_out_labels_used_for_training': False, 'weights_promoted': False,
           'encoder_start_sha256': original_encoder_hash,
           'protocol': {'arms': list(ARMS), 'epochs_per_arm': args.epochs, 'warmup_epochs': args.warmup_epochs,
              'mix_weights': list(MIX_WEIGHTS), 'minimum_dev_gain_fraction': args.minimum_dev_gain,
              'negative_plan': 'original: random/formula/fingerprint; generated: first three distinct wrong decoder candidates, original-plan fallback; same sampled anchors'
              , 'frozen_encoder_both_arms': True, 'candidate_bank_sha256': file_sha256(output / 'candidate_bank.npy'), 'generated_pools_sha256': file_sha256(output / 'generated_training_candidates.json'),
              'loss': '4-way matching CE + alignment_weight * symmetric identity-aware contrastive loss (temperature 0.1)',
              'selection': 'development Top1; stable ties prefer baseline/lower weight/earlier epoch; separate pure-matcher checkpoint',
              'assessment_used_for_selection': False, 'original_generator_updated': False},
           'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'rdkit': rdBase.rdkitVersion,
                        'cuda': torch.version.cuda, 'torch_num_threads': torch.get_num_threads(),
                        'float32_matmul_precision': torch.get_float32_matmul_precision(),
                        'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                        'device': args.device, 'gpu_name': torch.cuda.get_device_name() if args.device == 'cuda' else None},
           'limitations': ['Development and assessment partition previously inspected validation, not fresh holdouts.',
              'Both arms share frozen pretrained encoder and fresh identical heads; this is not continuation of old matcher weights.',
              'Only a seeded training subset is used for gradient anchors; original negatives draw from all training identities.',
              'The generator already saw all mining anchors; these are not out-of-fold negatives.',
              'Generated candidates with fewer than three wrong identities use original-plan fallback; inspect counts.',
              'Fixed candidate coverage bounds Top1; missing molecules cannot be recovered by reordering.',
              'Single training seed and multiple development choices; paired assessment intervals are exploratory.',
              'Exact graph identity ignores stereochemistry; no 3D or experimental-IR accuracy is established.']}
    atomic_json(output / 'configuration.json', cfg)
    configuration_hash = file_sha256(output / 'configuration.json')
    selections = {}
    with EvaluationProgress(output / 'progress', interval=args.progress_interval) as progress:
        plan_hashes = {arm: [] for arm in ARMS}
        plan_stats = []
        for epoch in range(1, args.epochs + 1):
            original = full_pairs.negative_plan(epoch, args.seed, progress=progress)[pairs.positive_indices]
            generated, stats = generated_plan(original, generated_indices, pairs.positive_indices, pairs.bank_identities)
            plan_stats.append(stats)
            for arm, plan in [('original', original), ('generated', generated)]:
                path = output / f'{arm}_negative_plan_{epoch:02d}.npy'
                np.save(path, plan, allow_pickle=False)
                plan_hashes[arm].append(file_sha256(path))
        atomic_json(output / 'negative_plans.json', {'indices_into_candidate_bank': True,
            'sha256': plan_hashes, 'generated_statistics': plan_stats, 'same_anchors_and_budget': True})
        base_correct = correct_count(dev, [list(range(len(r['candidates']))) for r in dev])
        initial_hash = None
        for arm in ARMS:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            matcher = TokenMatcher(generator.encoder, len(data['vocabulary']), max_len,
                n_heads=int(generator.encoder.spec['n_heads']), n_layers=args.matcher_layers,
                dropout=.1, train_encoder=False).to(args.device)
            start_hash = state_hash(matcher)
            if initial_hash is not None and initial_hash != start_hash:
                raise ValueError('Arms did not start from identical parameters.')
            initial_hash = start_hash
            directory = output / arm
            directory.mkdir()
            encoder_parameters = list(matcher.encoder.parameters())
            other_parameters = [p for n, p in matcher.named_parameters() if not n.startswith('encoder.')]
            optimizer = torch.optim.AdamW([{'params': other_parameters, 'lr': args.lr},
                {'params': encoder_parameters, 'lr': args.encoder_lr}], weight_decay=.01)
            history = []
            best = {'epoch': 0, 'weight': 0., 'correct': base_correct}
            pure = {'epoch': 0, 'weight': None, 'correct': -1}
            for epoch in range(1, args.epochs + 1):
                path = output / f'{arm}_negative_plan_{epoch:02d}.npy'
                if file_sha256(path) != plan_hashes[arm][epoch - 1]:
                    raise ValueError('Shared training negative plan changed.')
                entry = train_epoch(matcher, optimizer, data, pairs, np.load(path, allow_pickle=False), args, epoch, arm, progress)
                scores = score_pools(matcher, data, dev_rows, [r['candidates'] for r in dev], data['vocabulary'],
                    max_len, args.device, args.batch_size, progress, f'{arm}: development epoch {epoch}')
                choice, choices = select_weight(dev, scores)
                if choice['correct'] > best['correct']:
                    best = {'epoch': epoch, **choice}
                    save_weights(directory / 'best_combined.pt', matcher, epoch, configuration_hash)
                pure_correct = choices[-1]['correct']
                if pure_correct > pure['correct']:
                    pure = {'epoch': epoch, 'weight': None, 'correct': pure_correct}
                    save_weights(directory / 'best_matcher.pt', matcher, epoch, configuration_hash)
                entry.update(development_options=choices, negative_plan_sha256=plan_hashes[arm][epoch - 1])
                history.append(entry)
                atomic_json(directory / 'history.json', history)
                print(f'{arm} epoch {epoch}: loss={entry["loss"]:.5f}; dev best={choice["correct"]}/{len(dev)}; '
                      f'pure={pure_correct}/{len(dev)}', flush=True)
            final_encoder_hash = state_hash(matcher.encoder)
            if final_encoder_hash != original_encoder_hash:
                raise ValueError('Frozen encoder parameters changed.')
            if best['correct'] - base_correct < max(1, math.ceil(args.minimum_dev_gain * len(dev))):
                best = {'epoch': 0, 'weight': 0., 'correct': base_correct}
            best['baseline_retained'] = best['epoch'] == 0
            for mode, choice in [('combined', best), ('matcher', pure)]:
                if choice['epoch']:
                    choice['checkpoint'] = f'{arm}/best_{mode}.pt'
                    choice['checkpoint_sha256'] = file_sha256(output / choice['checkpoint'])
            selections[arm] = {'combined': best, 'matcher': pure, 'initial_state_sha256': start_hash,
                               'final_encoder_sha256': final_encoder_hash}
            del optimizer, matcher
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        # The winning arm is also selected on development, never on assessment.
        winning_arm = max(ARMS, key=lambda a: selections[a]['combined']['correct'])
        selected = {'arm': winning_arm, **selections[winning_arm]['combined']}
        selection = {'arms': selections, 'selected': selected, 'assessment_used_for_selection': False}
        atomic_json(output / 'selection.json', selection)
        selection_hash = file_sha256(output / 'selection.json')
        assessment_before = [row_record(r['row'], r['reference_smiles'], r['reference_formula'], r['candidates']) for r in assessment]
        results = {}
        combined_records = {}
        for arm in ARMS:
            results[arm] = {}
            for mode in ('combined', 'matcher'):
                choice = selections[arm][mode]
                if choice['epoch'] == 0:
                    records = assessment_before
                else:
                    path = output / choice['checkpoint']
                    if file_sha256(path) != choice['checkpoint_sha256']:
                        raise ValueError('Selected weights changed before assessment.')
                    payload = torch.load(path, map_location='cpu', weights_only=True)
                    matcher = TokenMatcher(generator.encoder, len(data['vocabulary']), max_len,
                        n_heads=int(generator.encoder.spec['n_heads']), n_layers=args.matcher_layers,
                        dropout=.1, train_encoder=False).to(args.device)
                    matcher.load_state_dict(payload['state_dict'], strict=True)
                    scores = score_pools(matcher, data, assessment_rows, [r['candidates'] for r in assessment],
                        data['vocabulary'], max_len, args.device, args.batch_size, progress, f'assessment: {arm}/{mode}, selection frozen')
                    records = prediction_records(assessment, scores, choice['weight'])
                    del matcher
                    if file_sha256(path) != choice['checkpoint_sha256']:
                        raise ValueError('Selected checkpoint changed during assessment.')
                if any(a['top5_identity'] != b['top5_identity'] for a, b in zip(assessment_before, records)):
                    raise ValueError('Fixed candidate coverage changed.')
                atomic_json(output / f'assessment_{arm}_{mode}.json', {'selection': choice, 'rows': records})
                results[arm][mode] = {'summary': summarize_with_coverage(records),
                    'paired': paired_metrics(assessment_before, records, BOOTSTRAP_SEED)}
                if mode == 'combined':
                    combined_records[arm] = records
        for path, sha in protected.items():
            if file_sha256(path) != sha:
                raise ValueError(f'Protected scientific input changed: {path}')
        for filename, key in [('candidate_bank.npy', 'candidate_bank_sha256'),
                              ('generated_training_candidates.json', 'generated_pools_sha256')]:
            if file_sha256(output / filename) != cfg['protocol'][key]:
                raise ValueError('Training candidate bank changed during experiment.')
        for name, sha in cfg['source_sha256'].items():
            if file_sha256(ROOT / name) != sha:
                raise ValueError(f'Experiment source changed: {name}')
        if state_hash(generator.encoder) != original_encoder_hash or file_sha256(output / 'selection.json') != selection_hash:
            raise ValueError('Original encoder or sealed selection changed.')
        comparison = {'diagnostic': diagnostic, 'weights_promoted': False, 'selection': selection,
            'assessment_rows': len(assessment), 'baseline': summarize_with_coverage(assessment_before),
            'arms': results, 'selected': results[winning_arm]['combined'],
            'generated_vs_original_combined': paired_metrics(combined_records['original'], combined_records['generated'], BOOTSTRAP_SEED),
            'limitations': cfg['limitations']}
        atomic_json(output / 'comparison.json', comparison)
        lines = ['# Decoder-negative token matcher controlled pilot', '',
            f'Diagnostic: {diagnostic}. Development: {len(dev)}; assessment: {len(assessment)}.',
            'All values below use the same assessment rows. Test was not evaluated.', '',
            '| Method | Top1 | Top5 | Formula |', '|---|---:|---:|---:|']
        summaries = [('decoder baseline', comparison['baseline'])] + [(f'{a}/{m}', results[a][m]['summary']) for a in ARMS for m in ('combined', 'matcher')]
        for name, summary in summaries:
            lines.append('| ' + name + ' | ' + ' | '.join(f'{100*summary[k+"_fraction"]:.2f}%' for k in ('top1_identity', 'top5_identity', 'top1_formula')) + ' |')
        delta = comparison['selected']['paired']['metrics']['top1_identity']
        contrast = comparison['generated_vs_original_combined']['metrics']['top1_identity']
        lines.extend(['', f'Generated-negative versus original-negative control: {contrast["delta_percentage_points"]:+.2f} pp; exploratory paired 95% interval: {contrast["paired_bootstrap_ci95_percentage_points"]}.',
            'Both arms use the same sampled training anchors, fresh head initialization, frozen encoder and update budget.',
            'Generated negatives use decoder order, not learned matcher-score mining. See negative_plans.json for fallback counts.'])
        lines.extend(['', f'Development-selected arm: {winning_arm}; epoch: {selected["epoch"]}; weight: {selected["weight"]}.',
            f'Selected assessment Top1 change: {delta["delta_percentage_points"]:+.2f} pp; exploratory paired 95% interval: {delta["paired_bootstrap_ci95_percentage_points"]}.',
            'Matcher-only uses its separately development-selected checkpoint. Combined includes a decoder-only fallback.',
            'No weights promoted. Review paired results before deploying. This experiment does not promise 90% accuracy.', '', *cfg['limitations'], ''])
        (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        with ZipFile(output / 'generated_matcher_results.zip', 'w', ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*.json')):
                if 'progress' not in path.parts:
                    archive.write(path, path.relative_to(output).as_posix())
            archive.write(output / 'RESULTS.md', 'RESULTS.md')
        atomic_json(output / 'completion.json', {'complete': True, 'results_zip_sha256': file_sha256(output / 'generated_matcher_results.zip')})
    print(f'Complete: {output / "RESULTS.md"}\nDownload: {output / "generated_matcher_results.zip"}', flush=True)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'baseline_report', 'output_dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--data', default='data/processed/squirl.npz')
    parser.add_argument('--splits', default='data/processed/splits.json')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    for name, default in [('epochs', 10), ('warmup_epochs', 1), ('batch_size', 64), ('dev_size', 2000),
                          ('training_limit', 5000), ('mining_seed', 20261112), ('seed', 78438379), ('partition_seed', 20261110), ('matcher_layers', 2)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('lr', 1e-4), ('encoder_lr', 1e-5), ('alignment_weight', .1),
                          ('minimum_dev_gain', .005), ('progress_interval', 15.)]:
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args(argv)
    if (min(args.epochs, args.dev_size, args.matcher_layers) < 1 or args.training_limit < 2 or args.mining_seed < 0 or args.batch_size < 2
            or not 0 <= args.warmup_epochs < args.epochs or min(args.seed, args.partition_seed) < 0
            or any(not math.isfinite(v) for v in (args.lr, args.encoder_lr, args.alignment_weight, args.minimum_dev_gain, args.progress_interval))
            or min(args.lr, args.encoder_lr) <= 0 or min(args.alignment_weight, args.minimum_dev_gain, args.progress_interval) < 0
            or args.minimum_dev_gain > 1):
        parser.error('Invalid training budget, warmup, seed, learning rate or selection settings.')
    try:
        with exclusive_report(Path(args.output_dir).with_suffix('.run')):
            return run(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
