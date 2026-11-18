"""Development-selected bounded decoder-score correction experiment."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from rdkit import rdBase
from diffusion.compatibility import cache_spectrum_features
from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.ranking import canonical_identity
from diffusion.progress import EvaluationProgress
from diffusion.spectrum import SpectrumStructureModel
from diffusion.spectrum_audit import atomic_json, exclusive_report, row_record, summarize_records, validate_selection
from diffusion.residual_selector import (CandidateBatch, decoder_scores, targets_for, partition_validation,
    corrections, ordered_candidates, fit_with_development)
from evaluation.evaluate_spectrum import valid_candidates
from training.train_hard_selector import validate_mined, FEATURE_SOURCES
from scripts.compare_spectrum_pilot import BOOTSTRAP_SEED, paired_metrics, same_json, validate_report
ROOT = Path(__file__).resolve().parents[1]


def run(args):
    print('Verifying checkpoint, saved candidate pools, split membership and source hashes...', flush=True)
    output = Path(args.output_dir)
    baseline_bytes = Path(args.baseline_report).read_bytes()
    baseline = json.loads(baseline_bytes.decode("utf-8"))
    checked = validate_report(baseline, "baseline candidate pool")
    if any(checked['provenance'].get(k) != v for k, v in
           [('beam_size', 5), ('length_penalty', .6), ('ema_weights', True),
            ('decode_protocol', 'legacy_generate_v1_valid_unique_candidates')]):
        raise ValueError('This controlled experiment requires EMA, legacy beam 5 and length penalty 0.6.')
    if checked["provenance"]["spectrum_control"] != "none":
        raise ValueError("Compatibility training requires an uncontrolled baseline candidate report.")
    data = load_native_npz(args.data)
    manifest = validate_split_manifest(json.loads(Path(args.splits).read_text(encoding="utf-8")), data["molecule_ids"], args.data)
    split_hash, checkpoint_hash = file_sha256(args.splits), file_sha256(args.checkpoint)
    if any(baseline[key] != value for key, value in (("data_sha256", manifest["data_sha256"]),
            ("split_sha256", split_hash), ("checkpoint_sha256", checkpoint_hash))):
        raise ValueError("Baseline report does not match checkpoint/data/split hashes.")
    val_rows = validate_selection(baseline["selection"], manifest, split_hash, "val")
    if any(record["reference_smiles"] != canonical_identity(data["molecule_ids"][row])
           for record, row in zip(checked["rows"], val_rows)):
        raise ValueError("Baseline report references differ from current validation identities.")
    for name in FEATURE_SOURCES:
        if checked["provenance"]["source_sha256"][name] != file_sha256(ROOT / name):
            raise ValueError(f"Baseline encoder/decoder/evaluation source differs: {name}; regenerate the baseline report.")
    model = SpectrumStructureModel.load_from_checkpoint(args.checkpoint)
    if file_sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError("Baseline checkpoint changed during loading.")
    for key, expected in (("data_sha256", manifest["data_sha256"]), ("split_sha256", split_hash),
                          ("preprocessing", data.get("preprocessing"))):
        if not same_json(model.run_metadata.get(key), expected) or not same_json(checked["provenance"][key], expected):
            raise ValueError(f"Checkpoint/report {key} differs from the data.")
    if list(model.hparams.vocabulary) != data["vocabulary"] or checked["provenance"]["vocabulary"] != data["vocabulary"]:
        raise ValueError("Vocabulary mismatch.")
    if model.run_metadata.get("spectrum_control", "none") != "none" or not model.use_ema_weights():
        raise ValueError("Frozen encoder requires an uncontrolled baseline with EMA weights.")
    diagnostic = bool(model.run_metadata.get("diagnostic") or baseline.get("checkpoint_training_diagnostic"))
    if diagnostic and not args.diagnostic:
        raise ValueError("Diagnostic checkpoint rejected; --diagnostic permits software testing only.")
    diagnostic = diagnostic or args.diagnostic
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; submit the experiment with one GPU.")
    model.to(args.device).eval().requires_grad_(False)
    if set(val_rows) != set(manifest['splits']['val']):
        raise ValueError('Use the completed full-validation report, not a pilot subset.')
    mining_dir = Path(args.mining_dir)
    mining_config_path, mining_path = mining_dir / 'configuration.json', mining_dir / 'training_candidates.json'
    mining_config = json.loads(mining_config_path.read_text())
    mining = json.loads(mining_path.read_text())
    if mining_config.get('held_out_labels_used_for_training') is not False or mining_config.get('beam_size') != 5:
        raise ValueError('Require the verified training-only beam-5 mining cache.')
    if mining_config.get('diagnostic') and not args.diagnostic:
        raise ValueError('Diagnostic mining cache requires --diagnostic; it is not scientific evidence.')
    diagnostic = diagnostic or bool(mining_config.get('diagnostic'))
    if mining.get('configuration_sha256') != file_sha256(mining_config_path):
        raise ValueError('Saved mining/configuration hash mismatch.')
    for key, expected in [('data_sha256', manifest['data_sha256']), ('split_sha256', split_hash),
                          ('checkpoint_sha256', checkpoint_hash)]:
        if mining_config.get(key) != expected:
            raise ValueError(f'Saved mining {key} differs from current inputs.')
    if (not isinstance(mining_config.get('source_sha256'), dict)
            or not set(FEATURE_SOURCES).issubset(mining_config['source_sha256'])):
        raise ValueError('Mining source provenance is missing.')
    for name, sha in mining_config['source_sha256'].items():
        if file_sha256(ROOT / name) != sha:
            raise ValueError(f'Mining source changed: {name}; do not mix incompatible caches.')
    train_rows = mining_config['training_rows']
    if (len(set(train_rows)) != len(train_rows) or not set(train_rows) <= set(manifest['splits']['train'])
            or len(mining['rows']) != len(train_rows)):
        raise ValueError('Mining must be a complete, unique training-only selection.')
    expected_train = np.random.default_rng(mining_config['arguments']['seed']).permutation(
        manifest['splits']['train'])[:mining_config['arguments']['limit']].tolist()
    if train_rows != expected_train:
        raise ValueError('Mining rows differ from their saved seeded training selection.')
    validate_mined(mining['rows'], train_rows, data)
    for record in baseline['rows']:
        if valid_candidates(record['candidates'], data['coords'].shape[1]) != record['candidates']:
            raise ValueError('Original validation candidate SELFIES/identity/formula are inconsistent.')
        decoder_scores(record['candidates'])
    development_rows, assessment_rows = partition_validation(val_rows, args.dev_size, args.partition_seed)
    by_row = {r['row']: r for r in baseline['rows']}
    development = [by_row[row] for row in development_rows]
    assessment = [by_row[row] for row in assessment_rows]
    names = set((*FEATURE_SOURCES, 'diffusion/compatibility.py', 'diffusion/hard_selector.py',
                 'diffusion/residual_selector.py', 'diffusion/ranking.py', 'diffusion/progress.py',
                 'training/train_residual_selector.py', 'training/train_hard_selector.py', 'scripts/compare_spectrum_pilot.py'))
    cfg = {'schema_version': 1, 'arguments': {k: v for k, v in vars(args).items() if k != 'resume'},
           'data_sha256': manifest['data_sha256'], 'split_sha256': split_hash,
           'checkpoint_sha256': checkpoint_hash, 'baseline_report_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
           'mining_config_sha256': file_sha256(mining_config_path), 'mining_sha256': file_sha256(mining_path),
           'source_sha256': {name: file_sha256(ROOT / name) for name in sorted(names)},
           'gradient_training_rows': train_rows, 'development_rows': development_rows, 'assessment_rows': assessment_rows,
           'diagnostic': diagnostic, 'test_used': False,
           'validation_labels_used_for_checkpoint_selection': True,
           'protocol': {'score': 'decoder log_probability / (SELFIES token count + BOS + EOS)^0.6 + bounded residual',
                        'correction_bound': .25, 'hidden': 32, 'dropout': .2, 'weight_decay': .01,
                        'minimum_development_gain_fraction': .005,
                        'selection': 'best development Top1 meeting gain threshold; ties retain earlier checkpoint; baseline epoch 0 allowed',
                        'early_stop': 'patience counts epochs without strict improvement over best observed development correct count',
                        'train_target_injection': False, 'assessment_used_for_selection': False},
           'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'rdkit': rdBase.rdkitVersion,
                        'cuda': torch.version.cuda},
           'limitations': ['Development and assessment are partitions of previously inspected validation data, not fresh independent holdouts.',
                           'The generator was trained on mining inputs; candidate-distribution mismatch can persist.',
                           'Development selection does not guarantee assessment improvement. Test remains untouched.']}
    if output.exists():
        if not args.resume or not (output / 'configuration.json').is_file():
            raise ValueError('Output exists; use a fresh directory or --resume for a matching interrupted run.')
        if not same_json(json.loads((output / 'configuration.json').read_text()), cfg):
            raise ValueError('Resume input/source/configuration mismatch.')
        if (output / 'completion.json').exists():
            raise ValueError('Run already completed; read RESULTS.md.')
    else:
        output.mkdir(parents=True)
        atomic_json(output / 'configuration.json', cfg)
    with EvaluationProgress(output / 'progress', interval=args.progress_interval) as progress:
        train_features = cache_spectrum_features(model.encoder, data, train_rows, args.device, args.batch_size,
                                                progress, 'Caching training-only features')
        train_batch = CandidateBatch(train_features, [r['candidates'] for r in mining['rows']])
        train_targets = targets_for(mining['rows'])
        dev_features = cache_spectrum_features(model.encoder, data, development_rows, args.device, args.batch_size,
                                              progress, 'Caching development features')
        dev_batch = CandidateBatch(dev_features, [r['candidates'] for r in development])
        dev_targets = targets_for(development)
        def save_checkpoint(name, head, epoch):
            temp = output / (name + '.tmp')
            torch.save({'schema_version': 1, 'spec': head.spec,
                        'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        'epoch': epoch, 'configuration': cfg}, temp)
            temp.replace(output / name)
        def save_epoch(history, head, selected_epoch):
            atomic_json(output / 'history.json', history)
            save_checkpoint('last.pt', head, history[-1]['epoch'])
            if history[-1]['epoch'] == selected_epoch:
                save_checkpoint('selected.pt', head, selected_epoch)
        head, history, selection = fit_with_development(train_batch, train_targets, dev_batch, dev_targets,
            device=args.device, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            seed=args.seed, lr=args.lr, progress=progress, on_epoch=save_epoch)
        save_checkpoint('selected.pt', head, selection['epoch'])
        # Freeze selection on disk before assessment feature extraction/scoring.
        selection['selected_checkpoint_sha256'] = file_sha256(output / 'selected.pt')
        atomic_json(output / 'selection.json', selection)
        assessment_features = cache_spectrum_features(model.encoder, data, assessment_rows, args.device,
            args.batch_size, progress, 'Assessing frozen selected checkpoint: feature extraction')
        assessment_batch = CandidateBatch(assessment_features, [r['candidates'] for r in assessment])
        outputs = {}
        for label, batch, records in [('development', dev_batch, development), ('assessment', assessment_batch, assessment)]:
            progress.start(f'Scoring {label} with frozen selected checkpoint', len(records), 'molecules')
            adjustment = corrections(head, batch, args.device, args.batch_size)
            reranked = [row_record(r['row'], r['reference_smiles'], r['reference_formula'],
                                  ordered_candidates(batch, adjustment, i)) for i, r in enumerate(records)]
            before = [row_record(r['row'], r['reference_smiles'], r['reference_formula'], r['candidates']) for r in records]
            if any(a['top5_identity'] != b['top5_identity'] or a['topk_identity'] != b['topk_identity']
                   for a, b in zip(before, reranked)):
                raise ValueError('Candidate coverage changed.')
            if selection['baseline_retained'] and any([c['smiles'] for c in a['candidates']] !=
                    [c['smiles'] for c in b['candidates']] for a, b in zip(before, reranked)):
                raise ValueError('Epoch-zero fallback did not preserve baseline ordering.')
            outputs[label] = {'baseline': summarize_records(before), 'selected': summarize_records(reranked),
                              'paired': paired_metrics(before, reranked, BOOTSTRAP_SEED),
                              'baseline_rows': before, 'selected_rows': reranked}
            atomic_json(output / f'{label}_predictions.json', {'scope': label,
                'used_for_checkpoint_selection': label == 'development', **outputs[label]})
            progress.update(len(records))
        if file_sha256(output / 'selected.pt') != selection['selected_checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed during assessment.')
        for path, sha in [(args.data, cfg['data_sha256']), (args.splits, split_hash),
                          (args.checkpoint, checkpoint_hash), (args.baseline_report, cfg['baseline_report_sha256']),
                          (mining_path, cfg['mining_sha256']), (mining_config_path, cfg['mining_config_sha256']),
                          *((ROOT / name, sha) for name, sha in cfg['source_sha256'].items())]:
            if file_sha256(path) != sha:
                raise ValueError(f'Input/source changed during run: {path}')
        comparison = {'selection': selection, 'diagnostic': diagnostic, 'weights_promoted': False,
                      'limitations': cfg['limitations'],
                      **{label: {k: v for k, v in result.items() if not k.endswith('_rows')}
                         for label, result in outputs.items()}}
        atomic_json(output / 'comparison.json', comparison)
        lines = ['# Development-selected decoder residual experiment', '',
                 f"Selected epoch: {selection['epoch']}. Baseline retained: {selection['baseline_retained']}.",
                 f"Development size: {len(development_rows)}; assessment size: {len(assessment_rows)}; diagnostic: {diagnostic}.", '',
                 '| Partition | Method | Top1 | Top5 | Formula |', '|---|---|---:|---:|---:|']
        for label in ('development', 'assessment'):
            for method in ('baseline', 'selected'):
                s = comparison[label][method]
                lines.append('| ' + label + ' | ' + method + ' | ' + ' | '.join(
                    f"{100*s[k+'_fraction']:.2f}%" for k in ('top1_identity', 'top5_identity', 'top1_formula')) + ' |')
        m = comparison['assessment']['paired']['metrics']['top1_identity']
        lines += ['', f"Assessment Top1 change: {m['delta_percentage_points']:+.2f} pp; paired exploratory 95% interval: {m['paired_bootstrap_ci95_percentage_points']}.",
                  f"Assessment gains/losses: {m['gained_rows']}/{m['lost_rows']}.", '',
                  'No weights promoted. Development results are used for selection and are not a generalization estimate.',
                  'Assessment is a subset: compare against its paired baseline, not the earlier 64.02% full-validation number.',
                  'The current fixed pools still bound Top1; this is not a promise of 90% accuracy.',
                  'Graph identity ignores stereochemistry; no 3D or experimental-IR claim.', *cfg['limitations'], '']
        (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        with ZipFile(output / 'residual_selector_results.zip', 'w', ZIP_DEFLATED) as archive:
            for path in sorted(output.glob('*.json')):
                if path.name != 'completion.json':
                    archive.write(path, path.name)
            archive.write(output / 'RESULTS.md', 'RESULTS.md')
        atomic_json(output / 'completion.json', {'complete': True,
                    'results_zip_sha256': file_sha256(output / 'residual_selector_results.zip')})
    print(f'Complete: {output / "RESULTS.md"}\nDownload: {output / "residual_selector_results.zip"}', flush=True)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'baseline_report', 'mining_dir', 'output_dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--data', default='data/processed/squirl.npz')
    parser.add_argument('--splits', default='data/processed/splits.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for name, default in [('dev_size', 2000), ('epochs', 20), ('patience', 3), ('batch_size', 64),
                          ('seed', 78438379), ('partition_seed', 20261110)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--progress_interval', type=float, default=15.)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (min(args.dev_size, args.epochs, args.patience, args.batch_size) < 1 or min(args.seed, args.partition_seed) < 0
            or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.progress_interval) or args.progress_interval < 0):
        parser.error('Invalid budgets, seeds, learning rate or progress interval.')
    try:
        with exclusive_report(Path(args.output_dir).with_suffix('.run')):
            return run(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
