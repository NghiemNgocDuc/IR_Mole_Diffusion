"""Train matched random/hard-negative selectors without modifying the generator."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from rdkit import rdBase
from diffusion.compatibility import cache_spectrum_features, rerank_candidates
from diffusion.hard_selector import make_groups, fit_selector
from diffusion.data import NativeNPZDataset, collate_native, file_sha256, load_native_npz, validate_split_manifest
from diffusion.progress import EvaluationProgress
from diffusion.ranking import canonical_identity
from diffusion.spectrum import SpectrumStructureModel
from diffusion.spectrum_audit import atomic_json, exclusive_report, row_record, summarize_records, validate_selection
from scripts.compare_spectrum_pilot import BOOTSTRAP_SEED, paired_metrics, same_json, validate_report
from evaluation.evaluate_spectrum import valid_candidates
ROOT = Path(__file__).resolve().parents[1]
FEATURE_SOURCES = ("diffusion/encoder.py", "diffusion/model.py", "diffusion/data.py",
                   "diffusion/spectrum.py", "evaluation/evaluate_spectrum.py", "diffusion/spectrum_audit.py")


def run(args):
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
    if len(manifest['splits']['train']) < args.limit:
        raise ValueError('Mining limit exceeds training split size.')
    selected = np.random.default_rng(args.seed).permutation(manifest['splits']['train'])[:args.limit].tolist()
    source_names = (*FEATURE_SOURCES, 'diffusion/compatibility.py', 'diffusion/hard_selector.py',
                    'diffusion/ranking.py', 'diffusion/spectrum.py', 'diffusion/spectrum_audit.py',
                    'diffusion/progress.py', 'evaluation/evaluate_spectrum.py', 'training/train_hard_selector.py',
                    'scripts/compare_spectrum_pilot.py')
    config = {'schema_version': 1, 'arguments': {k: v for k, v in vars(args).items() if k != 'resume'},
              'data_sha256': manifest['data_sha256'], 'split_sha256': split_hash,
              'checkpoint_sha256': checkpoint_hash, 'baseline_report_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
              'source_sha256': {name: file_sha256(ROOT / name) for name in source_names},
              'training_rows': selected, 'held_out_labels_used_for_training': False,
              'diagnostic': diagnostic, 'temperature': .1, 'beam_size': 5,
              'random_negative_pool': 'reference identities of the same selected training rows',
              'objective': 'per-input candidate cross entropy; positive injected in training only',
              'selection_policy': 'fixed final epoch; no validation tuning or score mixing',
              'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'rdkit': rdBase.rdkitVersion}}
    if output.exists():
        if not args.resume or not (output / 'configuration.json').is_file():
            raise ValueError('Output exists; use a fresh directory, or --resume for a matching interrupted run.')
        if not same_json(json.loads((output / 'configuration.json').read_text()), config):
            raise ValueError('Resume configuration, inputs, or source hashes differ.')
        if (output / 'RESULTS.md').exists():
            raise ValueError('Run already complete; read RESULTS.md rather than resubmitting.')
    else:
        output.mkdir(parents=True)
        atomic_json(output / 'configuration.json', config)
    mining_path = output / 'training_candidates.json'
    mined = []
    if mining_path.exists():
        saved = json.loads(mining_path.read_text())
        if saved.get('configuration_sha256') != file_sha256(output / 'configuration.json'):
            raise ValueError('Mining cache configuration mismatch.')
        mined = saved['rows']
        validate_mined(mined, selected, data)
    def save_mining():
        atomic_json(mining_path, {'configuration_sha256': file_sha256(output / 'configuration.json'), 'rows': mined})
    with EvaluationProgress(output / 'progress', interval=args.progress_interval) as progress:
        offset = len(mined)
        progress.start(f'Mining training candidates ({offset} saved)', len(selected) - offset, 'molecules')
        dataset = NativeNPZDataset(data, selected)
        try:
            with torch.inference_mode():
                for position in range(offset, len(selected)):
                    spectra = collate_native([dataset[position]])['spectra']
                    spectra = ({k: v.to(args.device) for k, v in spectra.items()}
                               if isinstance(spectra, dict) else spectra.to(args.device))
                    candidates = valid_candidates(model.generate(spectra, 5, .6), data['coords'].shape[1])
                    mined.append({'row': selected[position],
                                  'reference_smiles': canonical_identity(data['molecule_ids'][selected[position]]),
                                  'candidates': candidates})
                    if len(mined) % 10 == 0:
                        save_mining()
                    progress.update(position + 1 - offset)
        finally:
            save_mining()
        kept, groups = make_groups(mined, manifest['splits']['train'],
                                    [r['reference_smiles'] for r in mined], args.seed)
        atomic_json(output / 'training_groups.json', {'rows': kept, 'groups': groups,
                    'skipped_no_wrong_candidate': len(mined) - len(kept),
                    'reference_missing_from_generated_pool': sum(r['reference_smiles'] not in
                        {c['smiles'] for c in r['candidates']} for r in mined)})
        features = cache_spectrum_features(model.encoder, data, kept, args.device, args.batch_size,
                                           progress, 'Caching training-only spectra')
        heads = {}
        for arm in ('random', 'hard'):
            print(f'Training {arm} negatives: {len(kept)} inputs, {args.epochs} epochs', flush=True)
            head, history = fit_selector(features, groups[arm], device=args.device, epochs=args.epochs,
                batch_size=args.batch_size, seed=args.seed, lr=args.lr, projection_dim=args.projection_dim,
                progress=progress)
            atomic_json(output / f'{arm}_history.json', history)
            temp = output / f'{arm}.pt.tmp'
            torch.save({'spec': head.spec, 'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        'configuration': config, 'arm': arm}, temp)
            temp.replace(output / f'{arm}.pt')
            heads[arm] = head
        del features
        val_features = cache_spectrum_features(model.encoder, data, val_rows, args.device, args.batch_size,
                                               progress, 'Caching validation spectra for inference only')
        results = {}
        for arm, head in heads.items():
            records = []
            progress.start(f'Evaluating {arm} selector on fixed candidate pools', len(val_rows), 'molecules')
            for index, record in enumerate(checked['rows']):
                candidates = rerank_candidates(head, val_features[index], record['candidates'], args.device)
                # Evaluate references only after reference-blind ordering.
                records.append(row_record(record['row'], record['reference_smiles'], record['reference_formula'], candidates))
                progress.update(index + 1)
            if any(a['topk_identity'] != b['topk_identity'] for a, b in zip(checked['rows'], records)):
                raise ValueError('Candidate coverage changed.')
            results[arm] = {'summary': summarize_records(records), 'rows': records}
            atomic_json(output / f'{arm}_predictions.json', {'complete': True, 'selection': baseline['selection'],
                        'baseline_report_sha256': config['baseline_report_sha256'], **results[arm]})
        comparison = {'baseline': checked['summary'], 'random': results['random']['summary'],
                      'hard': results['hard']['summary'], 'validation_rows': len(val_rows),
                      'diagnostic': diagnostic, 'weights_promoted': False, 'candidate_pool_unchanged': True,
                      'checkpoint_sha256': checkpoint_hash, 'baseline_report_sha256': config['baseline_report_sha256'],
                      'random_vs_baseline': paired_metrics(checked['rows'], results['random']['rows'], BOOTSTRAP_SEED),
                      'hard_vs_baseline': paired_metrics(checked['rows'], results['hard']['rows'], BOOTSTRAP_SEED),
                      'hard_vs_random': paired_metrics(results['random']['rows'], results['hard']['rows'], BOOTSTRAP_SEED)}
        for path, expected in [(args.data, config['data_sha256']), (args.splits, split_hash),
                               (args.checkpoint, checkpoint_hash), (args.baseline_report, config['baseline_report_sha256']),
                               *((ROOT / name, value) for name, value in config['source_sha256'].items())]:
            if file_sha256(path) != expected:
                raise ValueError(f'Input or source changed during run: {path}')
        atomic_json(output / 'comparison.json', comparison)
        lines = ['# Matched hard-negative selector experiment', '',
                 f'Validation rows: {len(val_rows)}. Diagnostic: {diagnostic}. Training inputs retained: {len(kept)}.', '',
                 '| Method | Top1 | Top5 | Formula |', '|---|---:|---:|---:|']
        for arm in ('baseline', 'random', 'hard'):
            scores = comparison[arm]
            lines.append('| ' + arm + ' | ' + ' | '.join(f"{100*scores[k+'_fraction']:.2f}%" for k in
                         ('top1_identity', 'top5_identity', 'top1_formula')) + ' |')
        for name in ('hard_vs_baseline', 'hard_vs_random', 'random_vs_baseline'):
            metric = comparison[name]['metrics']['top1_identity']
            lines += ['', f"{name}: {metric['delta_percentage_points']:+.2f} pp; exploratory paired 95% interval: "
                      f"{metric['paired_bootstrap_ci95_percentage_points']}."]
        lines += ['', 'No weights promoted. Keep the existing generator/decoder until results are reviewed.',
                  'Both heads use identical inputs, initialization, architecture, negative counts and update budgets.',
                  'This compares negative construction under candidate cross entropy, not the previous in-batch loss.',
                  'Training positives are injected only into training groups. Validation pools are unchanged.',
                  'The random control draws identities from the selected training inputs; hard negatives are model outputs.',
                  'Final epoch is fixed in advance. No validation-based checkpoint selection or score blending.',
                  'The generator already saw these training inputs. Online training accuracy is diagnostic only.',
                  'One training seed; multiple exploratory comparisons. Validation has been used repeatedly.',
                  'Exact identity ignores stereochemistry. No experimental-IR or 3D accuracy claim. Test remains untouched.',
                  'Fixed candidate coverage bounds achievable accuracy; 90% Top1 is not promised.', '']
        (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        with ZipFile(output / 'hard_selector_results.zip', 'w', ZIP_DEFLATED) as archive:
            for path in sorted(output.glob('*.json')):
                archive.write(path, path.name)
            archive.write(output / 'RESULTS.md', 'RESULTS.md')
    print(f'Complete: {output / "RESULTS.md"}\nDownload: {output / "hard_selector_results.zip"}', flush=True)
    return comparison


def validate_mined(records, selected, data):
    if len(records) > len(selected) or [r['row'] for r in records] != selected[:len(records)]:
        raise ValueError('Mining cache is not the selected training-row prefix.')
    for record in records:
        if record['reference_smiles'] != canonical_identity(data['molecule_ids'][record['row']]):
            raise ValueError('Mining reference mismatch.')
        candidates = record['candidates']
        if (len(candidates) > 5 or valid_candidates(candidates, data['coords'].shape[1]) != candidates
                or any(not np.isfinite(c['log_probability']) for c in candidates)):
            raise ValueError('Invalid or inconsistent cached candidates.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'baseline_report', 'output_dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--data', default='data/processed/squirl.npz')
    parser.add_argument('--splits', default='data/processed/splits.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for name, default in [('limit', 5000), ('epochs', 20), ('batch_size', 64),
                          ('seed', 78438379), ('projection_dim', 128)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--progress_interval', type=float, default=15.)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (min(args.limit, args.epochs, args.batch_size, args.projection_dim) < 1 or args.limit < 2 or args.seed < 0
            or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.progress_interval) or args.progress_interval < 0):
        parser.error('Invalid training budget, learning rate, or progress interval.')
    try:
        with exclusive_report(Path(args.output_dir).with_suffix('.run')):
            return run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
