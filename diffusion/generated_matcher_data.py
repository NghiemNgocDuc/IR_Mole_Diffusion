"""Fixed decoder-negative training banks; no validation references are accessed."""
from types import SimpleNamespace
import numpy as np
import selfies as sf
import torch
from diffusion.ranking import canonical_identity
from diffusion.progress import EvaluationProgress
from diffusion.spectrum_audit import atomic_json
from diffusion.token_matcher_data import spectra_batch, tokenize_selfies
from evaluate_spectrum import valid_candidates


def generated_plan(original, generated, positives, identities):
    if original.shape != (len(positives), 3) or len(generated) != len(positives):
        raise ValueError('Plan shapes differ.')
    result = np.empty_like(original)
    counts = [0, 0, 0, 0]
    for i, positive in enumerate(positives):
        seen, chosen = {identities[int(positive)]}, []
        used = 0
        for source, indices in [('generated', generated[i]), ('fallback', original[i])]:
            for index in indices:
                if not 0 <= int(index) < len(identities):
                    raise ValueError('Candidate bank index out of bounds.')
                identity = identities[int(index)]
                if identity in seen:
                    continue
                chosen.append(int(index)); seen.add(identity)
                used += int(source == 'generated')
                if len(chosen) == 3:
                    break
            if len(chosen) == 3:
                break
        if len(chosen) != 3:
            raise ValueError('Three distinct wrong identities required.')
        result[i] = chosen
        counts[used] += 1
    return result, {'rows_by_generated_negative_count_0_to_3': counts,
                    'generated_slots': sum(i*n for i,n in enumerate(counts)),
                    'fallback_slots': 3*len(positives)-sum(i*n for i,n in enumerate(counts))}


def prepare_pairs(data, full, generator, args, output):
    selected = np.random.default_rng(args.mining_seed).permutation(len(full.rows))[:args.training_limit]
    if len(selected) < 2:
        raise ValueError('At least two training anchors required.')
    rows = [full.rows[int(i)] for i in selected]
    max_len = int(generator.hparams.max_length)
    bank = [t[t != 0].tolist() for t in full.tokens]
    identities = list(full.identities)
    generated, records = [], []
    torch.manual_seed(args.mining_seed)
    generator.to(args.device).eval().requires_grad_(False)
    with EvaluationProgress(output/'mining_progress', interval=args.progress_interval) as progress:
        progress.start('Generating training-only alternatives', len(rows), 'molecules')
        with torch.inference_mode():
            for n, (row, positive) in enumerate(zip(rows, selected)):
                pool = valid_candidates(generator.generate(spectra_batch(data, [row], args.device), 5, .6), data['coords'].shape[1])
                chosen, seen = [], {full.identities[int(positive)]}
                for c in pool:
                    identity = canonical_identity(sf.decoder(c['selfies']))
                    if not identity or identity != c['smiles']:
                        raise ValueError('Generated serialization and graph differ.')
                    tokens = tokenize_selfies([c['selfies']], data['vocabulary'], max_len)[0].tolist()
                    if identity in seen:
                        continue
                    chosen.append(len(bank)); bank.append(tokens); identities.append(identity); seen.add(identity)
                    if len(chosen) == 3:
                        break
                generated.append(chosen)
                records.append({'row': row, 'reference_smiles': full.identities[int(positive)],
                                'candidates': pool, 'negative_bank_indices': chosen})
                progress.update(n+1)
    tokens = torch.zeros((len(bank), max(map(len, bank))), dtype=torch.long)
    for i, sequence in enumerate(bank):
        tokens[i, :len(sequence)] = torch.tensor(sequence)
    np.save(output/'candidate_bank.npy', tokens.numpy(), allow_pickle=False)
    atomic_json(output/'generated_training_candidates.json', {'complete': True, 'rows': records,
        'anchor_local_indices': selected.tolist(), 'bank_identities': identities,
        'training_only': True, 'generator_saw_anchors': True})
    return SimpleNamespace(rows=rows, positive_indices=selected, tokens=tokens,
        identities=[full.identities[int(i)] for i in selected], bank_identities=identities), generated
