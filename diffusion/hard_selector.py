"""Matched random/generated-negative experiments for a frozen IR encoder."""
import numpy as np
import torch
from torch.nn import functional as F

from diffusion.compatibility import CompatibilityHead, molecular_fingerprints
from diffusion.ranking import canonical_identity


def make_groups(records, allowed_rows, random_pool, seed):
    """Inject a training positive; exclude its exact identity from every negative."""
    if len({r['row'] for r in records}) != len(records):
        raise ValueError('Duplicate mining rows.')
    allowed = set(allowed_rows)
    if any(r['row'] not in allowed for r in records):
        raise ValueError('Mining records must belong to the training split.')
    pool = sorted({canonical_identity(s) for s in random_pool})
    rng = np.random.default_rng(seed)
    hard, random, kept = [], [], []
    for record in records:
        positive = canonical_identity(record['reference_smiles'])
        negatives = list(dict.fromkeys(canonical_identity(c['smiles']) for c in record['candidates']))
        negatives = [s for s in negatives if s != positive]
        if not negatives:
            continue
        choices = [s for s in pool if s != positive]
        if len(choices) < len(negatives):
            raise ValueError('Insufficient distinct training identities for matched random negatives.')
        hard.append([positive, *negatives])
        random.append([positive, *rng.choice(choices, len(negatives), replace=False).tolist()])
        kept.append(record['row'])
    if not kept:
        raise ValueError('No valid wrong candidates were mined; no selector can be trained.')
    return kept, {'random': random, 'hard': hard}


def candidate_scores(head, features, fingerprints):
    """One spectrum against its own K candidates; no cross-batch negatives."""
    ir = F.normalize(head.ir_projection(features.float()), dim=-1)
    mol = F.normalize(head.molecule_projection(fingerprints.float()), dim=-1)
    return (ir[:, None, :] * mol).sum(-1)


def fit_selector(features, groups, *, device, epochs, batch_size, seed, lr,
                 projection_dim, progress=None):
    if len(features) != len(groups) or not groups or any(len(g) < 2 or len(set(g)) != len(g) for g in groups):
        raise ValueError('Each training row needs a unique positive and at least one distinct negative.')
    features = features.clone()
    identities = sorted({s for group in groups for s in group})
    fps, _ = molecular_fingerprints(identities)
    lookup = {s: i for i, s in enumerate(identities)}
    width = max(map(len, groups))
    indices = torch.zeros((len(groups), width), dtype=torch.long)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    for row, group in enumerate(groups):
        indices[row, :len(group)] = torch.tensor([lookup[s] for s in group])
        mask[row, :len(group)] = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = CompatibilityHead(features.shape[1], projection_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    rng = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        head.train()
        total, correct, seen, steps = 0., 0, 0, 0
        if progress:
            progress.start(f'Selector epoch {epoch + 1}/{epochs}', len(groups), 'training inputs')
        for batch in torch.randperm(len(groups), generator=rng).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            scores = candidate_scores(head, features[batch].to(device), fps[indices[batch]].to(device)) / .1
            scores = scores.masked_fill(~mask[batch].to(device), -torch.inf)
            loss = F.cross_entropy(scores, torch.zeros(len(batch), dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite selector loss.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            correct += int((scores.argmax(-1) == 0).sum())
            seen += len(batch)
            steps += 1
            if progress:
                progress.update(seen)
        record = {'epoch': epoch + 1, 'loss': total / seen, 'online_training_accuracy': correct / seen,
                  'training_rows': seen, 'optimizer_steps': steps, 'last_gradient_norm': float(norm)}
        history.append(record)
        print(record, flush=True)
    return head.eval(), history
