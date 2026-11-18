"""Bounded corrections to decoder scores; selection uses development rows only."""
import copy
import math
import numpy as np
import selfies as sf
import torch
from torch import nn
from torch.nn import functional as F
from diffusion.compatibility import molecular_fingerprints


def decoder_scores(candidates):
    values = []
    for c in candidates:
        probability = c.get('log_probability')
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or probability > 1e-6:
            raise ValueError('Finite decoder log probabilities are required.')
        if not isinstance(c.get('selfies'), str) or not c['selfies']:
            raise ValueError('Original generated SELFIES are required to recover decoder length normalization.')
        values.append(probability / (len(list(sf.split_selfies(c['selfies']))) + 2) ** .6)
    if any(a < b - 1e-10 for a, b in zip(values, values[1:])):
        raise ValueError('Candidate order differs from reconstructed decoder scores.')
    return values


def partition_validation(rows, dev_size, seed):
    if len(set(rows)) != len(rows) or not 0 < dev_size < len(rows):
        raise ValueError('Development and assessment need distinct nonempty selections.')
    development = set(np.random.default_rng(seed).permutation(rows)[:dev_size].tolist())
    return ([r for r in rows if r in development], [r for r in rows if r not in development])


class ResidualSelector(nn.Module):
    def __init__(self, ir_dim, hidden=32, bound=.25):
        super().__init__()
        if min(ir_dim, hidden) < 1 or not math.isfinite(bound) or bound <= 0:
            raise ValueError('Invalid head dimensions or correction bound.')
        self.spec = {'ir_dim': ir_dim, 'hidden': hidden, 'bound': bound}
        self.ir = nn.Sequential(nn.Linear(ir_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.molecule = nn.Sequential(nn.Linear(2048, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Dropout(.2), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, features, fingerprints):
        mol = self.molecule(fingerprints.float())
        ir = self.ir(features.float())[:, None, :].expand_as(mol)
        return self.spec['bound'] * torch.tanh(self.score(torch.cat((ir, mol, ir * mol), -1)).squeeze(-1))


class CandidateBatch:
    """No reference labels stored here: features and natural candidate pools only."""
    def __init__(self, features, candidates):
        if len(features) != len(candidates) or not len(features):
            raise ValueError('Candidate/features row count mismatch.')
        self.features = features.clone()
        self.candidates = copy.deepcopy(candidates)
        self.rows = len(candidates)
        width = max(1, max(map(len, candidates)))
        names = sorted({c['smiles'] for pool in candidates for c in pool})
        # Permit an all-empty development/assessment pool without special label handling.
        self.fingerprints = molecular_fingerprints(names)[0] if names else torch.zeros((1, 2048), dtype=torch.uint8)
        lookup = {s: i for i, s in enumerate(names)}
        self.indices = torch.zeros((self.rows, width), dtype=torch.long)
        self.mask = torch.zeros((self.rows, width), dtype=torch.bool)
        self.base = torch.full((self.rows, width), -torch.inf, dtype=torch.float64)
        for row, pool in enumerate(candidates):
            if len({c['smiles'] for c in pool}) != len(pool):
                raise ValueError('Duplicate candidates.')
            self.base[row, :len(pool)] = torch.tensor(decoder_scores(pool), dtype=torch.float64)
            self.indices[row, :len(pool)] = torch.tensor([lookup[c['smiles']] for c in pool], dtype=torch.long)
            self.mask[row, :len(pool)] = True

    def inputs(self, indices, device):
        return self.features[indices].to(device), self.fingerprints[self.indices[indices]].to(device)


def targets_for(records):
    """Missing targets remain -1; only training filters these out."""
    return torch.tensor([next((i for i, c in enumerate(r['candidates'])
                              if c['smiles'] == r['reference_smiles']), -1) for r in records], dtype=torch.long)


@torch.inference_mode()
def corrections(head, batch, device, batch_size=128):
    head.eval()
    values = []
    for indices in torch.arange(batch.rows).split(batch_size):
        result = head(*batch.inputs(indices, device)).cpu()
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite selector correction.')
        values.append(result)
    return torch.cat(values)


def measure(batch, adjustment, targets):
    scores = batch.base + adjustment.double()
    choices = scores.argmax(1)
    valid = batch.mask.any(1)
    correct = (choices == targets) & valid & (targets >= 0)
    present = targets >= 0
    # Conditional loss omits absent targets, but reported accuracy never omits them.
    loss = F.cross_entropy(scores[present], targets[present]).item() if present.any() else None
    return {'rows': batch.rows, 'correct': int(correct.sum()), 'top1': float(correct.double().mean()),
            'target_present': int(present.sum()), 'conditional_loss': loss}


def ordered_candidates(batch, adjustment, index):
    scores = (batch.base[index] + adjustment[index].double()).tolist()
    pool = batch.candidates[index]
    order = sorted(range(len(pool)), key=lambda j: (-scores[j], j))
    return [{**copy.deepcopy(pool[j]), 'decoder_rank': j + 1, 'decoder_score': float(batch.base[index, j]),
             'residual_score': float(adjustment[index, j]), 'combined_score': scores[j]} for j in order]


def fit_with_development(train, train_targets, development, dev_targets, *, device, epochs=20,
                         patience=3, batch_size=64, seed=78438379, lr=1e-4, hidden=32, bound=.25,
                         progress=None, on_epoch=None):
    if (epochs < 1 or patience < 1 or batch_size < 1 or seed < 0 or not math.isfinite(lr) or lr <= 0):
        raise ValueError('Invalid training settings.')
    eligible = torch.where((train_targets >= 0) & (train.mask.sum(1) >= 2))[0]
    if not len(eligible):
        raise ValueError('No trainable natural pools: need a present target and at least two candidates.')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = ResidualSelector(train.features.shape[1], hidden, bound).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    rng = torch.Generator().manual_seed(seed)
    zero = torch.zeros_like(development.base)
    baseline = measure(development, zero, dev_targets)
    required_gain = max(1, math.ceil(.005 * development.rows))
    selected_epoch, selected_correct = 0, baseline['correct']
    best_state = copy.deepcopy(head.state_dict())
    history = [{'epoch': 0, 'development': baseline, 'selected_epoch': 0,
                'selection': 'unchanged_decoder', 'required_development_gain': required_gain}]
    if on_epoch:
        on_epoch(history, head, selected_epoch)
    best_observed, stale = baseline['correct'], 0
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss, seen = 0., 0
        if progress:
            progress.start(f'Residual training epoch {epoch}/{epochs}', len(eligible), 'training inputs')
        for indices in eligible[torch.randperm(len(eligible), generator=rng)].split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            residual = head(*train.inputs(indices, device))
            base = train.base[indices].to(device=device, dtype=torch.float32)
            loss = F.cross_entropy(base + residual, train_targets[indices].to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite residual loss.')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            seen += len(indices)
            if progress:
                progress.update(seen)
        if progress:
            progress.start(f'Development check after epoch {epoch}', development.rows, 'molecules')
        current = measure(development, corrections(head, development, device, batch_size), dev_targets)
        if progress:
            progress.update(development.rows)
        if current['correct'] > best_observed:
            best_observed, stale = current['correct'], 0
        else:
            stale += 1
        accepted = (current['correct'] >= baseline['correct'] + required_gain and
                    current['correct'] > selected_correct)
        if accepted:
            best_state = copy.deepcopy(head.state_dict())
            selected_epoch, selected_correct = epoch, current['correct']
        history.append({'epoch': epoch, 'training_rows': seen, 'training_loss': total_loss / seen,
                        'last_gradient_norm': float(norm), 'development': current,
                        'selected_epoch': selected_epoch, 'new_selection': accepted, 'stale_epochs': stale})
        print(f"Epoch {epoch}: train loss {total_loss / seen:.5f}; development Top1 {current['top1']:.2%}; "
              f"selected epoch {selected_epoch}", flush=True)
        if on_epoch:
            on_epoch(history, head, selected_epoch)
        if stale >= patience:
            break
    head.load_state_dict(best_state)
    return head.eval(), history, {'epoch': selected_epoch, 'development_correct': selected_correct,
                                 'baseline_development_correct': baseline['correct'],
                                 'required_gain': required_gain, 'baseline_retained': selected_epoch == 0,
                                 'eligible_training_rows': len(eligible)}
