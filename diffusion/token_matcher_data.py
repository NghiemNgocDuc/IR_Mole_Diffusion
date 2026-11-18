"""Training-only molecular pairs and reproducible negative schedules."""
from collections import defaultdict
import numpy as np
import selfies as sf
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from diffusion.ranking import canonical_identity


def tokenize_selfies(strings, vocabulary, max_len):
    if list(vocabulary[:3]) != ['<pad>', '<bos>', '<eos>'] or len(set(vocabulary)) != len(vocabulary):
        raise ValueError('Expected unique vocabulary with PAD=0, BOS=1, EOS=2.')
    lookup = {token: i for i, token in enumerate(vocabulary)}
    sequences = []
    for text in strings:
        if not isinstance(text, str) or not text:
            raise ValueError('Candidate SELFIES must be nonempty.')
        pieces = list(sf.split_selfies(text))
        if not pieces or ''.join(pieces) != text or any(piece not in lookup or lookup[piece] < 3 for piece in pieces):
            raise ValueError('Candidate SELFIES contains malformed or unsupported vocabulary tokens.')
        ids = [1, *[lookup[piece] for piece in pieces], 2]
        if len(ids) > max_len:
            raise ValueError('Molecular token sequence exceeds checkpoint maximum length; no truncation permitted.')
        sequences.append(ids)
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long)
    # Trim inactive suffixes instead of making every pair attend to max_len padding.
    result = torch.zeros((len(sequences), max(map(len, sequences))), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        result[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return result


def spectra_batch(data, rows, device):
    indices = np.asarray(rows, dtype=np.int64)
    spectra = torch.as_tensor(data['spectra'][indices], dtype=torch.float32, device=device)
    if data['packed']:
        return {'packed': spectra, 'mask': torch.as_tensor(data['spectra_mask'][indices], dtype=torch.bool, device=device)}
    return spectra


class TrainingPairs:
    """Build features only from the explicit complete training selection.

    Graph fingerprints choose negative examples, never score validation candidates.
    The runner supplies the already validated split manifest as allowed_train_rows.
    """
    def __init__(self, data, train_rows, allowed_train_rows, max_len=None, on_progress=None):
        self.rows = list(train_rows)
        if (not self.rows or self.rows != list(allowed_train_rows) or len(set(self.rows)) != len(self.rows)
                or any(type(row) is not int or row < 0 for row in self.rows)):
            raise ValueError('Use exactly the unique ordered training split rows.')
        if len(self.rows) < 4:
            raise ValueError('Three distinct negatives require at least four training identities.')
        vocab = data['vocabulary']
        if list(vocab[:3]) != ['<pad>', '<bos>', '<eos>']:
            raise ValueError('Training vocabulary requires PAD=0, BOS=1, EOS=2.')
        self.identities, sequences, molecules = [], [], []
        for index, row in enumerate(self.rows):
            identity = canonical_identity(data['molecule_ids'][row])
            if not identity:
                raise ValueError('Invalid training identity.')
            supplied = np.asarray(data['selfies_tokens'][row])
            if supplied.ndim != 1 or supplied.dtype.kind not in 'iu' or np.any((supplied < 0) | (supplied >= len(vocab))):
                raise ValueError('Invalid stored training tokens.')
            live_length = int(np.count_nonzero(supplied))
            live = supplied[:live_length]
            if (live_length < 3 or live[0] != 1 or live[-1] != 2 or np.any(live[1:-1] < 3)
                    or np.any(supplied[live_length:] != 0)):
                raise ValueError('Training tokens must contain BOS, molecule, EOS, then right padding.')
            if max_len is not None and live_length > max_len:
                raise ValueError('Training tokens exceed checkpoint maximum length.')
            text = ''.join(vocab[int(i)] for i in live[1:-1])
            try:
                decoded = canonical_identity(sf.decoder(text))
            except (ValueError, sf.DecoderError) as exc:
                raise ValueError('Stored training SELFIES cannot be decoded.') from exc
            if decoded != identity:
                raise ValueError('Stored training SELFIES differs from its reference graph.')
            self.identities.append(identity)
            sequences.append(live.copy())
            molecules.append(Chem.MolFromSmiles(identity))
            if on_progress and ((index + 1) % 10000 == 0 or index + 1 == len(self.rows)):
                on_progress(index + 1, len(self.rows))
        if len(set(self.identities)) != len(self.identities):
            raise ValueError('Duplicate canonical training identities would create false negatives.')
        self.tokens = torch.zeros((len(self.rows), max(map(len, sequences))), dtype=torch.long)
        for i, sequence in enumerate(sequences):
            self.tokens[i, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
        self.formulas = [rdMolDescriptors.CalcMolFormula(mol) for mol in molecules]
        self.heavy_atoms = [mol.GetNumHeavyAtoms() for mol in molecules]
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
        self.fingerprints = [generator.GetFingerprint(mol) for mol in molecules]
        self.formula_groups, self.heavy_groups = {}, {}
        for values, target in [(self.formulas, self.formula_groups), (self.heavy_atoms, self.heavy_groups)]:
            groups = defaultdict(list)
            for i, value in enumerate(values):
                groups[value].append(i)
            target.update({key: np.asarray(group, dtype=np.int64) for key, group in groups.items()})
        self.all_indices = np.arange(len(self.rows), dtype=np.int64)
        self.last_plan_stats = {}

    @staticmethod
    def _sample_pool(pool, excluded, rng, limit):
        # Sampling limit+excluded guarantees enough available items when the pool
        # is large, without copying the entire training pool for every anchor.
        if len(pool) > limit + len(excluded):
            positions = rng.choice(len(pool), size=limit + len(excluded), replace=False)
            proposed = pool[positions]
        else:
            proposed = rng.permutation(pool)
        return [int(value) for value in proposed if int(value) not in excluded][:limit]

    def negative_plan(self, epoch, seed, progress=None):
        if type(epoch) is not int or epoch < 1 or type(seed) is not int or seed < 0:
            raise ValueError('Negative plans require a positive epoch and nonnegative seed.')
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 9403]))
        count = len(self.rows)
        plan = np.empty((count, 3), dtype=np.int64)
        stats = {'rows': count, 'same_formula_middle': 0, 'structural_same_formula_pool': 0,
                 'structural_same_heavy_pool': 0, 'structural_all_train_pool': 0, 'structural_search_limit': 32}
        if progress:
            progress.start(f'Preparing shared training negatives, epoch {epoch}', count, 'training pairs')
        for i in range(count):
            excluded = {i}
            random = self._sample_pool(self.all_indices, excluded, rng, 1)[0]
            excluded.add(random)
            middle_pool = self._sample_pool(self.formula_groups[self.formulas[i]], excluded, rng, 1)
            if middle_pool:
                middle = middle_pool[0]
                stats['same_formula_middle'] += 1
            else:
                middle = self._sample_pool(self.all_indices, excluded, rng, 1)[0]
            excluded.add(middle)
            structural_pool = self._sample_pool(self.formula_groups[self.formulas[i]], excluded, rng, 32)
            if structural_pool:
                stats['structural_same_formula_pool'] += 1
            else:
                structural_pool = self._sample_pool(self.heavy_groups[self.heavy_atoms[i]], excluded, rng, 32)
                if structural_pool:
                    stats['structural_same_heavy_pool'] += 1
                else:
                    structural_pool = self._sample_pool(self.all_indices, excluded, rng, 32)
                    stats['structural_all_train_pool'] += 1
            similarity = DataStructs.BulkTanimotoSimilarity(self.fingerprints[i], [self.fingerprints[j] for j in structural_pool])
            hard = structural_pool[int(np.argmax(similarity))]
            plan[i] = (random, middle, hard)
            if progress and ((i + 1) % 1000 == 0 or i + 1 == count):
                progress.update(i + 1)
        self.last_plan_stats = stats
        return plan
