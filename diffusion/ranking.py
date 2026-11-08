"""Train-only hard negatives for spectrum-to-structure score fine-tuning."""
import numpy as np


def build_negative_rows(data, training_rows, seed=78438379, max_pool=32, progress=None):
    """Choose a similar training identity, preferring the same element counts.

    Molecule identities/fingerprints are read only for training rows. Held-out
    labels never enter either the negative pools or the objective.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    rows = np.asarray(training_rows)
    if rows.dtype.kind not in "iu" or rows.ndim != 1 or len(rows) < 2 or max_pool < 1 or len(np.unique(rows)) != len(rows):
        raise ValueError("Hard negatives need at least two unique training rows and a positive pool size.")
    if np.any(rows < 0) or np.any(rows >= len(data["molecule_ids"])):
        raise ValueError("Training row indices are outside the dataset.")
    rows = rows.astype(np.int64)
    identities = data["molecule_ids"][rows].astype(str)
    if len(set(identities)) < 2:
        raise ValueError("Hard negatives require at least two training identities.")
    atoms = data["atom_types"][rows]
    counts = np.stack([(atoms == number).sum(1) for number in (1, 6, 7, 8, 9)], axis=1)
    formula_pools, heavy_pools = {}, {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprints = []
    for position, identity in enumerate(identities):
        molecule = Chem.MolFromSmiles(identity)
        if molecule is None:
            raise ValueError(f"Invalid training molecule at row {rows[position]}.")
        fingerprints.append(generator.GetFingerprint(molecule))
        formula_pools.setdefault(tuple(counts[position]), []).append(position)
        heavy_pools.setdefault(int(counts[position, 1:].sum()), []).append(position)
    negative_rows = np.full(len(rows), -1, dtype=np.int64)
    pool_types = np.zeros(len(rows), dtype=np.int8)
    all_positions = np.arange(len(rows))
    for position, row in enumerate(rows):
        pools = ((1, formula_pools[tuple(counts[position])]),
                 (2, heavy_pools[int(counts[position, 1:].sum())]), (3, all_positions))
        for kind, pool in pools:
            choices = np.asarray(pool, dtype=np.int64)
            choices = choices[identities[choices] != identities[position]]
            if not len(choices):
                continue
            rng = np.random.default_rng((seed + int(row)) % (2**32))
            if len(choices) > max_pool:
                choices = rng.choice(choices, max_pool, replace=False)
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[position], [fingerprints[index] for index in choices])
            negative_rows[position] = rows[choices[int(np.argmax(similarities))]]
            pool_types[position] = kind
            break
        if progress is not None:
            progress(position + 1)
    return negative_rows, pool_types


class RankingDataset:
    """Attach negative training SELFIES while retaining the standard batch schema."""
    def __init__(self, dataset, negative_rows=None, *, negative_tokens=None):
        self.dataset = dataset
        self.negative_tokens = None
        if (negative_rows is None) == (negative_tokens is None):
            raise ValueError("Supply exactly one verified negative row or token array.")
        if negative_tokens is not None:
            # Semantic and training-provenance checks belong to load_negative_tokens;
            # sparse mined replacements need checking only once when the file loads.
            self.negative_tokens = np.asarray(negative_tokens)
            if self.negative_tokens.shape != dataset.data["selfies_tokens"][dataset.indices].shape:
                raise ValueError("One full negative token sequence is required per training row.")
            validate_token_sequences(self.negative_tokens, len(dataset.data["vocabulary"]))
            return
        self.negative_rows = np.asarray(negative_rows)
        if self.negative_rows.dtype.kind not in "iu":
            raise ValueError("Negative row indices must be integers.")
        if self.negative_rows.shape != dataset.indices.shape:
            raise ValueError("One negative index is required per training row.")
        allowed = set(dataset.indices.tolist())
        ids = dataset.data["molecule_ids"]
        for row, negative in zip(dataset.indices, self.negative_rows):
            if negative not in allowed or ids[row] == ids[negative]:
                raise ValueError("A ranking negative must be a different identity within the training split.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["negative_selfies_tokens"] = (self.negative_tokens[index] if self.negative_tokens is not None
            else self.dataset.data["selfies_tokens"][self.negative_rows[index]])
        return item


def load_negative_rows(path, training_rows, data_hash, split_hash):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["data_sha256"].item()) != data_hash or str(archive["split_sha256"].item()) != split_hash:
            raise ValueError("Ranking negatives do not match the requested dataset/split hashes.")
        stored_rows, negative_rows = archive["training_rows"], archive["negative_rows"]
        if stored_rows.dtype.kind not in "iu" or negative_rows.dtype.kind not in "iu":
            raise ValueError("Ranking row arrays must be integer indices.")
        if not np.array_equal(stored_rows, np.asarray(training_rows)):
            raise ValueError("Ranking negatives have different training rows or order.")
        if negative_rows.shape != stored_rows.shape or not np.isin(negative_rows, stored_rows).all() or np.any(negative_rows == stored_rows):
            raise ValueError("Ranking negatives must be different rows confined to the training split.")
        return negative_rows.copy()


def validate_token_sequences(tokens, vocabulary_size):
    """Reject truncation, unknown IDs, and ambiguous BOS/EOS/PAD boundaries."""
    tokens = np.asarray(tokens)
    if tokens.dtype.kind not in "iu" or tokens.ndim != 2 or tokens.shape[1] < 3:
        raise ValueError("Negative tokens must be a two-dimensional integer array.")
    if np.any(tokens < 0) or np.any(tokens >= vocabulary_size):
        raise ValueError("Negative tokens contain unknown vocabulary IDs.")
    eos = tokens == 2
    if np.any(tokens[:, 0] != 1) or np.any(eos.sum(axis=1) != 1):
        raise ValueError("Negative sequences require BOS and exactly one EOS.")
    ends = eos.argmax(axis=1)
    positions = np.arange(tokens.shape[1])[None, :]
    if (np.any(ends < 2) or np.any((positions > ends[:, None]) & (tokens != 0))
            or np.any((positions > 0) & (positions < ends[:, None]) & (tokens < 3))):
        raise ValueError("Negative sequences require BOS, body, EOS, then PAD without truncation.")


def canonical_identity(smiles):
    """Strict graph identity, matching non-isomeric spectrum evaluation."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Negative candidate is not a valid connected molecule.")
    if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           or atom.GetAtomicNum() not in (1, 6, 7, 8, 9) for atom in mol.GetAtoms()):
        raise ValueError("Negative candidate must be neutral and within the training elements.")
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def encode_negative(selfies, vocabulary, width, reference_smiles, expected_smiles=None):
    """Encode the actual generated path, never truncating or substituting OOVs."""
    import selfies as sf
    lookup = {symbol: index for index, symbol in enumerate(vocabulary)}
    if list(vocabulary[:3]) != ["<pad>", "<bos>", "<eos>"] or len(lookup) != len(vocabulary):
        raise ValueError("Invalid negative vocabulary.")
    symbols = list(sf.split_selfies(selfies))
    if not symbols or len(symbols) + 2 > width:
        raise ValueError("Generated negative is empty or overlength; truncation is forbidden.")
    if any(symbol not in lookup or lookup[symbol] < 3 for symbol in symbols):
        raise ValueError("Generated negative contains an unknown vocabulary symbol.")
    try:
        identity = canonical_identity(sf.decoder(selfies))
    except (sf.DecoderError, ValueError) as exc:
        raise ValueError("Invalid generated negative SELFIES.") from exc
    if expected_smiles is not None and identity != canonical_identity(expected_smiles):
        raise ValueError("Generated SELFIES and declared negative identity differ.")
    if identity == canonical_identity(reference_smiles):
        raise ValueError("A generated negative must differ from its training target identity.")
    tokens = np.zeros(width, dtype=np.int64)
    body = [1, *(lookup[symbol] for symbol in symbols), 2]
    tokens[:len(body)] = body
    return tokens, identity


def load_negative_tokens(path, training_rows, data, data_hash, split_hash, checkpoint_hash=None):
    """Read old train-row negatives or v2 sparse model-generated replacements.

    The fallback remains one different training identity for every training row.
    Model-generated replacements are verified from their actual token sequences;
    they need not be existing dataset identities. No held-out labels are queried.
    """
    import json
    from diffusion.spectrum_audit import _same_json, select_rows
    rows = np.asarray(training_rows)
    fallback = load_negative_rows(path, rows, data_hash, split_hash)
    ids = data["molecule_ids"]
    if any(str(ids[row]) == str(ids[other]) for row, other in zip(rows, fallback)):
        raise ValueError("Fallback negative must differ from its training target identity.")
    tokens = np.asarray(data["selfies_tokens"][fallback]).copy()
    with np.load(path, allow_pickle=False) as archive:
        if "schema_version" not in archive:
            return tokens
        if archive["schema_version"].shape != () or archive["schema_version"].item() != 2:
            raise ValueError("Unsupported mined-negative archive schema.")
        provenance = json.loads(str(archive["provenance_json"].item()))
        if provenance.get("data_sha256") != data_hash or provenance.get("split_sha256") != split_hash:
            raise ValueError("Mined negative provenance has different dataset/split hashes.")
        if checkpoint_hash is not None and provenance.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Mined negatives came from a different checkpoint.")
        if not _same_json(provenance.get("vocabulary"), data["vocabulary"]):
            raise ValueError("Mined negative vocabulary differs.")
        if not _same_json(provenance.get("preprocessing"), data.get("preprocessing")):
            raise ValueError("Mined negative preprocessing differs.")
        if provenance.get("held_out_labels_used") is not False or provenance.get("split") != "train":
            raise ValueError("Mined negatives require an explicit training-only label declaration.")
        mined_rows = archive["mined_rows"]
        mined_tokens = archive["mined_tokens"]
        mined_smiles = archive["mined_smiles"]
        if (mined_rows.dtype.kind not in "iu" or mined_rows.ndim != 1
                or len(np.unique(mined_rows)) != len(mined_rows) or not np.isin(mined_rows, rows).all()):
            raise ValueError("Mined rows must be unique and confined to the training split.")
        selected = select_rows(rows, "random", provenance.get("limit"), provenance.get("seed"))
        if not np.isin(mined_rows, selected).all():
            raise ValueError("Mined replacements must belong to the frozen training selection.")
        if mined_tokens.shape != (len(mined_rows), tokens.shape[1]) or mined_smiles.shape != mined_rows.shape:
            raise ValueError("Mined rows, token sequences and identities have different shapes.")
        validate_token_sequences(mined_tokens, len(data["vocabulary"]))
        row_positions = {int(row): position for position, row in enumerate(rows)}
        for row, sequence, identity in zip(mined_rows, mined_tokens, mined_smiles):
            end = int(np.flatnonzero(sequence == 2)[0])
            selfies = "".join(data["vocabulary"][int(token)] for token in sequence[1:end])
            checked, _ = encode_negative(selfies, data["vocabulary"], tokens.shape[1], ids[row], str(identity))
            if not np.array_equal(checked, sequence):
                raise ValueError("Mined sequence has changed during semantic verification.")
            from rdkit import Chem
            if Chem.AddHs(Chem.MolFromSmiles(str(identity))).GetNumAtoms() > data["coords"].shape[1]:
                raise ValueError("Mined negative exceeds the dataset atom capacity.")
            tokens[row_positions[int(row)]] = sequence
    return tokens
