"""Convert trusted SQuIRL data to numeric arrays with verified graph alignment."""
import argparse
import json
from pathlib import Path

import numpy as np

from diffusion.data import file_sha256, load_native_npz, get_or_create_splits
from diffusion.preprocessing import aligned_bond_types, pack_peaks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/SQuIRL/squirl_ir_3d.npz")
    parser.add_argument("--output", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--trust_legacy", action="store_true", help="Allow object arrays in this trusted input")
    parser.add_argument("--max_peaks", type=int, default=81)
    parser.add_argument("--frequency_scale", type=float, default=4000.)
    parser.add_argument("--intensity_scale", type=float, default=1000.)
    parser.add_argument("--limit", type=int, help="Diagnostic subset only; use a separate output/splits path")
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--max_rejected_fraction", type=float, default=0.10,
                        help="Fail above this fraction; every rejected source row is recorded")
    args = parser.parse_args(argv)
    if not args.trust_legacy:
        parser.error("Conversion requires --trust_legacy for the supplied object-array NPZ.")
    output, split_path = Path(args.output), Path(args.splits)
    if output.exists() or split_path.exists():
        parser.error("Output or split manifest already exists. Reuse it, or choose fresh paths.")
    if not 0 <= args.max_rejected_fraction < 1 or (args.limit is not None and args.limit < 3):
        parser.error("Invalid rejection fraction or diagnostic limit.")
    from rdkit import Chem, rdBase
    import selfies as sf
    with np.load(args.input, allow_pickle=True) as archive:
        required = ("coords", "atom_types", "ir_frequencies", "ir_intensities", "smiles")
        if any(key not in archive.files for key in required):
            parser.error("Input lacks required SQuIRL geometry/IR fields.")
        raw = {key: archive[key] for key in required}
    rows = len(raw["smiles"]) if args.limit is None else min(args.limit, len(raw["smiles"]))
    if any(len(value) != len(raw["smiles"]) for value in raw.values()):
        raise ValueError("Input fields have inconsistent row counts.")
    accepted, rejected, strings, ids = [], [], [], []
    scale = dict(max_peaks=args.max_peaks, frequency_scale=args.frequency_scale,
                 intensity_scale=args.intensity_scale)
    for row in range(rows):
        try:
            coords = np.asarray(raw["coords"][row], dtype=np.float32)
            atoms = np.asarray(raw["atom_types"][row], dtype=np.int64)
            if coords.shape != (len(atoms), 3) or not np.isfinite(coords).all() or not len(atoms):
                raise ValueError("Invalid coordinate/atom shape or nonfinite coordinates.")
            if not np.isin(atoms, [1, 6, 7, 8, 9]).all():
                raise ValueError("This SQuIRL recipe supports neutral H/C/N/O/F molecules.")
            smiles = str(raw["smiles"][row])
            mol = Chem.MolFromSmiles(smiles)
            bonds = aligned_bond_types(smiles, atoms, coords)
            # Canonical achiral identity keeps enantiomers in the same held-out group.
            identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            target = sf.encoder(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
            restored = Chem.MolFromSmiles(sf.decoder(target))
            if Chem.MolToSmiles(restored, isomericSmiles=False) != identity:
                raise ValueError("SELFIES round trip changed molecular identity.")
            packed, peak_mask = pack_peaks(raw["ir_frequencies"][row], raw["ir_intensities"][row], **scale)
            coords = coords - coords.mean(0, keepdims=True)
            accepted.append((row, coords, atoms, bonds, packed, peak_mask))
            strings.append(list(sf.split_selfies(target)))
            ids.append(identity)
        except (ValueError, RuntimeError, sf.EncoderError, sf.DecoderError) as exc:
            rejected.append({"source_row": row, "smiles": str(raw["smiles"][row]), "reason": str(exc)})
        if (row + 1) % 5000 == 0:
            print(f"Validated {row+1}/{rows}; quarantined {len(rejected)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"conversion_version": 1, "source": str(Path(args.input).resolve()),
        "source_sha256": file_sha256(args.input), "source_rows_checked": rows,
        "accepted_rows": len(accepted), "rejected_rows": len(rejected), "rejected": rejected,
        "rdkit_version": rdBase.rdkitVersion, "selfies_version": sf.__version__,
        "preprocessing": scale, "diagnostic_subset": args.limit is not None,
        "identity_grouping": "canonical non-isomeric SMILES; stereoisomers grouped together",
        "graph_alignment": "coordinate connectivity matched to source SMILES by element graph isomorphism",
        "units": "angstrom (input convention; not independently determined)",
        "spectrum_provenance": "supplied SQuIRL pairing; not independently quantum-validated"}
    report_path = output.with_suffix(".metadata.json")
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if len(rejected) / rows > args.max_rejected_fraction or len(set(ids)) < 3:
        raise ValueError(f"Conversion rejected too many rows or has too few molecules; inspect {report_path}.")
    vocabulary = ["<pad>", "<bos>", "<eos>"] + sorted({token for sequence in strings for token in sequence})
    lookup = {token: index for index, token in enumerate(vocabulary)}
    size, atoms_max, length = len(accepted), max(len(item[2]) for item in accepted), max(map(len, strings)) + 2
    coords = np.zeros((size, atoms_max, 3), np.float32)
    atoms = np.zeros((size, atoms_max), np.int64)
    bonds = np.zeros((size, atoms_max, atoms_max), np.int8)
    tokens = np.zeros((size, length), np.int64)
    for i, ((_, xyz, numbers, topology, _, _), sequence) in enumerate(zip(accepted, strings)):
        count = len(numbers)
        coords[i, :count], atoms[i, :count], bonds[i, :count, :count] = xyz, numbers, topology
        tokens[i, :len(sequence)+2] = [1] + [lookup[t] for t in sequence] + [2]
    arrays = dict(coords=coords, atom_types=atoms, bond_types=bonds,
        num_atoms=np.array([len(item[2]) for item in accepted], dtype=np.int64),
        spectra_packed=np.stack([item[4] for item in accepted]),
        spectra_mask=np.stack([item[5] for item in accepted]), molecule_ids=np.asarray(ids),
        selfies_tokens=tokens, vocabulary=np.asarray(vocabulary),
        source_rows=np.asarray([item[0] for item in accepted], dtype=np.int64),
        preprocessing_json=np.asarray(json.dumps(scale)))
    # Save only numeric/unicode arrays. The training path never loads pickle.
    np.savez_compressed(output, **arrays)
    verified = load_native_npz(output)
    manifest = get_or_create_splits(split_path, verified["molecule_ids"], output, seed=args.seed)
    metadata.update(data_sha256=manifest["data_sha256"], max_atoms=atoms_max,
                    max_sequence_length=length, vocabulary=vocabulary)
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data": str(output), "splits": str(split_path), "accepted": size,
                      "quarantined": len(rejected), "metadata": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
