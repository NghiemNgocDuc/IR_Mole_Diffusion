"""Explicit numeric NPZ schema and molecule-grouped split manifests.

NPZ files are loaded with ``allow_pickle=False``. Atomic types are actual atomic
numbers, coordinates are Angstroms, and active atoms occupy a prefix of each row.
The schema never guesses legacy dataset field names or molecular identities.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_array(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "iu":
        raise ValueError(f"{name} must use an integer dtype, got {value.dtype}.")
    return value.astype(np.int64, copy=False)


def load_native_npz(path, require_molecule_ids=True):
    """Load and validate only the documented padded numeric schema."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except ValueError as exc:
        raise ValueError(
            "Native NPZ loading prohibits object/pickled arrays. Convert the data "
            "to the numeric schema; inspect trusted legacy data through its explicit adapter."
        ) from exc
    for key in ("coords", "atom_types"):
        if key not in arrays:
            raise ValueError(f"Missing native schema field {key!r}; no legacy aliases are inferred.")
    coords = np.asarray(arrays["coords"])
    atoms = _integer_array(arrays["atom_types"], "atom_types")
    if coords.ndim != 3 or coords.shape[-1] != 3 or coords.dtype.kind not in "fi":
        raise ValueError("coords must be a numeric [samples, padded_atoms, 3] array in Angstroms.")
    batch_size, max_atoms, _ = coords.shape
    if batch_size == 0 or max_atoms == 0 or atoms.shape != (batch_size, max_atoms):
        raise ValueError("Nonempty coords and atom_types [samples, padded_atoms] must agree.")
    if not np.isfinite(coords).all():
        raise ValueError("coords contains non-finite values, including padding.")
    if "num_atoms" in arrays:
        counts = _integer_array(arrays["num_atoms"], "num_atoms")
        if counts.shape != (batch_size,) or np.any((counts < 1) | (counts > max_atoms)):
            raise ValueError("num_atoms must be [samples] with counts between 1 and padded_atoms.")
        mask = np.arange(max_atoms)[None, :] < counts[:, None]
    elif "mask" in arrays:
        supplied_mask = np.asarray(arrays["mask"])
        if supplied_mask.dtype.kind != "b" or supplied_mask.shape != atoms.shape:
            raise ValueError("mask must be a boolean [samples, padded_atoms] array.")
        counts = supplied_mask.sum(axis=1).astype(np.int64)
        if np.any(counts < 1):
            raise ValueError("Every sample must contain at least one atom.")
        mask = np.arange(max_atoms)[None, :] < counts[:, None]
        if not np.array_equal(mask, supplied_mask):
            raise ValueError("Active atoms must occupy a contiguous prefix; mask has internal gaps.")
    else:
        raise ValueError("Provide num_atoms and/or boolean mask explicitly; padding is not inferred.")
    if "mask" in arrays:
        supplied_mask = np.asarray(arrays["mask"])
        if supplied_mask.dtype.kind != "b" or not np.array_equal(supplied_mask, mask):
            raise ValueError("mask disagrees with num_atoms or is not boolean.")
    if np.any((atoms[mask] < 1) | (atoms[mask] > 118)):
        raise ValueError("Active atom_types must be actual atomic numbers in 1..118, not vocabulary IDs.")
    if np.any(atoms[~mask] != 0) or np.any(coords[~mask] != 0):
        raise ValueError("Padded atom_types and coordinates must be exactly zero.")
    if "spectra_packed" in arrays and "spectra" in arrays:
        raise ValueError("Provide either spectra or spectra_packed, not both.")
    packed = "spectra_packed" in arrays or "spectra_mask" in arrays
    spectrum_key = "spectra_packed" if "spectra_packed" in arrays else "spectra"
    if spectrum_key not in arrays:
        raise ValueError("Provide spectra or spectra_packed explicitly.")
    spectra = np.asarray(arrays[spectrum_key])
    if spectra.ndim < 2 or spectra.dtype.kind not in "fi" or spectra.shape[0] != batch_size or not np.isfinite(spectra).all():
        raise ValueError("Spectral values must be finite numeric arrays with the same sample count.")
    spectra_mask = None
    if packed:
        if spectra.ndim != 3 or "spectra_mask" not in arrays:
            raise ValueError("Packed spectra require [samples, peaks, features] and spectra_mask.")
        spectra_mask = np.asarray(arrays["spectra_mask"])
        if spectra_mask.dtype.kind != "b" or spectra_mask.shape != spectra.shape[:2]:
            raise ValueError("spectra_mask must be boolean [samples, peaks].")
        if np.any(spectra_mask.sum(axis=1) == 0):
            raise ValueError("Every packed spectrum requires at least one unmasked peak.")
        if np.any(spectra[~spectra_mask] != 0):
            raise ValueError("Masked packed spectral values must be exactly zero.")
    elif spectra.ndim not in (2, 3) or np.any(np.asarray(spectra.shape[1:]) == 0):
        raise ValueError("Dense spectra must be [samples, points] or [samples, channels, points].")
    molecule_ids = None
    if "molecule_ids" in arrays:
        identifiers = np.asarray(arrays["molecule_ids"])
        if identifiers.shape != (batch_size,) or identifiers.dtype.kind not in "USiu":
            raise ValueError("molecule_ids must be a non-object string/integer [samples] array.")
        molecule_ids = identifiers.astype(str)
        if np.any(np.char.str_len(molecule_ids) == 0):
            raise ValueError("molecule_ids cannot be empty.")
    elif require_molecule_ids:
        raise ValueError("molecule_ids is required for grouped splits; no row-level fallback is allowed.")
    bond_types = None
    if "bond_types" in arrays:
        bond_types = _integer_array(arrays["bond_types"], "bond_types")
        if bond_types.shape != (batch_size, max_atoms, max_atoms):
            raise ValueError("bond_types must have shape [samples, padded_atoms, padded_atoms].")
        if np.any((bond_types < 0) | (bond_types > 4)):
            raise ValueError("bond_types codes are 0 none, 1 single, 2 double, 3 triple, 4 aromatic.")
        if not np.array_equal(bond_types, bond_types.swapaxes(1, 2)):
            raise ValueError("bond_types must be symmetric.")
        if np.any(np.diagonal(bond_types, axis1=1, axis2=2) != 0):
            raise ValueError("bond_types diagonal must be zero.")
        pair_mask = mask[:, :, None] & mask[:, None, :]
        if np.any(bond_types[~pair_mask] != 0):
            raise ValueError("Padded atoms cannot have bonds.")
    graph_targets = {}
    if bond_types is not None:
        graph_targets["bond_types"] = bond_types
    tuple_fields = {"angles": 3, "planarity": 4, "torsions": 4, "chirality": 4}
    companions = {"angles": ("angle_cos",), "torsions": ("torsion_periodicity", "torsion_phase", "torsion_amplitude"),
                  "chirality": ("chirality_sign",), "planarity": ()}
    for name, width in tuple_fields.items():
        if name not in arrays:
            if any(companion in arrays for companion in companions[name]):
                raise ValueError(f"{name} indices are required with their target arrays.")
            continue
        if bond_types is None:
            raise ValueError(f"{name} requires explicit bond_types.")
        indices = _integer_array(arrays[name], name)
        if indices.ndim != 3 or indices.shape[0] != batch_size or indices.shape[2] != width:
            raise ValueError(f"{name} must be [samples, tuples, {width}], padded with complete -1 rows.")
        padded = (indices == -1).all(axis=2)
        if np.any((indices == -1).any(axis=2) & ~padded):
            raise ValueError(f"{name} padding must use complete -1 rows.")
        for row in range(batch_size):
            live = indices[row, ~padded[row]]
            if np.any((live < 0) | (live >= counts[row])):
                raise ValueError(f"{name} references padded or out-of-range atoms.")
            if any(len(set(values)) != width for values in live.tolist()):
                raise ValueError(f"{name} tuples must reference distinct atoms.")
        graph_targets[name] = indices
        for companion in companions[name]:
            if companion not in arrays:
                raise ValueError(f"{name} requires target field {companion}.")
            values = np.asarray(arrays[companion])
            if values.shape != indices.shape[:2] or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError(f"{companion} must be finite numeric [samples, tuples].")
            active_values = values[~padded]
            if companion == "angle_cos" and np.any(np.abs(active_values) > 1):
                raise ValueError("angle_cos must lie in [-1, 1].")
            if companion == "torsion_periodicity" and (values.dtype.kind not in "iu" or np.any((active_values < 1) | (active_values > 12))):
                raise ValueError("torsion_periodicity must be integers in 1..12.")
            if companion == "torsion_amplitude" and np.any(active_values < 0):
                raise ValueError("torsion_amplitude must be nonnegative.")
            if companion == "chirality_sign" and np.any(~np.isin(active_values, (-1, 1))):
                raise ValueError("chirality_sign must be -1 or +1.")
            graph_targets[companion] = values.astype(np.int64 if companion == "torsion_periodicity" else np.float32)
    for name in ("bond_mean", "bond_std"):
        if name in arrays:
            values = np.asarray(arrays[name])
            if bond_types is None or values.shape != bond_types.shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError(f"{name} requires finite [samples, atoms, atoms] values and bond_types.")
            if not np.allclose(values, values.swapaxes(1, 2)) or np.any(values[bond_types > 0] <= 0):
                raise ValueError(f"{name} must be symmetric and positive on bonded pairs.")
            graph_targets[name] = values.astype(np.float32)
    extras = {}
    if "selfies_tokens" in arrays:
        tokens = _integer_array(arrays["selfies_tokens"], "selfies_tokens")
        vocabulary = np.asarray(arrays.get("vocabulary"))
        if tokens.ndim != 2 or tokens.shape[0] != batch_size or tokens.shape[1] < 3 or vocabulary.ndim != 1 or vocabulary.dtype.kind not in "US":
            raise ValueError("selfies_tokens requires [samples,length] integers and a string vocabulary.")
        if vocabulary[:3].tolist() != ["<pad>", "<bos>", "<eos>"] or len(np.unique(vocabulary)) != len(vocabulary) or np.any((tokens < 0) | (tokens >= len(vocabulary))):
            raise ValueError("Invalid SELFIES vocabulary or token IDs.")
        for sequence in tokens:
            eos = np.flatnonzero(sequence == 2)
            if sequence[0] != 1 or len(eos) != 1 or eos[0] < 2 or np.any(sequence[eos[0]+1:] != 0) or np.any(sequence[1:eos[0]] < 3):
                raise ValueError("SELFIES sequences require BOS, body, EOS, then PAD.")
        extras.update(selfies_tokens=tokens, vocabulary=vocabulary.tolist())
    if "preprocessing_json" in arrays:
        extras["preprocessing"] = json.loads(str(arrays["preprocessing_json"].item()))
    if "source_rows" in arrays:
        extras["source_rows"] = _integer_array(arrays["source_rows"], "source_rows")
        if extras["source_rows"].shape != (batch_size,) or np.any(extras["source_rows"] < 0):
            raise ValueError("source_rows must contain one nonnegative source index per sample.")
    return {
        "coords": coords.astype(np.float32), "atom_types": atoms,
        "num_atoms": counts, "mask": mask, "spectra": spectra.astype(np.float32),
        "spectra_mask": spectra_mask, "molecule_ids": molecule_ids,
        "bond_types": bond_types, "graph": graph_targets or None, "packed": packed, **extras,
    }


def audit_native_data(data):
    """Describe validated fields without claiming spectrum/geometry provenance."""
    ids = data["molecule_ids"]
    if ids is not None:
        _, per_molecule = np.unique(ids, return_counts=True)
        groups = {"unique_molecules": int(len(per_molecule)),
                  "molecules_with_multiple_rows": int((per_molecule > 1).sum()),
                  "max_rows_per_molecule": int(per_molecule.max())}
    else:
        groups = {"unique_molecules": None, "grouped_split_possible": False}
    active_coords = data["coords"][data["mask"]]
    return {
        "schema_version": SCHEMA_VERSION, "samples": int(len(data["coords"])),
        "padded_atoms": int(data["coords"].shape[1]),
        "min_atoms": int(data["num_atoms"].min()), "max_atoms": int(data["num_atoms"].max()),
        "atomic_numbers": [int(z) for z in np.unique(data["atom_types"][data["mask"]])],
        "coordinate_absolute_max_angstrom": float(np.abs(active_coords).max()),
        "spectral_shape": list(data["spectra"].shape), "packed_spectra": bool(data["packed"]),
        "has_bond_types": data["bond_types"] is not None, **groups,
        "provenance_checks_required": ["coordinate units are Angstroms", "atom_types are actual atomic numbers",
            "molecule_ids group every conformer of the same molecular identity",
            "each spectrum matches its geometry or is explicitly identified as an ensemble",
            "spectral grid, normalization, and broadening match the encoder training data"],
    }


def grouped_split_manifest(molecule_ids, data_path, seed=78438379, val_fraction=0.1, test_fraction=0.1):
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than 1.")
    identifiers = np.asarray(molecule_ids).astype(str)
    unique = np.unique(identifiers)
    if len(unique) < 3:
        raise ValueError("At least three unique molecules are required for nonempty train/validation/test splits.")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    n_test = max(1, int(round(len(unique) * test_fraction)))
    if n_val + n_test >= len(unique):
        raise ValueError("Requested fractions leave no training molecules; lower held-out fractions.")
    groups = {"test": shuffled[:n_test], "val": shuffled[n_test:n_test + n_val],
              "train": shuffled[n_test + n_val:]}
    return {"schema_version": SCHEMA_VERSION, "seed": int(seed),
            "data_sha256": file_sha256(data_path), "sample_count": int(len(identifiers)),
            "val_fraction": float(val_fraction), "test_fraction": float(test_fraction),
            "splits": {name: np.flatnonzero(np.isin(identifiers, group)).tolist()
                       for name, group in groups.items()},
            "molecule_counts": {name: int(len(group)) for name, group in groups.items()}}


def validate_split_manifest(manifest, molecule_ids, data_path):
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported split manifest schema_version.")
    if manifest.get("data_sha256") != file_sha256(data_path):
        raise ValueError("Split manifest data hash differs from this NPZ; create a new manifest explicitly.")
    identifiers = np.asarray(molecule_ids).astype(str)
    splits = manifest.get("splits", {})
    seen_rows, seen_groups = set(), set()
    for name in ("train", "val", "test"):
        rows = splits.get(name)
        if not isinstance(rows, list) or not rows or any(type(row) is not int for row in rows):
            raise ValueError(f"Split {name} must be a nonempty integer row-index list.")
        if any(row < 0 or row >= len(identifiers) for row in rows) or len(set(rows)) != len(rows):
            raise ValueError(f"Split {name} contains invalid or repeated rows.")
        if seen_rows.intersection(rows):
            raise ValueError("Split manifest contains overlapping rows.")
        groups = set(identifiers[rows])
        if seen_groups.intersection(groups):
            raise ValueError("Molecule identity crosses splits; all conformers must stay together.")
        seen_rows.update(rows)
        seen_groups.update(groups)
    if seen_rows != set(range(len(identifiers))) or manifest.get("sample_count") != len(identifiers):
        raise ValueError("Split manifest must cover each data row exactly once.")
    return manifest


def get_or_create_splits(path, molecule_ids, data_path, **kwargs):
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return validate_split_manifest(manifest, molecule_ids, data_path)
    manifest = grouped_split_manifest(molecule_ids, data_path, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class NativeNPZDataset:
    """Dataset protocol with controls confined to one selected split."""

    def __init__(self, data, indices=None, graph_conditioning=False, spectrum_control="none", seed=0):
        self.data = data
        self.indices = np.asarray(indices if indices is not None else np.arange(len(data["coords"])), dtype=np.int64)
        self.graph_conditioning = graph_conditioning
        if graph_conditioning and data["bond_types"] is None:
            raise ValueError("Graph conditioning requires explicit bond_types; bonds are never inferred from targets.")
        if spectrum_control not in ("none", "removed", "shuffled"):
            raise ValueError("spectrum_control must be none, removed, or shuffled.")
        self.spectrum_control = spectrum_control
        self.spectrum_indices = self.indices.copy()
        if spectrum_control == "shuffled":
            ids = data["molecule_ids"]
            groups = np.unique(ids[self.indices]) if ids is not None else None
            if groups is None or len(groups) < 2:
                raise ValueError("A shuffled spectral control requires at least two molecular identities in this split.")
            rng = np.random.default_rng(seed)
            order = rng.permutation(groups)
            donors = {group: order[(position + 1) % len(order)] for position, group in enumerate(order)}
            pools = {}
            for row in self.indices:
                pools.setdefault(ids[row], []).append(row)
            for position, row in enumerate(self.indices):
                self.spectrum_indices[position] = rng.choice(pools[donors[ids[row]]])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row, spectrum_row = int(self.indices[index]), int(self.spectrum_indices[index])
        spectra = self.data["spectra"][spectrum_row]
        if self.spectrum_control == "removed":
            spectra = np.zeros_like(spectra)
        if self.data["packed"]:
            spectral_mask = self.data["spectra_mask"][spectrum_row]
            if self.spectrum_control == "removed":
                spectral_mask = np.zeros_like(spectral_mask)
                spectral_mask[0] = True
            spectra = {"packed": spectra, "mask": spectral_mask}
        item = {"spectra": spectra, **{key: self.data[key][row]
                for key in ("coords", "atom_types", "num_atoms", "mask")}}
        if "selfies_tokens" in self.data:
            item["selfies_tokens"] = self.data["selfies_tokens"][row]
        if self.data["graph"] is not None:
            item["graph"] = {key: values[row] for key, values in self.data["graph"].items()}
        return item


def collate_native(batch):
    import torch
    from torch.utils.data._utils.collate import default_collate
    result = default_collate(batch)
    result["coords"] = result["coords"].float()
    result["atom_types"] = result["atom_types"].long()
    result["num_atoms"] = result["num_atoms"].long()
    result["mask"] = result["mask"].bool()
    return result
