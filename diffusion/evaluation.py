"""Transparent geometry metrics with optional RDKit graph checks/refinement."""
 
import numpy as np

def kabsch_rmsd(predicted, reference):
    """Fixed-correspondence RMSD after translation and a proper rotation.

    This never permits reflections and is not an atom-permutation-aware metric.
    It cannot establish chemical identity or physical validity.
    """
    predicted, reference = np.asarray(predicted, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 3 or not len(predicted):
        raise ValueError("RMSD inputs must be matching nonempty [atoms, 3] arrays.")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        return float("nan")
    left, right = predicted - predicted.mean(axis=0), reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(left.T @ right)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotated = left @ (u @ correction @ vt)
    return float(np.sqrt(np.square(rotated - right).sum() / len(left)))


def geometry_summary(samples, references, atom_types, mask, thresholds=(0.5, 1.0, 2.0), progress_callback=None):
    """Summarize [rows, candidates, atoms, 3], retaining failures in hit rates."""
    samples = np.asarray(samples)
    references, atom_types, mask = np.asarray(references), np.asarray(atom_types), np.asarray(mask, dtype=bool)
    if samples.ndim != 4 or samples.shape[0] != len(references) or samples.shape[2:] != references.shape[1:]:
        raise ValueError("Samples must have shape [rows, candidates, padded_atoms, 3] matching references.")
    if samples.shape[1] < 1 or len(samples) == 0 or atom_types.shape != mask.shape or mask.shape != references.shape[:2]:
        raise ValueError("Nonempty sample dimensions, atom_types, mask, and references must agree.")
    rmsds = np.full(samples.shape[:2], np.nan)
    finite = np.zeros(samples.shape[:2], dtype=bool)
    near_coincident = np.zeros(samples.shape[:2], dtype=bool)
    centroids = np.full(samples.shape[:2], np.nan)
    for row in range(len(samples)):
        active = mask[row]
        heavy = active & (atom_types[row] > 1)
        # All-H inputs remain evaluable, but are explicitly counted below.
        rmsd_mask = heavy if heavy.any() else active
        for candidate in range(samples.shape[1]):
            coords = samples[row, candidate, active]
            finite[row, candidate] = np.isfinite(coords).all()
            if not finite[row, candidate]:
                continue
            rmsds[row, candidate] = kabsch_rmsd(samples[row, candidate, rmsd_mask], references[row, rmsd_mask])
            centroids[row, candidate] = np.linalg.norm(coords.mean(axis=0))
            if len(coords) > 1:
                distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                near_coincident[row, candidate] = bool(np.any(distances[np.triu_indices(len(coords), 1)] < 0.2))
        if progress_callback is not None:
            progress_callback(row + 1)
    valid_rmsds = rmsds[np.isfinite(rmsds)]
    summary = {
        "metric": "fixed_correspondence_proper_rotation_heavy_atom_RMSD_angstrom",
        "rows": int(len(samples)), "candidates_per_row": int(samples.shape[1]),
        "total_candidates": int(finite.size), "finite_candidates": int(finite.sum()),
        "finite_candidate_fraction": float(finite.mean()),
        "all_hydrogen_rows_using_all_atoms": int((~((atom_types > 1) & mask).any(axis=1)).sum()),
        "mean_rmsd_finite_candidates": float(valid_rmsds.mean()) if len(valid_rmsds) else None,
        "median_rmsd_finite_candidates": float(np.median(valid_rmsds)) if len(valid_rmsds) else None,
        "near_coincident_atom_fraction_all_candidates": float(near_coincident.mean()),
        "near_coincident_threshold_angstrom": 0.2,
        "max_centroid_norm_angstrom": float(np.nanmax(centroids)) if finite.any() else None,
        "first_sample_rmsd_success_fraction": {}, "best_of_k_rmsd_success_fraction": {},
        "interpretation": "RMSD uses supplied atom correspondence; graph identity, atom symmetry, stereochemistry, and chemical validity are unverified. The first sample is unranked.",
    }
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError("RMSD thresholds must be positive.")
        hits = np.isfinite(rmsds) & (rmsds <= threshold)
        summary["first_sample_rmsd_success_fraction"][str(threshold)] = float(hits[:, 0].mean())
        summary["best_of_k_rmsd_success_fraction"][str(threshold)] = float(hits.any(axis=1).mean())
    return summary, rmsds


def _reference_molecule(atom_types, coords, bond_types):
    from rdkit import Chem
    molecule = Chem.RWMol()
    for atomic_number in atom_types:
        molecule.AddAtom(Chem.Atom(int(atomic_number)))
    bond_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for left in range(len(atom_types)):
        for right in range(left + 1, len(atom_types)):
            order = int(bond_types[left, right])
            if order:
                molecule.AddBond(left, right, bond_map[order])
    molecule = molecule.GetMol()
    Chem.SanitizeMol(molecule)
    conformer = Chem.Conformer(len(atom_types))
    for index, point in enumerate(coords):
        conformer.SetAtomPosition(index, [float(value) for value in point])
    molecule.AddConformer(conformer)
    Chem.AssignStereochemistryFrom3D(molecule, replaceExistingTags=True)
    return molecule


def chemical_summary(samples, references, atom_types, mask, bond_types, charge=0, thresholds=(0.5, 1.0, 2.0), progress_callback=None):
    """Optional neutral-graph reconstruction and symmetry-aware RMSD.

    Reference stereochemistry is assigned from its supplied geometry. Charged
    references are unsupported by the native bond-only schema and rejected.
    This is a check under RDKit bond perception, not quantum stability.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds, rdMolAlign
    except ImportError as exc:
        raise RuntimeError("--chemistry requires RDKit. Omit the flag for explicitly labeled geometric metrics.") from exc
    if charge != 0:
        raise ValueError("Native bond-only chemistry evaluation currently supports neutral molecules only.")
    if bond_types is None:
        raise ValueError("--chemistry requires reference bond_types; target identity is never guessed.")
    num_rows, num_candidates = samples.shape[:2]
    valid, matched = np.zeros((num_rows, num_candidates), dtype=bool), np.zeros((num_rows, num_candidates), dtype=bool)
    rmsds = np.full((num_rows, num_candidates), np.nan)
    for row in range(num_rows):
        active = np.flatnonzero(mask[row])
        atom_row, reference_coords = atom_types[row, active], references[row, active]
        try:
            reference = _reference_molecule(atom_row, reference_coords, bond_types[row][np.ix_(active, active)])
        except Exception as exc:
            raise ValueError(f"Reference graph row {row} cannot be sanitized: {exc}") from exc
        reference_smiles = Chem.MolToSmiles(reference, isomericSmiles=True)
        for candidate in range(num_candidates):
            coords = samples[row, candidate, active]
            if not np.isfinite(coords).all():
                continue
            try:
                xyz = str(len(atom_row)) + "\n\n" + "\n".join(
                    f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}"
                    for z, p in zip(atom_row, coords))
                molecule = Chem.MolFromXYZBlock(xyz)
                rdDetermineBonds.DetermineBonds(molecule, charge=charge, maxIterations=1000)
                Chem.SanitizeMol(molecule)
                Chem.AssignStereochemistryFrom3D(molecule, replaceExistingTags=True)
                valid[row, candidate] = len(Chem.GetMolFrags(molecule)) == 1
                matched[row, candidate] = valid[row, candidate] and Chem.MolToSmiles(molecule, isomericSmiles=True) == reference_smiles
                if matched[row, candidate]:
                    probe, target = Chem.RemoveHs(molecule), Chem.RemoveHs(reference)
                    if probe.GetNumAtoms() == 0:
                        probe, target = molecule, reference
                    rmsds[row, candidate] = rdMolAlign.GetBestRMS(probe, target, maxMatches=100000)
            except Exception:
                # Every failed perception/alignment remains in the denominator.
                continue
        if progress_callback is not None:
            progress_callback(row + 1)
    result = {
        "backend": "RDKit distance/valence bond perception, neutral molecules",
        "valid_connected_fraction_all_candidates": float(valid.mean()),
        "graph_and_geometry_assigned_stereo_match_fraction": float(matched.mean()),
        "first_unranked_graph_match_fraction": float(matched[:, 0].mean()),
        "any_graph_match_fraction_best_of_k": float(matched.any(axis=1).mean()),
        "symmetry_aware_heavy_atom_rmsd_first_sample_success": {},
        "symmetry_aware_heavy_atom_rmsd_best_of_k_success": {},
        "interpretation": "Valid means RDKit-sanitized and connected, not energetic stability. Stereo is derived from supplied reference 3D geometry; ordinary IR alone may not identify its handedness.",
    }
    for threshold in thresholds:
        hits = matched & np.isfinite(rmsds) & (rmsds <= threshold)
        result["symmetry_aware_heavy_atom_rmsd_first_sample_success"][str(threshold)] = float(hits[:, 0].mean())
        result["symmetry_aware_heavy_atom_rmsd_best_of_k_success"][str(threshold)] = float(hits.any(axis=1).mean())
    return result, rmsds


def refine_with_known_graph(samples, atom_types, mask, bond_types, max_iterations=200, progress_callback=None):
    """MMFF refinement only with an explicitly available inference graph.

    Failed/unsupported candidates retain their raw coordinates and an explicit
    status, so refinement never silently removes unsuccessful samples.
    """
    try:
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("--refine requires RDKit and an explicit inference graph.") from exc
    refined = np.array(samples, copy=True)
    statuses = np.full(samples.shape[:2], "not_run", dtype="U64")
    for row in range(len(samples)):
        active = np.flatnonzero(mask[row])
        for candidate in range(samples.shape[1]):
            coords = samples[row, candidate, active]
            if not np.isfinite(coords).all():
                statuses[row, candidate] = "nonfinite_raw"
                continue
            try:
                molecule = _reference_molecule(atom_types[row, active], coords, bond_types[row][np.ix_(active, active)])
                if not AllChem.MMFFHasAllMoleculeParams(molecule):
                    statuses[row, candidate] = "unsupported_mmff_parameters"
                    continue
                code = AllChem.MMFFOptimizeMolecule(molecule, maxIters=max_iterations)
                if code != 0:
                    statuses[row, candidate] = "optimization_not_converged_raw_retained"
                    continue
                refined[row, candidate, active] = molecule.GetConformer().GetPositions()
                statuses[row, candidate] = "converged"
            except Exception:
                statuses[row, candidate] = "refinement_failed_raw_retained"
        if progress_callback is not None:
            progress_callback(row + 1)
    return refined, statuses
