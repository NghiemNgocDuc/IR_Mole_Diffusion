"""Shared training/inference peak preprocessing and verified atom mapping."""
import numpy as np


def pack_peaks(frequencies, intensities, max_peaks=81, frequency_scale=4000., intensity_scale=1000.):
    x, y = np.asarray(frequencies, dtype=np.float32), np.asarray(intensities, dtype=np.float32)
    if x.ndim != 1 or x.shape != y.shape or not len(x):
        raise ValueError("Frequencies/intensities must be matching nonempty vectors.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x <= 0).any() or (y < 0).any():
        raise ValueError("Peaks require positive finite frequencies and nonnegative finite intensities.")
    if max_peaks < 1 or frequency_scale <= 0 or intensity_scale <= 0:
        raise ValueError("Peak capacity and scales must be positive.")
    if len(x) > max_peaks:
        selected = np.argsort(y, kind="stable")[-max_peaks:]
        x, y = x[selected], y[selected]
    order = np.argsort(x, kind="stable")
    packed, mask = np.zeros((max_peaks, 2), np.float32), np.zeros(max_peaks, bool)
    packed[:len(x), 0], packed[:len(x), 1] = x[order] / frequency_scale, y[order] / intensity_scale
    mask[:len(x)] = True
    return packed, mask


def bond_array(molecule):
    from rdkit import Chem
    codes = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
             Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    result = np.zeros((molecule.GetNumAtoms(),) * 2, dtype=np.int64)
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in codes:
            raise ValueError("Unsupported chemical bond type.")
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        result[i, j] = result[j, i] = codes[bond.GetBondType()]
    return result


def aligned_bond_types(smiles, atoms, coordinates):
    """Transfer known bond orders only after connectivity/element isomorphism.

    Distance perception is used to validate an atom mapping, not to invent a
    target molecular identity. Invalid or rearranged geometries are rejected.
    """
    import networkx as nx
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    reference = Chem.MolFromSmiles(smiles)
    if reference is None or len(Chem.GetMolFrags(reference)) != 1 or any(a.GetFormalCharge() for a in reference.GetAtoms()):
        raise ValueError("Expected a connected molecule with no formal atomic charges; charged graphs need an extended schema.")
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in reference.GetAtoms()):
        raise ValueError("Radicals and isotope-labeled molecules are outside this recipe's domain.")
    reference = Chem.AddHs(reference)
    if reference.GetNumAtoms() != len(atoms):
        raise ValueError("SMILES/coordinate atom counts differ.")
    block = str(len(atoms)) + "\n\n" + "\n".join(
        f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
        for z, p in zip(atoms, coordinates))
    perceived = Chem.MolFromXYZBlock(block)
    if perceived is None:
        raise ValueError("Coordinates could not be read as XYZ.")
    rdDetermineBonds.DetermineConnectivity(perceived)
    def connectivity(molecule):
        graph = nx.Graph()
        graph.add_nodes_from((a.GetIdx(), {"z": a.GetAtomicNum()}) for a in molecule.GetAtoms())
        graph.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds())
        return graph
    source, target = connectivity(reference), connectivity(perceived)
    matcher = nx.isomorphism.GraphMatcher(source, target,
        node_match=nx.isomorphism.categorical_node_match("z", 0))
    mapping = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        raise ValueError("Coordinate connectivity does not match the reference SMILES.")
    original, aligned = bond_array(reference), np.zeros((len(atoms), len(atoms)), dtype=np.int64)
    for i, j in source.edges:
        aligned[mapping[i], mapping[j]] = aligned[mapping[j], mapping[i]] = original[i, j]
    return aligned
