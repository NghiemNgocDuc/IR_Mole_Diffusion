"""Achiral molecular graph comparison with consistent explicit hydrogens."""


def graph_key(value):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(value) if isinstance(value, str) else Chem.Mol(value)
    if molecule is None:
        raise ValueError('Cannot parse molecular graph')
    Chem.SanitizeMol(molecule)
    molecule = Chem.AddHs(molecule)
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    # Preserve isotope/charge/bond distinctions while intentionally ignoring stereo.
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True, allHsExplicit=True)


def same_achiral_graph(left, right):
    return graph_key(left) == graph_key(right)
