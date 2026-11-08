"""Count explicitly defined structural motifs before/after data preparation."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "carbonyl_C_equals_O": "[C,c]=[O]",
    "hydroxyl_O_H": "[O;H1]",
    "oxygen_bonded_to_two_carbons_including_esters": "[OD2]([#6])[#6]",
    "amide_N_C_equals_O": "[NX3][CX3](=[OX1])",
    "nitrile_C_triple_N": "[C]#[N]",
    "nitro_charge_separated": "[N+](=[O])[O-]",
    "nitrogen_bearing_hydrogen": "[N,n;!H0]",
}


def main():
    metadata = json.loads((ROOT / "data/processed/squirl.metadata.json").read_text())
    manifest = json.loads((ROOT / "data/processed/splits.json").read_text())
    # This is the explicitly trusted legacy input already used by prepare_squirl.
    with np.load(ROOT / "data/SQuIRL/squirl_ir_3d.npz", allow_pickle=True) as z:
        smiles = z["smiles"]
    with np.load(ROOT / "data/processed/squirl.npz", allow_pickle=False) as z:
        source_rows = z["source_rows"]
    split_for_source = {}
    first500_sources = set(map(int, source_rows[np.asarray(manifest["splits"]["val"][:500])]))
    for name, rows in manifest["splits"].items():
        split_for_source.update({int(source_rows[i]): name for i in rows})
    reject_reason = {r["source_row"]: r["reason"] for r in metadata["rejected"]}
    patterns = {name: Chem.MolFromSmarts(s) for name, s in PATTERNS.items()}
    assert all(p is not None for p in patterns.values())
    counts, sizes = defaultdict(Counter), Counter()
    examples = defaultdict(list)
    for i, text in enumerate(smiles):
        labels = ["raw"]
        if i in split_for_source:
            labels += ["accepted", split_for_source[i]]
            if i in first500_sources:
                labels += ["first500_val"]
        else:
            labels += ["rejected", "rejected: " + reject_reason[i]]
        mol = Chem.MolFromSmiles(str(text))
        features = {"unparseable_smiles": mol is None}
        if mol is not None:
            features.update({k: mol.HasSubstructMatch(p) for k, p in patterns.items()})
            features.update({
                "aromatic_atoms": any(a.GetIsAromatic() for a in mol.GetAtoms()),
                "any_formally_charged_atom": any(a.GetFormalCharge() for a in mol.GetAtoms()),
                "multiple_fragments": len(Chem.GetMolFrags(mol)) > 1,
                "contains_fluorine": any(a.GetAtomicNum() == 9 for a in mol.GetAtoms()),
            })
        for label in labels:
            sizes[label] += 1
            for name, value in features.items():
                counts[label][name] += int(value)
        if i not in split_for_source and len(examples[reject_reason[i]]) < 5:
            examples[reject_reason[i]].append({"source_row": i, "smiles": str(text)})
        if (i + 1) % 25000 == 0:
            print(f"Functional coverage {i+1}/{len(smiles)}", flush=True)
    result = {"rdkit_version": rdBase.rdkitVersion, "smarts_patterns": PATTERNS,
              "description": "Overlapping motif presence per molecule, not disjoint classes. Counts measure dataset coverage, not model accuracy.",
              "groups": {label: {"rows": n, "motif_counts": dict(counts[label]),
                                    "motif_fractions": {k: c/n for k, c in counts[label].items()}}
                         for label, n in sizes.items()},
              "rejection_examples": dict(examples)}
    path = ROOT / "outputs/dataset_review/functional_coverage_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
