"""Read-only chemical and split audit of the prepared SQuIRL data.

Run from the repository root with the bundled Python. This measures dataset
coverage, not predictive accuracy. RDKit default tautomer enumeration caps are
retained and cap-hit rows are reported explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "histogram": {str(k): v for k, v in sorted(Counter(values).items())}}


def overlap(records, indices, training_indices, field, skip_empty=False):
    training_values = {records[i][field] for i in training_indices}
    eligible = [i for i in indices if not skip_empty or records[i][field]]
    matched = [i for i in eligible if records[i][field] in training_values]
    unique = {records[i][field] for i in eligible}
    return {"eligible_rows": len(eligible), "matching_train_rows": len(matched),
            "fraction_rows_matching_train": len(matched) / len(eligible) if eligible else None,
            "unique_values": len(unique), "unique_values_matching_train": len(unique & training_values)}


def run(args):
    start = time.monotonic()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.splits).read_text())
    with np.load(args.data, allow_pickle=False) as data:
        identities = data["molecule_ids"]
        source_rows = data["source_rows"]
        tokens = data["selfies_tokens"]
        vocabulary = data["vocabulary"]
        num_atoms = data["num_atoms"]
    splits = {name: np.asarray(rows, dtype=int) for name, rows in manifest["splits"].items()}
    assert len(identities) == manifest["sample_count"]
    assert len(np.unique(np.concatenate(list(splits.values())))) == len(identities)
    data_hash = file_sha256(args.data)
    assert data_hash == manifest["data_sha256"]
    row_split = [None] * len(identities)
    for name, indices in splits.items():
        for i in indices:
            assert row_split[i] is None
            row_split[i] = name
    enum = rdMolStandardize.TautomerEnumerator()
    tautomer_status = Counter()
    records = []
    fields = ["row", "source_row", "split", "smiles", "formula", "heavy_atoms", "total_atoms", "hetero_atoms",
              "N", "O", "F", "rings", "aromatic_rings", "aromatic_atoms", "rotatable_bonds", "formal_charge",
              "scaffold", "tautomer_family", "tautomer_scaffold", "tautomer_count", "tautomer_status", "target_tokens"]
    cap_rows = []
    with (output / "chemical_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, text in enumerate(identities):
            mol = Chem.MolFromSmiles(str(text))
            if mol is None:
                raise ValueError(f"Unparseable prepared identity at row {i}")
            elements = Counter(a.GetSymbol() for a in mol.GetAtoms())
            tautomer_result = enum.Enumerate(mol)
            status = str(tautomer_result.status)
            tautomer_status[status] += 1
            if status != "Completed":
                cap_rows.append({"row": i, "status": status})
            canonical_tautomer = enum.PickCanonical(tautomer_result)
            row = {"row": i, "source_row": int(source_rows[i]), "split": row_split[i], "smiles": str(text),
                   "formula": rdMolDescriptors.CalcMolFormula(mol), "heavy_atoms": mol.GetNumHeavyAtoms(),
                   "total_atoms": int(num_atoms[i]), "hetero_atoms": sum(elements[e] for e in ("N", "O", "F")),
                   **{e: elements[e] for e in ("N", "O", "F")}, "rings": rdMolDescriptors.CalcNumRings(mol),
                   "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                   "aromatic_atoms": sum(a.GetIsAromatic() for a in mol.GetAtoms()),
                   "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol), "formal_charge": Chem.GetFormalCharge(mol),
                   "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
                   "tautomer_family": Chem.MolToSmiles(canonical_tautomer, isomericSmiles=False),
                   "tautomer_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=canonical_tautomer, includeChirality=False),
                   "tautomer_count": len(tautomer_result), "tautomer_status": status,
                   "target_tokens": int(np.count_nonzero(tokens[i])) - 2}
            records.append(row)
            writer.writerow(row)
            if (i + 1) % 5000 == 0 or i + 1 == len(identities):
                elapsed = time.monotonic() - start
                print(f"Chemical audit {i+1}/{len(identities)} ({(i+1)/len(identities):.1%}); elapsed {elapsed:.0f}s", flush=True)
                handle.flush()
    analysis_splits = dict(splits)
    analysis_splits["val_first_500"] = splits["val"][:500]
    summaries = {}
    numeric_fields = ["heavy_atoms", "total_atoms", "hetero_atoms", "N", "O", "F", "rings", "aromatic_rings",
                      "aromatic_atoms", "rotatable_bonds", "target_tokens", "tautomer_count"]
    train_token_set = set(map(int, tokens[splits["train"]].ravel()))
    for name, indices in analysis_splits.items():
        rows = [records[i] for i in indices]
        token_counts = Counter(map(int, tokens[indices].ravel()))
        summaries[name] = {"rows": len(rows), "unique_identity_count": len({r["smiles"] for r in rows}),
                           "unique_formulas": len({r["formula"] for r in rows}),
                           "unique_tautomer_families": len({r["tautomer_family"] for r in rows}),
                           "unique_nonempty_scaffolds": len({r["scaffold"] for r in rows} - {""}),
                           "acyclic_rows": sum(not r["scaffold"] for r in rows),
                           "rows_with_aromatic_atoms": sum(bool(r["aromatic_atoms"]) for r in rows),
                           "rows_with_each_element": {e: sum(bool(r[e]) for r in rows) for e in ("N", "O", "F")},
                           "charged_rows": sum(r["formal_charge"] != 0 for r in rows),
                           "source_row_min_max": [int(source_rows[indices].min()), int(source_rows[indices].max())],
                           "row_indices_ascending": bool(np.all(np.diff(indices) > 0)),
                           "distributions": {key: distribution([r[key] for r in rows]) for key in numeric_fields},
                           "top_formulas": Counter(r["formula"] for r in rows).most_common(12),
                           "top_scaffolds": Counter(r["scaffold"] for r in rows if r["scaffold"]).most_common(12),
                           "token_counts": {str(vocabulary[k]): n for k, n in sorted(token_counts.items())},
                           "tokens_absent_from_train": [str(vocabulary[k]) for k in sorted(set(token_counts) - train_token_set)]}
        if name != "train":
            summaries[name]["overlap_with_train"] = {
                field: overlap(records, indices, splits["train"], field, skip_empty=field.endswith("scaffold"))
                for field in ("smiles", "formula", "tautomer_family", "scaffold", "tautomer_scaffold")}
    exact_pairwise = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        exact_pairwise[f"{left}__{right}"] = len(set(identities[splits[left]]) & set(identities[splits[right]]))
    first = summaries["val_first_500"]
    full = summaries["val"]
    first500_comparison = {}
    for field in numeric_fields:
        a = np.asarray([records[i][field] for i in splits["val"][:500]])
        b = np.asarray([records[i][field] for i in splits["val"]])
        levels = np.unique(b)
        ks = max(abs(np.mean(a <= v) - np.mean(b <= v)) for v in levels)
        first500_comparison[field] = {"first_500_mean": float(a.mean()), "all_val_mean": float(b.mean()),
                                      "absolute_ecdf_max_difference": float(ks),
                                      "note": "Descriptive difference; subsets overlap, so this is not a two-sample significance test."}
    train_families = {}
    for i in splits["train"]:
        train_families.setdefault(records[i]["tautomer_family"], int(i))
    overlap_examples = []
    for i in splits["test"]:
        key = records[i]["tautomer_family"]
        if key in train_families:
            j = train_families[key]
            overlap_examples.append({"test_row": int(i), "test_smiles": records[i]["smiles"],
                                     "train_row": j, "train_smiles": records[j]["smiles"], "canonical_tautomer": key})
            if len(overlap_examples) == 10:
                break
    report = {"schema_version": 1, "data_path": str(args.data), "data_sha256": data_hash,
              "splits_sha256": file_sha256(args.splits), "rdkit_version": rdBase.rdkitVersion,
              "rows": len(identities), "elapsed_seconds": time.monotonic()-start,
              "definitions": {"identity": "Prepared canonical non-isomeric SMILES; exact tautomer-specific graph identity.",
                              "scaffold": "RDKit Bemis-Murcko scaffold retaining atom types and bond orders, without chirality; acyclic empty scaffolds excluded from overlap denominator.",
                              "tautomer_family": "RDKit default canonical tautomer followed by non-isomeric SMILES. Diagnostic chemical grouping, not a replacement for strict identity accuracy.",
                              "tautomer_scaffold": "Bemis-Murcko scaffold of RDKit canonical tautomer; excludes acyclic rows.",
                              "target_tokens": "SELFIES target non-padding length excluding BOS and EOS.",
                              "overlap": "Rows whose grouping key exists in train. Repeated scaffold/formula is a generalization characteristic, not by itself label leakage."},
              "tautomer_enumeration": {"max_tautomers": enum.GetMaxTautomers(), "max_transforms": enum.GetMaxTransforms(),
                                       "statuses": dict(tautomer_status), "non_completed_rows": cap_rows},
              "exact_identity_overlap": exact_pairwise, "splits": summaries,
              "first_500_val_comparison": first500_comparison,
              "test_train_tautomer_overlap_examples": overlap_examples,
              "challenge_row_24778": records[24778],
              "limitations": ["This audit measures input/target coverage, not model accuracy by molecular class.",
                              "A scaffold overlap is not an exact identity overlap; scaffold-disjoint performance needs a separately designed evaluation.",
                              "Tautomer canonicalization is implementation-dependent and can alter stereo annotations; this audit uses non-isomeric grouping.",
                              "First 500 validation rows are a subset of full validation; the ECDF difference is descriptive, not an independent-sample test."]}
    target = output / "chemical_split_audit.json"
    target.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Wrote {target}", flush=True)
    for name in ("val", "test", "val_first_500"):
        print(name, json.dumps({"rows": summaries[name]["rows"], "heavy_atoms_mean": summaries[name]["distributions"]["heavy_atoms"]["mean"],
                                "overlap": summaries[name]["overlap_with_train"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--output-dir", default="outputs/dataset_review")
    run(parser.parse_args())
