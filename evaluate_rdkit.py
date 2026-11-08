"""Graph-only ETKDG/MMFF baseline on the identical held-out molecules."""
import argparse
import json
from pathlib import Path

import numpy as np

from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.evaluation import _reference_molecule, chemical_summary, geometry_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--output_dir", default="evaluation/rdkit_val")
    args = parser.parse_args(argv)
    if min(args.limit, args.num_samples) < 1:
        parser.error("Limit and number of samples must be positive.")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    data = load_native_npz(args.data)
    manifest = validate_split_manifest(json.loads(Path(args.splits).read_text()), data["molecule_ids"], args.data)
    if data["bond_types"] is None:
        parser.error("Baseline needs an explicit reference graph.")
    rows = np.array(manifest["splits"][args.split][:args.limit])
    samples = np.full((len(rows), args.num_samples, *data["coords"].shape[1:]), np.nan, dtype=np.float32)
    statuses = np.full((len(rows), args.num_samples), "embedding_failed", dtype="U64")
    for index, row in enumerate(rows):
        count = int(data["num_atoms"][row])
        # Build connectivity in coordinate order, then discard reference geometry
        # and stereo before embedding. No IR or target conformer is used by ETKDG.
        molecule = _reference_molecule(data["atom_types"][row, :count],
            data["coords"][row, :count], data["bond_types"][row, :count, :count])
        molecule.RemoveAllConformers()
        Chem.RemoveStereochemistry(molecule)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = (args.seed + index) % (2**31-1)
        parameters.numThreads = 1
        conformers = AllChem.EmbedMultipleConfs(molecule, numConfs=args.num_samples, params=parameters)
        for candidate, conf_id in enumerate(conformers):
            status = "etkdg_only"
            if AllChem.MMFFHasAllMoleculeParams(molecule):
                try:
                    code = AllChem.MMFFOptimizeMolecule(molecule, confId=conf_id, maxIters=500)
                    status = "mmff_converged" if code == 0 else "mmff_not_converged"
                except (ValueError, RuntimeError):
                    status = "mmff_failed_etkdg_retained"
            xyz = molecule.GetConformer(conf_id).GetPositions()
            samples[index, candidate] = 0
            samples[index, candidate, :count] = xyz - xyz.mean(0)
            statuses[index, candidate] = status
    refs, atoms, mask, bonds = (data[key][rows] for key in ["coords", "atom_types", "mask", "bond_types"])
    summary, rmsds = geometry_summary(samples, refs, atoms, mask)
    chemistry, symmetry = chemical_summary(samples, refs, atoms, mask, bonds)
    names, counts = np.unique(statuses, return_counts=True)
    report = {"configuration": vars(args), "task": "graph-only ETKDG/MMFF baseline; known graph, no IR",
        "data_sha256": manifest["data_sha256"], "split_sha256": file_sha256(args.splits),
        "geometry": summary, "chemistry": chemistry,
        "statuses": dict(zip(names.tolist(), counts.tolist()))}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "samples.npz", samples=samples, row_indices=rows,
        statuses=statuses, fixed_correspondence_rmsd=rmsds, symmetry_aware_rmsd=symmetry)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
