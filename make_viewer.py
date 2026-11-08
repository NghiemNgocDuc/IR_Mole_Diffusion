"""Create an offline 3D viewer from saved predictions or an explicit 2D SMILES.

Saved predictions retain their coordinates. The separate SMILES mode generates
conformers and labels them as estimates, without claiming an IR prediction.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np

from plot_prediction import read_input_ir


ASSETS = Path(__file__).resolve().parent / "viewer_assets"


def _molecule_frame(molecule, coords, comment):
    from rdkit import Chem
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite [atoms, 3].")
    molecule = Chem.Mol(molecule)
    molecule.RemoveAllConformers()
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for index, point in enumerate(coords):
        conformer.SetAtomPosition(index, point.tolist())
    molecule.AddConformer(conformer)
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    xyz = str(len(symbols)) + "\n" + comment + "\n" + "\n".join(
        f"{symbol} {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}" for symbol, point in zip(symbols, coords)) + "\n"
    return {"molblock": Chem.MolToMolBlock(molecule), "coords": coords.tolist(), "xyz": xyz}


def _drawing(molecule):
    from rdkit.Chem.Draw import rdMolDraw2D
    drawing = rdMolDraw2D.MolDraw2DSVG(440, 240)
    drawing.DrawMolecule(molecule)
    drawing.FinishDrawing()
    return re.sub(r"<\?xml[^>]*\?>", "", drawing.GetDrawingText())


def _frame(numbers, bonds, coords, expected_smiles=None):
    from rdkit import Chem
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (len(numbers), 3) or not np.isfinite(coords).all():
        raise ValueError("Saved coordinates must be finite [atoms, 3]; refusing replacement coordinates.")
    molecule = Chem.RWMol()
    for number in numbers:
        molecule.AddAtom(Chem.Atom(int(number)))
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for left, right in np.argwhere(np.triu(bonds > 0, 1)):
        molecule.AddBond(int(left), int(right), kinds[int(bonds[left, right])])
    molecule = molecule.GetMol()
    Chem.SanitizeMol(molecule)
    if expected_smiles is not None:
        saved_graph = Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=False)
        declared_graph = Chem.MolToSmiles(Chem.MolFromSmiles(expected_smiles), isomericSmiles=False)
        if saved_graph != declared_graph:
            raise ValueError("Saved atom labels/topology do not match the declared candidate SMILES.")
    return _molecule_frame(molecule, coords, "Saved prediction coordinates in Angstrom")


def prediction_payload(directory):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    directory = Path(directory)
    report = json.loads((directory / "prediction.json").read_text(encoding="utf-8"))
    frequencies, intensities, description = read_input_ir(directory, report)
    frequencies, intensities = np.asarray(frequencies, float), np.asarray(intensities, float)
    if (frequencies.ndim != 1 or frequencies.shape != intensities.shape or not len(frequencies)
            or not np.isfinite(frequencies).all() or not np.isfinite(intensities).all()
            or (frequencies <= 0).any() or (intensities < 0).any()):
        raise ValueError("Input IR must contain matching finite positive frequencies and nonnegative intensities.")
    payload = {"source": directory.name, "status": report.get("status", "unknown"),
        "provenance": {"kind": "saved_prediction", "coordinate_source": "coordinates.npz"},
        "ranking": report.get("ranking", "Graphs in saved order; conformers unranked."),
        "input_ir": {"frequencies": frequencies.tolist(), "intensities": intensities.tolist(),
                     "description": description}, "candidates": []}
    candidates = report.get("graph_candidates", [])
    if not candidates:
        return payload
    with np.load(directory / "coordinates.npz", allow_pickle=False) as arrays:
        seen = set()
        for candidate in candidates:
            graph_index = candidate["graph_index"]
            if type(graph_index) is not int or graph_index < 0 or graph_index in seen:
                raise ValueError("Graph indices must be unique nonnegative integers.")
            seen.add(graph_index)
            mol = Chem.MolFromSmiles(candidate["smiles"])
            if mol is None:
                raise ValueError("A saved candidate has invalid SMILES.")
            numbers = arrays[f"graph{graph_index}_atom_types"]
            bonds = arrays[f"graph{graph_index}_bond_types"]
            if (numbers.ndim != 1 or not 0 < len(numbers) <= 256 or numbers.dtype.kind not in "iu"
                    or not np.isin(numbers, [1, 6, 7, 8, 9]).all()
                    or bonds.shape != (len(numbers), len(numbers)) or bonds.dtype.kind not in "iu"
                    or not np.isin(bonds, [0, 1, 2, 3, 4]).all()
                    or not np.array_equal(bonds, bonds.T) or np.diag(bonds).any()):
                raise ValueError("Saved atom labels or topology are malformed.")
            entry = {"graph_index": graph_index, "smiles": candidate["smiles"],
                "formula": rdMolDescriptors.CalcMolFormula(mol), "svg": _drawing(mol),
                "elements": [Chem.GetPeriodicTable().GetElementSymbol(int(number)) for number in numbers],
                "decoder_log_probability": candidate.get("log_probability"), "conformers": []}
            seen_conformers = set()
            for conformer in candidate.get("conformers", []):
                index = conformer["conformer_index"]
                if type(index) is not int or index < 0 or index in seen_conformers:
                    raise ValueError("Conformer indices must be unique nonnegative integers.")
                seen_conformers.add(index)
                item = {key: conformer.get(key) for key in ("conformer_index", "geometry_screen_passed",
                    "refinement_status", "mmff_energy_kcal_mol", "bond_penalty", "clash_penalty", "error")}
                for label, suffix in (("raw", "raw_coords"), ("final", "coords")):
                    key = f"graph{graph_index}_conformer{index}_{suffix}"
                    item[label] = _frame(numbers, bonds, arrays[key], candidate["smiles"]) if key in arrays else None
                entry["conformers"].append(item)
            payload["candidates"].append(entry)
    return payload


def smiles_payload(smiles, num_conformers=3):
    """Generate reproducible ETKDGv3 conformers, separate from model outputs."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdMolDescriptors
    if type(num_conformers) is not int or not 1 <= num_conformers <= 20:
        raise ValueError("num_conformers must be an integer from 1 to 20.")
    graph = Chem.MolFromSmiles(smiles)
    if graph is None or not graph.GetNumAtoms():
        raise ValueError("Provide a valid, nonempty SMILES structure.")
    if len(Chem.GetMolFrags(graph)) != 1 or any(atom.GetAtomicNum() == 0 for atom in graph.GetAtoms()):
        raise ValueError("3D generation requires one connected structure with explicit element identities.")
    molecule = Chem.AddHs(graph)
    if molecule.GetNumAtoms() > 256:
        raise ValueError("Interactive 3D generation supports at most 256 atoms including hydrogens.")
    canonical = Chem.MolToSmiles(graph, isomericSmiles=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = 78438379
    params.numThreads = 1
    params.pruneRmsThresh = -1.0
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=num_conformers, params=params))
    if not conformer_ids:
        raise ValueError("RDKit could not embed this structure in 3D; no substitute geometry was created.")
    raw_coordinates = {index: np.array(molecule.GetConformer(index).GetPositions(), copy=True)
                       for index in conformer_ids}
    refined = Chem.Mol(molecule)
    supported = AllChem.MMFFHasAllMoleculeParams(refined)
    properties = AllChem.MMFFGetMoleculeProperties(refined, mmffVariant="MMFF94") if supported else None
    candidate = {"graph_index": 0, "smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(graph), "svg": _drawing(graph),
        "elements": [atom.GetSymbol() for atom in molecule.GetAtoms()],
        "decoder_log_probability": None, "conformers": []}
    for number, conformer_id in enumerate(conformer_ids):
        status, energy, error = "unsupported", None, None
        final = raw_coordinates[conformer_id]
        if properties is not None:
            try:
                field = AllChem.MMFFGetMoleculeForceField(refined, properties, confId=conformer_id)
                if field is None:
                    raise ValueError("MMFF force field could not be constructed.")
                outcome = field.Minimize(maxIts=500)
                calculated_energy = float(field.CalcEnergy())
                optimized = np.array(refined.GetConformer(conformer_id).GetPositions(), copy=True)
                if not np.isfinite(calculated_energy) or not np.isfinite(optimized).all():
                    raise ValueError("MMFF returned nonfinite geometry or energy.")
                final, energy = optimized, calculated_energy
                status = "converged" if outcome == 0 else "iteration_limit"
            except (RuntimeError, ValueError) as exc:
                status, error = "failed_using_embedded_coordinates", str(exc)
        candidate["conformers"].append({"conformer_index": number,
            "geometry_screen_passed": None, "refinement_status": status,
            "mmff_energy_kcal_mol": energy, "bond_penalty": None, "clash_penalty": None,
            "error": error,
            "raw": _molecule_frame(molecule, raw_coordinates[conformer_id], "Generated ETKDGv3 coordinates in Angstrom; not a model prediction"),
            "final": _molecule_frame(molecule, final, "Generated conformer coordinates in Angstrom; not a model prediction")})
    return {"source": "2D structure: " + canonical, "status": "generated_from_2d",
        "provenance": {"kind": "generated_from_2d", "method": "RDKit ETKDGv3",
            "rdkit_version": rdBase.rdkitVersion, "seed": params.randomSeed,
            "refinement": "MMFF94 when supported", "requested_conformers": num_conformers,
            "generated_conformers": len(conformer_ids),
            "interpretation": "Generated conformers are estimates from a supplied molecular graph; no IR prediction, experimental geometry, or accuracy claim."},
        "ranking": "Conformers in embedding order; unranked.", "input_ir": None, "candidates": [candidate]}


def _export_payload(payload, output, title):
    provenance = json.loads((ASSETS / "3dmol_provenance.json").read_text())
    for name, expected in provenance["files_sha256"].items():
        if hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundled 3Dmol asset hash mismatch: {name}")
    template = (ASSETS / "viewer.html").read_text(encoding="utf-8")
    safe_json = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    replacements = {"__TITLE__": html.escape(title), "__DATA_JSON__": safe_json,
        "__ENGINE__": (ASSETS / "3Dmol-min.js").read_text(encoding="utf-8"),
        "__APP_JS__": (ASSETS / "viewer.js").read_text(encoding="utf-8"),
        "__LICENSE__": html.escape((ASSETS / "3Dmol_LICENSE.txt").read_text(encoding="utf-8"))}
    for name in ("__ENGINE__", "__APP_JS__"):
        replacements[name] = re.sub(r"</script", r"<\\/script", replacements[name], flags=re.IGNORECASE)
    # One replacement pass prevents data from being reinterpreted as a template.
    document = re.sub("|".join(re.escape(key) for key in replacements), lambda match: replacements[match[0]], template)
    output = Path(output)
    if output.suffix.lower() != ".html":
        raise ValueError("Interactive viewer output must end in .html.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def export_viewer(directory, output=None, title="IR molecule explorer"):
    return _export_payload(prediction_payload(directory), output or Path(directory) / "viewer.html", title)


def export_smiles_viewer(smiles, output, title="Generated 3D molecule explorer", num_conformers=3):
    return _export_payload(smiles_payload(smiles, num_conformers), output, title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction_dir", help="Export existing prediction coordinates unchanged")
    source.add_argument("--smiles", help="Generate estimated 3D conformers from a 2D SMILES structure")
    parser.add_argument("--output", help="Standalone HTML path; defaults to prediction_dir/viewer.html")
    parser.add_argument("--title")
    parser.add_argument("--num_conformers", type=int, default=3, help="SMILES mode only; 1 to 20, default 3")
    args = parser.parse_args()
    if args.smiles is not None and not args.output:
        parser.error("--output is required with --smiles")
    try:
        if args.smiles is not None:
            output = export_smiles_viewer(args.smiles, args.output,
                args.title or "Generated 3D molecule explorer", args.num_conformers)
        else:
            output = export_viewer(args.prediction_dir, args.output, args.title or "IR molecule explorer")
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"Cannot create viewer: {exc}\n")
    print(f"Interactive viewer: {output}\nDownload this HTML file and open it in Chrome or Edge. No server or internet is needed.", flush=True)


if __name__ == "__main__":
    main()
