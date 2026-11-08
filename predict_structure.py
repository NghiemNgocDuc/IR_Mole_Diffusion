"""IR-only prediction: SELFIES candidates, coordinate diffusion, geometry checks.

Chemical plausibility is a filter, not proof of an IR match or quantum stability.
No reference atoms, formula, SMILES, or coordinates are used to generate outputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion.data import load_native_npz
from diffusion.model import EquivariantDiffusion3D_v4
from diffusion.preprocessing import bond_array, pack_peaks
from diffusion.physics import PhysicsLoss
from diffusion.spectrum import SpectrumStructureModel
from evaluate_spectrum import valid_candidates


def predict(spectrum_model, diffusion_model, spectra, beam_size=5, conformers=5,
            n_steps=200, refine=True, refinement_iterations=500, physics_guidance=0.01):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    candidates = valid_candidates(spectrum_model.generate(spectra, beam_size), max_atoms=29)
    records, molecules, arrays = [], [], {}
    physics = PhysicsLoss(bond_weight=1., clash_weight=1., angle_weight=0., ring_weight=0.).to(diffusion_model.device)
    for graph_index, candidate in enumerate(candidates):
        template = Chem.AddHs(Chem.MolFromSmiles(candidate["smiles"]))
        numbers = np.array([atom.GetAtomicNum() for atom in template.GetAtoms()])
        topology = bond_array(template)
        atoms = torch.tensor(numbers[None], dtype=torch.long, device=diffusion_model.device)
        graph = {"bond_types": torch.tensor(topology[None], dtype=torch.long, device=diffusion_model.device)}
        count = torch.tensor([len(numbers)], device=diffusion_model.device)
        conformer_records = []
        for conformer_index in range(conformers):
            try:
                coords, mask = diffusion_model.sample(spectra, atoms, count, n_steps=n_steps,
                    graph=graph, physics_guidance=physics_guidance)
            except FloatingPointError as exc:
                conformer_records.append({"graph_index": graph_index,
                    "conformer_index": conformer_index, "geometry_screen_passed": False,
                    "sampling_status": "failed", "error": str(exc)})
                continue
            raw = coords[0].detach().cpu().numpy()
            xyz = raw.copy()
            molecule = Chem.Mol(template)
            conf = Chem.Conformer(len(numbers))
            for index, point in enumerate(xyz):
                conf.SetAtomPosition(index, tuple(map(float, point)))
            molecule.AddConformer(conf)
            status, energy = "not_requested", None
            if refine:
                try:
                    if not AllChem.MMFFHasAllMoleculeParams(molecule):
                        status = "unsupported_parameters_raw_retained"
                    elif AllChem.MMFFOptimizeMolecule(molecule, maxIters=refinement_iterations) == 0:
                        status = "converged"
                        xyz = molecule.GetConformer().GetPositions().astype(np.float32)
                        xyz -= xyz.mean(0)
                        for index, point in enumerate(xyz):
                            molecule.GetConformer().SetAtomPosition(index, tuple(map(float, point)))
                        properties = AllChem.MMFFGetMoleculeProperties(molecule)
                        energy = float(AllChem.MMFFGetMoleculeForceField(molecule, properties).CalcEnergy())
                    else:
                        status = "not_converged_raw_retained"
                except (ValueError, RuntimeError):
                    status = "refinement_failed_raw_retained"
                if status != "converged":
                    for index, point in enumerate(raw):
                        molecule.GetConformer().SetAtomPosition(index, tuple(map(float, point)))
            tensor = torch.tensor(xyz[None], device=atoms.device)
            _, components = physics(tensor, atoms, mask, graph=graph, return_components=True)
            bond_penalty, clash_penalty = float(components["bond"]), float(components["clash"])
            # This threshold is a transparent screening heuristic, not a learned probability.
            screened = bool(bond_penalty < 1. and clash_penalty < 0.1 and np.isfinite(xyz).all())
            record = {"graph_index": graph_index, "conformer_index": conformer_index,
                "geometry_screen_passed": screened, "bond_penalty": bond_penalty,
                "clash_penalty": clash_penalty, "refinement_status": status,
                "mmff_energy_kcal_mol": energy}
            conformer_records.append(record)
            key = f"graph{graph_index}_conformer{conformer_index}"
            arrays[key + "_raw_coords"] = raw
            arrays[key + "_coords"] = xyz
            molecule.SetProp("graph_index", str(graph_index))
            molecule.SetProp("conformer_index", str(conformer_index))
            molecule.SetProp("refinement_status", status)
            molecules.append(molecule)
        arrays[f"graph{graph_index}_atom_types"] = numbers
        arrays[f"graph{graph_index}_bond_types"] = topology
        records.append({**candidate, "graph_index": graph_index, "conformers": conformer_records})
    return records, molecules, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum_checkpoint", required=True)
    parser.add_argument("--diffusion_checkpoint", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--peaks_json", help="JSON with frequencies (cm^-1) and intensities in training units")
    inputs.add_argument("--data", help="Processed NPZ; only the selected IR spectrum is used for generation")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="predictions")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--conformers", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refinement_iterations", type=int, default=500)
    parser.add_argument("--physics_guidance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=78438379)
    args = parser.parse_args(argv)
    if min(args.conformers, args.beam_size, args.n_steps, args.refinement_iterations) < 1:
        parser.error("Candidate, step, and refinement counts must be positive.")
    torch.manual_seed(args.seed)
    spectrum_model = SpectrumStructureModel.load_from_checkpoint(args.spectrum_checkpoint)
    diffusion_model = EquivariantDiffusion3D_v4.load_from_checkpoint(args.diffusion_checkpoint)
    for key in ("data_sha256", "split_sha256", "preprocessing"):
        if spectrum_model.run_metadata.get(key) != diffusion_model.run_metadata.get(key):
            parser.error(f"Stage checkpoints differ in {key}; use a matched training pipeline.")
    for model in (spectrum_model, diffusion_model):
        model.use_ema_weights()
        model.to(args.device).eval()
    scale = spectrum_model.run_metadata.get("preprocessing")
    if scale is None:
        parser.error("Checkpoint lacks preprocessing metadata.")
    if args.peaks_json:
        peaks = json.loads(Path(args.peaks_json).read_text())
        packed, mask = pack_peaks(peaks["frequencies"], peaks["intensities"], **scale)
    else:
        data = load_native_npz(args.data)
        if data.get("preprocessing") != scale or not data["packed"]:
            parser.error("Input preprocessing differs from checkpoint.")
        if not 0 <= args.row < len(data["coords"]):
            parser.error("row is outside the input data.")
        packed, mask = data["spectra"][args.row], data["spectra_mask"][args.row]
    spectra = {"packed": torch.tensor(packed[None], device=args.device),
               "mask": torch.tensor(mask[None], device=args.device)}
    records, molecules, arrays = predict(spectrum_model, diffusion_model, spectra,
        args.beam_size, args.conformers, args.n_steps, args.refine,
        args.refinement_iterations, args.physics_guidance)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the exact retained IR input so a downloaded prediction can be plotted alone.
    input_ir = {"frequencies": (packed[mask, 0] * scale["frequency_scale"]).tolist(),
        "intensities": (packed[mask, 1] * scale["intensity_scale"]).tolist(),
        "frequency_units": "cm^-1", "intensity_units": "training intensity units",
        "description": "Retained input peaks passed to the model; not a forward-calculated IR spectrum.",
        "preprocessing": scale}
    (output / "input_ir.json").write_text(json.dumps(input_ir, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from rdkit import Chem
    with Chem.SDWriter(str(output / "candidates.sdf")) as writer:
        for molecule in molecules:
            writer.write(molecule)
    np.savez_compressed(output / "coordinates.npz", **arrays)
    status = "candidates_generated" if molecules else "all_coordinate_samples_failed" if records else "no_valid_completed_graph"
    report = {"configuration": vars(args), "input_ir": "input_ir.json", "graph_candidates": records,
        "valid_graph_candidates": len(records), "status": status,
        "ranking": "Graphs ordered by decoder sequence score; conformers are unranked. MMFF energies only compare conformers of the same graph.",
        "limitations": ["Geometry screens and MMFF are approximate physical checks, not quantum stability.",
            "No forward IR calculation or spectral-match guarantee is implemented.",
            "Ordinary achiral IR may not determine absolute stereochemistry; this pipeline reports achiral graph identities.",
            "This recipe is restricted to neutral, closed-shell H/C/N/O/F molecules with at most 29 total atoms."]}
    (output / "prediction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "valid_graph_candidates": len(records),
        "output": str(output / "prediction.json")}, indent=2))


if __name__ == "__main__":
    main()
