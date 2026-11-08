"""Render input IR peaks, a predicted graph, and saved 3D coordinates.

This does not calculate an IR spectrum from a structure or generate coordinates.
Run from the project root when plotting an older report without input_ir.json.
"""
import argparse
import json
import textwrap
from pathlib import Path

import numpy as np


def read_input_ir(directory, report):
    """Read only IR input, never reference identities or reference coordinates."""
    saved = directory / "input_ir.json"
    if saved.is_file():
        peaks = json.loads(saved.read_text(encoding="utf-8"))
        return peaks["frequencies"], peaks["intensities"], "Retained peaks used by the model"
    config = report["configuration"]
    if config.get("peaks_json"):
        peaks = json.loads(Path(config["peaks_json"]).read_text(encoding="utf-8"))
        return peaks["frequencies"], peaks["intensities"], "Supplied peaks; older report may have retained a subset"
    if config.get("data"):
        with np.load(config["data"], allow_pickle=False) as data:
            row = config["row"]
            scale = json.loads(str(data["preprocessing_json"].item()))
            mask = data["spectra_mask"][row]
            packed = data["spectra_packed"][row]
            return (packed[mask, 0] * scale["frequency_scale"],
                    packed[mask, 1] * scale["intensity_scale"], "Retained peaks used by the model")
    raise ValueError("No saved input IR or accessible source. Copy input_ir.json with this prediction.")


def _message(axis, message):
    axis.set_axis_off()
    draw = getattr(axis, "text2D", axis.text)
    draw(.5, .5, textwrap.fill(message, 40), ha="center", va="center",
         transform=axis.transAxes, fontsize=12, color="#475569")


def plot_prediction(directory, graph_index=0, conformer_index=None, output=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import Draw

    directory = Path(directory)
    report = json.loads((directory / "prediction.json").read_text(encoding="utf-8"))
    frequencies, intensities, input_description = read_input_ir(directory, report)
    frequencies, intensities = np.asarray(frequencies, float), np.asarray(intensities, float)
    if (frequencies.ndim != 1 or not len(frequencies) or frequencies.shape != intensities.shape
            or not np.isfinite(frequencies).all() or not np.isfinite(intensities).all()
            or (frequencies <= 0).any() or (intensities < 0).any()):
        raise ValueError("Input IR must contain matching finite frequencies > 0 and intensities >= 0.")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(17, 5.8), facecolor="white")
    grid = figure.add_gridspec(1, 3, width_ratios=(1.5, 1., 1.1))
    ir_axis = figure.add_subplot(grid[0])
    graph_axis = figure.add_subplot(grid[1])
    xyz_axis = figure.add_subplot(grid[2], projection="3d")
    figure.subplots_adjust(left=.055, right=.97, top=.82, bottom=.30, wspace=.3)
    figure.suptitle(title or "IR input and molecular prediction", fontsize=18, x=.055, ha="left", y=.96)
    ir_axis.vlines(frequencies, 0, intensities, color="#126c91", linewidth=1.5)
    ir_axis.set_xlim(max(4000., frequencies.max() + 100.), max(0., min(400., frequencies.min() - 50.)))
    ir_axis.set_ylim(0, max(float(intensities.max()) * 1.12, 1.))
    ir_axis.set(title="Input IR peaks", xlabel=r"Wavenumber (cm$^{-1}$)", ylabel="Intensity (training units)")
    ir_axis.grid(axis="y", alpha=.2)
    figure.text(.055, .18, textwrap.fill(input_description, 80), fontsize=9,
                color="#475569", va="center")
    candidates = report.get("graph_candidates", [])
    candidate = next((item for item in candidates if item["graph_index"] == graph_index), None)
    graph_axis.set_title(f"Graph candidate {graph_index + 1}")
    xyz_axis.set_title("Saved 3D sample")
    geometry_status = "No 3D sample was returned"
    if candidate is None:
        _message(graph_axis, f"No graph candidate {graph_index + 1} was returned. Status: {report.get('status', 'unknown')}")
        _message(xyz_axis, "No corresponding 3D structure is available")
    else:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        if molecule is None:
            raise ValueError("The reported graph cannot be read as SMILES.")
        graph_axis.imshow(Draw.MolToImage(molecule, size=(600, 600)))
        graph_axis.set_axis_off()
        graph_axis.text(.5, -.07, textwrap.fill(candidate["smiles"], 34), transform=graph_axis.transAxes,
                        fontsize=9, ha="center", va="top", family="monospace")
        with np.load(directory / "coordinates.npz", allow_pickle=False) as arrays:
            samples = [item for item in candidate["conformers"] if
                f"graph{graph_index}_conformer{item['conformer_index']}_coords" in arrays]
            if conformer_index is not None:
                samples = [item for item in samples if item["conformer_index"] == conformer_index]
            if not samples:
                _message(xyz_axis, "No saved 3D coordinates for the selected graph/conformer")
            else:
                sample = samples[0]  # Conformers are unranked; never claim this is the best geometry.
                index = sample["conformer_index"]
                xyz = arrays[f"graph{graph_index}_conformer{index}_coords"]
                atoms = arrays[f"graph{graph_index}_atom_types"]
                bonds = arrays[f"graph{graph_index}_bond_types"]
                if (xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all()
                        or bonds.shape != (len(atoms), len(atoms))):
                    raise ValueError("Saved coordinates or topology are malformed; refusing to invent a replacement.")
                colors = {1: "#e2e8f0", 6: "#475569", 7: "#2563eb", 8: "#dc2626", 9: "#16a34a"}
                for first, second in np.argwhere(np.triu(bonds > 0, 1)):
                    xyz_axis.plot(*xyz[[first, second]].T, color="#64748b", linewidth=2)
                xyz_axis.scatter(*xyz.T, c=[colors.get(int(z), "#a855f7") for z in atoms],
                                 s=[48 if z == 1 else 125 for z in atoms], edgecolors="#334155", linewidths=.5)
                # Keep atom names out of the 3D drawing; put the color key below.
                from matplotlib.lines import Line2D
                element_names = {1: "Hydrogen (H)", 6: "Carbon (C)", 7: "Nitrogen (N)",
                                 8: "Oxygen (O)", 9: "Fluorine (F)"}
                present = set(int(z) for z in atoms)
                handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                           markerfacecolor=colors[z], markeredgecolor="#64748b",
                           label=element_names[z]) for z in (6, 7, 8, 1, 9) if z in present]
                xyz_axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.12),
                                ncol=min(3, len(handles)), frameon=False, fontsize=8,
                                handletextpad=.4, columnspacing=1.)
                center, radius = xyz.mean(0), max(float(np.ptp(xyz, axis=0).max()), 1.) * .6
                xyz_axis.set(xlim=(center[0]-radius, center[0]+radius),
                             ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius),
                             xlabel=r"x ($\AA$)", ylabel=r"y ($\AA$)", zlabel=r"z ($\AA$)")
                xyz_axis.set_box_aspect((1, 1, 1))
                xyz_axis.view_init(elev=22, azim=35)
                xyz_axis.set_title(f"3D conformer {index + 1} (unranked)")
                passed = "PASS" if sample.get("geometry_screen_passed") else "FAIL"
                geometry_status = f"Geometry screen: {passed}; refinement: {sample.get('refinement_status', 'unknown')}"
    status_color = "#b91c1c" if "FAIL" in geometry_status else "#166534" if "PASS" in geometry_status else "#334155"
    figure.text(.055, .12, geometry_status, fontsize=10, color=status_color)
    figure.text(.055, .055, "IR panel shows input peaks. No IR spectrum has been calculated from the predicted structure.",
                fontsize=10, color="#475569")
    output = Path(output) if output else directory / "result.png"
    if output.suffix.lower() != ".png":
        plt.close(figure)
        raise ValueError("Use a .png output path; the companion .svg is saved automatically.")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    figure.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--graph_index", type=int, default=0)
    parser.add_argument("--conformer_index", type=int, help="Default: first saved sample, not energy-ranked")
    parser.add_argument("--output", help="PNG path; a companion SVG is also written")
    parser.add_argument("--title", help="Optional display title")
    args = parser.parse_args()
    if args.graph_index < 0 or (args.conformer_index is not None and args.conformer_index < 0):
        parser.error("Graph/conformer indices must be nonnegative.")
    try:
        output = plot_prediction(args.prediction_dir, args.graph_index, args.conformer_index, args.output, args.title)
    except ImportError as exc:
        parser.exit(1, f"Missing visualization dependency: {exc}. Install requirements-visualization.txt.\n")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Cannot plot prediction: {exc}\n")
    print(f"Picture: {output}\nSVG picture: {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
