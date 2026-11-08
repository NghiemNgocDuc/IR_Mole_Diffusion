"""Read-only quantitative audit of the prepared SQuIRL data; no model training.

Run from the repository root using its Python environment. Outputs go to
outputs/dataset_review. Only numerical, non-object arrays are loaded.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dataset_review"


def distribution(x):
    x = np.asarray(x)
    return {"n": int(x.size), "min": float(np.min(x)),
            "p01": float(np.quantile(x, .01)), "p10": float(np.quantile(x, .1)),
            "median": float(np.median(x)), "mean": float(np.mean(x)),
            "p90": float(np.quantile(x, .9)), "p99": float(np.quantile(x, .99)),
            "max": float(np.max(x))}


def histogram(x):
    return dict(sorted(Counter(map(int, x)).items()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/processed/squirl.npz"
    split_file = ROOT / "data/processed/splits.json"
    manifest = json.loads(split_file.read_text())
    with np.load(path, allow_pickle=False) as z:
        spectra, mask = z["spectra_packed"], z["spectra_mask"]
        atoms, sizes, coords = z["atom_types"], z["num_atoms"], z["coords"]
        bonds, source_rows = z["bond_types"], z["source_rows"]
        schema = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
    assert np.isfinite(spectra).all() and np.isfinite(coords).all()
    assert not np.any(spectra[~mask])
    assert np.array_equal((atoms > 0).sum(1), sizes)
    counts = mask.sum(1)
    heavy = (atoms > 1).sum(1)
    frequencies = spectra[:, :, 0] * 4000
    intensities = spectra[:, :, 1] * 1000
    groups = {k: np.array(v) for k, v in manifest["splits"].items()}
    groups["first500_val"] = groups["val"][:500]
    groups["remaining_val"] = groups["val"][500:]
    active_f, active_i = frequencies[mask], intensities[mask]
    report = {"scope": "Local prepared file; descriptive audit, no accuracy experiment",
              "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
              "schema": schema, "n_rows": len(sizes),
              "frequency_cm_inverse": distribution(active_f),
              "intensity_training_units": distribution(active_i),
              "active_peak_count": distribution(counts),
              "nonzero_intensity_below_0_5": int(((active_i > 0) & (active_i < .5 - 1e-6)).sum()),
              "peak_slots_occupied_fraction": float(mask.mean()),
              "rows_reaching_capacity_81": int((counts == 81).sum()),
              "rows_with_3N_minus_6_peaks": int((counts == 3 * sizes - 6).sum()),
              "peak_count_total_atom_count_correlation": float(np.corrcoef(counts, sizes)[0, 1]),
              "by_split": {}}
    for name, rows in groups.items():
        report["by_split"][name] = {
            "rows": len(rows), "source_row": distribution(source_rows[rows]),
            "heavy_atoms": distribution(heavy[rows]), "heavy_atom_histogram": histogram(heavy[rows]),
            "total_atoms": distribution(sizes[rows]), "peak_count": distribution(counts[rows]),
            "peak_count_histogram": histogram(counts[rows]),
            "max_intensity_per_row": distribution(intensities[rows].max(1)),
            "sum_intensity_per_row": distribution(intensities[rows].sum(1))}

    # Descriptive train-only lookup, not a model fine-tune. Does peak count carry
    # total atom-count information? Compare to a constant train-majority guess.
    train = groups["train"]
    majority = Counter(map(int, sizes[train])).most_common(1)[0][0]
    count_to_size = {int(k): Counter(map(int, sizes[train][counts[train] == k])).most_common(1)[0][0]
                     for k in np.unique(counts[train])}
    report["peak_count_size_lookup"] = {"description": "Train-only majority atom-count lookup by retained peak count; diagnostic, not IR model accuracy",
                                       "constant_train_majority_total_atoms": majority}
    for name in ("val", "test", "first500_val"):
        rows = groups[name]
        pred = np.array([count_to_size.get(int(k), majority) for k in counts[rows]])
        report["peak_count_size_lookup"][name] = {
            "exact_total_atom_count_fraction": float(np.mean(pred == sizes[rows])),
            "mean_absolute_atom_count_error": float(np.abs(pred - sizes[rows]).mean()),
            "constant_majority_exact_fraction": float(np.mean(sizes[rows] == majority))}

    print("Checking exact model-input duplicates...", flush=True)
    input_hashes = [hashlib.sha256(s.tobytes() + m.tobytes()).hexdigest() for s, m in zip(spectra, mask)]
    input_counter = Counter(input_hashes)
    report["exact_input_duplicates"] = {
        "unique_model_inputs": len(input_counter),
        "rows_in_repeated_inputs": sum(v for v in input_counter.values() if v > 1)}
    train_hashes = {input_hashes[i] for i in train}
    for name in ("val", "test"):
        report["exact_input_duplicates"][name + "_inputs_also_in_train"] = sum(input_hashes[i] in train_hashes for i in groups[name])

    print("Checking coordinate padding, centering and pair distances...", flush=True)
    active = atoms > 0
    assert not np.any(coords[~active])
    centroids = coords.sum(1) / sizes[:, None]
    min_separation, max_bond = np.empty(len(sizes)), np.empty(len(sizes))
    bond_distances = defaultdict(list)
    near_rows = []
    for start in range(0, len(sizes), 1024):
        end = min(start + 1024, len(sizes))
        c = coords[start:end].astype(np.float64)
        distance = np.linalg.norm(c[:, :, None] - c[:, None, :], axis=-1)
        pair_mask = active[start:end, :, None] & active[start:end, None, :]
        pair_mask &= np.triu(np.ones((coords.shape[1], coords.shape[1]), bool), 1)
        ds = np.where(pair_mask, distance, np.inf)
        min_separation[start:end] = ds.min((1, 2))
        max_bond[start:end] = np.where(bonds[start:end] > 0, distance, 0).max((1, 2))
        near_rows.extend((start + np.flatnonzero(ds.min((1, 2)) < .2)).tolist())
        for i, j, k in zip(*np.where((bonds[start:end] > 0) & pair_mask)):
            z1, z2 = sorted((int(atoms[start+i, j]), int(atoms[start+i, k])))
            bond_distances[f"{z1}-{z2}:order{bonds[start+i,j,k]}"].append(distance[i, j, k])
        if start % 20480 == 0:
            print(f"Geometry {end}/{len(sizes)}", flush=True)
    report["geometry"] = {
        "max_centroid_norm_angstrom": float(np.linalg.norm(centroids, axis=1).max()),
        "min_interatomic_distance_angstrom": distribution(min_separation),
        "rows_with_distance_below_0_2_angstrom": near_rows,
        "maximum_bond_length_per_row_angstrom": distribution(max_bond),
        "longest_bond_rows": [{"row": int(i), "source_row": int(source_rows[i]), "max_bond_angstrom": float(max_bond[i])}
                               for i in np.argsort(max_bond)[-10:][::-1]],
        "bond_lengths_by_atomic_numbers_and_order": {k: distribution(v) for k, v in sorted(bond_distances.items())}}
    out = OUT / "spectral_geometry_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for key, color, label in [("train", "#777777", "Training"), ("val", "#0072B2", "All validation"),
                                 ("first500_val", "#D55E00", "First 500 validation")]:
            rows = groups[key]
            axes[0, 0].hist(heavy[rows], bins=np.arange(.5, 10.5), density=True,
                            histtype="step", linewidth=2, color=color, label=label)
            axes[0, 1].hist(counts[rows], bins=np.arange(0, 76, 3), density=True,
                            histtype="step", linewidth=2, color=color)
        axes[0, 0].set(xlabel="Heavy atoms", ylabel="Density", title="Molecular size")
        axes[0, 0].legend(fontsize=9)
        axes[0, 1].set(xlabel="Retained peaks per molecule", ylabel="Density", title="IR input size")
        axes[1, 0].hist(active_i, bins=np.geomspace(.49, 1800, 60), color="#0072B2")
        axes[1, 0].set(xscale="log", xlabel="Intensity (stored training units)", ylabel="Number of peaks", title="Input intensities start at 0.5")
        for key, color, label in [("val", "#0072B2", "All validation"), ("first500_val", "#D55E00", "First 500 validation")]:
            axes[1, 1].hist(source_rows[groups[key]], bins=np.linspace(0, 133795, 35),
                            density=True, histtype="step", linewidth=2, color=color, label=label)
        axes[1, 1].set(xlabel="Original source row", ylabel="Density", title="Evaluation subset comes from file start")
        axes[1, 1].legend(fontsize=9)
        fig.suptitle("Prepared SQuIRL dataset: measured local distributions", fontsize=15)
        fig.savefig(OUT / "dataset_distributions.png", dpi=170)
        plt.close(fig)
    except ImportError as exc:
        print(f"Optional plot unavailable: {exc}", flush=True)


if __name__ == "__main__":
    main()
