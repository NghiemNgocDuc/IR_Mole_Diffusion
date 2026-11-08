"""Verify trusted legacy SQuIRL source against every prepared example."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion.preprocessing import pack_peaks


def main():
    raw_path = ROOT / "data/SQuIRL/squirl_ir_3d.npz"
    prepared_path = ROOT / "data/processed/squirl.npz"
    meta = json.loads((ROOT / "data/processed/squirl.metadata.json").read_text())
    # Explicitly trusted original project archive, already used by converter.
    with np.load(raw_path, allow_pickle=True) as z:
        raw_schema = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
        raw = {k: z[k] for k in ("ir_frequencies", "ir_intensities", "coords", "atom_types", "smiles")}
    with np.load(prepared_path, allow_pickle=False) as z:
        data = {k: z[k] for k in ("source_rows", "coords", "atom_types", "num_atoms", "spectra_packed", "spectra_mask")}
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == meta["source_sha256"]
    rejected = {r["source_row"]: r["reason"] for r in meta["rejected"]}
    source_rows = data["source_rows"]
    assert len(source_rows) + len(rejected) == len(raw["smiles"])
    assert set(map(int, source_rows)).isdisjoint(rejected)
    assert set(map(int, source_rows)) | set(rejected) == set(range(len(raw["smiles"])))
    problems, negative_rows = Counter(), []
    peak_lengths, mode_delta = [], Counter()
    intensity_min, intensity_max, total_peaks, below_half, exactly_half = float("inf"), 0., 0, 0, 0
    for i, (x, y, a) in enumerate(zip(raw["ir_frequencies"], raw["ir_intensities"], raw["atom_types"])):
        x, y = np.asarray(x), np.asarray(y)
        peak_lengths.append(len(x))
        mode_delta[int(len(x) - (3*len(a) - 6))] += 1
        if (x < 0).any():
            negative_rows.append(i)
        for key, flag in {"negative_frequency_rows": (x < 0).any(), "zero_frequency_rows": (x == 0).any(),
                          "nonfinite_frequency_rows": not np.isfinite(x).all(), "negative_intensity_rows": (y < 0).any(),
                          "zero_intensity_rows": (y == 0).any(), "nonfinite_intensity_rows": not np.isfinite(y).all()}.items():
            problems[key] += int(flag)
        intensity_min = min(intensity_min, float(y.min()))
        intensity_max = max(intensity_max, float(y.max()))
        total_peaks += len(x)
        below_half += int((y < .5).sum())
        exactly_half += int((y == .5).sum())
    mismatches = Counter(peaks=0, masks=0, centered_coordinates=0, atom_labels=0, atom_counts=0)
    for i, j in enumerate(source_rows):
        n = int(data["num_atoms"][i])
        raw_c = np.asarray(raw["coords"][j], dtype=np.float32)
        raw_a = np.asarray(raw["atom_types"][j], dtype=np.int64)
        packed, mask = pack_peaks(raw["ir_frequencies"][j], raw["ir_intensities"][j], **meta["preprocessing"])
        mismatches["peaks"] += int(not np.array_equal(packed, data["spectra_packed"][i]))
        mismatches["masks"] += int(not np.array_equal(mask, data["spectra_mask"][i]))
        mismatches["atom_counts"] += int(n != len(raw_a))
        mismatches["atom_labels"] += int(not np.array_equal(raw_a, data["atom_types"][i, :n]))
        mismatches["centered_coordinates"] += int(not np.array_equal(raw_c - raw_c.mean(axis=0), data["coords"][i, :n]))
        if (i+1) % 25000 == 0:
            print(f"Source reconstruction {i+1}/{len(source_rows)}", flush=True)
    assert not any(mismatches.values()), mismatches
    report = {
        "source_sha256": meta["source_sha256"], "prepared_sha256": meta["data_sha256"],
        "raw_schema": raw_schema, "source_rows": len(raw["smiles"]), "accepted_rows": len(source_rows),
        "rejected_rows": len(rejected), "rejection_reason_counts": dict(Counter(rejected.values())),
        "source_row_accounting_complete_disjoint": True,
        "full_accepted_reconstruction_mismatches": dict(mismatches),
        "raw_spectral_problems": dict(problems), "negative_frequency_source_rows": negative_rows,
        "negative_frequency_rows_by_first_rejection_reason": dict(Counter(rejected.get(i, "ACCEPTED") for i in negative_rows)),
        "raw_peak_count_min": min(peak_lengths), "raw_peak_count_max": max(peak_lengths),
        "raw_total_peaks": total_peaks, "raw_intensity_min": intensity_min, "raw_intensity_max": intensity_max,
        "raw_intensities_below_0_5": below_half, "raw_intensities_exactly_0_5": exactly_half,
        "raw_mode_count_minus_3N_minus_6_histogram": dict(sorted(mode_delta.items())),
        "raw_smiles_with_at_stereo_marker": sum("@" in str(s) for s in raw["smiles"]),
        "raw_smiles_with_directional_bond_markers": sum("/" in str(s) or "\\" in str(s) for s in raw["smiles"]),
        "preprocessing": meta["preprocessing"],
        "limitations": [
            "This verifies conversion against the local legacy archive, not agreement with the original published HDF5.",
            "The archive lacks original QM9 IDs, warning flags, computational-method metadata and upstream filtering recipe.",
            "An intensity cutoff at 0.5 is inferred from the observed lower bound; its provenance is not established locally.",
            "3N-6 is a nonlinear-molecule reference count; linear molecules can have 3N-5 modes.",
            "Rejection reasons record the first failing converter check; a row can have additional problems.",
            "Graph alignment rejection counts are from the saved converter metadata, not a rerun of every graph-isomorphism check."]}
    path = ROOT / "outputs/dataset_review/source_preprocessing_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
