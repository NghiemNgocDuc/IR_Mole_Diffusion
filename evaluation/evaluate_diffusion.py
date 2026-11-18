"""Save every sampled coordinate and report explicitly labeled evaluation metrics."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from diffusion.data import NativeNPZDataset, collate_native, file_sha256, load_native_npz, validate_split_manifest
from diffusion.evaluation import chemical_summary, geometry_summary, refine_with_known_graph
from diffusion.progress import EvaluationProgress


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value.to(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted corrected diffusion checkpoint")
    parser.add_argument("--checkpoint_policy", choices=("reject",), default="reject")
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output_dir", default="evaluation")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Explicit first-N held-out rows for a quick diagnostic")
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--spectrum_control", choices=("none", "removed", "shuffled"), default="none")
    parser.add_argument("--use_inference_graph", action="store_true", help="Declare that reference graph labels are actually available at inference")
    parser.add_argument("--physics_guidance", type=float, default=0.0)
    parser.add_argument("--chemistry", action="store_true", help="Optional RDKit bond perception and symmetry-aware same-graph RMSD")
    parser.add_argument("--refine", action="store_true", help="Optional MMFF relaxation with declared inference graph")
    parser.add_argument("--refine_iterations", type=int, default=200)
    parser.add_argument("--raw_weights", action="store_true", help="Evaluate raw training weights instead of EMA")
    parser.add_argument("--progress_interval", type=float, default=15.0, help="Seconds between flushed progress lines (default: 15; 0 logs every update)")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress lines and progress.json")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.num_samples, args.refine_iterations) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch/sample counts, refinement iterations, and limit must be positive.")
    if (args.refine or args.physics_guidance > 0) and not args.use_inference_graph:
        parser.error("Refinement and physics guidance require --use_inference_graph; evaluation-only reference topology cannot be used for generation.")
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
        parser.error("--progress_interval must be finite and nonnegative.")
    with EvaluationProgress(args.output_dir, args.progress_interval, enabled=not args.no_progress) as progress:
        evaluate(args, parser, progress)


def evaluate(args, parser, progress):
    progress.start("Loading dataset and validating split")
    data = load_native_npz(args.data)
    manifest = validate_split_manifest(json.loads(Path(args.splits).read_text(encoding="utf-8")), data["molecule_ids"], args.data)
    rows = np.asarray(manifest["splits"][args.split], dtype=np.int64)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not len(rows):
        parser.error("The selected split contains no rows.")
    if args.use_inference_graph and data["bond_types"] is None:
        parser.error("--use_inference_graph requires explicit bond_types in the data.")
    dataset = NativeNPZDataset(data, rows, args.use_inference_graph, args.spectrum_control, args.seed)
    import torch
    from torch.utils.data import DataLoader
    from diffusion.model import EquivariantDiffusion3D_v4
    progress.start("Loading checkpoint and preparing device")
    torch.manual_seed(args.seed)
    model = EquivariantDiffusion3D_v4.load_from_checkpoint(args.checkpoint, map_location="cpu", checkpoint_policy=args.checkpoint_policy)
    for key, value in (("data_sha256", manifest["data_sha256"]), ("split_sha256", file_sha256(args.splits))):
        if model.run_metadata.get(key) != value:
            parser.error(f"Checkpoint {key} differs from the requested data/split manifest.")
    used_ema = model.use_ema_weights() if not args.raw_weights else False
    model = model.to(device=args.device, dtype=torch.float32).eval()
    if getattr(model.hparams, "graph_conditioning", False) and not args.use_inference_graph:
        parser.error("This checkpoint requires graph conditioning; explicitly declare --use_inference_graph.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_native)
    chunks, sampling_failures = [], []
    progress.start("Sampling coordinates", len(rows) * args.num_samples, "candidates")
    completed_candidates = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, args.device)
            graph_kwargs = {"graph": batch["graph"]} if args.use_inference_graph else {}
            candidates = []
            for candidate_index in range(args.num_samples):
                try:
                    coords, _ = model.sample(batch["spectra"], batch["atom_types"], batch["num_atoms"],
                        n_steps=args.n_steps, sampler=args.sampler, eta=args.eta,
                        physics_guidance=args.physics_guidance, **graph_kwargs)
                    candidates.append(coords.detach().cpu().numpy())
                except FloatingPointError as exc:
                    sampling_failures.append({"batch": len(chunks), "candidate": candidate_index, "error": str(exc)})
                    candidates.append(np.full(tuple(batch["coords"].shape), np.nan, dtype=np.float32))
                completed_candidates += len(batch["coords"])
                progress.update(completed_candidates)
            chunks.append(np.stack(candidates, axis=1))
    samples = np.concatenate(chunks, axis=0)
    references, atoms, mask = data["coords"][rows], data["atom_types"][rows], data["mask"][rows]
    bonds = data["bond_types"][rows] if data["bond_types"] is not None else None
    progress.start("Raw geometry metrics", len(rows), "molecules")
    raw_summary, raw_rmsd = geometry_summary(samples, references, atoms, mask, progress_callback=progress.update)
    payload = {"raw_samples": samples, "reference_coords": references, "atom_types": atoms, "mask": mask,
               "row_indices": rows, "molecule_ids": data["molecule_ids"][rows], "raw_fixed_correspondence_rmsd": raw_rmsd}
    report = {"schema_version": 1, "configuration": vars(args), "ema_weights": used_ema,
        "evaluation_task": "coordinate generation with supplied atom counts and declared reference graph; not IR-only identification",
        "data_sha256": manifest["data_sha256"],
        "checkpoint_sha256": file_sha256(args.checkpoint), "split_manifest": str(Path(args.splits).resolve()),
        "raw": raw_summary, "refined": None, "ranking": None,
        "sampling_failures": sampling_failures,
        "limitations": ["No learned or quantum spectral ranking is implemented; first sample is unranked.",
            "Finite coordinates and geometric RMSD alone do not establish chemical validity or physical stability.",
            "Chemical identity is unverified unless --chemistry succeeds; geometry metrics do not exclude wrong graphs."]}
    if args.chemistry:
        progress.start("Raw chemistry and symmetry checks", len(rows), "molecules")
        raw_summary["chemistry"], payload["raw_symmetry_aware_same_graph_rmsd"] = chemical_summary(
            samples, references, atoms, mask, bonds, progress_callback=progress.update)
    if args.refine:
        progress.start("MMFF refinement", len(rows), "molecules")
        refined, statuses = refine_with_known_graph(samples, atoms, mask, bonds, args.refine_iterations,
                                                   progress_callback=progress.update)
        progress.start("Refined geometry metrics", len(rows), "molecules")
        report["refined"], refined_rmsd = geometry_summary(refined, references, atoms, mask, progress_callback=progress.update)
        names, counts = np.unique(statuses, return_counts=True)
        report["refined"]["refinement_status_counts"] = dict(zip(names.tolist(), counts.tolist()))
        report["refined"]["method"] = "MMFF, graph available at inference; unsupported/nonconverged samples retain raw coordinates"
        payload.update(refined_samples=refined, refinement_status=statuses, refined_fixed_correspondence_rmsd=refined_rmsd)
        if args.chemistry:
            progress.start("Refined chemistry and symmetry checks", len(rows), "molecules")
            report["refined"]["chemistry"], payload["refined_symmetry_aware_same_graph_rmsd"] = chemical_summary(
                refined, references, atoms, mask, bonds, progress_callback=progress.update)
    progress.start("Writing samples and summary")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "samples.npz", **payload)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"samples": str(output / "samples.npz"), "summary": str(output / "summary.json"),
                      "evaluated_rows": len(rows), "candidates_per_row": args.num_samples}, indent=2))


if __name__ == "__main__":
    main()
