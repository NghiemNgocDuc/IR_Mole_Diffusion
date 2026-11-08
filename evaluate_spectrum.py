"""Held-out molecular identity/formula evaluation of the IR-only first stage."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from diffusion.data import NativeNPZDataset, collate_native, file_sha256, load_native_npz, validate_split_manifest
from diffusion.spectrum import SpectrumStructureModel
from diffusion.progress import EvaluationProgress
from diffusion.spectrum_audit import (atomic_json, exclusive_report, make_selection, row_record,
    summarize_records, validate_resume, validate_selection)


def valid_candidates(decoded, max_atoms=29):
    import selfies as sf
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    candidates, seen = [], set()
    for entry in decoded:
        try:
            mol = Chem.MolFromSmiles(sf.decoder(entry["selfies"]))
            if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
                continue
            if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetAtomicNum() not in [1, 6, 7, 8, 9] for a in mol.GetAtoms()):
                continue
            molecule = Chem.AddHs(mol)
            if molecule.GetNumAtoms() > max_atoms:
                continue
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smiles in seen:
                continue
            seen.add(smiles)
            candidates.append({**entry, "smiles": smiles,
                               "formula": rdMolDescriptors.CalcMolFormula(mol)})
        except (ValueError, RuntimeError, sf.DecoderError):
            continue
    return candidates


def _provenance(args, data, model, split_hash, ema_weights, checkpoint_hash):
    import selfies
    from rdkit import rdBase
    root = Path(__file__).resolve().parent
    return {
        "data_sha256": model.run_metadata["data_sha256"], "split_sha256": split_hash,
        "checkpoint_sha256": checkpoint_hash, "ema_weights": ema_weights,
        "checkpoint_training_diagnostic": model.run_metadata.get("diagnostic"),
        "preprocessing": data.get("preprocessing"), "vocabulary": data["vocabulary"],
        "split": args.split, "beam_size": args.beam_size,
        "length_penalty": getattr(args, "length_penalty", 0.6),
        "decode_protocol": f"{getattr(args, 'score_mode', 'legacy')}_generate_v1_valid_unique_candidates",
        "spectrum_control": args.spectrum_control, "control_seed": args.control_seed,
        "device": args.device, "torch_num_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                     "rdkit": rdBase.rdkitVersion, "selfies": selfies.__version__},
        "source_sha256": {name: file_sha256(root / name) for name in (
            "evaluate_spectrum.py", "diffusion/spectrum_audit.py", "diffusion/spectrum.py",
            "diffusion/encoder.py", "diffusion/model.py", "diffusion/data.py")},
    }


def evaluate(args):
    """Persist a provenance-checked prefix with explicit, frozen decoding settings."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    # Preserve callers that built a Namespace before decoding experiments were
    # added, while writing the actual settings into every new report.
    args.score_mode = getattr(args, "score_mode", "legacy")
    args.length_penalty = getattr(args, "length_penalty", 0.6)
    if args.score_mode not in ("legacy", "consistent"):
        raise ValueError("score_mode must be legacy or consistent")
    if not math.isfinite(args.length_penalty) or args.length_penalty < 0:
        raise ValueError("length_penalty must be finite and nonnegative")
    output = Path(args.output)
    protected = [Path(args.data), Path(args.splits), Path(args.checkpoint)]
    selection_path = Path(args.selection_manifest) if args.selection_manifest else None
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Report output would overwrite an input")
    if selection_path and selection_path.resolve() in {p.resolve() for p in [*protected, output]}:
        raise ValueError("Selection manifest would overwrite an input or report")
    if output.exists() and not args.resume:
        raise ValueError("Report exists; use --resume to validate and continue, or choose a new output")
    with exclusive_report(output), EvaluationProgress(output.with_suffix(".progress"), interval=args.progress_interval) as progress:
        progress.start("Loading data and verifying split hashes")
        data = load_native_npz(args.data)
        manifest = validate_split_manifest(json.loads(Path(args.splits).read_text(encoding="utf-8")),
                                           data["molecule_ids"], args.data)
        split_hash = file_sha256(args.splits)
        selection = make_selection(manifest, split_hash, args.split, args.selection, args.limit, args.seed)
        if selection_path and selection_path.exists():
            saved_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            validate_selection(saved_selection, manifest, split_hash, args.split)
            if saved_selection != selection:
                raise ValueError("Saved selection differs from requested policy/seed/limit; use the original settings")
            selection = saved_selection
        elif selection_path:
            atomic_json(selection_path, selection)
        rows = validate_selection(selection, manifest, split_hash, args.split)
        progress.start("Loading checkpoint and checking vocabulary/preprocessing")
        checkpoint_hash = file_sha256(args.checkpoint)
        model = SpectrumStructureModel.load_from_checkpoint(args.checkpoint)
        if file_sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed during loading; retry with a stable checkpoint")
        if model.run_metadata.get("data_sha256") != manifest["data_sha256"] or model.run_metadata.get("split_sha256") != split_hash:
            raise ValueError("Checkpoint data/split provenance differs")
        if model.run_metadata.get("preprocessing") != data.get("preprocessing"):
            raise ValueError("Checkpoint preprocessing differs from the dataset")
        if list(model.hparams.vocabulary) != data["vocabulary"]:
            raise ValueError("Checkpoint vocabulary differs from the dataset")
        ema_weights = bool(model.use_ema_weights())
        model.to(args.device).eval()
        provenance = _provenance(args, data, model, split_hash, ema_weights, checkpoint_hash)
        records = []
        if output.exists():
            previous = json.loads(output.read_text(encoding="utf-8"))
            records = validate_resume(previous, provenance, selection, data["molecule_ids"])
            for record in records:
                expected = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(record["reference_smiles"]))
                if record["reference_formula"] != expected:
                    raise ValueError("Resumed reference formula differs from the dataset")
            if previous["complete"]:
                print(f"Already complete and provenance verified: {output}", flush=True)
                return previous
        # Build the entire selection even on resume: shuffled donors must not change.
        dataset = NativeNPZDataset(data, rows, spectrum_control=args.spectrum_control, seed=args.control_seed)
        configuration = vars(args).copy()
        def report(complete=False):
            return {"schema_version": 2, "configuration": configuration,
                "complete": complete, "selected_rows": rows, "selection": selection,
                "provenance": provenance, "ema_weights": ema_weights,
                "checkpoint_training_diagnostic": provenance["checkpoint_training_diagnostic"],
                "identity_metric": "canonical non-isomeric SMILES; no ground-truth formula supplied",
                "candidate_ranking": f"{args.score_mode.capitalize()} decoder order after validity filtering and identity deduplication; no reference-based ranking",
                "data_sha256": manifest["data_sha256"], "split_sha256": split_hash,
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                **summarize_records(records), "rows": records}
        atomic_json(output, report())
        offset = len(records)
        print(f"Frozen {args.split} selection: {len(rows)} rows; {offset} already saved. "
              f"Policy={args.selection}, seed={args.seed}, EMA={ema_weights}. "
              f"Progress: {output.with_suffix('.progress') / 'progress.json'}", flush=True)
        if provenance["checkpoint_training_diagnostic"]:
            print("DIAGNOSTIC CHECKPOINT: these results test software, not trained-model accuracy.", flush=True)
        progress.start(f"Generating remaining rows ({offset} previously saved of {len(rows)})", len(rows)-offset, "molecules")
        try:
            for position in range(offset, len(rows)):
                row = rows[position]
                spectra = collate_native([dataset[position]])["spectra"]
                spectra = {key: value.to(args.device) for key, value in spectra.items()} if isinstance(spectra, dict) else spectra.to(args.device)
                candidates = valid_candidates(model.generate(spectra, args.beam_size,
                    length_penalty=args.length_penalty, score_mode=args.score_mode), data["coords"].shape[1])
                identity = str(data["molecule_ids"][row])
                formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(identity))
                records.append(row_record(row, identity, formula, candidates))
                if len(records) % args.save_every == 0:
                    atomic_json(output, report())
                progress.update(position + 1 - offset)
        except BaseException:
            atomic_json(output, report())
            raise
        result = report(complete=True)
        atomic_json(output, result)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("rows", "selected_rows", "selection", "provenance")}, indent=2), flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--score_mode", choices=("legacy", "consistent"), default="legacy",
                        help="Legacy scoring or the same length-normalized score for pruning and final ranking")
    parser.add_argument("--length_penalty", type=float, default=0.6,
                        help="Nonnegative score length exponent; sequence length includes BOS/EOS")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="evaluation/spectrum.json")
    parser.add_argument("--spectrum_control", choices=("none", "removed", "shuffled"), default="none")
    parser.add_argument("--selection", choices=("first", "random", "all"), default="first",
                        help="Legacy first rows, uniformly random subset, or the entire split (random order)")
    parser.add_argument("--seed", type=int, default=20261105, help="Frozen selection seed")
    parser.add_argument("--control_seed", type=int, default=0, help="Separate seed for shuffled-spectrum donors")
    parser.add_argument("--selection_manifest", help="Write or verify a frozen row-selection JSON")
    parser.add_argument("--resume", action="store_true", help="Continue only a matching report; a verified complete report is reused")
    parser.add_argument("--progress_interval", type=float, default=15.0)
    parser.add_argument("--save_every", type=int, default=10, help="Save every N molecules; at most N-1 repeat after a hard kill")
    args = parser.parse_args(argv)
    if args.limit < 1 or not 1 <= args.beam_size <= 100 or args.save_every < 1:
        parser.error("Limit/save_every must be positive and beam size must be in 1..100.")
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0 or min(args.seed, args.control_seed) < 0:
        parser.error("Progress interval and seeds must be finite/nonnegative.")
    if not math.isfinite(args.length_penalty) or args.length_penalty < 0:
        parser.error("Length penalty must be finite and nonnegative.")
    try:
        return evaluate(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
