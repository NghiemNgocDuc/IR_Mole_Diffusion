"""Mine a frozen model's wrong candidates using training inputs and labels only.

The first valid wrong candidate in the legacy decoder's order replaces a static
fallback for a seeded subset. Remaining rows retain verified training-only
fingerprint negatives. Generated identities need not occur in the dataset.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from diffusion.data import NativeNPZDataset, collate_native, file_sha256, load_native_npz, validate_split_manifest
from diffusion.progress import EvaluationProgress
from diffusion.ranking import canonical_identity, encode_negative, load_negative_rows, load_negative_tokens
from diffusion.spectrum import SpectrumStructureModel
from diffusion.spectrum_audit import _same_json, atomic_json, exclusive_report, select_rows
from evaluate_spectrum import valid_candidates


def choose_negative(candidates, reference, vocabulary, width):
    """Keep decoder order; never use the reference to reorder valid wrong graphs."""
    reference = canonical_identity(reference)
    for candidate in candidates:
        if canonical_identity(candidate["smiles"]) == reference:
            continue
        try:
            tokens, identity = encode_negative(candidate["selfies"], vocabulary, width,
                                               reference, candidate["smiles"])
        except ValueError:
            # A valid decoded molecule can still have a path outside this
            # training tokenizer's width. Keep the fallback; never truncate it.
            continue
        return {"smiles": identity, "selfies": candidate["selfies"], "tokens": tokens.tolist()}
    return None


def validate_mining_resume(previous, provenance, selected_rows, data):
    """Accept only a matching prefix and semantically verified training negatives."""
    if (not isinstance(previous, dict) or type(previous.get("schema_version")) is not int
            or previous["schema_version"] != 1 or not _same_json(previous.get("provenance"), provenance)
            or not _same_json(previous.get("selected_rows"), selected_rows)):
        raise ValueError("Mining resume provenance or frozen training selection differs.")
    records = previous.get("rows")
    if not isinstance(records, list) or len(records) > len(selected_rows) or type(previous.get("complete")) is not bool:
        raise ValueError("Mining resume must contain a valid ordered prefix and completion flag.")
    if previous["complete"] and len(records) != len(selected_rows):
        raise ValueError("Completed mining report does not cover its selection.")
    for position, record in enumerate(records):
        row = selected_rows[position]
        if not isinstance(record, dict) or type(record.get("row")) is not int or record["row"] != row:
            raise ValueError("Mining resume row order differs from its training selection.")
        reference = str(data["molecule_ids"][row])
        if record.get("reference_smiles") != reference or "negative" not in record:
            raise ValueError("Mining resume reference differs from its training label.")
        negative = record["negative"]
        if negative is not None:
            if not isinstance(negative, dict) or not isinstance(negative.get("tokens"), list):
                raise ValueError("Invalid saved generated negative.")
            tokens, _ = encode_negative(negative["selfies"], data["vocabulary"],
                data["selfies_tokens"].shape[1], reference, negative["smiles"])
            if any(type(value) is not int for value in negative["tokens"]) or tokens.tolist() != negative["tokens"]:
                raise ValueError("Saved generated tokens differ from the declared SELFIES.")
    return records.copy()


def atomic_archive(path, **arrays):
    """Commit a complete compressed archive atomically; no partial NPZ reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _provenance(args, data, manifest, model, checkpoint_hash, split_hash, fallback_hash, ema_weights):
    import selfies
    from rdkit import rdBase
    root = Path(__file__).resolve().parent
    return {"method": "model_generated_wrong_candidate_with_training_only_static_fallback_v1",
        "data_sha256": manifest["data_sha256"], "split_sha256": split_hash,
        "checkpoint_sha256": checkpoint_hash, "fallback_sha256": fallback_hash,
        "ema_weights": ema_weights, "checkpoint_training_diagnostic": model.run_metadata.get("diagnostic"),
        "vocabulary": data["vocabulary"], "preprocessing": data.get("preprocessing"),
        "split": "train", "held_out_labels_used": False, "selection": "seeded_random_prefix",
        "seed": args.seed, "limit": args.limit, "beam_size": args.beam_size, "length_penalty": .6,
        "decode_protocol": "legacy_generate_v1_valid_unique_candidates",
        "negative_policy": "first valid wrong identity in decoder order; no formula-based candidate reranking",
        "token_width": data["selfies_tokens"].shape[1], "device": args.device,
        "torch_num_threads": torch.get_num_threads(), "cuda_version": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                     "rdkit": rdBase.rdkitVersion, "selfies": selfies.__version__},
        "source_sha256": {name: file_sha256(root / name) for name in (
            "mine_ranking.py", "diffusion/ranking.py", "evaluate_spectrum.py", "diffusion/spectrum.py",
            "diffusion/encoder.py", "diffusion/model.py", "diffusion/data.py")}}


def mine(args):
    output = Path(args.output)
    records_path = output.with_suffix(".mining.json")
    summary_path = output.with_suffix(".json")
    protected = {Path(value).resolve() for value in (args.data, args.splits, args.checkpoint, args.fallback)}
    targets = [output.resolve(), records_path.resolve(), summary_path.resolve()]
    if len(set(targets)) != 3 or any(path in protected for path in targets):
        raise ValueError("Mining outputs would overwrite an input or each other; use an .npz output path.")
    if (output.exists() or records_path.exists() or summary_path.exists()) and not args.resume:
        raise ValueError("Mining output exists; use --resume to verify it or choose a fresh output.")
    with exclusive_report(output), EvaluationProgress(output.with_suffix(".progress"), interval=args.progress_interval) as progress:
        progress.start("Loading training data, fallback pairs and frozen checkpoint")
        data = load_native_npz(args.data)
        manifest = validate_split_manifest(json.loads(Path(args.splits).read_text(encoding="utf-8")),
                                           data["molecule_ids"], args.data)
        training_rows = np.asarray(manifest["splits"]["train"], dtype=np.int64)
        selected_rows = select_rows(training_rows, "random", args.limit, args.seed)
        split_hash, fallback_hash = file_sha256(args.splits), file_sha256(args.fallback)
        fallback = load_negative_rows(args.fallback, training_rows, manifest["data_sha256"], split_hash)
        # Validate training identity differences, not just integer membership.
        load_negative_tokens(args.fallback, training_rows, data, manifest["data_sha256"], split_hash)
        if file_sha256(args.fallback) != fallback_hash:
            raise ValueError("Fallback archive changed during loading.")
        checkpoint_hash = file_sha256(args.checkpoint)
        model = SpectrumStructureModel.load_from_checkpoint(args.checkpoint)
        if file_sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed during loading.")
        for key, expected in (("data_sha256", manifest["data_sha256"]), ("split_sha256", split_hash),
                              ("preprocessing", data.get("preprocessing"))):
            if not _same_json(model.run_metadata.get(key), expected):
                raise ValueError(f"Checkpoint {key} differs from mining inputs.")
        if list(model.hparams.vocabulary) != data["vocabulary"]:
            raise ValueError("Checkpoint vocabulary differs from mining inputs.")
        if model.run_metadata.get("spectrum_control", "none") != "none":
            raise ValueError("Mining requires an ordinary spectrum-conditioned checkpoint.")
        if model.run_metadata.get("diagnostic") and not args.allow_diagnostic:
            raise ValueError("Diagnostic checkpoints require --allow_diagnostic and cannot establish accuracy.")
        if data["selfies_tokens"].shape[1] > model.hparams.max_length:
            raise ValueError("Training token width exceeds model sequence capacity.")
        ema_weights = bool(model.use_ema_weights())
        if not ema_weights and not args.allow_diagnostic:
            raise ValueError("Mining requires the baseline EMA weights used by the pilot.")
        model.to(args.device).eval()
        provenance = _provenance(args, data, manifest, model, checkpoint_hash, split_hash, fallback_hash, ema_weights)
        records = []
        if records_path.exists():
            records = validate_mining_resume(json.loads(records_path.read_text(encoding="utf-8")),
                                             provenance, selected_rows, data)
        elif output.exists() or summary_path.exists():
            raise ValueError("Existing archive/summary has no resumable mining record.")
        def report(complete=False):
            return {"schema_version": 1, "provenance": provenance, "selected_rows": selected_rows,
                    "complete": complete, "rows": records}
        if output.exists():
            prior_report = json.loads(records_path.read_text(encoding="utf-8"))
            if not prior_report["complete"]:
                raise ValueError("Existing archive has an incomplete mining record.")
            load_negative_tokens(output, training_rows, data, manifest["data_sha256"], split_hash, checkpoint_hash)
            with np.load(output, allow_pickle=False) as existing:
                if (not _same_json(json.loads(str(existing["provenance_json"].item())), provenance)
                        or str(existing["mining_report_sha256"].item()) != file_sha256(records_path)):
                    raise ValueError("Existing archive provenance or mining record hash differs.")
                mined = [record for record in records if record["negative"] is not None]
                expected_rows = np.asarray([record["row"] for record in mined], dtype=np.int64)
                expected_tokens = np.asarray([record["negative"]["tokens"] for record in mined], dtype=np.int64).reshape(
                    len(mined), data["selfies_tokens"].shape[1])
                expected_smiles = np.asarray([record["negative"]["smiles"] for record in mined], dtype=str)
                if (not np.array_equal(existing["negative_rows"], fallback)
                        or not np.array_equal(existing["mined_rows"], expected_rows)
                        or not np.array_equal(existing["mined_tokens"], expected_tokens)
                        or not np.array_equal(existing["mined_smiles"], expected_smiles)):
                    raise ValueError("Existing archive pairs differ from the saved mining record or fallback.")
        else:
            atomic_json(records_path, report())
            dataset = NativeNPZDataset(data, selected_rows)
            offset = len(records)
            progress.start(f"Mining model mistakes on training rows ({offset} already saved)", len(selected_rows)-offset,
                           "training molecules")
            try:
                with torch.inference_mode():
                    for position in range(offset, len(selected_rows)):
                        row = selected_rows[position]
                        spectra = collate_native([dataset[position]])["spectra"]
                        spectra = ({key: value.to(args.device) for key, value in spectra.items()}
                                   if isinstance(spectra, dict) else spectra.to(args.device))
                        candidates = valid_candidates(model.generate(spectra, args.beam_size, .6), data["coords"].shape[1])
                        reference = str(data["molecule_ids"][row])
                        negative = choose_negative(candidates, reference, data["vocabulary"], data["selfies_tokens"].shape[1])
                        records.append({"row": row, "reference_smiles": reference, "negative": negative})
                        if len(records) % args.save_every == 0:
                            atomic_json(records_path, report())
                        progress.update(position + 1 - offset)
            except BaseException:
                atomic_json(records_path, report())
                raise
            atomic_json(records_path, report(complete=True))
            mined = [record for record in records if record["negative"] is not None]
            atomic_archive(output, schema_version=np.asarray(2, dtype=np.int64),
                training_rows=training_rows, negative_rows=fallback,
                data_sha256=np.asarray(manifest["data_sha256"]), split_sha256=np.asarray(split_hash),
                provenance_json=np.asarray(json.dumps(provenance, sort_keys=True, allow_nan=False)),
                mining_report_sha256=np.asarray(file_sha256(records_path)),
                mined_rows=np.asarray([record["row"] for record in mined], dtype=np.int64),
                mined_tokens=np.asarray([record["negative"]["tokens"] for record in mined], dtype=np.int64)
                    .reshape(len(mined), data["selfies_tokens"].shape[1]),
                mined_smiles=np.asarray([record["negative"]["smiles"] for record in mined], dtype=str))
            load_negative_tokens(output, training_rows, data, manifest["data_sha256"], split_hash, checkpoint_hash)
        count = sum(record["negative"] is not None for record in records)
        summary = {"schema_version": 2, "provenance": provenance, "training_rows": len(training_rows),
            "attempted_training_rows": len(selected_rows), "model_generated_pairs": count,
            "static_fallback_pairs": len(training_rows)-count, "held_out_labels_used": False,
            "complete": True, "output_sha256": file_sha256(output),
            "mining_report_sha256": file_sha256(records_path),
            "interpretation": "One wrong candidate per mined input; no held-out labels or formula supplied at inference. "
                              "Other training rows retain the same static negatives as the comparison arm. "
                              "Mining correctness is relative to training labels, not measured generalization."}
        if summary_path.exists() and not _same_json(json.loads(summary_path.read_text(encoding="utf-8")), summary):
            raise ValueError("Existing mining summary differs from the verified archive.")
        atomic_json(summary_path, summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--fallback", required=True, help="Verified prepare_ranking.py training-only archive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=5000, help="Seeded training-input budget; all other rows retain static negatives")
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--progress_interval", type=float, default=15.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_diagnostic", action="store_true", help="Software checks only; never trained-model evidence")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.seed < 0 or args.save_every < 1 or not 1 <= args.beam_size <= 100:
        parser.error("Limit/save_every must be positive, seed nonnegative, and beam_size in 1..100.")
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
        parser.error("Progress interval must be finite and nonnegative.")
    try:
        return mine(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
