"""Prepare train-only, same-formula-preferred hard negatives for fine-tuning."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from diffusion.data import file_sha256, load_native_npz, validate_split_manifest
from diffusion.progress import EvaluationProgress
from diffusion.ranking import build_negative_rows, load_negative_rows
from diffusion.spectrum_audit import _same_json, atomic_json, exclusive_report


METHOD = "Morgan radius-2 fingerprints; nearest identity in a seeded capped training pool; same formula preferred"


def atomic_archive(output, **arrays):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{output.name}.", suffix=".tmp",
                                         dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def verify_configuration(configuration, args):
    if not isinstance(configuration, dict):
        raise ValueError("Negative preparation configuration is missing.")
    for name in ("seed", "max_pool"):
        if type(configuration.get(name)) is not int or configuration[name] != getattr(args, name):
            raise ValueError(f"Negative preparation {name} differs; choose a new output.")
    for name in ("data", "splits", "output"):
        if not isinstance(configuration.get(name), str) or Path(configuration[name]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError(f"Negative preparation {name} path differs; use the original inputs or a new output.")


def prepare(args):
    output, sidecar = Path(args.output), Path(args.output).with_suffix(".json")
    protected = {Path(value).resolve() for value in (args.data, args.splits)}
    if output.suffix != ".npz" or any(path.resolve() in protected for path in (output, sidecar)):
        raise ValueError("Use an .npz output separate from the input data and split files.")
    with exclusive_report(output):
        if (output.exists() or sidecar.exists()) and not args.resume:
            raise ValueError("Negative files already exist; use --resume to verify them, or choose a fresh output.")
        if sidecar.exists() and not output.exists():
            raise ValueError("Negative metadata exists but its archive is missing; choose a fresh output.")
        data_hash, split_hash = file_sha256(args.data), file_sha256(args.splits)
        data = load_native_npz(args.data)
        manifest = validate_split_manifest(json.loads(Path(args.splits).read_text(encoding="utf-8")), data["molecule_ids"], args.data)
        if manifest["data_sha256"] != data_hash or file_sha256(args.splits) != split_hash:
            raise ValueError("Preparation inputs changed while loading.")
        rows = manifest["splits"]["train"]
        existing = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None
        configuration = dict(sorted((key, value) for key, value in vars(args).items() if key != "resume"))
        if output.exists():
            try:
                negatives = load_negative_rows(output, rows, data_hash, split_hash)
                with np.load(output, allow_pickle=False) as archive:
                    kinds = archive["pool_types"].copy()
                    if "configuration_json" in archive:
                        configuration = json.loads(str(archive["configuration_json"].item()))
                    elif existing is not None:
                        configuration = existing.get("configuration")
                    else:
                        raise ValueError("Legacy archive has no embedded settings and no metadata; choose a fresh output.")
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                raise ValueError(f"Cannot safely reuse negative archive {output}: {exc}") from exc
            verify_configuration(configuration, args)
            if kinds.dtype.kind not in "iu" or kinds.shape != (len(rows),) or not np.isin(kinds, [1, 2, 3]).all():
                raise ValueError("Negative archive has invalid pool annotations; choose a fresh output.")
            if np.any(data["molecule_ids"][negatives] == data["molecule_ids"][rows]):
                raise ValueError("Negative archive contains the same training identity as its target.")
        else:
            with EvaluationProgress(output.parent / "negative_preparation") as progress:
                progress.start("Selecting training-only hard negatives", len(rows), "training molecules")
                negatives, kinds = build_negative_rows(data, rows, args.seed, args.max_pool, progress.update)
            if file_sha256(args.data) != data_hash or file_sha256(args.splits) != split_hash:
                raise ValueError("Preparation inputs changed during negative selection.")
            # Do not add schema_version: that field identifies mined archives in
            # load_negative_tokens. Static readers ignore this additional field.
            atomic_archive(output, training_rows=np.asarray(rows, dtype=np.int64), negative_rows=negatives,
                pool_types=kinds, data_sha256=np.asarray(data_hash), split_sha256=np.asarray(split_hash),
                configuration_json=np.asarray(json.dumps(configuration, sort_keys=True)))
        names = {1: "same_formula", 2: "same_heavy_atom_count", 3: "other_training_identity"}
        report = {"configuration": configuration, "training_rows": len(rows),
            "negative_pools": {names[int(kind)]: int((kinds == kind).sum()) for kind in np.unique(kinds)},
            "data_sha256": data_hash, "split_sha256": split_hash, "output_sha256": file_sha256(output),
            "method": METHOD, "held_out_labels_used": False}
        if existing is not None:
            if not _same_json(existing, report):
                raise ValueError("Existing negative metadata differs from the verified archive; choose a fresh output.")
        else:
            atomic_json(sidecar, report)
        print(json.dumps(report, indent=2), flush=True)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--output", default="runs/spectrum_rank_finetune/ranking_negatives.npz")
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--max_pool", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Verify complete artifacts or recover a missing metadata file from an atomic archive")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.max_pool < 1:
        parser.error("Seed must be nonnegative and max_pool positive.")
    try:
        return prepare(args)
    except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
