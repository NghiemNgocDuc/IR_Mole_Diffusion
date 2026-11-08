"""Safely inspect numeric native NPZ data and persist molecule-grouped splits."""

import argparse
import json
from pathlib import Path

from diffusion.data import audit_native_data, file_sha256, get_or_create_splits, load_native_npz


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="dataset_audit.json")
    parser.add_argument("--splits", help="Create or validate a grouped train/val/test manifest")
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    data = load_native_npz(args.data, require_molecule_ids=bool(args.splits))
    report = {"data": str(Path(args.data).resolve()), "data_sha256": file_sha256(args.data), **audit_native_data(data)}
    if args.splits:
        report["split_manifest"] = get_or_create_splits(args.splits, data["molecule_ids"], args.data,
            seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
