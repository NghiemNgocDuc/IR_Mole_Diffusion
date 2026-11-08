"""Train corrected diffusion using explicit data and molecule-grouped splits."""

import argparse
import json
from pathlib import Path

from diffusion.data import (NativeNPZDataset, audit_native_data, collate_native,
                            file_sha256, get_or_create_splits, load_native_npz)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON defaults; explicit CLI options override them")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--ckpt", help="Trusted pretrained external spectral encoder checkpoint")
    modes.add_argument("--finetune_from", help="Corrected diffusion checkpoint; reset optimizer")
    modes.add_argument("--resume_from_checkpoint", help="Corrected diffusion checkpoint; restore training state")
    parser.add_argument("--checkpoint_policy", choices=("reject", "migrate"), default="reject")
    parser.add_argument("--encoder_spec", help="JSON specification for a self-contained spectral encoder")
    parser.add_argument("--data", required=False)
    parser.add_argument("--dataset_format", choices=("native", "external"), default="native")
    parser.add_argument("--trust_legacy_data", action="store_true", help="Explicitly trust data consumed by external QMugsDatasetV2")
    parser.add_argument("--molecule_ids", help="Non-pickled molecule ID .npy sidecar required for external datasets")
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--spectrum_control", choices=("none", "removed", "shuffled"), default="none")
    parser.add_argument("--model_type", default="Transformer")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_timesteps", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--conditioning_mode", choices=("pooled", "node", "node_edge"), default="pooled")
    parser.add_argument("--readout", choices=("residual", "vector"), default="residual")
    parser.add_argument("--graph_conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unfreeze_at_epoch", type=int, default=5)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--save_dir", default="./ckpt/corrected_diffusion")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--ir_rep", choices=("gaussian", "peaklist", "interp"), default="gaussian")
    parser.add_argument("--ir_max_peaks", type=int, default=128)
    parser.add_argument("--ir_peak_freq_scale", type=float, default=4000.0)
    parser.add_argument("--ir_peak_int_scale", type=float)
    parser.add_argument("--symbol_to_idx_path", default="./data/qmugs/symbol_to_idx_qmugs.json")
    parser.add_argument("--physics_weight", type=float, default=0.0)
    parser.add_argument("--bond_weight", type=float, default=0.1)
    parser.add_argument("--clash_weight", type=float, default=0.1)
    parser.add_argument("--angle_weight", type=float, default=0.0)
    parser.add_argument("--ring_weight", type=float, default=0.0)
    parser.add_argument("--torsion_weight", type=float, default=0.0)
    parser.add_argument("--chirality_weight", type=float, default=0.0)
    parser.add_argument("--bond_stats_json")
    parser.add_argument("--self_conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cosine_schedule", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive_cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cutoff_radius", type=float, default=6.0)
    parser.add_argument("--use_attention_pooling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true", help="Validate data/configuration and persist manifests without training")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    initial, _ = parser.parse_known_args(argv)
    if initial.config:
        defaults = json.loads(Path(initial.config).read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            parser.error("--config must contain a JSON object.")
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown config fields: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if not args.data:
        parser.error("--data is required (on the CLI or in --config).")
    sources = sum(bool(getattr(args, key)) for key in ("ckpt", "finetune_from", "resume_from_checkpoint"))
    if sources > 1:
        parser.error("--ckpt, --finetune_from, and --resume_from_checkpoint are mutually exclusive, including JSON defaults.")
    if not sources and not args.encoder_spec:
        parser.error("New training requires --ckpt or --encoder_spec.")
    if args.encoder_spec and sources:
        parser.error("--encoder_spec is only for new training without a checkpoint source.")
    if min(args.batch_size, args.epochs, args.n_timesteps, args.devices, args.accumulate_grad_batches) < 1 or args.num_workers < 0:
        parser.error("Batch size, epochs, timesteps, devices and accumulation must be positive; workers nonnegative.")
    return args

def prepare_datasets(args):
    if args.dataset_format == "native":
        data = load_native_npz(args.data)
        manifest = get_or_create_splits(args.splits, data["molecule_ids"], args.data,
            seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction)
        if args.physics_weight > 0 and data["bond_types"] is None:
            raise ValueError("Positive physics_weight requires explicit graph labels in native data.")
        train = NativeNPZDataset(data, manifest["splits"]["train"], args.graph_conditioning, args.spectrum_control, args.seed)
        val = NativeNPZDataset(data, manifest["splits"]["val"], args.graph_conditioning, args.spectrum_control, args.seed + 1)
        return train, val, collate_native, manifest, audit_native_data(data)
    if not args.trust_legacy_data:
        raise ValueError("External loading requires --trust_legacy_data; its loader may deserialize objects.")
    if not args.molecule_ids:
        raise ValueError("External datasets require --molecule_ids .npy sidecar to prevent row-level splitting.")
    if args.spectrum_control != "none":
        raise ValueError("Spectral controls require native data; convert legacy data first.")
    if args.physics_weight > 0 or args.graph_conditioning:
        raise ValueError("External tuple batches have no verified graph labels; convert to native data for graph features.")
    try:
        from dataset_utils_qmugs import QMugsDatasetV2, collate_variable_length
    except ImportError as exc:
        raise ImportError("External mode needs dataset_utils_qmugs.QMugsDatasetV2 and collate_variable_length on PYTHONPATH; use --dataset_format native with the documented NPZ schema otherwise.") from exc
    import numpy as np
    from torch.utils.data import Subset
    dataset = QMugsDatasetV2(args.data, mode="diffusion", ir_rep=args.ir_rep,
        ir_max_peaks=args.ir_max_peaks, ir_peak_freq_scale=args.ir_peak_freq_scale,
        ir_peak_int_scale=args.ir_peak_int_scale, symbol_to_idx_path=args.symbol_to_idx_path)
    identifiers = np.load(args.molecule_ids, allow_pickle=False)
    if identifiers.shape != (len(dataset),) or identifiers.dtype.kind not in "USiu":
        raise ValueError("External molecule_ids must be non-object strings/integers, one ID per dataset row.")
    manifest = get_or_create_splits(args.splits, identifiers, args.data, seed=args.seed,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    return Subset(dataset, manifest["splits"]["train"]), Subset(dataset, manifest["splits"]["val"]), collate_variable_length, manifest, {"external_dataset": True, "molecule_ids_sha256": file_sha256(args.molecule_ids)}


def main(argv=None):
    args = parse_args(argv)
    train_ds, val_ds, collate, manifest, audit = prepare_datasets(args)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder_spec = json.loads(Path(args.encoder_spec).read_text(encoding="utf-8")) if args.encoder_spec else None
    resolved = {**vars(args), "data": str(Path(args.data).resolve()), "splits": str(Path(args.splits).resolve()),
        "data_sha256": manifest["data_sha256"], "split_manifest": manifest, "data_audit": audit,
        "encoder_spec_resolved": encoder_spec,
        "validation_selection": "val_loss tracks denoising; final physical/conformer selection requires evaluate_diffusion.py",
        "test_usage": "test split is held out and excluded from trainer.fit"}
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "train_rows": len(train_ds), "val_rows": len(val_ds),
                          "test_rows": len(manifest["splits"]["test"]), "resolved_config": str(output / "resolved_config.json")}, indent=2))
        return
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from diffusion.model import EquivariantDiffusion3D_v4
    pl.seed_everything(args.seed, workers=True)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "collate_fn": collate,
                   "pin_memory": torch.cuda.is_available(), "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    checkpoint = args.finetune_from or args.resume_from_checkpoint
    if checkpoint:
        overrides = {"lr": args.lr, "encoder_lr": args.encoder_lr, "unfreeze_at_epoch": args.unfreeze_at_epoch} if args.finetune_from else {}
        model = EquivariantDiffusion3D_v4.load_from_checkpoint(checkpoint, map_location="cpu", checkpoint_policy=args.checkpoint_policy, **overrides)
        resolved["loaded_model_hparams"] = dict(model.hparams)
        (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        fields = ("model_type", "lr", "encoder_lr", "unfreeze_at_epoch", "n_timesteps", "hidden_dim", "n_layers",
            "k_neighbors", "physics_weight", "bond_weight", "clash_weight", "angle_weight", "ring_weight", "torsion_weight",
            "chirality_weight", "self_conditioning", "use_cosine_schedule", "adaptive_cutoff", "cutoff_radius", "use_attention_pooling",
            "bond_stats_json", "conditioning_mode", "readout", "graph_conditioning")
        kwargs = {key: getattr(args, key) for key in fields}
        if encoder_spec is not None:
            kwargs["encoder_spec"] = encoder_spec
        model = EquivariantDiffusion3D_v4(pretrained_ckpt_path=args.ckpt, **kwargs)
    callbacks = [ModelCheckpoint(dirpath=str(output), filename="corrected-{epoch:03d}-{val_loss:.4f}",
        save_top_k=3, monitor="val_loss", mode="min", save_last=True), LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=args.patience, mode="min", min_delta=0.001)]
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=args.devices,
        precision=args.precision, callbacks=callbacks, gradient_clip_val=1.0, log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_grad_batches, default_root_dir=str(output))
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
