"""One trainer implementation for both supported stages."""
import argparse
import inspect
import json
import platform
import sys
import time
from pathlib import Path

from diffusion.data import (NativeNPZDataset, audit_native_data, collate_native,
                            file_sha256, get_or_create_splits, load_native_npz)


def parser_for(stage):
    parser = argparse.ArgumentParser(description=f"Train the self-contained {stage} model")
    parser.add_argument("--config", help="JSON defaults; CLI overrides individual fields")
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--save_dir", default=f"runs/{stage}")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--resume_from_checkpoint")
    sources.add_argument("--finetune_from")
    parser.add_argument("--encoder_spec", help="JSON file describing the native spectrum encoder")
    parser.add_argument("--seed", type=int, default=78438379)
    parser.add_argument("--batch_size", type=int, default=32 if stage == "diffusion" else 64)
    parser.add_argument("--lr", type=float, default=1e-4 if stage == "diffusion" else 2e-4)
    parser.add_argument("--epochs", type=int, default=200 if stage == "diffusion" else 100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=0.)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", choices=("32-true", "bf16-mixed", "16-mixed"), default="32-true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--spectrum_control", choices=("none", "removed", "shuffled"), default="none")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_train_batches", type=int, help="Diagnostic first N batches")
    parser.add_argument("--limit_val_batches", type=int, help="Diagnostic first N validation batches")
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--checkpoint_every_steps", type=int, default=1000)
    parser.add_argument("--dry_run", action="store_true")
    if stage == "spectrum":
        parser.add_argument("--decoder_layers", type=int, default=4)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--label_smoothing", type=float, default=0.05)
        parser.add_argument("--ranking_weight", type=float, default=0.0)
        parser.add_argument("--ranking_margin", type=float, default=0.1)
        parser.add_argument("--ranking_negatives", help="Verified train-only negative-row NPZ prepared by prepare_ranking.py")
    else:
        parser.add_argument("--encoder_checkpoint", help="New spectrum-stage checkpoint; initialize its encoder")
        parser.add_argument("--encoder_lr", type=float, default=1e-5)
        parser.add_argument("--unfreeze_at_epoch", type=int, default=5)
        parser.add_argument("--n_timesteps", type=int, default=1000)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--n_layers", type=int, default=6)
        parser.add_argument("--k_neighbors", type=int, default=32)
        parser.add_argument("--conditioning_mode", choices=("pooled", "node", "node_edge"), default="node")
        parser.add_argument("--readout", choices=("vector", "residual"), default="vector")
        parser.add_argument("--coordinate_scale", type=float, default=1.)
        parser.add_argument("--physics_weight", type=float, default=0.)
        parser.add_argument("--physics_min_alpha", type=float, default=0.05)
        parser.add_argument("--warmup_steps", type=int, default=1000)
        for name, default in (("bond_weight", .1), ("clash_weight", .1), ("angle_weight", 0.),
                              ("ring_weight", 0.), ("torsion_weight", 0.), ("chirality_weight", 0.)):
            parser.add_argument("--" + name, type=float, default=default)
        for name, default in (("graph_conditioning", True), ("self_conditioning", True),
                              ("use_cosine_schedule", True), ("use_attention_pooling", True),
                              ("adaptive_cutoff", False)):
            parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=default)
        parser.add_argument("--cutoff_radius", type=float, default=6.)
    return parser


def train(stage, argv=None):
    parser = parser_for(stage)
    initial, _ = parser.parse_known_args(argv)
    if initial.config:
        defaults = json.loads(Path(initial.config).read_text(encoding="utf-8"))
        actions = {action.dest: action for action in parser._actions}
        if not isinstance(defaults, dict) or set(defaults) - set(actions):
            parser.error("Config must be a JSON object containing supported option names only.")
        for key, value in defaults.items():
            if actions[key].choices and value not in actions[key].choices:
                parser.error(f"Invalid config value for {key}.")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.resume_from_checkpoint and args.finetune_from:
        parser.error("Resume and fine-tune sources are mutually exclusive, including config defaults.")
    if min(args.batch_size, args.epochs, args.devices, args.accumulate_grad_batches, args.checkpoint_every_steps) < 1 or args.num_workers < 0:
        parser.error("Batch/epoch/device/accumulation/checkpoint counts must be positive; workers nonnegative.")
    if args.devices != 1:
        parser.error("This validated recipe uses one GPU/device. Multi-device training is not enabled.")
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from diffusion.model import EquivariantDiffusion3D_v4
    from diffusion.spectrum import SpectrumStructureModel
    pl.seed_everything(args.seed, workers=True)
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        parser.error("GPU requested, but PyTorch cannot access CUDA. Check the job allocation and wheel.")
    if args.accelerator == "cpu" and args.precision != "32-true":
        parser.error("Use 32-true precision for the validated CPU path.")
    data = load_native_npz(args.data)
    manifest = get_or_create_splits(args.splits, data["molecule_ids"], args.data, seed=args.seed)
    split_hash = file_sha256(args.splits)
    provenance = {"data_sha256": manifest["data_sha256"], "split_sha256": split_hash,
        "spectrum_control": args.spectrum_control, "preprocessing": data.get("preprocessing"),
        "diagnostic": bool(args.max_steps > 0 or args.limit_train_batches or args.limit_val_batches or args.overfit_batches)}
    ranking_tokens = None
    if stage == "spectrum":
        import math
        if not math.isfinite(args.ranking_weight) or not math.isfinite(args.ranking_margin) or min(args.ranking_weight, args.ranking_margin) < 0:
            parser.error("Ranking weight/margin must be finite and nonnegative.")
        if args.ranking_weight > 0:
            if not args.ranking_negatives:
                parser.error("--ranking_weight requires --ranking_negatives prepared from training rows only.")
            from diffusion.ranking import load_negative_tokens
            # Generated alternatives need not be existing dataset rows. Verify
            # tokenizer, identity, training membership and frozen mining weights.
            mining_checkpoint = args.finetune_from
            ranking_tokens = load_negative_tokens(args.ranking_negatives, manifest["splits"]["train"], data,
                manifest["data_sha256"], split_hash,
                file_sha256(mining_checkpoint) if mining_checkpoint else None)
            provenance["ranking_negatives_sha256"] = file_sha256(args.ranking_negatives)
            provenance["ranking_weight"] = args.ranking_weight
            provenance["ranking_margin"] = args.ranking_margin
    model_class = SpectrumStructureModel if stage == "spectrum" else EquivariantDiffusion3D_v4
    if stage == "spectrum" and "selfies_tokens" not in data:
        parser.error("Spectrum training needs SELFIES labels from prepare_squirl.py.")
    if stage == "diffusion":
        if (args.graph_conditioning or args.physics_weight > 0) and data["bond_types"] is None:
            parser.error("Graph conditioning/physics requires verified bond_types.")
        for weight, field in (("angle_weight", "angles"), ("ring_weight", "planarity"),
                              ("torsion_weight", "torsions"), ("chirality_weight", "chirality")):
            if args.physics_weight > 0 and getattr(args, weight) > 0 and field not in (data["graph"] or {}):
                parser.error(f"{weight} requires explicit {field} targets; they are not guessed.")
    source = args.resume_from_checkpoint or args.finetune_from
    finetune_initialization = None
    spec = json.loads(Path(args.encoder_spec).read_text(encoding="utf-8")) if args.encoder_spec else None
    encoder_source = None
    if stage == "diffusion" and args.encoder_checkpoint and not source:
        encoder_source = SpectrumStructureModel.load_from_checkpoint(args.encoder_checkpoint)
        for key in ("data_sha256", "split_sha256", "spectrum_control", "preprocessing"):
            if encoder_source.run_metadata.get(key) != provenance[key]:
                parser.error(f"Encoder provenance differs in {key}; use the same dataset, splits, and spectral control.")
        encoder_source.use_ema_weights()
        spec = dict(encoder_source.hparams.encoder_spec)
    if source:
        model = model_class.load_from_checkpoint(source)
        if stage == "spectrum" and list(model.hparams.vocabulary) != data["vocabulary"]:
            parser.error("Loaded vocabulary differs from this dataset; explicit vocabulary transfer is required.")
        if args.resume_from_checkpoint:
            for key, value in provenance.items():
                if model.run_metadata.get(key) != value:
                    parser.error(f"Resume provenance differs in {key}; use the original run settings.")
        else:
            # Start a new fine-tune from the same EMA weights used in baseline
            # inference, with a fresh optimizer/EMA lifecycle.
            initialized_from_ema = model.use_ema_weights()
            finetune_initialization = "EMA" if initialized_from_ema else "raw (EMA unavailable)"
            model.hparams.lr = args.lr
            model.hparams.ema_decay = args.ema_decay
            model.hparams.weight_decay = args.weight_decay
            if stage == "spectrum":
                model.hparams.ranking_weight = args.ranking_weight
                model.hparams.ranking_margin = args.ranking_margin
                model.hparams.label_smoothing = args.label_smoothing
            if stage == "diffusion":
                model.hparams.encoder_lr = args.encoder_lr
                model.hparams.unfreeze_at_epoch = args.unfreeze_at_epoch
                model.hparams.physics_weight = args.physics_weight
                model.hparams.physics_min_alpha = args.physics_min_alpha
                for key in ("bond_weight", "clash_weight", "angle_weight", "ring_weight", "torsion_weight", "chirality_weight"):
                    value = getattr(args, key)
                    if value < 0:
                        parser.error("Physics weights must be nonnegative.")
                    model.hparams[key] = value
                    setattr(model.physics_loss, key, value)
            model._ema = {}
            model._ema_step = -1
        # Reconcile all architecture/data checks with the loaded checkpoint.
        if stage == "diffusion" and model.hparams.graph_conditioning != args.graph_conditioning:
            parser.error("Loaded checkpoint graph_conditioning differs from the requested configuration.")
    else:
        parameters = inspect.signature(model_class.__init__).parameters
        kwargs = {key: value for key, value in vars(args).items() if key in parameters and key != "encoder_spec"}
        if spec is not None:
            kwargs["encoder_spec"] = spec
        if stage == "spectrum":
            kwargs.update(vocabulary=data["vocabulary"], max_length=data["selfies_tokens"].shape[1])
        model = model_class(**kwargs)
        if encoder_source is not None:
            model.encoder.load_state_dict(encoder_source.encoder.state_dict(), strict=True)
        elif stage == "diffusion" and args.unfreeze_at_epoch != 0:
            parser.error("A randomly initialized encoder must train from epoch 0; set --unfreeze_at_epoch 0 or supply --encoder_checkpoint.")
    if stage == "spectrum" and float(model.hparams.ranking_weight) != args.ranking_weight:
        parser.error("Loaded ranking weight differs from requested settings; resume with the original objective.")
    model.run_metadata = provenance
    if stage == "diffusion" and (model.hparams.graph_conditioning or model.hparams.physics_weight > 0) and data["bond_types"] is None:
        parser.error("Loaded model requires graph targets absent from this data.")
    if stage == "diffusion" and model.hparams.physics_weight > 0:
        for weight, field in (("angle_weight", "angles"), ("ring_weight", "planarity"),
                              ("torsion_weight", "torsions"), ("chirality_weight", "chirality")):
            if model.hparams[weight] > 0 and field not in (data["graph"] or {}):
                parser.error(f"Loaded model's {weight} requires missing {field} targets.")
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = {"stage": stage, "arguments": vars(args), "model_hparams": dict(model.hparams),
        "provenance": provenance, "data_audit": audit_native_data(data),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "lightning": pl.__version__, "validation": "EMA weights when available; test excluded from fit",
        "source_sha256": {str(path): file_sha256(path) for path in
            sorted([*Path("diffusion").glob("*.py"), *Path(".").glob("*.py")])}}
    if source:
        resolved["checkpoint_source_sha256"] = file_sha256(source)
    if finetune_initialization is not None:
        resolved["finetune_initial_weights"] = finetune_initialization
    if encoder_source is not None:
        resolved["encoder_checkpoint_sha256"] = file_sha256(args.encoder_checkpoint)
    # Preserve prior manifests when resuming instead of overwriting their evidence.
    config_path = output / (f"resolved_config_resume_{time.time_ns()}.json" if args.resume_from_checkpoint else "resolved_config.json")
    if not args.resume_from_checkpoint and any((output / "checkpoints" / name).exists() for name in ("last.ckpt", "best.ckpt")):
        parser.error("This save_dir already has a checkpoint; resume explicitly or choose a new directory.")
    if args.finetune_from and Path(args.finetune_from).resolve().is_relative_to(output.resolve()):
        parser.error("Fine-tuning must use a new save_dir outside the source checkpoint directory.")
    config_path.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "train_rows": len(manifest["splits"]["train"]),
        "val_rows": len(manifest["splits"]["val"]), "test_rows": len(manifest["splits"]["test"]),
        "config": str(config_path), "parameters": sum(p.numel() for p in model.parameters())}, indent=2))
    if args.dry_run:
        return
    graph_mode = stage == "diffusion" and model.hparams.graph_conditioning
    train_ds = NativeNPZDataset(data, manifest["splits"]["train"], graph_mode, args.spectrum_control, args.seed)
    if ranking_tokens is not None:
        from diffusion.ranking import RankingDataset
        train_ds = RankingDataset(train_ds, negative_tokens=ranking_tokens)
    val_ds = NativeNPZDataset(data, manifest["splits"]["val"], graph_mode, args.spectrum_control, args.seed + 1)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_native,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    callbacks = [ModelCheckpoint(dirpath=str(output / "checkpoints"), filename="best", auto_insert_metric_name=False,
        save_top_k=1, monitor="val_loss", mode="min", save_last=True, save_on_train_epoch_end=False,
        enable_version_counter=False),
        ModelCheckpoint(dirpath=str(output / "recovery"), save_top_k=0, save_last=True,
                        every_n_train_steps=args.checkpoint_every_steps),
        EarlyStopping(monitor="val_loss", patience=args.patience, min_delta=args.min_delta,
                      check_on_train_epoch_end=False),
        LearningRateMonitor(logging_interval="step")]
    limits = {key: getattr(args, key) for key in ("limit_train_batches", "limit_val_batches") if getattr(args, key) is not None}
    trainer = pl.Trainer(max_epochs=args.epochs, max_steps=args.max_steps, accelerator=args.accelerator,
        devices=args.devices, precision=args.precision, callbacks=callbacks,
        logger=CSVLogger(str(output), name="metrics"), gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches, overfit_batches=args.overfit_batches,
        log_every_n_steps=10, num_sanity_val_steps=2, **limits)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)
    # A short max_steps diagnostic can finish before an epoch checkpoint is due.
    trainer.save_checkpoint(str(output / "checkpoints/last.ckpt"))
