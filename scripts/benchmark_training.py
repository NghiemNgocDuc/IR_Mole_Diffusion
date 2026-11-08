"""Measure this recipe's training throughput on the allocated GPU.

No useful weights are saved. Estimates include validation plus an explicit
planning allowance, and exclude queueing, beam decoding, and scientific tuning.
Run from the repository root inside a GPU allocation.
"""
import argparse
from contextlib import nullcontext
import gc
import inspect
import json
import math
from pathlib import Path
import sys
import time

# Also supports an ordinary Python installation, without the embedded runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from diffusion.data import NativeNPZDataset, collate_native, load_native_npz, validate_split_manifest
from diffusion.model import EquivariantDiffusion3D_v4
from diffusion.spectrum import SpectrumStructureModel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/squirl.npz")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("32-true", "bf16-mixed"), default="32-true")
    parser.add_argument("--stage", choices=("both", "spectrum", "diffusion"), default="both")
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--timed_batches", type=int, default=50)
    parser.add_argument("--validation_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, help="Override recipe size for diagnostics")
    parser.add_argument("--overhead_fraction", type=float, default=.15)
    parser.add_argument("--output", default="runs/benchmark.json")
    args = parser.parse_args(argv)
    if min(args.warmup_batches, args.timed_batches, args.validation_batches) < 1 or args.overhead_fraction < 0:
        parser.error("Batch counts must be positive and overhead nonnegative.")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch_size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable: run inside a GPU allocation with a compatible PyTorch wheel.")
    if args.precision == "bf16-mixed" and (args.device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("This benchmark's BF16 mode requires a supported CUDA GPU.")
    torch.manual_seed(78438379)
    data = load_native_npz(args.data)
    manifest = validate_split_manifest(json.loads(Path(args.splits).read_text()), data["molecule_ids"], args.data)
    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
    def move(value):
        return {key: move(item) for key, item in value.items()} if isinstance(value, dict) else value.to(args.device)
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16-mixed" else nullcontext()
    def benchmark(stage):
        config = json.loads(Path(f"configs/{stage}.json").read_text())
        batch_size = args.batch_size or config["batch_size"]
        cls = SpectrumStructureModel if stage == "spectrum" else EquivariantDiffusion3D_v4
        spec = json.loads(Path(config.get("encoder_spec", "configs/encoder.json")).read_text())
        parameters = inspect.signature(cls.__init__).parameters
        kwargs = {key: value for key, value in config.items() if key in parameters and key != "encoder_spec"}
        kwargs["encoder_spec"] = spec
        if stage == "spectrum":
            kwargs.update(vocabulary=data["vocabulary"], max_length=data["selfies_tokens"].shape[1])
        else:
            # Most production epochs occur after unfreezing. Measure that phase.
            kwargs["unfreeze_at_epoch"] = 0
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = cls(**kwargs).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-5))
        accumulation = config.get("accumulate_grad_batches", 1)
        datasets = {split: NativeNPZDataset(data, manifest["splits"][split]) for split in ("train", "val")}
        loaders = {split: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                     collate_fn=collate_native) for split, dataset in datasets.items()}
        def loss(batch, training):
            if stage == "diffusion":
                return model.loss(batch, self_condition=training)[0]
            tokens = batch["selfies_tokens"]
            logits = model(batch["spectra"], tokens[:, :-1])
            labels = tokens[:, 1:]
            values = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=0,
                reduction="none", label_smoothing=model.hparams.label_smoothing if training else 0.)
            return (values.sum(1) / labels.ne(0).sum(1).clamp_min(1)).mean()
        shadow = {}
        def train_batches(number):
            iterator = iter(loaders["train"])
            optimizer.zero_grad(set_to_none=True)
            for index in range(number):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loaders["train"])
                    batch = next(iterator)
                batch = move(batch)
                with autocast():
                    value = loss(batch, True) / accumulation
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite benchmark loss.")
                value.backward()
                if (index + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if model.hparams.ema_decay > 0:
                        with torch.no_grad():
                            for name, parameter in model.named_parameters():
                                if name not in shadow:
                                    shadow[name] = parameter.detach().clone()
                                else:
                                    shadow[name].lerp_(parameter.detach(), 1-model.hparams.ema_decay)
        model.train()
        train_batches(max(args.warmup_batches, accumulation))
        sync()
        started = time.perf_counter()
        train_batches(args.timed_batches)
        sync()
        train_seconds = (time.perf_counter() - started) / args.timed_batches
        model.eval()
        sync()
        started = time.perf_counter()
        measured = min(args.validation_batches, len(loaders["val"]))
        with torch.no_grad():
            for index, batch in enumerate(loaders["val"]):
                if index >= measured:
                    break
                with autocast():
                    loss(move(batch), False)
        sync()
        val_seconds = (time.perf_counter() - started) / measured
        batches = math.ceil(len(datasets["train"]) / batch_size)
        val_batches = math.ceil(len(datasets["val"]) / batch_size)
        epoch_seconds = batches * train_seconds + val_batches * val_seconds
        hours = config["epochs"] * epoch_seconds / 3600
        result = {"stage": stage, "parameters": sum(p.numel() for p in model.parameters()),
            "batch_size": batch_size, "gradient_accumulation": accumulation,
            "training_microbatches_per_epoch": batches, "validation_batches_per_epoch": val_batches,
            "seconds_per_training_microbatch": train_seconds, "seconds_per_validation_batch": val_seconds,
            "estimated_minutes_per_epoch": epoch_seconds / 60, "configured_max_epochs": config["epochs"],
            "estimated_max_run_hours": hours, "planning_hours_with_overhead": hours*(1+args.overhead_fraction),
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None}
        print(json.dumps(result, indent=2), flush=True)
        del model, optimizer, shadow
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return result
    stages = ["spectrum", "diffusion"] if args.stage == "both" else [args.stage]
    report = {"configuration": vars(args), "torch": str(torch.__version__),
        "hardware": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "stages": [benchmark(stage) for stage in stages],
        "limitations": ["Short steady-state timing extrapolation, not a measured full run.",
            "Manual loop approximates the trainer; logging and checkpoint I/O are covered only by the stated allowance.",
            "Uses fresh weights and an unfrozen diffusion encoder; includes transfer, clipping, accumulation, and EMA.",
            "Excludes queue time, beam-search evaluation, sample generation, tuning, and dataset preparation."]}
    report["total_planning_compute_hours"] = sum(item["planning_hours_with_overhead"] for item in report["stages"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Timing report: {output}", flush=True)


if __name__ == "__main__":
    main()
