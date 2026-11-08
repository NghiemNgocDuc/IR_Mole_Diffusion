"""Real-data CPU/GPU smoke: conversion -> both trainers -> reload -> sample.

Run from repository root. This is a diagnostic, not a useful trained model.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/smoke")
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists():
        parser.error("Choose a fresh output_dir to preserve previous diagnostic evidence.")
    root.mkdir(parents=True)
    spec = root / "encoder.json"
    spec.write_text(json.dumps({"hidden_dim": 32, "n_layers": 1, "n_heads": 4, "dropout": 0.0}))
    def run(*arguments):
        print("RUN", " ".join(map(str, arguments)), flush=True)
        subprocess.run([sys.executable, *map(str, arguments)], check=True)
    data, splits = root / "data.npz", root / "splits.json"
    run("prepare_squirl.py", "--trust_legacy", "--limit", 300, "--output", data, "--splits", splits)
    common = ["--data", data, "--splits", splits, "--accelerator", args.accelerator,
              "--batch_size", 4, "--epochs", 2, "--max_steps", 3,
              "--limit_val_batches", 2, "--checkpoint_every_steps", 2]
    run("train_spectrum.py", *common, "--encoder_spec", spec, "--decoder_layers", 1,
        "--dropout", 0., "--save_dir", root / "spectrum")
    encoder = root / "spectrum/checkpoints/last.ckpt"
    run("train_diffusion.py", *common, "--encoder_checkpoint", encoder,
        "--hidden_dim", 32, "--n_layers", 2, "--n_timesteps", 20,
        "--unfreeze_at_epoch", 0, "--warmup_steps", 0, "--physics_weight", .001,
        "--save_dir", root / "diffusion")
    checkpoint = root / "diffusion/checkpoints/last.ckpt"
    run("train_diffusion.py", *common, "--resume_from_checkpoint", checkpoint,
        "--max_steps", 6, "--save_dir", root / "diffusion")
    device = "cuda" if args.accelerator == "gpu" else "cpu"
    run("evaluate_diffusion.py", "--checkpoint", checkpoint, "--data", data, "--splits", splits,
        "--limit", 2, "--num_samples", 2, "--n_steps", 4, "--device", device,
        "--use_inference_graph", "--physics_guidance", .01, "--chemistry",
        "--output_dir", root / "evaluation")
    run("evaluate_spectrum.py", "--checkpoint", encoder, "--data", data, "--splits", splits,
        "--limit", 2, "--beam_size", 2, "--device", device, "--output", root / "spectrum_evaluation.json")
    run("predict_structure.py", "--spectrum_checkpoint", encoder, "--diffusion_checkpoint", checkpoint,
        "--data", data, "--row", 0, "--device", device, "--beam_size", 2,
        "--conformers", 1, "--n_steps", 4, "--no-refine", "--output_dir", root / "prediction")
    print(f"Smoke complete: {root}. These short-run weights are diagnostic only.", flush=True)


if __name__ == "__main__":
    main()
