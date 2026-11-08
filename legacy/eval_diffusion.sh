#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=./logs/eval_diffusion.out
#SBATCH --error=./logs/eval_diffusion.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load conda/latest
conda activate ir

DATA_NPZ="./data/SQuIRL/squirl_ir_3d.npz"
VOCAB="./data/symbol_to_idx.json"
OUT_BASE="./diffusion_eval"

BEST_CKPT="./ckpt/v4-epoch=156-val_loss=0.2450.ckpt"

COMMON="--checkpoint $BEST_CKPT --data_path $DATA_NPZ --symbol_to_idx_path $VOCAB \
    --ir_rep peaklist --ir_max_peaks 81 --ir_peak_int_scale 1000.0 \
    --mode evaluate --n_samples 500"

python inference_v2.py $COMMON --n_steps 200 --sampler ddim --output_dir "$OUT_BASE/ddim_steps200"
python inference_v2.py $COMMON --n_steps 100 --sampler ddim --output_dir "$OUT_BASE/ddim_steps100"
python inference_v2.py $COMMON --n_steps 100 --sampler ddim \
    --physics_guidance 0.05 --guidance_start_frac 0.3 \
    --output_dir "$OUT_BASE/ddim_steps100_guided"
