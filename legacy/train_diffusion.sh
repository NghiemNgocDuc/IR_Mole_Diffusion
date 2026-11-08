#!/bin/bash
#SBATCH --job-name=diff
#SBATCH --output=./logs/train_diffusion.out
#SBATCH --error=./logs/train_diffusion.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load conda/latest
conda activate ir

DATA_NPZ="./data/SQuIRL/squirl_ir_3d.npz"
VOCAB="./data/symbol_to_idx.json"
BOND_STATS="./data/SQuIRL/bond_stats.json"
SAVE_DIR="./ckpt/diffusion_squirl"
ENCODER_CKPT="./models_formula/squirl/best-epoch=033-val_loss=0.102336.ckpt"

python train_diffusion_v2.py \
    --ckpt "$ENCODER_CKPT" \
    --data "$DATA_NPZ" \
    --symbol_to_idx_path "$VOCAB" \
    --ir_rep peaklist \
    --ir_max_peaks 81 \
    --ir_peak_int_scale 1000.0 \
    --hidden_dim 512 \
    --n_layers 8 \
    --k_neighbors 16 \
    --batch_size 32 \
    --accumulate_grad_batches 2 \
    --lr 1e-4 \
    --encoder_lr 1e-5 \
    --unfreeze_at_epoch 10 \
    --epochs 200 \
    --patience 30 \
    --min_delta 0.0005 \
    --precision bf16-mixed \
    --physics_weight 0.2 \
    --bond_stats_json "$BOND_STATS" \
    --self_conditioning \
    --use_cosine_schedule \
    --use_attention_pooling \
    --use_bond_graph \
    --bond_emb_dim 16 \
    --save_dir "$SAVE_DIR"
