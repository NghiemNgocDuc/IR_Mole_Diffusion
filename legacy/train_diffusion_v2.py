"""Training script for E(3)-equivariant diffusion model (v2 clean imports).

Usage:
    # New training (from pretrained encoder):
    python train_diffusion_v2.py --ckpt <encoder.ckpt> --data <data.npz> ...

    # Fine-tune (load diffusion weights, reset optimizer with new LR):
    python train_diffusion_v2.py --finetune_from <diffusion.ckpt> --data <data.npz> --lr 3e-5 ...

    # Resume (restore full state including optimizer):
    python train_diffusion_v2.py --resume_from_checkpoint <diffusion.ckpt> --data <data.npz> ...
"""

import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset_utils_qmugs_v2 import QMugsDatasetV2, collate_variable_length
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from diffusion_v2 import EquivariantDiffusion3D_v4


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Pretrained encoder checkpoint (for new training)')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Diffusion checkpoint to fine-tune from (loads weights, resets optimizer)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Diffusion checkpoint to resume from (restores full state)')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--k_neighbors', type=int, default=16)
    parser.add_argument('--unfreeze_at_epoch', type=int, default=5)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--save_dir', type=str, default='./ckpt/diffusion_v4')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=0.0005)

    parser.add_argument('--ir_rep', type=str, default='gaussian',
                        choices=['gaussian', 'peaklist', 'interp'])
    parser.add_argument('--ir_max_peaks', type=int, default=128)
    parser.add_argument('--ir_peak_freq_scale', type=float, default=4000.0)
    parser.add_argument('--ir_peak_int_scale', type=float, default=None)
    parser.add_argument('--symbol_to_idx_path', type=str,
                        default='./data/qmugs/symbol_to_idx_qmugs.json')

    parser.add_argument('--physics_weight', type=float, default=0.2)
    parser.add_argument('--bond_weight', type=float, default=0.1)
    parser.add_argument('--clash_weight', type=float, default=0.1)
    parser.add_argument('--angle_weight', type=float, default=0.05)
    parser.add_argument('--ring_weight', type=float, default=0.05)
    parser.add_argument('--torsion_weight', type=float, default=0.0)
    parser.add_argument('--bond_stats_json', type=str, default=None,
                        help='Path to dataset-derived bond stats JSON')

    parser.add_argument('--self_conditioning', action='store_true')
    parser.add_argument('--use_cosine_schedule', action='store_true')
    parser.add_argument('--adaptive_cutoff', action='store_true')
    parser.add_argument('--cutoff_radius', type=float, default=6.0)
    parser.add_argument('--use_attention_pooling', action='store_true')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--use_bond_graph', action='store_true',
                        help='Enable bond graph conditioning from GT SMILES '
                             '(training) / predicted SELFIES (inference)')
    parser.add_argument('--bond_emb_dim', type=int, default=16,
                        help='Embedding dimension for bond type features')

    args = parser.parse_args()
    pl.seed_everything(78438379)

    print(f"Loading dataset (ir_rep={args.ir_rep})...")
    dataset = QMugsDatasetV2(
        args.data, mode='diffusion',
        ir_rep=args.ir_rep,
        ir_max_peaks=args.ir_max_peaks,
        ir_peak_freq_scale=args.ir_peak_freq_scale,
        ir_peak_int_scale=args.ir_peak_int_scale,
        symbol_to_idx_path=args.symbol_to_idx_path,
    )
    if args.use_bond_graph:
        dataset.emit_smiles = True
        print("[Bond graph] Dataset will emit SMILES for bond conditioning")
    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(78438379)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_variable_length, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_variable_length, pin_memory=True, persistent_workers=True
    )

    print("Creating model...")
    if args.finetune_from:
        print(f"Fine-tuning from: {args.finetune_from}")
        print(f"  New LR: {args.lr}, encoder_lr: {args.encoder_lr}")
        model = EquivariantDiffusion3D_v4.load_from_checkpoint(
            args.finetune_from, map_location='cpu', strict=False
        )
        model.hparams.lr = args.lr
        model.hparams.encoder_lr = args.encoder_lr
        model.hparams.unfreeze_at_epoch = args.unfreeze_at_epoch
        if args.bond_stats_json:
            model.physics_loss = __import__('diffusion.physics', fromlist=['PhysicsLoss']).PhysicsLoss(
                bond_weight=args.bond_weight, clash_weight=args.clash_weight,
                angle_weight=args.angle_weight, ring_weight=args.ring_weight,
                torsion_weight=args.torsion_weight, bond_stats_path=args.bond_stats_json,
            )
    elif args.resume_from_checkpoint:
        print(f"Resuming from: {args.resume_from_checkpoint}")
        model = EquivariantDiffusion3D_v4.load_from_checkpoint(
            args.resume_from_checkpoint, map_location='cpu', strict=False
        )
    else:
        if args.ckpt is None:
            raise ValueError("--ckpt required when not resuming or fine-tuning.")
        model = EquivariantDiffusion3D_v4(
            pretrained_ckpt_path=args.ckpt,
            lr=args.lr, encoder_lr=args.encoder_lr,
            unfreeze_at_epoch=args.unfreeze_at_epoch,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            k_neighbors=args.k_neighbors,
            physics_weight=args.physics_weight,
            bond_weight=args.bond_weight, clash_weight=args.clash_weight,
            angle_weight=args.angle_weight, ring_weight=args.ring_weight,
            torsion_weight=args.torsion_weight,
            self_conditioning=args.self_conditioning,
            use_cosine_schedule=args.use_cosine_schedule,
            adaptive_cutoff=args.adaptive_cutoff, cutoff_radius=args.cutoff_radius,
            use_attention_pooling=args.use_attention_pooling,
            bond_stats_json=args.bond_stats_json,
            use_bond_graph=args.use_bond_graph,
            bond_emb_dim=args.bond_emb_dim,
        )

    save_dir = f"{args.save_dir}/h{args.hidden_dim}_L{args.n_layers}_k{args.k_neighbors}"
    callbacks = [
        ModelCheckpoint(dirpath=save_dir, filename='v4-{epoch:03d}-{val_loss:.4f}',
                        save_top_k=3, monitor='val_loss', mode='min', save_last=True),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', min_delta=args.min_delta)
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        precision=args.precision,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        val_check_interval=0.5,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    print("Starting training...")
    fit_kwargs = {}
    if args.resume_from_checkpoint:
        fit_kwargs['ckpt_path'] = args.resume_from_checkpoint
    trainer.fit(model, train_loader, val_loader, **fit_kwargs)
    print("Training complete!")


if __name__ == '__main__':
    main()
