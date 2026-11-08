import torch
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.distance import cdist

from dataset_utils_qmugs_v2 import QMugsDatasetV2
from diffusion_v2 import EquivariantDiffusion3D_v4 as Diffusion3D


def _prep_single_spectrum(spectrum, device):
    if isinstance(spectrum, dict):
        out = {k: v.unsqueeze(0).to(device) for k, v in spectrum.items()}
        if 'packed' in out and 'peaks' not in out:
            out['peaks'] = out.pop('packed')
        return out
    return spectrum.unsqueeze(0).to(device)


ATOMIC_MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
                 15: 30.974, 16: 32.06, 17: 35.45, 35: 79.904, 53: 126.90}


def _kabsch_align(coords_pred, coords_true, weights=None):
    if weights is not None:
        w = weights / weights.sum()
        com_pred = (w[:, None] * coords_pred).sum(axis=0)
        com_true = (w[:, None] * coords_true).sum(axis=0)
        coords_pred = coords_pred - com_pred
        coords_true = coords_true - com_true
        H = coords_pred.T @ (w[:, None] * coords_true)
    else:
        coords_pred = coords_pred - coords_pred.mean(axis=0)
        coords_true = coords_true - coords_true.mean(axis=0)
        H = coords_pred.T @ coords_true

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return coords_pred @ R, coords_true


def compute_rmsd(coords_pred, coords_true, mask):
    coords_pred = coords_pred[mask]
    coords_true = coords_true[mask]
    if len(coords_pred) == 0:
        return float('nan')

    aligned, true = _kabsch_align(coords_pred.copy(), coords_true.copy())
    return np.sqrt(((aligned - true) ** 2).sum(axis=1).mean())


def compute_mass_weighted_rmsd(coords_pred, coords_true, atom_types, mask):
    coords_pred = coords_pred[mask]
    coords_true = coords_true[mask]
    at = atom_types[mask]
    if len(coords_pred) == 0:
        return float('nan')

    masses = np.array([ATOMIC_MASSES.get(int(a), 12.0) for a in at])
    aligned, true = _kabsch_align(coords_pred.copy(), coords_true.copy(), weights=masses)
    w = masses / masses.sum()
    return np.sqrt((w * ((aligned - true) ** 2).sum(axis=1)).sum())


def compute_bond_distribution_error(coords, atom_types, mask):
    coords = coords[mask]
    atom_types = atom_types[mask]

    if len(coords) < 2:
        return 0.0

    dists = cdist(coords, coords)
    np.fill_diagonal(dists, 1e6)

    typical_bonds = {
        (1, 6): 1.09, (1, 7): 1.01, (1, 8): 0.96,
        (6, 6): 1.54, (6, 7): 1.47, (6, 8): 1.43,
    }

    errors = []
    for i in range(len(coords)):
        if atom_types[i] == 1:
            j = np.argmin(dists[i])
            pair = tuple(sorted([1, int(atom_types[j])]))
            if pair in typical_bonds:
                errors.append(abs(dists[i, j] - typical_bonds[pair]))
        else:
            neighbors = np.where(dists[i] < 1.8)[0]
            for j in neighbors:
                if j <= i or atom_types[j] == 1:
                    continue
                pair = tuple(sorted([int(atom_types[i]), int(atom_types[j])]))
                if pair in typical_bonds:
                    errors.append(abs(dists[i, j] - typical_bonds[pair]))

    return np.mean(errors) if errors else 0.0


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"Found hyperparameters in checkpoint:")
        print(f"  model_type: {hparams.get('model_type', 'unknown')}")
        print(f"  hidden_dim: {hparams.get('hidden_dim', 'unknown')}")
        print(f"  n_layers: {hparams.get('n_layers', 'unknown')}")
        print(f"  k_neighbors: {hparams.get('k_neighbors', 'unknown')}")

    try:
        model = Diffusion3D.load_from_checkpoint(checkpoint_path, strict=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Error loading model: {e}")
        print("Attempting to load with relaxed constraints...")

        if 'hyper_parameters' in checkpoint:
            model = Diffusion3D(**hparams)
        else:
            raise RuntimeError(
                "Checkpoint is missing 'hyper_parameters'. Please provide a checkpoint "
                "saved by PyTorch Lightning (which stores hparams automatically), or "
                "construct Diffusion3D manually with the appropriate pretrained_ckpt_path "
                "and model_type before loading the state_dict."
            )

        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded with strict=False")

    model.eval()
    model.to(device)

    return model


def _generate_one(model, spectrum, atom_types_batch, num_atoms_tensor, coords_true,
                   mask_np, atom_types_np, sample_kwargs):
    coords_pred, mask = model.sample(
        spectrum, atom_types_batch, num_atoms_tensor, **sample_kwargs
    )
    coords_pred = coords_pred[0].cpu().numpy()

    valid = coords_pred[mask_np]
    if len(valid) > 0:
        coords_pred[mask_np] -= valid.mean(axis=0, keepdims=True)

    rmsd = compute_rmsd(coords_pred, coords_true, mask_np)
    bond_error = compute_bond_distribution_error(coords_pred, atom_types_np, mask_np)
    coord_mag = np.linalg.norm(coords_pred[mask_np], axis=1).mean()
    return coords_pred, rmsd, bond_error, coord_mag


def evaluate_generation(model, dataset, n_samples=100, n_steps=100, device='cuda',
                        sampler='ddim', eta=0.0, physics_guidance=0.0,
                        guidance_start_frac=0.3, n_conformers=1,
                        oracle_smiles=False):
    model.eval()
    model.to(device)

    results = {
        'rmsds': [],
        'mw_rmsds': [],
        'bond_errors': [],
        'coord_magnitudes': [],
        'success_count': 0
    }
    if n_conformers > 1:
        results['rmsds_best'] = []

    label = f"sampler={sampler}, eta={eta}"
    if n_conformers > 1:
        label += f", K={n_conformers} (best-of-K)"
    if oracle_smiles:
        label += ", oracle_smiles=GT"
    print(f"Evaluating on {n_samples} samples ({label})...")

    for i in range(min(n_samples, len(dataset))):
        try:
            spectrum, coords_true, atom_types, num_atoms = dataset[i]

            spectrum = _prep_single_spectrum(spectrum, device)
            atom_types_batch = atom_types.unsqueeze(0).to(device)
            num_atoms_tensor = torch.tensor([num_atoms]).to(device)
            coords_true_np = coords_true.numpy()
            mask_np = None
            atom_types_np = atom_types.cpu().numpy()

            sample_kwargs = dict(
                n_steps=n_steps, sampler=sampler, eta=eta,
            )
            if physics_guidance > 0:
                sample_kwargs['physics_guidance'] = physics_guidance
                sample_kwargs['guidance_start_frac'] = guidance_start_frac
            if oracle_smiles and hasattr(dataset, 'smiles'):
                sample_kwargs['smiles_list'] = [str(dataset.smiles[i])]

            if n_conformers <= 1:
                coords_pred, mask = model.sample(
                    spectrum, atom_types_batch, num_atoms_tensor, **sample_kwargs
                )
                coords_pred = coords_pred[0].cpu().numpy()
                mask_np = mask[0].cpu().numpy()

                valid = coords_pred[mask_np]
                if len(valid) > 0:
                    coords_pred[mask_np] -= valid.mean(axis=0, keepdims=True)

                rmsd = compute_rmsd(coords_pred, coords_true_np, mask_np)
                mw_rmsd = compute_mass_weighted_rmsd(coords_pred, coords_true_np, atom_types_np, mask_np)
                bond_error = compute_bond_distribution_error(coords_pred, atom_types_np, mask_np)
                coord_mag = np.linalg.norm(coords_pred[mask_np], axis=1).mean()
            else:
                conformers = []
                for k in range(n_conformers):
                    coords_pred_k, mask_k = model.sample(
                        spectrum, atom_types_batch, num_atoms_tensor, **sample_kwargs
                    )
                    coords_pred_k = coords_pred_k[0].cpu().numpy()
                    mk = mask_k[0].cpu().numpy()
                    if mask_np is None:
                        mask_np = mk

                    valid = coords_pred_k[mask_np]
                    if len(valid) > 0:
                        coords_pred_k[mask_np] -= valid.mean(axis=0, keepdims=True)

                    r = compute_rmsd(coords_pred_k, coords_true_np, mask_np)
                    be = compute_bond_distribution_error(coords_pred_k, atom_types_np, mask_np)
                    cm = np.linalg.norm(coords_pred_k[mask_np], axis=1).mean()
                    conformers.append((r, be, cm, coords_pred_k))

                valid_confs = [(r, be, cm, c) for r, be, cm, c in conformers if not np.isnan(r)]
                if valid_confs:
                    all_rmsds = [r for r, _, _, _ in valid_confs]
                    results['rmsds_best'].append(min(all_rmsds))
                    rmsd = np.mean(all_rmsds)
                    best = min(valid_confs, key=lambda x: x[0])
                    _, bond_error, coord_mag, coords_pred = best
                else:
                    rmsd = float('nan')
                    bond_error = 0.0
                    coord_mag = 0.0

            if n_conformers > 1:
                if not np.isnan(rmsd) and mask_np is not None:
                    mw_rmsd = compute_mass_weighted_rmsd(
                        coords_pred, coords_true_np, atom_types_np, mask_np)
                else:
                    mw_rmsd = float('nan')

            if not np.isnan(rmsd):
                results['rmsds'].append(rmsd)
            if not np.isnan(mw_rmsd):
                results['mw_rmsds'].append(mw_rmsd)

            results['bond_errors'].append(bond_error)
            results['coord_magnitudes'].append(coord_mag)

            if not np.isnan(rmsd) and rmsd < 5.0 and coord_mag < 20.0:
                results['success_count'] += 1

            if (i + 1) % 10 == 0:
                if len(results['rmsds']) > 0:
                    avg_rmsd = np.mean(results['rmsds'])
                    extra = ""
                    if n_conformers > 1 and results['rmsds_best']:
                        extra = f", Best-of-{n_conformers}: {np.mean(results['rmsds_best']):.3f}A"
                    print(f"  {i+1}/{n_samples} - Avg RMSD: {avg_rmsd:.3f}A{extra}, Success: {results['success_count']}/{i+1}")
                else:
                    print(f"  {i+1}/{n_samples} - No valid samples yet")

        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    if len(results['rmsds']) == 0:
        print("No valid samples evaluated!")
        return results

    print(f"Total Samples: {len(results['rmsds'])}")
    print(f"Success Rate: {results['success_count'] / len(results['rmsds']) * 100:.1f}%")
    print(f"\nRMSD (A):")
    print(f"  Mean: {np.mean(results['rmsds']):.3f}")
    print(f"  Median: {np.median(results['rmsds']):.3f}")
    print(f"  Std: {np.std(results['rmsds']):.3f}")
    print(f"  Min: {np.min(results['rmsds']):.3f}")
    print(f"  Max: {np.max(results['rmsds']):.3f}")

    if results['mw_rmsds']:
        print(f"\nMass-Weighted RMSD (A):")
        print(f"  Mean: {np.mean(results['mw_rmsds']):.3f}")
        print(f"  Median: {np.median(results['mw_rmsds']):.3f}")

    if n_conformers > 1 and results.get('rmsds_best'):
        print(f"\nBest-of-{n_conformers} RMSD (A):")
        print(f"  Mean: {np.mean(results['rmsds_best']):.3f}")
        print(f"  Median: {np.median(results['rmsds_best']):.3f}")

    if results['bond_errors']:
        print(f"\nBond Length Error (A):")
        print(f"  Mean: {np.mean(results['bond_errors']):.3f}")
        print(f"  Median: {np.median(results['bond_errors']):.3f}")

    print(f"\nCoordinate Magnitude (A):")
    print(f"  Mean: {np.mean(results['coord_magnitudes']):.3f}")
    print(f"  Std: {np.std(results['coord_magnitudes']):.3f}")
    print("=" * 60)

    return results


def generate_single_molecule(model, dataset, sample_idx, n_steps=100, device='cuda',
                             sampler='ddim', eta=0.0):
    model.eval()
    model.to(device)

    spectrum, coords_true, atom_types, num_atoms = dataset[sample_idx]
    smiles = str(dataset.smiles[sample_idx]) if hasattr(dataset, 'smiles') else f"Sample_{sample_idx}"

    spectrum = _prep_single_spectrum(spectrum, device)
    atom_types_batch = atom_types.unsqueeze(0).to(device)
    num_atoms_tensor = torch.tensor([num_atoms]).to(device)

    coords_pred, mask = model.sample(
        spectrum, atom_types_batch, num_atoms_tensor, n_steps=n_steps,
        sampler=sampler, eta=eta,
    )

    coords_pred = coords_pred[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    valid = coords_pred[mask_np]
    if len(valid) > 0:
        coords_pred[mask_np] -= valid.mean(axis=0, keepdims=True)

    return {
        'coords_pred': coords_pred,
        'coords_true': coords_true.numpy(),
        'atom_types': atom_types.numpy(),
        'mask': mask_np,
        'smiles': smiles,
        'num_atoms': num_atoms
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Evaluation (v2 clean)")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to QMugs dataset')
    parser.add_argument('--mode', type=str, default='evaluate',
                        choices=['evaluate', 'generate'],
                        help='Mode')

    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of diffusion steps')

    parser.add_argument('--output_dir', type=str, default='./generated_mols',
                        help='Output directory')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index for generation')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')

    parser.add_argument('--ir_rep', type=str, default='gaussian',
                        choices=['gaussian', 'peaklist', 'interp'])
    parser.add_argument('--ir_max_peaks', type=int, default=128)
    parser.add_argument('--ir_peak_freq_scale', type=float, default=4000.0)
    parser.add_argument('--ir_peak_int_scale', type=float, default=None)
    parser.add_argument('--symbol_to_idx_path', type=str,
                        default='./data/qmugs/symbol_to_idx_qmugs.json')

    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddim', 'ddpm'],
                        help='Sampling method (ddim=deterministic, ddpm=stochastic)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM stochasticity (0=deterministic, 1=DDPM-like)')
    parser.add_argument('--physics_guidance', type=float, default=0.0,
                        help='Physics-guided sampling scale (0=off, try 0.01-0.1)')
    parser.add_argument('--guidance_start_frac', type=float, default=0.3,
                        help='Fraction of total timesteps where physics guidance starts')
    parser.add_argument('--n_conformers', type=int, default=1,
                        help='Generate K conformers per molecule, report best-of-K RMSD (requires eta>0 for diversity)')
    parser.add_argument('--oracle_smiles', action='store_true',
                        help='Use GT SMILES for bond-graph conditioning instead of predicted SELFIES')

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset (ir_rep={args.ir_rep})...")
    dataset = QMugsDatasetV2(
        args.data_path,
        mode='diffusion',
        ir_rep=args.ir_rep,
        ir_max_peaks=args.ir_max_peaks,
        ir_peak_freq_scale=args.ir_peak_freq_scale,
        ir_peak_int_scale=args.ir_peak_int_scale,
        symbol_to_idx_path=args.symbol_to_idx_path,
    )

    print(f"\nLoading model...")
    model = load_model_from_checkpoint(args.checkpoint, args.device)

    if hasattr(model, 'pretrained') and not hasattr(model.pretrained, 'symbol_to_idx'):
        import json
        vocab_path = getattr(dataset, 'symbol_to_idx_path', None)
        if vocab_path is None:
            vocab_path = args.symbol_to_idx_path
        with open(vocab_path) as f:
            model.pretrained.symbol_to_idx = json.load(f)

    if args.mode == 'evaluate':
        results = evaluate_generation(
            model, dataset,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            device=args.device,
            sampler=args.sampler,
            eta=args.eta,
            physics_guidance=args.physics_guidance,
            guidance_start_frac=args.guidance_start_frac,
            n_conformers=args.n_conformers,
            oracle_smiles=args.oracle_smiles,
        )

        if len(results['rmsds']) > 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / 'evaluation_results.npz'
            np.savez(
                out_path,
                rmsds=results['rmsds'],
                mw_rmsds=results['mw_rmsds'],
                bond_errors=results['bond_errors'],
                coord_magnitudes=results['coord_magnitudes']
            )
            print(f"\nResults saved to {out_path}")
        else:
            print("\nNo results to save")

    elif args.mode == 'generate':
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"\nGenerating molecule {args.sample_idx}...")

        result = generate_single_molecule(
            model, dataset, args.sample_idx, args.n_steps, args.device
        )

        print(f"SMILES: {result['smiles']}")
        print(f"Atoms: {result['num_atoms']}")

        rmsd = compute_rmsd(result['coords_pred'], result['coords_true'], result['mask'])
        print(f"RMSD: {rmsd:.3f} A")

        output_path = output_dir / f"mol_{args.sample_idx}.npz"
        np.savez(
            output_path,
            coords_pred=result['coords_pred'],
            coords_true=result['coords_true'],
            atom_types=result['atom_types'],
            mask=result['mask'],
            smiles=result['smiles']
        )
        print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
