"""
Diffusion model (clean version) — imports pretrained encoder from src_v2.

This is identical to diffusion/model.py except:
  1. Imports Transformer from src_v2.models (clean version)
  2. Reuses diffusion/ submodules (denoiser, physics, utils, layers)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from diffusion.denoiser import EquivariantDenoiser
from diffusion.physics import PhysicsLoss
from diffusion.utils import cosine_beta_schedule


def smiles_to_bond_adj(smiles: str, max_atoms: int) -> np.ndarray:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((max_atoms, max_atoms), dtype=np.int64)
    mol = Chem.AddHs(mol)
    adj = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    bond_map = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            bt = bond_map.get(bond.GetBondType(), 1)
            adj[i, j] = bt
            adj[j, i] = bt
    return adj


def batch_smiles_to_bond_adj(smiles_list, max_atoms, device):
    adjs = [smiles_to_bond_adj(s, max_atoms) for s in smiles_list]
    return torch.as_tensor(np.stack(adjs), dtype=torch.long, device=device)


def selfies_to_smiles(selfies_str: str) -> str:
    try:
        import selfies as sf
        return sf.decoder(selfies_str)
    except Exception:
        return ""


class EquivariantDiffusion3D_v4(pl.LightningModule):
    """E(3)-equivariant diffusion model with pretrained spectrum encoder.

    Architecture
    ------------
    Pretrained encoder (frozen → unfrozen at epoch N):
        IR peaklist → PeakMLPEncoderV2 → Transformer encoder → condition vector

    Denoiser (trainable):
        (noisy_coords, atom_types, t, condition) → EGNN → predicted noise

    The model learns to denoise 3D coordinates conditioned on the IR spectrum.
    """

    def __init__(self, pretrained_ckpt_path, model_type='Transformer',
                 lr=1e-4, encoder_lr=1e-5, unfreeze_at_epoch=5,
                 n_timesteps=1000, hidden_dim=256, n_layers=6, k_neighbors=16,
                 physics_weight=0.2, bond_weight=0.1, clash_weight=0.1,
                 angle_weight=0.05, ring_weight=0.05, torsion_weight=0.0,
                 self_conditioning=True, use_cosine_schedule=True,
                 adaptive_cutoff=False, cutoff_radius=6.0,
                 use_attention_pooling=True, bond_stats_json=None,
                 use_bond_graph=False, bond_emb_dim=16):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_is_unfrozen = False

        # Load pretrained encoder from clean src_v2
        from src_v2.models import Transformer
        self.pretrained = Transformer.load_from_checkpoint(
            pretrained_ckpt_path, strict=False,
        )
        self.pretrained.eval()
        self.cond_dim = getattr(self.pretrained.hparams, 'hidden_dim', 768)
        self.pretrained_encoder_type = str(
            getattr(self.pretrained.hparams, 'encoder_type', 'cnn')
        )
        print(f"[Diffusion] cond_dim={self.cond_dim}, encoder={self.pretrained_encoder_type}")

        self._set_encoder_trainable(False)

        # Attention pooling: encoder tokens → single condition vector
        self.use_attention_pooling = use_attention_pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim // 2),
            nn.SiLU(),
            nn.Linear(self.cond_dim // 2, 1),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.SiLU(),
            nn.LayerNorm(self.cond_dim),
        )

        # EGNN denoiser
        self.denoiser = EquivariantDenoiser(
            cond_dim=self.cond_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            k_neighbors=k_neighbors,
            self_conditioning=self_conditioning,
            adaptive_cutoff=adaptive_cutoff,
            cutoff_radius=cutoff_radius,
            use_bond_graph=use_bond_graph,
            bond_emb_dim=bond_emb_dim,
        )

        # Physics loss for guided sampling
        self.physics_loss = PhysicsLoss(
            bond_weight=bond_weight, clash_weight=clash_weight,
            angle_weight=angle_weight, ring_weight=ring_weight,
            torsion_weight=torsion_weight, bond_stats_path=bond_stats_json,
        )
        self.physics_weight = physics_weight

        # Noise schedule
        self.n_timesteps = n_timesteps
        betas = (cosine_beta_schedule(n_timesteps) if use_cosine_schedule
                 else torch.linspace(1e-4, 0.02, n_timesteps))
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=1e-5)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        one_minus_ac = (1.0 - alphas_cumprod).clamp(min=1e-5)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / one_minus_ac
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / one_minus_ac)
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / one_minus_ac)

    # ---- Encoder management ----

    def _encoder_modules(self):
        pre = self.pretrained
        modules = []
        for attr in ('feature_extractor', 'input_projection', 'positional_encoding'):
            if hasattr(pre, attr):
                modules.append(getattr(pre, attr))
        if hasattr(pre, 'transformer') and hasattr(pre.transformer, 'encoder'):
            modules.append(pre.transformer.encoder)
        return modules

    def _set_encoder_trainable(self, trainable: bool):
        for p in self.pretrained.parameters():
            p.requires_grad = False
        for m in self._encoder_modules():
            m.train(trainable)
            for p in m.parameters():
                p.requires_grad = trainable

    def on_train_epoch_start(self):
        if not self.encoder_is_unfrozen and self.current_epoch >= self.hparams.unfreeze_at_epoch:
            self.encoder_is_unfrozen = True
            self._set_encoder_trainable(True)
            print(f"[Unfreeze] encoder at epoch={self.current_epoch}")

    # ---- Encoder forward ----

    def _run_pretrained_encoder(self, spectra):
        """Run pretrained encoder → [B, L, D] token features."""
        pre = self.pretrained
        if isinstance(spectra, dict):
            packed = spectra.get('packed', spectra.get('peaks'))
            mask = spectra.get('mask')
            return pre.feature_extractor(packed)
        if spectra.dim() == 2:
            spectra = spectra.unsqueeze(1)
        feats = pre.feature_extractor(spectra).transpose(1, 2)
        if hasattr(pre, 'input_projection'):
            feats = pre.input_projection(feats)
        if hasattr(pre, 'transformer') and hasattr(pre.transformer, 'encoder'):
            if hasattr(pre, 'positional_encoding'):
                feats = pre.positional_encoding(feats)
            feats = pre.transformer.encoder(feats)
        return feats

    def extract_condition(self, spectra):
        if self.encoder_is_unfrozen:
            feats = self._run_pretrained_encoder(spectra)
        else:
            with torch.no_grad():
                feats = self._run_pretrained_encoder(spectra)
        if self.use_attention_pooling:
            scores = self.attn_pool(feats).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            cond = (feats * weights).sum(dim=1)
        else:
            cond = feats.mean(dim=1)
        return self.cond_proj(cond)

    # ---- SELFIES prediction for bond graph ----

    @torch.no_grad()
    def predict_selfies_greedy(self, spectra) -> list[str]:
        pre = self.pretrained
        pre.eval()
        B = self._spectra_batch_size(spectra)
        max_len = pre.max_word_len
        device = self._spectra_device(spectra)
        inp = torch.zeros((B, max_len), dtype=torch.long, device=device)
        idx_to_symbol = {v: k for k, v in pre.symbol_to_idx.items()}
        for i in range(max_len - 1):
            logits, _ = pre(spectra, inp)
            next_token = logits[:, i].argmax(dim=-1)
            if i + 1 < max_len:
                inp[:, i + 1] = next_token
        results = []
        for b in range(B):
            tokens = []
            for j in range(1, max_len):
                tok_id = inp[b, j].item()
                sym = idx_to_symbol.get(tok_id, '')
                if sym in ('[nop]', '[pad]', ''):
                    break
                tokens.append(sym)
            results.append(''.join(tokens))
        return results

    def get_bond_adj_from_smiles_batch(self, smiles_list, max_atoms):
        return batch_smiles_to_bond_adj(smiles_list, max_atoms, self.device)

    # ---- Diffusion operations ----

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def predict_x0_from_noise(self, x_t, t, noise_pred):
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t][:, None, None].float()
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t][:, None, None].float()
        return (sqrt_recip * x_t.float() - sqrt_recipm1 * noise_pred.float()).to(x_t.dtype)

    # ---- Sampling ----

    @torch.no_grad()
    def p_sample(self, x_t, t, condition, atom_types, mask, self_cond=None, bond_adj=None):
        noise_pred = self.denoiser(
            x_t, atom_types, t, condition, mask, self_cond=self_cond, bond_adj=bond_adj,
        ).clamp(-5, 5)
        x0_pred = self.predict_x0_from_noise(x_t, t, noise_pred).clamp(-30, 30)
        coef1 = self.posterior_mean_coef1[t][:, None, None].float()
        coef2 = self.posterior_mean_coef2[t][:, None, None].float()
        model_mean = coef1 * x0_pred.float() + coef2 * x_t.float()
        posterior_var = self.posterior_variance[t][:, None, None].float()
        noise = torch.randn_like(x_t).float()
        nonzero_mask = (t > 0).float()[:, None, None]
        x_prev = (model_mean + nonzero_mask * torch.sqrt(posterior_var) * noise).to(x_t.dtype).clamp(-30, 30)
        if mask is not None:
            x_prev = x_prev * mask.unsqueeze(-1).float()
        return x_prev, x0_pred

    @staticmethod
    def _spectra_batch_size(spectra):
        if isinstance(spectra, dict):
            return next(iter(spectra.values())).shape[0]
        return spectra.shape[0]

    @staticmethod
    def _spectra_device(spectra):
        if isinstance(spectra, dict):
            return next(iter(spectra.values())).device
        return spectra.device

    @torch.no_grad()
    def ddim_sample_step(self, x_t, t, t_prev, condition, atom_types, mask,
                         self_cond=None, eta=0.0, bond_adj=None):
        noise_pred = self.denoiser(
            x_t, atom_types, t, condition, mask, self_cond=self_cond, bond_adj=bond_adj,
        ).clamp(-5, 5)
        x0_pred = self.predict_x0_from_noise(x_t, t, noise_pred).clamp(-30, 30)
        alpha_t = self.alphas_cumprod[t][:, None, None].float()
        alpha_prev = (self.alphas_cumprod[t_prev][:, None, None].float()
                      if t_prev is not None else torch.ones_like(alpha_t))
        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t).clamp(min=1e-8) *
            (1 - alpha_t / alpha_prev.clamp(min=1e-8))
        ).clamp(max=1.0)
        dir_xt = torch.sqrt((1 - alpha_prev - sigma ** 2).clamp(min=0)) * noise_pred.float()
        x_prev = torch.sqrt(alpha_prev) * x0_pred.float() + dir_xt
        if eta > 0:
            noise = torch.randn_like(x_t).float()
            nonzero_mask = (t > 0).float()[:, None, None]
            x_prev = x_prev + nonzero_mask * sigma * noise
        x_prev = x_prev.to(x_t.dtype).clamp(-30, 30)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            x_prev = x_prev * mask_f
            n_atoms = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            com = (x_prev * mask_f).sum(dim=1, keepdim=True) / n_atoms
            x_prev = (x_prev - com) * mask_f
        return x_prev, x0_pred

    @torch.no_grad()
    def sample(self, spectra, atom_types, num_atoms, n_steps=None,
               sampler='ddim', eta=0.0, physics_guidance=0.0,
               guidance_start_frac=0.3, smiles_list=None):
        """Generate 3D conformers from IR spectrum."""
        self.eval()
        B = self._spectra_batch_size(spectra)
        N = atom_types.shape[1]
        device = self._spectra_device(spectra)

        condition = self.extract_condition(spectra)
        mask = torch.arange(N, device=device)[None, :] < num_atoms[:, None]

        bond_adj = None
        if self.hparams.use_bond_graph:
            if smiles_list is None:
                selfies_list = self.predict_selfies_greedy(spectra)
                smiles_list = [selfies_to_smiles(s) for s in selfies_list]
            bond_adj = self.get_bond_adj_from_smiles_batch(smiles_list, N)

        x = torch.randn((B, N, 3), device=device) * mask.unsqueeze(-1).float()
        if n_steps is None:
            n_steps = self.n_timesteps
        step_size = max(1, self.n_timesteps // n_steps)
        timesteps = list(range(self.n_timesteps - 1, -1, -step_size))
        guidance_start_t = int(self.n_timesteps * guidance_start_frac)

        self_cond = None
        if sampler == 'ddim':
            for i, t_val in enumerate(timesteps):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else None
                t_prev = (torch.full((B,), t_prev_val, device=device, dtype=torch.long)
                          if t_prev_val is not None else None)
                x, self_cond = self.ddim_sample_step(
                    x, t, t_prev, condition, atom_types, mask,
                    self_cond=self_cond, eta=eta, bond_adj=bond_adj,
                )
                if physics_guidance > 0 and t_val < guidance_start_t:
                    x = self._apply_physics_guidance(x, atom_types, mask, physics_guidance)
        else:
            for t_val in timesteps:
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                x, self_cond = self.p_sample(
                    x, t, condition, atom_types, mask, self_cond=self_cond, bond_adj=bond_adj,
                )

        mask_f = mask.unsqueeze(-1).float()
        n_atoms_f = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        com = (x * mask_f).sum(dim=1, keepdim=True) / n_atoms_f
        x = (x - com) * mask_f
        return x, mask

    def _apply_physics_guidance(self, x, atom_types, mask, scale):
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            ploss = self.physics_loss(x_g, atom_types, mask)
            grad = torch.autograd.grad(ploss, x_g)[0]
        return (x - scale * grad) * mask.unsqueeze(-1).float()

    # ---- Training / validation ----

    def training_step(self, batch, batch_idx):
        if len(batch) == 6:
            spectra, coords, atom_types, num_atoms, mask, smiles = batch
        else:
            spectra, coords, atom_types, num_atoms, mask = batch
            smiles = None

        B = self._spectra_batch_size(spectra)
        condition = self.extract_condition(spectra)
        t = torch.randint(0, self.n_timesteps, (B,), device=self.device).long()

        mask_f = mask.unsqueeze(-1).float()
        n_atoms = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        coords_centered = (coords - (coords * mask_f).sum(dim=1, keepdim=True) / n_atoms) * mask_f

        noisy_coords, noise = self.forward_diffusion(coords_centered, t)
        noisy_coords = noisy_coords.clamp(-50, 50)

        bond_adj = None
        if self.hparams.use_bond_graph and smiles is not None:
            bond_adj = self.get_bond_adj_from_smiles_batch(list(smiles), atom_types.shape[1])

        self_cond = None
        if self.denoiser.self_conditioning and torch.rand(1, device=self.device) < 0.5:
            with torch.no_grad():
                tmp = self.denoiser(noisy_coords, atom_types, t, condition, mask, bond_adj=bond_adj)
                with torch.amp.autocast('cuda', enabled=False):
                    self_cond = self.predict_x0_from_noise(
                        noisy_coords.float(), t, tmp.float()
                    ).clamp(-30, 30).detach()

        noise_pred = self.denoiser(
            noisy_coords, atom_types, t, condition, mask,
            self_cond=self_cond, bond_adj=bond_adj,
        )

        diff = (noise_pred - noise) * mask_f
        mse_loss = (diff ** 2).sum() / (mask.sum() * 3 + 1e-8)

        if torch.isnan(mse_loss) or torch.isinf(mse_loss):
            self.log('train_loss', torch.tensor(0.0, device=self.device), prog_bar=True)
            return None

        if self.physics_weight > 0 and batch_idx % 5 == 0:
            with torch.amp.autocast('cuda', enabled=False):
                x0_pred = self.predict_x0_from_noise(
                    noisy_coords.float(), t, noise_pred.float()
                ).clamp(-30, 30)
            phys_loss = self.physics_loss(x0_pred, atom_types, mask)
            if torch.isnan(phys_loss) or torch.isinf(phys_loss):
                phys_loss = torch.tensor(0.0, device=self.device)
            total_loss = mse_loss + self.physics_weight * phys_loss
            self.log('physics_loss', phys_loss, prog_bar=False)
        else:
            total_loss = mse_loss

        self.log('train_loss', mse_loss, prog_bar=True)
        return total_loss.clamp(max=100.0)

    def validation_step(self, batch, batch_idx):
        if len(batch) == 6:
            spectra, coords, atom_types, num_atoms, mask, smiles = batch
        else:
            spectra, coords, atom_types, num_atoms, mask = batch
            smiles = None

        B = self._spectra_batch_size(spectra)
        condition = self.extract_condition(spectra)
        t = torch.randint(0, self.n_timesteps, (B,), device=self.device).long()

        mask_f = mask.unsqueeze(-1).float()
        n_atoms = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        coords_centered = (coords - (coords * mask_f).sum(dim=1, keepdim=True) / n_atoms) * mask_f

        noisy_coords, noise = self.forward_diffusion(coords_centered, t)

        bond_adj = None
        if self.hparams.use_bond_graph and smiles is not None:
            bond_adj = self.get_bond_adj_from_smiles_batch(list(smiles), atom_types.shape[1])

        noise_pred = self.denoiser(
            noisy_coords, atom_types, t, condition, mask, bond_adj=bond_adj,
        )
        loss = ((noise_pred - noise) ** 2 * mask_f).sum() / (mask.sum() * 3 + 1e-8)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=self.device)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        for pg in optimizer.param_groups:
            for p in pg['params']:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad.zero_()

    def configure_optimizers(self):
        denoiser_params = list(self.denoiser.parameters())
        cond_params = list(self.attn_pool.parameters()) + list(self.cond_proj.parameters())
        encoder_params = []
        for m in self._encoder_modules():
            encoder_params.extend(list(m.parameters()))
        seen = set()
        uniq_enc = [p for p in encoder_params if not (id(p) in seen or seen.add(id(p)))]

        param_groups = [
            {'params': denoiser_params + cond_params, 'lr': self.hparams.lr, 'weight_decay': 1e-4},
        ]
        if uniq_enc:
            param_groups.append({'params': uniq_enc, 'lr': self.hparams.encoder_lr, 'weight_decay': 1e-5})

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.99))

        def base_lr_lambda(step):
            warmup = 1000
            if step < warmup:
                return float(step) / float(max(1, warmup))
            total = max(1, int(self.trainer.estimated_stepping_batches))
            progress = (step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        if len(param_groups) == 1:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=base_lr_lambda)
        else:
            def encoder_lr_lambda(step):
                if self.current_epoch < self.hparams.unfreeze_at_epoch:
                    return 0.0
                return base_lr_lambda(step)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[base_lr_lambda, encoder_lr_lambda],
            )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
