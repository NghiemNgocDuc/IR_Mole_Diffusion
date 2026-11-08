"""Portable spectrum/graph-conditioned diffusion on centered coordinates."""
import math

import pytorch_lightning as pl
import torch
from torch import nn

from diffusion.denoiser import EquivariantDenoiser
from diffusion.encoder import SpectrumTokenEncoder
from diffusion.physics import PhysicsLoss
from diffusion.utils import (build_diffusion_schedule, centered_randn_like,
                             masked_center, sampling_timesteps)

CHECKPOINT_VERSION = 2


def read_checkpoint(path):
    # Own the tensor storage: a resumed run may overwrite the source last.ckpt.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Legacy/incompatible checkpoint. Train the self-contained models "
                         "using README.md; legacy weights cannot be silently migrated.")
    return checkpoint


class PortableModule(pl.LightningModule):
    """Strict portable loading, provenance, and optimizer-step EMA."""
    def __init__(self):
        super().__init__()
        self.run_metadata, self._ema = {}, {}
        self._ema_step, self._validation_backup = -1, None

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location="cpu", strict=True,
                             checkpoint_policy="reject", **overrides):
        if not strict or checkpoint_policy != "reject":
            raise ValueError("Only strict versioned checkpoint loading is supported.")
        checkpoint = read_checkpoint(checkpoint_path)
        if checkpoint.get("model_class") != cls.__name__:
            raise ValueError(f"Expected {cls.__name__}, found {checkpoint.get('model_class')}.")
        model = cls(**{**checkpoint["hyper_parameters"], **overrides})
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.on_load_checkpoint(checkpoint)
        return model.to(map_location)

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(format_version=CHECKPOINT_VERSION, model_class=type(self).__name__,
                          run_metadata=self.run_metadata, ema_state_dict=self._ema)

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Cannot resume a legacy checkpoint with the corrected model.")
        self.run_metadata = checkpoint.get("run_metadata", {})
        self._ema = checkpoint.get("ema_state_dict", {})

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Accumulated microbatches must not update EMA more than once per optimizer step.
        if self.hparams.ema_decay <= 0 or self.global_step in (0, self._ema_step):
            return
        self._ema_step = self.global_step
        with torch.no_grad():
            for name, value in self.named_parameters():
                if name not in self._ema:
                    self._ema[name] = value.detach().clone()
                else:
                    self._ema[name] = self._ema[name].to(value.device)
                    self._ema[name].lerp_(value.detach(), 1 - self.hparams.ema_decay)

    def use_ema_weights(self):
        if not self._ema:
            return False
        with torch.no_grad():
            parameters = dict(self.named_parameters())
            if set(parameters) != set(self._ema):
                raise ValueError("EMA parameters do not match this architecture.")
            for name, value in self._ema.items():
                parameters[name].copy_(value.to(parameters[name]))
        return True

    def on_validation_epoch_start(self):
        if self._ema and not self.trainer.sanity_checking:
            self._validation_backup = {n: p.detach().clone() for n, p in self.named_parameters()}
            self.use_ema_weights()

    def on_validation_epoch_end(self):
        if self._validation_backup is not None:
            with torch.no_grad():
                for name, value in self.named_parameters():
                    value.copy_(self._validation_backup[name])
            self._validation_backup = None

    def on_before_optimizer_step(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Nonfinite gradient; stopped instead of silently zeroing it.")


class EquivariantDiffusion3D_v4(PortableModule):
    def __init__(self, encoder_spec=None, hidden_dim=256, n_layers=6, k_neighbors=32,
                 n_timesteps=1000, lr=1e-4, encoder_lr=1e-4, unfreeze_at_epoch=0,
                 conditioning_mode="node", readout="vector", graph_conditioning=True,
                 self_conditioning=True, use_cosine_schedule=True,
                 adaptive_cutoff=False, cutoff_radius=6.0, use_attention_pooling=True,
                 coordinate_scale=1.0, physics_weight=0.0, bond_weight=0.1,
                 clash_weight=0.1, angle_weight=0.0, ring_weight=0.0,
                 torsion_weight=0.0, chirality_weight=0.0, physics_min_alpha=0.05,
                 ema_decay=0.999, warmup_steps=1000, weight_decay=1e-5):
        super().__init__()
        encoder_spec = dict(encoder_spec or {"hidden_dim": 128, "n_layers": 2, "dropout": 0.0})
        encoder_spec.pop("kind", None)
        if coordinate_scale <= 0 or not 0 <= ema_decay < 1 or warmup_steps < 0:
            raise ValueError("Invalid coordinate scale, EMA decay, or warmup.")
        if lr <= 0 or encoder_lr <= 0 or unfreeze_at_epoch < 0 or not 0 < physics_min_alpha <= 1 or physics_weight < 0:
            raise ValueError("Invalid learning rate, epoch, or physics settings.")
        self.save_hyperparameters()
        self.encoder = SpectrumTokenEncoder(**encoder_spec)
        dim = self.encoder.spec["hidden_dim"]
        self.attn_pool = nn.Linear(dim, 1)
        self.cond_proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.LayerNorm(dim))
        self.denoiser = EquivariantDenoiser(cond_dim=dim, hidden_dim=hidden_dim,
            n_layers=n_layers, k_neighbors=k_neighbors, conditioning_mode=conditioning_mode,
            readout=readout, graph_conditioning=graph_conditioning, self_conditioning=self_conditioning,
            adaptive_cutoff=adaptive_cutoff, cutoff_radius=cutoff_radius)
        self.physics_loss = PhysicsLoss(bond_weight=bond_weight, clash_weight=clash_weight,
            angle_weight=angle_weight, ring_weight=ring_weight, torsion_weight=torsion_weight,
            chirality_weight=chirality_weight)
        self.n_timesteps = n_timesteps
        for name, values in build_diffusion_schedule(n_timesteps, use_cosine_schedule).items():
            self.register_buffer(name, values.float())
        self._set_encoder_trainable(unfreeze_at_epoch == 0)

    def _set_encoder_trainable(self, enabled):
        self.encoder.requires_grad_(enabled)
        self.encoder.train(self.training and enabled)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def on_train_epoch_start(self):
        self._set_encoder_trainable(self.current_epoch >= self.hparams.unfreeze_at_epoch)

    def condition(self, spectra):
        tokens, mask = self.encoder.encode_spectra(spectra)
        if self.hparams.use_attention_pooling:
            logits = self.attn_pool(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
            pooled = (tokens * logits.softmax(-1).unsqueeze(-1)).sum(1)
        else:
            pooled = tokens.sum(1) / mask.sum(1, keepdim=True)
        return self.cond_proj(pooled), tokens, mask

    def _predict(self, x, atoms, t, conditioning, mask, graph=None, self_cond=None):
        pooled, tokens, token_mask = conditioning
        return self.denoiser(x, atoms, t, pooled, mask, self_cond=self_cond,
            condition_tokens=tokens, condition_mask=token_mask, graph=graph)

    def forward_diffusion(self, x0, t, mask=None, generator=None):
        x0 = masked_center(x0, mask)
        noise = centered_randn_like(x0, mask, generator)
        return (self.sqrt_alphas_cumprod[t, None, None] * x0 +
                self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise), noise

    def predict_x0_from_noise(self, x, t, noise):
        return (self.sqrt_recip_alphas_cumprod[t, None, None] * x.float() -
                self.sqrt_recipm1_alphas_cumprod[t, None, None] * noise.float())

    def loss(self, batch, *, generator=None, self_condition=True):
        mask, atoms = batch["mask"], batch["atom_types"]
        coords = masked_center(batch["coords"].float(), mask) / self.hparams.coordinate_scale
        t = torch.randint(self.n_timesteps, (len(coords),), generator=generator, device=coords.device)
        noisy, target = self.forward_diffusion(coords, t, mask, generator)
        condition, graph, previous = self.condition(batch["spectra"]), batch.get("graph"), None
        if self_condition and self.hparams.self_conditioning and torch.rand((), device=coords.device) < 0.5:
            with torch.no_grad():
                estimate = self._predict(noisy, atoms, t, condition, mask, graph)
                previous = masked_center(self.predict_x0_from_noise(noisy, t, estimate), mask).detach()
        predicted = self._predict(noisy, atoms, t, condition, mask, graph, previous)
        squared = ((predicted.float() - target).square() * mask.unsqueeze(-1)).sum((1, 2))
        denoising = (squared / (3 * mask.sum(1)).clamp_min(1)).mean()
        physics = denoising.new_zeros(())
        if self.hparams.physics_weight > 0:
            if graph is None:
                raise ValueError("Physics training requires explicit graph targets.")
            selected = self.alphas_cumprod[t] >= self.hparams.physics_min_alpha
            if selected.any():
                x0 = masked_center(self.predict_x0_from_noise(noisy, t, predicted), mask)
                topology = {key: value[selected] for key, value in graph.items()}
                physics = self.physics_loss(x0[selected] * self.hparams.coordinate_scale,
                    atoms[selected], mask[selected], graph=topology)
        total = denoising + self.hparams.physics_weight * physics
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite loss; inspect data and configuration.")
        return total, denoising, physics

    def training_step(self, batch, batch_idx):
        total, mse, physics = self.loss(batch)
        for name, value in (("train_loss", total), ("train_noise_mse", mse), ("train_physics", physics)):
            self.log(name, value, batch_size=len(batch["coords"]), on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        generator = torch.Generator(device=self.device).manual_seed(92017 + batch_idx)
        total, mse, physics = self.loss(batch, generator=generator, self_condition=False)
        self.log("val_loss", mse, batch_size=len(batch["coords"]), sync_dist=True, prog_bar=True)
        self.log("val_total_loss", total, batch_size=len(batch["coords"]), sync_dist=True)
        return mse

    @torch.no_grad()
    def sample(self, spectra, atom_types, num_atoms, n_steps=None, sampler="ddim", eta=0.0,
               physics_guidance=0.0, guidance_start_frac=0.3, graph=None, generator=None):
        if not 0 <= eta <= 1 or physics_guidance < 0 or not 0 <= guidance_start_frac <= 1:
            raise ValueError("Invalid DDIM eta or physics guidance.")
        steps = sampling_timesteps(self.n_timesteps, n_steps, sampler)
        if num_atoms.shape != (len(atom_types),) or (num_atoms < 1).any() or (num_atoms > atom_types.shape[1]).any():
            raise ValueError("num_atoms must be valid counts matching atom_types.")
        mask = torch.arange(atom_types.shape[1], device=atom_types.device)[None] < num_atoms[:, None]
        if physics_guidance > 0 and graph is None:
            raise ValueError("Guidance requires a graph available at inference.")
        condition = self.condition(spectra)
        x = centered_randn_like(torch.zeros(*atom_types.shape, 3, device=atom_types.device), mask, generator)
        previous = None
        for index, time in enumerate(steps):
            t = torch.full((len(x),), time, device=x.device, dtype=torch.long)
            epsilon = self._predict(x, atom_types, t, condition, mask, graph, previous)
            x0 = masked_center(self.predict_x0_from_noise(x, t, epsilon), mask)
            if sampler == "ddpm":
                x = self.posterior_mean_coef1[time] * x0 + self.posterior_mean_coef2[time] * x
                if time > 0:
                    x = x + self.posterior_variance[time].sqrt() * centered_randn_like(x, mask, generator)
            else:
                alpha = self.alphas_cumprod[time]
                alpha_previous = self.alphas_cumprod[steps[index + 1]] if index + 1 < len(steps) else x.new_tensor(1.)
                variance = ((1-alpha_previous)/(1-alpha) * (1-alpha/alpha_previous)).clamp_min(0)
                sigma = eta * variance.sqrt()
                x = alpha_previous.sqrt() * x0 + (1-alpha_previous-sigma.square()).clamp_min(0).sqrt() * epsilon
                if eta > 0:
                    x = x + sigma * centered_randn_like(x, mask, generator)
            x, previous = masked_center(x, mask), x0 if self.hparams.self_conditioning else None
            if physics_guidance > 0 and time < self.n_timesteps * guidance_start_frac:
                with torch.enable_grad():
                    physical = (x.detach() * self.hparams.coordinate_scale).requires_grad_(True)
                    energy = self.physics_loss(physical, atom_types, mask, graph=graph, reduction="sum")
                    gradient = torch.autograd.grad(energy, physical)[0]
                    norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1.)
                    x = masked_center(x - physics_guidance * gradient / norm / self.hparams.coordinate_scale, mask).detach()
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Nonfinite coordinates at timestep {time}.")
        return x * self.hparams.coordinate_scale, mask

    def configure_optimizers(self):
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        groups = [{"params": [p for p in self.parameters() if id(p) not in encoder_ids], "lr": self.hparams.lr},
                  {"params": list(self.encoder.parameters()), "lr": self.hparams.encoder_lr}]
        optimizer = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
        def factor(step):
            warmup = self.hparams.warmup_steps
            if step < warmup:
                return (step + 1) / max(1, warmup)
            total = max(warmup + 1, self.trainer.estimated_stepping_batches)
            progress = min(1., (step-warmup) / (total-warmup))
            return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, factor), "interval": "step"}}
