"""Coordinate projections and a single, consistent diffusion schedule."""

import math
from numbers import Integral

import numpy as np
import torch


# Older TensorBoard releases reference the pre-NumPy-2 names.
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_


def coordinate_mask(x, mask=None):
    if x.ndim != 3 or x.shape[1] == 0:
        raise ValueError('Coordinates must have shape [batch, atoms, dimensions] with atoms > 0.')
    if mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.shape != x.shape[:2] or mask.dtype != torch.bool or mask.device != x.device:
        raise ValueError('mask must be a boolean [batch, atoms] tensor on the coordinate device.')
    return mask


def masked_center(x, mask=None):
    """Project valid atoms onto the zero-centroid subspace, leaving padding zero.

    This is the arithmetic centroid used by molecular coordinate diffusion,
    rather than a mass-weighted center. Empty masks produce differentiable zeros.
    """
    mask = coordinate_mask(x, mask)
    work = x if x.dtype == torch.float64 else x.float()
    selected = torch.where(mask.unsqueeze(-1), work, torch.zeros_like(work))
    count = mask.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1)
    mean = selected.sum(1, keepdim=True) / count
    return torch.where(mask.unsqueeze(-1), selected - mean, torch.zeros_like(work)).to(x.dtype)


def centered_randn_like(x, mask=None, generator=None):
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return masked_center(noise, mask)


def cosine_beta_schedule(timesteps, s=0.008):
    if isinstance(timesteps, bool) or not isinstance(timesteps, Integral) or timesteps < 2:
        raise ValueError('timesteps must be an integer >= 2.')
    if not math.isfinite(s) or s < 0:
        raise ValueError('cosine offset must be finite and nonnegative.')
    x = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos((x + s) / (1 + s) * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    # A finite terminal SNR avoids division by near-zero alpha in epsilon
    # reconstruction. Adjust the marginals BEFORE deriving transition betas,
    # rather than clamping a cumulative product independently in the model.
    cumulative = 1e-5 + (1 - 1e-5) * cumulative
    # Stabilize the step ratios, then reconstruct every marginal from them.
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)


def build_diffusion_schedule(timesteps, use_cosine=True):
    if isinstance(timesteps, bool) or not isinstance(timesteps, Integral) or timesteps < 2:
        raise ValueError('timesteps must be an integer >= 2.')
    betas = (cosine_beta_schedule(timesteps) if use_cosine else
             torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64))
    alphas = 1 - betas
    cumulative = torch.cumprod(alphas, 0)
    previous = torch.cat([torch.ones(1, dtype=torch.float64), cumulative[:-1]])
    denominator = 1 - cumulative
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': cumulative,
        'sqrt_alphas_cumprod': cumulative.sqrt(),
        'sqrt_one_minus_alphas_cumprod': denominator.sqrt(),
        'sqrt_recip_alphas_cumprod': cumulative.rsqrt(),
        'sqrt_recipm1_alphas_cumprod': (1 / cumulative - 1).sqrt(),
        'posterior_variance': betas * (1 - previous) / denominator,
        'posterior_mean_coef1': betas * previous.sqrt() / denominator,
        'posterior_mean_coef2': (1 - previous) * alphas.sqrt() / denominator,
    }


SCHEDULE_NAMES = tuple(build_diffusion_schedule(2))


def sampling_timesteps(timesteps, n_steps=None, sampler='ddim'):
    if sampler not in ('ddim', 'ddpm'):
        raise ValueError("sampler must be 'ddim' or 'ddpm'.")
    n_steps = timesteps if n_steps is None else n_steps
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or not 1 <= n_steps <= timesteps:
        raise ValueError('n_steps must be an integer in [1, n_timesteps].')
    if sampler == 'ddpm' and n_steps != timesteps:
        raise ValueError('DDPM requires the full adjacent-step schedule; use DDIM for fewer steps.')
    if n_steps == 1:
        return [timesteps - 1]
    values = torch.linspace(timesteps - 1, 0, n_steps, dtype=torch.float64).round().long().tolist()
    if len(set(values)) != n_steps:
        raise RuntimeError('Sampling schedule contains repeated timesteps.')
    return values
