"""A self-contained spectral encoder for new training and portable checkpoints."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class SpectrumTokenEncoder(nn.Module):
    """Encode a dense spectrum or masked peak list into invariant tokens.

    Both supported stages use this encoder and embed its full specification
    in their portable checkpoints. No external src package is required.
    """

    def __init__(self, hidden_dim=128, input_channels=1, patch_size=16,
                 n_layers=2, n_heads=4, peak_feature_dim=2, dropout=0.1):
        super().__init__()
        dimensions = (hidden_dim, input_channels, patch_size, n_layers, n_heads, peak_feature_dim)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError('Encoder dimensions must be positive integers.')
        if hidden_dim < 4 or hidden_dim % n_heads:
            raise ValueError('Invalid native encoder dimensions or attention head count.')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        self.spec = dict(kind='native', hidden_dim=hidden_dim, input_channels=input_channels,
                         patch_size=patch_size, n_layers=n_layers, n_heads=n_heads,
                         peak_feature_dim=peak_feature_dim, dropout=dropout)
        self.hparams = SimpleNamespace(hidden_dim=hidden_dim, encoder_type='native')
        self.dense_projection = nn.Linear(input_channels * patch_size, hidden_dim)
        self.peak_projection = nn.Linear(peak_feature_dim, hidden_dim)
        block = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def encode_spectra(self, spectra):
        if isinstance(spectra, dict):
            packed = spectra['packed']
            if packed.ndim != 3 or packed.shape[-1] != self.spec['peak_feature_dim']:
                raise ValueError('Peak lists must have shape [B, peaks, peak_feature_dim].')
            mask = spectra.get('mask')
            if mask is None:
                mask = torch.ones(packed.shape[:2], device=packed.device, dtype=torch.bool)
            if mask.dtype != torch.bool or mask.shape != packed.shape[:2]:
                raise ValueError('Peak mask must be a boolean [B, peaks] tensor.')
            safe = torch.where(mask.unsqueeze(-1), packed, torch.zeros_like(packed))
            tokens = self.peak_projection(safe)
        else:
            if spectra.ndim == 2:
                spectra = spectra.unsqueeze(1)
            if spectra.ndim != 3 or spectra.shape[1] != self.spec['input_channels'] or spectra.shape[-1] == 0:
                raise ValueError('Dense spectra must have shape [B, input_channels, length].')
            batch, channels, length = spectra.shape
            patch = self.spec['patch_size']
            padded = F.pad(spectra, (0, (-length) % patch))
            patches = padded.unfold(-1, patch, patch).permute(0, 2, 1, 3).reshape(batch, -1, channels * patch)
            tokens = self.dense_projection(patches)
            mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        if tokens.shape[1] == 0 or (~mask.any(1)).any():
            raise ValueError('Each spectrum must contain at least one valid token.')
        if not torch.isfinite(tokens[mask]).all():
            raise ValueError('Spectra contain nonfinite active features.')
        if not isinstance(spectra, dict):
            length = tokens.shape[1]
            positions = torch.arange(length, device=tokens.device, dtype=torch.float32).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, tokens.shape[-1], 2, device=tokens.device).float()
                                  * (-math.log(10000) / tokens.shape[-1]))
            positional = torch.zeros(length, tokens.shape[-1], device=tokens.device)
            positional[:, 0::2] = torch.sin(positions * frequency)
            positional[:, 1::2] = torch.cos(positions * frequency[:positional[:, 1::2].shape[1]])
            tokens = tokens + positional.to(tokens.dtype)
        tokens = self.encoder_blocks(tokens, src_key_padding_mask=~mask)
        tokens = self.output_norm(tokens)
        return torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens)), mask
