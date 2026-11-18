"""Full-token IR/SELFIES matching experiment, separate from the generator.

The spectral encoder is copied before any change. Molecule self-attention is
bidirectional; cross-attention sees every valid spectral token. Scores are
learned compatibility logits, not calibrated probabilities or calculated IR.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def identity_contrastive_loss(scores, identities, temperature=0.1):
    """Symmetric multi-positive InfoNCE; duplicate identities are positives."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if not identities or scores.ndim != 2 or scores.shape != (len(identities), len(identities)):
        raise ValueError("Alignment requires a square matrix of paired identities.")
    if not torch.isfinite(scores).all():
        raise ValueError("Alignment scores must be finite.")
    positive = torch.tensor([[left == right for right in identities] for left in identities],
                            device=scores.device, dtype=torch.bool)
    logits = scores / temperature
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    return 0.5 * ((torch.logsumexp(logits, 1) - torch.logsumexp(positive_logits, 1)).mean()
                  + (torch.logsumexp(logits, 0) - torch.logsumexp(positive_logits, 0)).mean())


class TokenMatcher(nn.Module):
    """Match paired full spectra and molecules, with independent alignment heads.

    ``encode_molecule`` expects BOS at position zero and right padding. The BOS
    representation aggregates the entire molecule with no causal attention
    mask. ``alignment`` returns all-pairs cosine similarities; ``match`` takes
    equal-sized batches and returns one unbounded logit per pair.
    """

    def __init__(self, encoder, vocab_size, max_len, hidden_dim=None, pad_idx=0,
                 n_heads=4, n_layers=2, dropout=0.1, train_encoder=False):
        super().__init__()
        encoder_dim = getattr(getattr(encoder, "hparams", None), "hidden_dim", None)
        if encoder_dim is None:
            encoder_dim = getattr(encoder, "spec", {}).get("hidden_dim")
        hidden_dim = encoder_dim if hidden_dim is None else hidden_dim
        dimensions = (encoder_dim, hidden_dim, vocab_size, max_len, n_heads, n_layers)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Encoder, token and attention dimensions must be positive integers.")
        if hidden_dim % n_heads or type(pad_idx) is not int or not 0 <= pad_idx < vocab_size:
            raise ValueError("Invalid attention head count or padding index.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")
        self.spec = dict(encoder_dim=encoder_dim, hidden_dim=hidden_dim, vocab_size=vocab_size,
                         max_len=max_len, pad_idx=pad_idx, n_heads=n_heads,
                         n_layers=n_layers, dropout=dropout, train_encoder=bool(train_encoder))
        self.encoder = copy.deepcopy(encoder)
        self.spectrum_adapter = nn.Identity() if encoder_dim == hidden_dim else nn.Linear(encoder_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.position = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.molecule_encoder = nn.TransformerEncoder(layer, n_layers,
                                                     norm=nn.LayerNorm(hidden_dim),
                                                     enable_nested_tensor=False)
        cross_layer = nn.TransformerDecoderLayer(hidden_dim, n_heads, hidden_dim * 4,
                                                dropout=dropout, batch_first=True, norm_first=True)
        self.cross_attention = nn.TransformerDecoder(cross_layer, n_layers,
                                                    norm=nn.LayerNorm(hidden_dim))
        self.match_score = nn.Linear(hidden_dim, 1)
        # Alignment uses learned attention pooling, while matching retains every
        # peak token. No pooled feature cache restricts the matching pathway.
        self.spectrum_pool = nn.Linear(hidden_dim, 1, bias=False)
        self.spectrum_projection = nn.Linear(hidden_dim, hidden_dim)
        self.molecule_projection = nn.Linear(hidden_dim, hidden_dim)
        self.set_encoder_trainable(train_encoder)

    def set_encoder_trainable(self, value):
        """Enable encoder learning after warmup without changing its source copy."""
        self.train_encoder = bool(value)
        self.spec["train_encoder"] = self.train_encoder
        self.encoder.requires_grad_(self.train_encoder)
        self.encoder.train(self.training and self.train_encoder)
        return self

    def train(self, mode=True):
        super().train(mode)
        if not self.train_encoder:
            self.encoder.eval()
        return self

    @staticmethod
    def _validate_hidden(hidden, mask, label):
        if (hidden.ndim != 3 or mask.dtype != torch.bool or mask.shape != hidden.shape[:2]
                or hidden.shape[0] == 0 or hidden.shape[1] == 0 or (~mask.any(1)).any()):
            raise ValueError(f"{label} requires nonempty hidden states and a valid boolean mask.")
        if not torch.isfinite(hidden[mask]).all():
            raise ValueError(f"{label} contains nonfinite active tokens.")
        # Ignored padding must not contaminate projections or attention values.
        return torch.where(mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))

    def encode_spectrum(self, spectra):
        if self.train_encoder:
            memory, mask = self.encoder.encode_spectra(spectra)
        else:
            with torch.no_grad():
                memory, mask = self.encoder.encode_spectra(spectra)
        memory = self._validate_hidden(memory, mask, "Spectrum")
        memory = self.spectrum_adapter(memory)
        return torch.where(mask.unsqueeze(-1), memory, torch.zeros_like(memory)), mask

    def encode_molecule(self, tokens):
        if (tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64)
                or tokens.shape[0] == 0 or not 0 < tokens.shape[1] <= self.spec["max_len"]):
            raise ValueError("Molecule tokens require a nonempty integer [B, length] tensor within max_len.")
        if ((tokens < 0) | (tokens >= self.spec["vocab_size"])).any():
            raise ValueError("Molecule token is outside the trained vocabulary.")
        mask = tokens != self.spec["pad_idx"]
        if not mask[:, 0].all() or ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError("Molecule tokens must start with BOS and use right padding only.")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        hidden = self.molecule_encoder(hidden, src_key_padding_mask=~mask)
        return torch.where(mask.unsqueeze(-1), hidden, torch.zeros_like(hidden)), mask

    def alignment(self, memory, memory_mask, hidden, molecule_mask):
        memory = self._validate_hidden(memory, memory_mask, "Spectrum")
        hidden = self._validate_hidden(hidden, molecule_mask, "Molecule")
        if not molecule_mask[:, 0].all():
            raise ValueError("Molecule alignment requires an active BOS token.")
        weights = self.spectrum_pool(memory).squeeze(-1).masked_fill(~memory_mask, -torch.inf).softmax(-1)
        ir = F.normalize(self.spectrum_projection((memory * weights.unsqueeze(-1)).sum(1)), dim=-1)
        molecule = F.normalize(self.molecule_projection(hidden[:, 0]), dim=-1)
        return ir @ molecule.T

    def match(self, memory, memory_mask, hidden, molecule_mask):
        memory = self._validate_hidden(memory, memory_mask, "Spectrum")
        hidden = self._validate_hidden(hidden, molecule_mask, "Molecule")
        if memory.shape[0] != hidden.shape[0] or not molecule_mask[:, 0].all():
            raise ValueError("Matching needs equal paired batches and an active molecule BOS token.")
        cross = self.cross_attention(hidden, memory, tgt_key_padding_mask=~molecule_mask,
                                     memory_key_padding_mask=~memory_mask)
        return self.match_score(cross[:, 0]).squeeze(-1)

    def forward(self, spectra, tokens):
        memory, memory_mask = self.encode_spectrum(spectra)
        hidden, molecule_mask = self.encode_molecule(tokens)
        return self.match(memory, memory_mask, hidden, molecule_mask)
