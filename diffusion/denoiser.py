"""Spectrum-conditioned, mean-free equivariant coordinate-noise prediction."""

import torch
import torch.nn as nn

from diffusion.layers import SinusoidalTimeEmb, StableEGNNLayer
from diffusion.utils import masked_center


class EquivariantDenoiser(nn.Module):
    """EGNN noise predictor with pooled, node, or node-and-edge token conditioning.

    graph['bond_types'] contains a symmetric [B, N, N] integer matrix:
    0=no bond, 1=single, 2=double, 3=triple, 4=aromatic.
    Vector readout has unbounded learned scalar coefficients; residual readout
    retains the original parameter layout for explicit legacy configurations.
    """

    def __init__(self, cond_dim=384, hidden_dim=256, time_dim=128,
                 n_layers=6, k_neighbors=16, self_conditioning=True, self_cond_dim=32,
                 adaptive_cutoff=False, cutoff_radius=6.0,
                 conditioning_mode='pooled', attention_heads=4,
                 graph_conditioning=False, readout='vector'):
        super().__init__()
        for name, value in [('cond_dim', cond_dim), ('hidden_dim', hidden_dim),
                            ('n_layers', n_layers), ('k_neighbors', k_neighbors),
                            ('attention_heads', attention_heads)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if hidden_dim < 2 or cutoff_radius <= 0:
            raise ValueError('hidden_dim must be >= 2 and cutoff_radius must be positive')
        if self_conditioning and (not isinstance(self_cond_dim, int) or self_cond_dim < 1):
            raise ValueError('self_cond_dim must be a positive integer')
        if conditioning_mode not in ('pooled', 'node', 'node_edge'):
            raise ValueError('conditioning_mode must be pooled, node, or node_edge')
        if readout not in ('vector', 'residual'):
            raise ValueError('readout must be vector or residual')
        if readout == 'residual' and n_layers < 2:
            raise ValueError('residual readout needs at least two layers')
        if conditioning_mode != 'pooled' and hidden_dim % attention_heads:
            raise ValueError('hidden_dim must be divisible by attention_heads for token attention')

        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.self_conditioning = self_conditioning
        self.self_cond_dim = self_cond_dim if self_conditioning else 0
        self.adaptive_cutoff = adaptive_cutoff
        self.cutoff_radius = cutoff_radius
        self.conditioning_mode = conditioning_mode
        self.graph_conditioning = graph_conditioning
        self.readout = readout

        self.atom_emb = nn.Embedding(119, 64, padding_idx=0)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_dim), nn.Linear(time_dim, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        if self.self_conditioning:
            self.self_cond_mlp = nn.Sequential(
                nn.Linear(1, self_cond_dim), nn.SiLU(),
                nn.Linear(self_cond_dim, self_cond_dim),
            )
        self.node_init = nn.Sequential(
            nn.Linear(64 + time_dim + hidden_dim + self.self_cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
        )

        graph_dim = 16 if graph_conditioning else 0
        if graph_conditioning:
            self.bond_emb = nn.Embedding(5, graph_dim, padding_idx=0)
        self.edge_dim = graph_dim + (hidden_dim if conditioning_mode == 'node_edge' else 0)
        self.egnn_layers = nn.ModuleList([
            StableEGNNLayer(hidden_dim, edge_dim=self.edge_dim,
                            update_coords=(i < n_layers - 1), coord_scale=0.1)
            for i in range(n_layers)
        ])

        if conditioning_mode != 'pooled':
            self.token_proj = nn.Linear(cond_dim, hidden_dim)
            self.node_attn = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, attention_heads, dropout=0.0, batch_first=True)
                for _ in range(n_layers)
            ])
            self.node_attn_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        if conditioning_mode == 'node_edge':
            self.edge_query = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim),
                              nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(n_layers)
            ])
            self.edge_attn = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, attention_heads, dropout=0.0, batch_first=True)
                for _ in range(n_layers)
            ])

        self.output_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.output_scale[-1].weight)
        nn.init.zeros_(self.output_scale[-1].bias)
        if readout == 'vector':
            self.vector_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1 + self.edge_dim, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, 1),
            )
            self.radial_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for head in (self.vector_mlp, self.radial_mlp):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _bond_types(graph, x, mask):
        if not isinstance(graph, dict) or 'bond_types' not in graph:
            raise ValueError('graph conditioning requires graph["bond_types"]')
        bonds = graph['bond_types']
        B, N, _ = x.shape
        if not isinstance(bonds, torch.Tensor) or bonds.shape != (B, N, N):
            raise ValueError('bond_types must have shape [B, N, N]')
        if bonds.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('bond_types must be an integer tensor')
        bonds = bonds.to(device=x.device, dtype=torch.long)
        if (bonds < 0).any() or (bonds > 4).any():
            raise ValueError('bond_types must use codes 0 through 4')
        if not torch.equal(bonds, bonds.transpose(1, 2)):
            raise ValueError('bond_types must be symmetric')
        if bonds.diagonal(dim1=1, dim2=2).any():
            raise ValueError('bond_types cannot contain self bonds')
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        if bonds.masked_select(~valid).any():
            raise ValueError('bond_types cannot connect padded atoms')
        return bonds

    def compute_edges(self, x, mask, k, return_mask=False, graph=None):
        """kNN/radius neighbors union all bonds, with explicit validity.

        Include every tie at the kNN boundary to preserve permutation symmetry.
        Invalid padding slots carry arbitrary indices and an always-false mask.
        """
        B, N, _ = x.shape
        if N < 1 or k < 1:
            raise ValueError('at least one atom slot and a positive k are required')
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask = mask.bool()
        dist_sq = (x.unsqueeze(2) - x.unsqueeze(1)).square().sum(-1)
        eligible = mask.unsqueeze(1) & mask.unsqueeze(2)
        eligible = eligible & ~torch.eye(N, device=x.device, dtype=torch.bool)[None]
        if self.adaptive_cutoff:
            eligible = eligible & (dist_sq <= self.cutoff_radius ** 2)
        distances = dist_sq.masked_fill(~eligible, float('inf'))
        count = min(k, N)
        threshold = distances.topk(count, dim=-1, largest=False).values[..., -1:]
        tolerance = 8 * torch.finfo(dist_sq.dtype).eps * threshold.abs().clamp(min=1)
        selected = eligible & (dist_sq <= threshold + tolerance)
        if graph is not None:
            bonds = self._bond_types(graph, x, mask)
            selected = selected | (bonds > 0)
        width = max(1, int(selected.sum(-1).max().item()))
        indices = dist_sq.masked_fill(~selected, float('inf')).topk(
            width, dim=-1, largest=False
        ).indices
        valid = selected.gather(2, indices)
        return (indices, valid) if return_mask else indices

    def _tokens(self, tokens, token_mask, B):
        if tokens is None or tokens.ndim != 3 or tokens.shape[0] != B or tokens.shape[2] != self.cond_dim:
            raise ValueError('token conditioning requires condition_tokens [B, L, cond_dim]')
        if token_mask is None:
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if token_mask.shape != tokens.shape[:2]:
            raise ValueError('condition_mask must have shape [B, L]')
        token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        tokens = self.token_proj(torch.where(token_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens)))
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        has_tokens = token_mask.any(dim=1)
        # PyTorch attention needs one unmasked key to avoid NaNs for empty spectra.
        safe_mask = token_mask.clone()
        if tokens.shape[1]:
            safe_mask[~has_tokens, 0] = True
        return tokens, ~safe_mask, has_tokens

    @staticmethod
    def _attend(attention, queries, tokens, padding_mask, has_tokens):
        if tokens.shape[1] == 0:
            return torch.zeros_like(queries)
        context = attention(queries, tokens, tokens, key_padding_mask=padding_mask,
                            need_weights=False)[0]
        return context * has_tokens[:, None, None].to(context.dtype)

    def _edge_features(self, h, x, indices, valid, bonds, layer_idx, token_data):
        B, N, K = indices.shape
        batch_index = torch.arange(B, device=x.device)[:, None, None]
        features = []
        if bonds is not None:
            features.append(self.bond_emb(bonds.gather(2, indices)))
        if self.conditioning_mode == 'node_edge':
            h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
            h_j = h[batch_index, indices]
            distance = torch.sqrt(
                (x.unsqueeze(2) - x[batch_index, indices]).square().sum(-1, keepdim=True) + 1e-8
            )
            query = self.edge_query[layer_idx](
                torch.cat([h_i, h_j, distance.to(h.dtype) / self.cutoff_radius], dim=-1)
            ).reshape(B, N * K, self.hidden_dim)
            context = self._attend(self.edge_attn[layer_idx], query, *token_data)
            features.append(context.reshape(B, N, K, self.hidden_dim))
        if not features:
            return None
        return torch.cat(features, dim=-1) * valid.unsqueeze(-1).to(h.dtype)

    def forward(self, x, atom_types, t, condition, mask=None, self_cond=None,
                condition_tokens=None, condition_mask=None, graph=None):
        if x.ndim != 3 or x.shape[-1] != 3 or x.shape[1] < 1:
            raise ValueError('x must have shape [B, N, 3] with N >= 1')
        B, N, _ = x.shape
        if atom_types.shape != (B, N) or t.shape != (B,) or condition.shape != (B, self.cond_dim):
            raise ValueError('atom_types, t, and condition must match the coordinate batch')
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        if mask.shape != (B, N):
            raise ValueError('mask must have shape [B, N]')
        mask = mask.to(device=x.device, dtype=torch.bool)
        x_input = masked_center(x, mask)
        bonds = self._bond_types(graph, x, mask) if self.graph_conditioning else None
        edge_graph = {'bond_types': bonds} if bonds is not None else None
        indices, valid = self.compute_edges(x_input, mask, self.k_neighbors,
                                            return_mask=True, graph=edge_graph)
        token_data = None
        if self.conditioning_mode != 'pooled':
            token_data = self._tokens(condition_tokens, condition_mask, B)

        atom_feat = self.atom_emb(torch.where(mask, atom_types, torch.zeros_like(atom_types)))
        t_feat = self.time_mlp(t).unsqueeze(1).expand(-1, N, -1)
        c_feat = self.cond_mlp(condition).unsqueeze(1).expand(-1, N, -1)
        features = [atom_feat, t_feat, c_feat]
        if self.self_conditioning:
            if self_cond is None:
                sc = torch.zeros(B, N, self.self_cond_dim, device=x.device, dtype=c_feat.dtype)
            else:
                if self_cond.shape != x.shape:
                    raise ValueError('self_cond must match x')
                delta = masked_center(self_cond, mask) - x_input
                distance = torch.sqrt(delta.square().sum(-1, keepdim=True) + 1e-8)
                sc = self.self_cond_mlp(distance.to(c_feat.dtype))
            features.append(sc)
        h = self.node_init(torch.cat(features, dim=-1))
        h = h * mask.unsqueeze(-1).to(h.dtype)

        x_curr = x_input
        edge_attr = None
        for layer_idx, layer in enumerate(self.egnn_layers):
            edge_attr = self._edge_features(h, x_curr, indices, valid, bonds, layer_idx, token_data)
            h, x_curr = layer(h, x_curr, indices, mask, edge_attr=edge_attr, edge_mask=valid)
            if self.conditioning_mode != 'pooled':
                # Message passing supplies geometry-aware invariant atom queries.
                context = self._attend(self.node_attn[layer_idx], h, *token_data)
                h = self.node_attn_norm[layer_idx](h + context)
                h = h * mask.unsqueeze(-1).to(h.dtype)

        scale = 1.0 + torch.tanh(self.output_scale(h))
        if self.readout == 'residual':
            noise = (x_input - x_curr) * scale
        else:
            K = indices.shape[-1]
            batch_index = torch.arange(B, device=x.device)[:, None, None]
            relative = x_input.unsqueeze(2) - x_input[batch_index, indices]
            distance = torch.sqrt(relative.square().sum(-1, keepdim=True) + 1e-8)
            edge_features = [
                h.unsqueeze(2).expand(-1, -1, K, -1), h[batch_index, indices],
                distance.to(h.dtype) / self.cutoff_radius,
            ]
            if edge_attr is not None:
                edge_features.append(edge_attr)
            weights = self.vector_mlp(torch.cat(edge_features, dim=-1)).to(x.dtype)
            vectors = weights * relative / distance * valid.unsqueeze(-1).to(x.dtype)
            degree = valid.sum(-1, keepdim=True).clamp(min=1).to(x.dtype)
            noise = vectors.sum(dim=2) / degree + self.radial_mlp(h).to(x.dtype) * x_input
            noise = noise * scale.to(x.dtype)
        return masked_center(noise, mask)
