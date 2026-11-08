"""Invariant messages and equivariant coordinate updates for molecular diffusion."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmb(nn.Module):
    """Time embedding with exactly dim channels, including odd and small dims."""

    def __init__(self, dim):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError('time embedding dimension must be a positive integer')
        self.dim = dim

    def forward(self, t):
        half = (self.dim + 1) // 2
        frequencies = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000) / max(half - 1, 1))
        )
        angles = t[:, None].float() * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)[:, :self.dim]


class StableEGNNLayer(nn.Module):
    """EGNN layer accepting explicit neighbor validity and invariant edge features."""

    def __init__(self, hidden_dim, edge_dim=0, update_coords=True,
                 coord_scale=0.1, cutoff=10.0):
        super().__init__()
        if hidden_dim < 2 or edge_dim < 0 or cutoff <= 0 or coord_scale < 0:
            raise ValueError('invalid EGNN dimensions, cutoff, or coordinate scale')
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.update_coords = update_coords
        self.coord_scale = coord_scale
        self.cutoff = cutoff

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Tanh(),
            )
            nn.init.zeros_(self.coord_mlp[-2].weight)
            nn.init.zeros_(self.coord_mlp[-2].bias)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, x, edge_index, mask=None, edge_attr=None, edge_mask=None):
        B, N, D = h.shape
        K = edge_index.shape[2]
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=h.device)

        batch_index = torch.arange(B, device=h.device)[:, None, None]
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
        h_j = h[batch_index, edge_index]
        rel_pos = x.unsqueeze(2) - x[batch_index, edge_index]
        dist = torch.sqrt(rel_pos.square().sum(-1, keepdim=True) + 1e-8)

        valid = mask.unsqueeze(2) & mask[batch_index, edge_index]
        valid = valid & (edge_index != torch.arange(N, device=h.device)[None, :, None])
        if edge_mask is not None:
            if edge_mask.shape != (B, N, K):
                raise ValueError('edge_mask must have shape [B, N, K]')
            valid = valid & edge_mask.bool()
        valid_f = valid.unsqueeze(-1).to(h.dtype)

        features = [h_i, h_j, dist.to(h.dtype) / self.cutoff]
        if self.edge_dim:
            if edge_attr is None or edge_attr.shape != (B, N, K, self.edge_dim):
                raise ValueError('edge_attr must match [B, N, K, edge_dim]')
            features.append(edge_attr.to(h.dtype))
        elif edge_attr is not None and edge_attr.shape[-1]:
            raise ValueError('this layer was configured without edge features')

        messages = self.edge_mlp(torch.cat(features, dim=-1)) * valid_f
        aggregate = messages.sum(dim=2)
        h_out = self.norm(h + self.node_mlp(torch.cat([h, aggregate], dim=-1)))
        h_out = h_out * mask.unsqueeze(-1).to(h_out.dtype)

        if self.update_coords:
            weights = self.coord_mlp(messages).to(x.dtype)
            update = (weights * (rel_pos / dist) * valid_f.to(x.dtype)).sum(dim=2)
            x_out = (x + self.coord_scale * update) * mask.unsqueeze(-1).to(x.dtype)
        else:
            x_out = x
        return h_out, x_out
