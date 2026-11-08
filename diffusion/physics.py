"""Differentiable, graph-defined geometry penalties for coordinates in Angstrom."""

import json
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Approximate fallbacks. Explicit graph targets or training statistics take
# precedence; chemical environments cannot be inferred from elements alone.
DEFAULT_BOND_LENGTHS = {
    (1, 1): 0.74, (1, 6): 1.09, (1, 7): 1.01, (1, 8): 0.96,
    (6, 6): 1.54, (6, 7): 1.47, (6, 8): 1.43, (6, 9): 1.35,
    (7, 7): 1.45, (7, 8): 1.40, (8, 8): 1.48,
    (1, 16): 1.34, (6, 16): 1.82, (6, 17): 1.77, (16, 16): 2.05,
}
DEFAULT_BOND_LENGTHS_BY_TYPE = {
    1: DEFAULT_BOND_LENGTHS,
    2: {(6, 6): 1.34, (6, 7): 1.28, (6, 8): 1.23, (7, 7): 1.25},
    3: {(6, 6): 1.20, (6, 7): 1.16, (7, 7): 1.10},
    4: {(6, 6): 1.40, (6, 7): 1.35, (7, 7): 1.35, (6, 8): 1.37},
}

# Bondi, A. J. Phys. Chem. 1964, 68, 441-451.
VDW_RADII = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 16: 1.80, 17: 1.75}
_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


class PhysicsLoss(nn.Module):
    """Robust geometry losses with explicit connectivity and optional stereo.

    forward(coords, atom_types, mask, graph=None, reduction='mean',
    return_components=False) expects coordinates [B,N,3] in Angstrom,
    integer atomic numbers [B,N], and a boolean atom mask [B,N]. Every active
    term requires an explicit graph; this class never guesses connectivity.

    Graph schema (all tensors include the batch dimension):
      * bond_types [B,N,N]: symmetric integers, zero diagonal/padding; 0 means
        no bond, 1/2/3/4 mean single/double/triple/aromatic. This is mandatory.
      * optional bond_mean/bond_std [B,N,N]: positive, symmetric targets on
        bonds. They override element-and-bond-type fallback/statistical targets.
      * angles [B,A,3], angle_cos [B,A]: bonded i-j-k tuples and cosine targets.
      * planarity [B,P,4]: ordered a-b-c-d tuples; d lies in the plane of a,b,c.
        Supply only groups whose chemistry requires planarity, not every ring.
      * torsions [B,T,4]: bonded a-b-c-d chains, with torsion_periodicity,
        torsion_phase and torsion_amplitude [B,T]. Phase is in radians and
        periodicity is an integer 1..12. Penalty: amplitude *
        (1 - cos(periodicity * phi - phase)). Multiple terms for a bond can
        represent a multimodal periodic potential.
      * chirality [B,S,4], chirality_sign [B,S]: four ordered substituents and
        the desired +/-1 sign of their oriented tetrahedral volume. Ordering
        must match the supplied sign; it is not automatically an R/S label.

    Tuple rows are padded entirely with -1. Fields for an active angle,
    planarity, torsion or chirality term are required; empty tuple tensors are
    valid. Nonbonded clashes exclude 1-2 and 1-3 pairs. 1-4 pairs are included;
    this geometric clash term is not a complete force field or energy model.

    Bond-statistics JSON uses keys 'atomic_number-atomic_number-bond_type' and
    objects with positive finite 'mean' and optional 'std'. Legacy two-number
    keys are accepted for SINGLE bonds only, with a warning. Missing/invalid
    entries retain fallbacks. Invalid JSON structure raises ValueError.

    reduction='none' returns [B]; 'mean'/'sum' reduce molecule losses equally.
    return_components=True returns (total, unweighted_components), using the
    same reduction for each component. All geometry math runs in FP32.
    """

    def __init__(self, bond_weight=0.1, clash_weight=0.1,
                 angle_weight=0.05, ring_weight=0.05, torsion_weight=0.0,
                 bond_stats_path=None, default_bond_std=0.08,
                 chirality_weight=0.0, huber_delta=1.0,
                 min_chirality_volume=1e-3):
        super().__init__()
        weights = dict(bond=bond_weight, clash=clash_weight, angle=angle_weight,
                       planarity=ring_weight, torsion=torsion_weight,
                       chirality=chirality_weight)
        if any(not math.isfinite(float(v)) or v < 0 for v in weights.values()):
            raise ValueError('Physics weights must be finite and nonnegative.')
        if not math.isfinite(default_bond_std) or default_bond_std <= 0:
            raise ValueError('default_bond_std must be positive and finite.')
        if not math.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError('huber_delta must be positive and finite.')
        if not math.isfinite(min_chirality_volume) or min_chirality_volume <= 0:
            raise ValueError('min_chirality_volume must be positive and finite.')
        self.bond_weight = float(bond_weight)
        self.clash_weight = float(clash_weight)
        self.angle_weight = float(angle_weight)
        self.ring_weight = float(ring_weight)
        self.torsion_weight = float(torsion_weight)
        self.chirality_weight = float(chirality_weight)
        self.huber_delta = float(huber_delta)
        self.min_chirality_volume = float(min_chirality_volume)
        self.eps = 1e-6

        # Index by chemical bond type before the two atomic numbers.
        self.register_buffer('bond_mean_by_type', torch.zeros(5, 120, 120))
        self.register_buffer('bond_std_by_type',
                             torch.full((5, 120, 120), max(default_bond_std, 0.03)))
        self.register_buffer('bond_mask_by_type', torch.zeros(5, 120, 120, dtype=torch.bool))
        for bond_type, entries in DEFAULT_BOND_LENGTHS_BY_TYPE.items():
            for (a, b), mean in entries.items():
                self._set_bond_stat(a, b, bond_type, mean, max(default_bond_std, 0.03))
        if bond_stats_path:
            self._load_bond_stats(bond_stats_path, default_bond_std)

        self.register_buffer('vdw_radii', torch.full((120,), 1.7))
        for atom, radius in VDW_RADII.items():
            self.vdw_radii[atom] = radius

    def _set_bond_stat(self, a, b, bond_type, mean, std):
        for i, j in ((a, b), (b, a)):
            self.bond_mean_by_type[bond_type, i, j] = mean
            self.bond_std_by_type[bond_type, i, j] = std
            self.bond_mask_by_type[bond_type, i, j] = True

    def _load_bond_stats(self, path, default_std):
        path = Path(path)
        if not path.is_file():
            warnings.warn(f'Bond statistics file not found: {path}; retaining fallbacks.',
                          UserWarning, stacklevel=2)
            return
        try:
            with path.open('r', encoding='utf-8') as handle:
                stats = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Cannot read bond statistics JSON: {path}') from exc
        if not isinstance(stats, dict):
            raise ValueError('Bond statistics JSON must contain an object.')
        if 'bond_stats' in stats:
            units = stats.get('units', 'angstrom')
            if not isinstance(units, str) or units.lower() not in ('angstrom', 'angstroms', 'å'):
                raise ValueError('Bond statistics must use Angstrom units.')
            stats = stats['bond_stats']
            if not isinstance(stats, dict):
                raise ValueError('bond_stats must contain an object.')
        invalid, legacy = [], False
        for key, entry in stats.items():
            try:
                parts = key.split('-')
                if len(parts) == 2:
                    a, b = map(int, parts)
                    bond_type, legacy = 1, True
                elif len(parts) == 3:
                    a, b, bond_type = map(int, parts)
                else:
                    raise ValueError('bad key')
                mean = float(entry['mean'])
                std = float(entry.get('std', default_std))
                if not (1 <= a <= 118 and 1 <= b <= 118 and 1 <= bond_type <= 4):
                    raise ValueError('bad element or bond type')
                if not (math.isfinite(mean) and mean > 0 and math.isfinite(std) and std > 0):
                    raise ValueError('bad mean/std')
                self._set_bond_stat(a, b, bond_type, mean, max(std, 0.03))
            except (AttributeError, KeyError, TypeError, ValueError):
                invalid.append(str(key))
        if legacy:
            warnings.warn('Legacy bond statistics apply to single bonds only; regenerate '
                          'statistics with bond-type keys.', UserWarning, stacklevel=2)
        if invalid or not stats:
            warnings.warn('Invalid or empty bond statistics entries retain fallbacks: '
                          + ', '.join(invalid[:5]), UserWarning, stacklevel=2)

    def _tensor(self, graph, key, device, shape=None, integer=False):
        if key not in graph:
            raise ValueError(f'Active physics term requires graph["{key}"].')
        value = torch.as_tensor(graph[key], device=device)
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f'graph["{key}"] must have shape {tuple(shape)}.')
        if integer and value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f'graph["{key}"] must contain integers.')
        return value.long() if integer else value.float()

    def _tuples(self, graph, key, width, mask):
        batch_size, n_atoms = mask.shape
        indices = self._tensor(graph, key, mask.device, integer=True)
        if indices.ndim != 3 or indices.shape[0] != batch_size or indices.shape[-1] != width:
            raise ValueError(f'graph["{key}"] must have shape [B, count, {width}].')
        padded = (indices == -1).all(dim=-1)
        valid = ~padded
        if ((indices < 0) & valid.unsqueeze(-1)).any() or ((indices >= n_atoms) & valid.unsqueeze(-1)).any():
            raise ValueError(f'graph["{key}"] has invalid indices or partially padded rows.')
        batch, rows = valid.nonzero(as_tuple=True)
        selected = indices[batch, rows]
        if selected.numel():
            if not mask[batch.unsqueeze(-1), selected].all():
                raise ValueError(f'graph["{key}"] references masked atoms.')
            sorted_indices = selected.sort(dim=-1).values
            if (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any():
                raise ValueError(f'graph["{key}"] repeats an atom within a tuple.')
        return batch, rows, selected, indices.shape[:2]

    def _targets(self, graph, key, shape, batch, rows, device):
        values = self._tensor(graph, key, device, shape=shape)[batch, rows]
        if not torch.isfinite(values).all():
            raise ValueError(f'graph["{key}"] must have finite values on valid tuples.')
        return values

    def _huber(self, value):
        absolute = value.abs()
        delta = self.huber_delta
        return torch.where(absolute <= delta, 0.5 * value.square(),
                           delta * (absolute - 0.5 * delta))

    def _norm(self, vector):
        return torch.sqrt(vector.square().sum(dim=-1) + self.eps ** 2)

    @staticmethod
    def _average(values, batch, zeros):
        sums = zeros.scatter_add(0, batch, values)
        counts = torch.zeros_like(zeros).scatter_add(0, batch, torch.ones_like(values))
        return sums / counts.clamp(min=1)

    @staticmethod
    def _reduce(value, reduction):
        if reduction == 'none':
            return value
        return value.sum() if reduction == 'sum' else value.mean()

    def forward(self, coords, atom_types, mask, graph=None, reduction='mean',
                return_components=False):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] == 0:
            raise ValueError('coords must have shape [B,N,3] with B > 0.')
        if tuple(atom_types.shape) != tuple(coords.shape[:2]) or tuple(mask.shape) != tuple(coords.shape[:2]):
            raise ValueError('atom_types and mask must have shape [B,N].')
        if mask.dtype != torch.bool or atom_types.dtype not in _INTEGER_DTYPES:
            raise ValueError('mask must be boolean and atom_types must contain integers.')
        if mask.device != coords.device or atom_types.device != coords.device:
            raise ValueError('coords, atom_types, and mask must be on the same device.')
        if ((atom_types[mask] < 1) | (atom_types[mask] > 118)).any():
            raise ValueError('Valid atoms require atomic numbers from 1 to 118.')
        weights = dict(bond=self.bond_weight, clash=self.clash_weight,
                       angle=self.angle_weight, planarity=self.ring_weight,
                       torsion=self.torsion_weight, chirality=self.chirality_weight)
        active = any(weight > 0 for weight in weights.values())
        if active and (not isinstance(graph, dict) or 'bond_types' not in graph):
            raise ValueError('Active physics losses require an explicit graph with bond_types; '
                             'disable every physics term when connectivity is unavailable.')
        with torch.autocast(device_type=coords.device.type, enabled=False):
            x = torch.where(mask.unsqueeze(-1), coords.float(), torch.zeros_like(coords, dtype=torch.float32))
            if not torch.isfinite(x).all():
                raise ValueError('Coordinates must be finite on valid atoms.')
            zeros = x.sum(dim=(1, 2)) * 0.0
            components = {key: zeros for key in weights}
            if active:
                self._compute_components(x, atom_types.long(), mask, graph, zeros, components)
            total = sum(weights[key] * value for key, value in components.items())
            total = self._reduce(total, reduction)
            if return_components:
                return total, {key: self._reduce(value, reduction) for key, value in components.items()}
            return total

    def _compute_components(self, x, atom_types, mask, graph, zeros, components):
        batch_size, n_atoms = mask.shape
        bonds = self._tensor(graph, 'bond_types', x.device,
                             shape=(batch_size, n_atoms, n_atoms), integer=True)
        if ((bonds < 0) | (bonds > 4)).any() or not torch.equal(bonds, bonds.transpose(1, 2)):
            raise ValueError('bond_types must be symmetric with values from 0 to 4.')
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        if (bonds.diagonal(dim1=1, dim2=2) != 0).any() or ((bonds != 0) & ~pair_mask).any():
            raise ValueError('bond_types must have a zero diagonal and no bonds to masked atoms.')
        unique = torch.ones(n_atoms, n_atoms, device=x.device, dtype=torch.bool).triu(1)
        connected = bonds > 0

        if self.bond_weight > 0:
            batch, i, j = (connected & unique).nonzero(as_tuple=True)
            types, atoms_i, atoms_j = bonds[batch, i, j], atom_types[batch, i], atom_types[batch, j]
            mean = self.bond_mean_by_type[types, atoms_i, atoms_j]
            std = self.bond_std_by_type[types, atoms_i, atoms_j]
            if 'bond_mean' in graph:
                targets = self._tensor(graph, 'bond_mean', x.device, shape=bonds.shape)
                mean = targets[batch, i, j]
                if not torch.equal(mean, targets[batch, j, i]):
                    raise ValueError('bond_mean must be symmetric on bonded pairs.')
            elif not self.bond_mask_by_type[types, atoms_i, atoms_j].all():
                raise ValueError('No bond target for an element/bond-type pair; supply graph["bond_mean"] '
                                 'or bond-type-aware statistics.')
            if 'bond_std' in graph:
                targets = self._tensor(graph, 'bond_std', x.device, shape=bonds.shape)
                std = targets[batch, i, j]
                if not torch.equal(std, targets[batch, j, i]):
                    raise ValueError('bond_std must be symmetric on bonded pairs.')
            if not (torch.isfinite(mean).all() and (mean > 0).all()
                    and torch.isfinite(std).all() and (std > 0).all()):
                raise ValueError('Bond means and standard deviations must be positive and finite.')
            distances = self._norm(x[batch, i] - x[batch, j])
            components['bond'] = self._average(self._huber((distances - mean) / std.clamp(min=0.03)), batch, zeros)

        if self.clash_weight > 0:
            # Matrix multiplication counts shared covalent neighbors (1-3 pairs).
            one_three = torch.bmm(connected.float(), connected.float()) > 0
            batch, i, j = (pair_mask & unique & ~connected & ~one_three).nonzero(as_tuple=True)
            radii = self.vdw_radii[atom_types.clamp(min=0, max=119)]
            minimum = 0.7 * (radii[batch, i] + radii[batch, j])
            overlap = F.relu(minimum - self._norm(x[batch, i] - x[batch, j]))
            components['clash'] = self._average(self._huber(overlap), batch, zeros)

        if self.angle_weight > 0:
            batch, rows, indices, shape = self._tuples(graph, 'angles', 3, mask)
            if not (connected[batch, indices[:, 0], indices[:, 1]].all()
                    and connected[batch, indices[:, 1], indices[:, 2]].all()):
                raise ValueError('angles must describe consecutively bonded atoms.')
            target = self._targets(graph, 'angle_cos', shape, batch, rows, x.device)
            if ((target < -1) | (target > 1)).any():
                raise ValueError('angle_cos targets must lie in [-1,1].')
            points = x[batch.unsqueeze(-1), indices]
            first, second = points[:, 0] - points[:, 1], points[:, 2] - points[:, 1]
            cosine = (first * second).sum(-1) / (self._norm(first) * self._norm(second))
            components['angle'] = self._average(self._huber(cosine.clamp(-1, 1) - target), batch, zeros)

        if self.ring_weight > 0:
            batch, _, indices, _ = self._tuples(graph, 'planarity', 4, mask)
            points = x[batch.unsqueeze(-1), indices]
            normal = torch.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0], dim=-1)
            height = (normal * (points[:, 3] - points[:, 0])).sum(-1) / self._norm(normal)
            components['planarity'] = self._average(self._huber(height), batch, zeros)

        if self.torsion_weight > 0:
            batch, rows, indices, shape = self._tuples(graph, 'torsions', 4, mask)
            for edge in range(3):
                if not connected[batch, indices[:, edge], indices[:, edge + 1]].all():
                    raise ValueError('torsions must describe four consecutively bonded atoms.')
            periodicity = self._tensor(graph, 'torsion_periodicity', x.device, shape=shape, integer=True)[batch, rows]
            phase = self._targets(graph, 'torsion_phase', shape, batch, rows, x.device)
            amplitude = self._targets(graph, 'torsion_amplitude', shape, batch, rows, x.device)
            if (periodicity <= 0).any() or (periodicity > 12).any() or (amplitude < 0).any():
                raise ValueError('Torsion periodicities must be integers 1..12 and amplitudes nonnegative.')
            points = x[batch.unsqueeze(-1), indices]
            first, central, last = points[:, 1] - points[:, 0], points[:, 2] - points[:, 1], points[:, 3] - points[:, 2]
            n1, n2 = torch.cross(first, central, dim=-1), torch.cross(central, last, dim=-1)
            denominator = self._norm(n1) * self._norm(n2)
            cosine = ((n1 * n2).sum(-1) / denominator).clamp(-1, 1)
            sine = (torch.cross(n1, n2, dim=-1) * (central / self._norm(central).unsqueeze(-1))).sum(-1) / denominator
            cos_n, sin_n = torch.ones_like(cosine), torch.zeros_like(sine)
            chosen_cos, chosen_sin = torch.zeros_like(cosine), torch.zeros_like(sine)
            max_order = int(periodicity.max().item()) if periodicity.numel() else 0
            for order in range(1, max_order + 1):
                cos_n, sin_n = cos_n * cosine - sin_n * sine, sin_n * cosine + cos_n * sine
                chosen_cos = torch.where(periodicity == order, cos_n, chosen_cos)
                chosen_sin = torch.where(periodicity == order, sin_n, chosen_sin)
            energy = amplitude * (1 - chosen_cos * torch.cos(phase) - chosen_sin * torch.sin(phase)).clamp(min=0)
            components['torsion'] = self._average(energy, batch, zeros)

        if self.chirality_weight > 0:
            batch, rows, indices, shape = self._tuples(graph, 'chirality', 4, mask)
            sign = self._targets(graph, 'chirality_sign', shape, batch, rows, x.device)
            if ((sign != -1) & (sign != 1)).any():
                raise ValueError('chirality_sign must contain +/-1 on valid tuples.')
            points = x[batch.unsqueeze(-1), indices]
            a, b, c = points[:, 0] - points[:, 3], points[:, 1] - points[:, 3], points[:, 2] - points[:, 3]
            volume = (a * torch.cross(b, c, dim=-1)).sum(-1) / 6
            deficit = F.relu(self.min_chirality_volume - sign * volume)
            components['chirality'] = self._average(self._huber(deficit), batch, zeros)
