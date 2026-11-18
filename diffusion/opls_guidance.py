"""Opt-in, late clean-coordinate OPLS corrections. No early rejection policy."""
import math
import numpy as np
import torch


class OPLSGuidance:
    def __init__(self, clients, min_alpha=.8, max_step_angstrom=.02):
        if not math.isfinite(min_alpha) or not 0 < min_alpha <= 1:
            raise ValueError('OPLS min_alpha must be in (0, 1]')
        if not math.isfinite(max_step_angstrom) or max_step_angstrom < 0:
            raise ValueError('OPLS step cap must be finite and nonnegative')
        self.clients = list(clients)
        self.min_alpha, self.max_step = min_alpha, max_step_angstrom
        self.trace = []

    def validate(self, atoms, counts, graph):
        if len(self.clients) != len(atoms) or 'bond_types' not in graph:
            raise ValueError('One OPLS graph backend per batch member required')
        for i, client in enumerate(self.clients):
            n = int(counts[i])
            if (atoms[i, :n].tolist() != client.metadata['atom_numbers'] or
                    graph['bond_types'][i, :n, :n].tolist() != client.metadata['bond_types']):
                raise ValueError('Sampler graph/atom order differs from parameterized OPLS graph')

    def correct(self, coordinates, mask, *, alpha, timestep):
        self.last_input = coordinates
        self.last_output = coordinates
        if alpha < self.min_alpha or self.max_step == 0:
            return coordinates
        result = coordinates.clone()
        for row, client in enumerate(self.clients):
            xyz = coordinates[row, mask[row]].detach().cpu().double().numpy()
            response = client.request('correct', coordinates=xyz.tolist(), max_step=self.max_step)
            corrected = np.asarray(response['coordinates'], dtype=float)
            if corrected.shape != xyz.shape or not np.isfinite(corrected).all():
                raise FloatingPointError('OPLS worker returned invalid coordinates')
            if np.linalg.norm(corrected-xyz, axis=1).max() > self.max_step + 1e-7:
                raise FloatingPointError('OPLS correction exceeds displacement budget')
            if not np.allclose(corrected.mean(0), xyz.mean(0), atol=1e-7, rtol=0):
                raise FloatingPointError('OPLS correction changed coordinate centroid')
            result[row, mask[row]] = torch.as_tensor(corrected, dtype=coordinates.dtype, device=coordinates.device)
            self.trace.append({'timestep': timestep, 'alpha': alpha, 'batch_row': row, **response['record']})
        self.last_output = result
        return result
