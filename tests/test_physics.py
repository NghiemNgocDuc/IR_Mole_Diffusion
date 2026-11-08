"""Behavioral checks for graph-defined geometry, exclusions, and gradients."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from diffusion.physics import PhysicsLoss


def physics(**kwargs):
    options = dict(bond_weight=0, clash_weight=0, angle_weight=0,
                   ring_weight=0, torsion_weight=0, chirality_weight=0)
    options.update(kwargs)
    return PhysicsLoss(**options)


def inputs(points, mask=None, dtype=torch.float32):
    coords = torch.tensor(points, dtype=dtype).unsqueeze(0).requires_grad_(True)
    valid = torch.ones(coords.shape[:2], dtype=torch.bool) if mask is None else torch.tensor([mask], dtype=torch.bool)
    atoms = torch.where(valid, torch.full(valid.shape, 6), torch.zeros_like(valid, dtype=torch.long))
    return coords, atoms, valid


def graph(n_atoms, edges=()):
    bond_types = torch.zeros(1, n_atoms, n_atoms, dtype=torch.long)
    for i, j, order in edges:
        bond_types[0, i, j] = bond_types[0, j, i] = order
    return {'bond_types': bond_types}


class PhysicsLossTests(unittest.TestCase):
    def test_graph_is_required_even_for_clashes(self):
        args = inputs([[0, 0, 0], [1.54, 0, 0]])
        for loss in (physics(bond_weight=1), physics(clash_weight=1)):
            with self.assertRaisesRegex(ValueError, 'explicit graph'):
                loss(*args)

    def test_disabled_loss_is_differentiable_and_ignores_padding(self):
        args = inputs([[0, 0, 0], [float('nan'), 1, 2]], mask=[True, False])
        total, components = physics()(*args, return_components=True)
        self.assertEqual(total.item(), 0)
        self.assertEqual(total.ndim, 0)
        self.assertTrue(all(value.item() == 0 for value in components.values()))
        total.backward()
        self.assertTrue(torch.isfinite(args[0].grad).all())
        self.assertTrue((args[0].grad == 0).all())

    def test_bond_targets_have_no_self_penalty_or_proximity_inference(self):
        args = inputs([[0, 0, 0], [1.54, 0, 0], [0.2, 0.1, 0]])
        topology = graph(3, [(0, 1, 1)])
        total = physics(bond_weight=1)(*args, graph=topology)
        self.assertLess(total.item(), 1e-8)
        total.backward()
        self.assertTrue((args[0].grad[0, 2] == 0).all())
        self.assertEqual(physics(bond_weight=1)(*args, graph=graph(3)).item(), 0)

    def test_stretched_bond_retains_restoring_gradient(self):
        args = inputs([[0, 0, 0], [4, 0, 0]])
        loss = physics(bond_weight=1)(*args, graph=graph(2, [(0, 1, 1)]))
        self.assertGreater(loss.item(), 1)
        loss.backward()
        self.assertGreater(args[0].grad[0, 1, 0].item(), 0)
        self.assertLess(args[0].grad[0, 0, 0].item(), 0)

    def test_bond_order_and_explicit_targets(self):
        args = inputs([[0, 0, 0], [1.34, 0, 0]])
        loss = physics(bond_weight=1)
        self.assertLess(loss(*args, graph=graph(2, [(0, 1, 2)])).item(), 1e-8)
        self.assertGreater(loss(*args, graph=graph(2, [(0, 1, 1)])).item(), 1)
        topology = graph(2, [(0, 1, 1)])
        topology['bond_mean'] = torch.tensor([[[0, 1.34], [1.34, 0]]])
        topology['bond_std'] = torch.tensor([[[0, 0.05], [0.05, 0]]])
        self.assertLess(loss(*args, graph=topology).item(), 1e-8)

    def test_reductions_normalize_each_molecule(self):
        coords = torch.tensor([[[0., 0, 0], [2.54, 0, 0], [0, 0, 0], [0, 0, 0]],
                               [[0., 0, 0], [2.54, 0, 0], [5.08, 0, 0], [7.62, 0, 0]]])
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        atoms = torch.where(mask, torch.full(mask.shape, 6), torch.zeros_like(mask, dtype=torch.long))
        bonds = torch.zeros(2, 4, 4, dtype=torch.long)
        for batch, edges in enumerate(([(0, 1)], [(0, 1), (1, 2), (2, 3)])):
            for i, j in edges:
                bonds[batch, i, j] = bonds[batch, j, i] = 1
        loss = physics(bond_weight=1)
        each, components = loss(coords, atoms, mask, graph={'bond_types': bonds}, reduction='none', return_components=True)
        torch.testing.assert_close(each[0], each[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(each, components['bond'])
        torch.testing.assert_close(loss(coords, atoms, mask, graph={'bond_types': bonds}), each.mean())
        torch.testing.assert_close(loss(coords, atoms, mask, graph={'bond_types': bonds}, reduction='sum'), each.sum())

    def test_clashes_exclude_bonds_and_shared_neighbors(self):
        loss = physics(clash_weight=1)
        args = inputs([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
        self.assertEqual(loss(*args, graph=graph(3, [(0, 1, 1), (1, 2, 1)])).item(), 0)
        self.assertGreater(loss(*args, graph=graph(3)).item(), 0)
        args4 = inputs([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
        self.assertGreater(loss(*args4, graph=graph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])).item(), 0)

    def test_angles_use_only_bonded_tuples_and_explicit_cosines(self):
        args = inputs([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
        topology = graph(3, [(0, 1, 1), (1, 2, 1)])
        topology.update(angles=torch.tensor([[[0, 1, 2], [-1, -1, -1]]]),
                        angle_cos=torch.tensor([[0., float('nan')]]))
        self.assertEqual(physics(angle_weight=1)(*args, graph=topology).item(), 0)
        topology['angle_cos'][0, 0] = -0.5
        self.assertGreater(physics(angle_weight=1)(*args, graph=topology).item(), 0)
        topology['bond_types'].zero_()
        with self.assertRaisesRegex(ValueError, 'consecutively bonded'):
            physics(angle_weight=1)(*args, graph=topology)

    def test_planarity_is_explicit_and_does_not_flatten_all_carbon(self):
        args = inputs([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        topology = graph(4)
        topology['planarity'] = torch.empty(1, 0, 4, dtype=torch.long)
        self.assertEqual(physics(ring_weight=1)(*args, graph=topology).item(), 0)
        topology['planarity'] = torch.tensor([[[0, 1, 2, 3]]])
        self.assertGreater(physics(ring_weight=1)(*args, graph=topology).item(), 0)

    def test_periodic_torsion_has_multiple_minima_and_radian_phase(self):
        topology = graph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        topology.update(torsions=torch.tensor([[[0, 1, 2, 3]]]),
                        torsion_periodicity=torch.tensor([[2]]),
                        torsion_phase=torch.tensor([[0.]]),
                        torsion_amplitude=torch.tensor([[1.]]))
        loss = physics(torsion_weight=1)
        for last in ([1, 1, 0], [1, -1, 0]):
            args = inputs([[0, 1, 0], [0, 0, 0], [1, 0, 0], last])
            self.assertLess(loss(*args, graph=topology).item(), 1e-7)
        args = inputs([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]])
        self.assertGreater(loss(*args, graph=topology).item(), 1.9)
        topology['torsion_periodicity'][:] = 1
        topology['torsion_phase'][:] = torch.pi / 2
        self.assertLess(loss(*args, graph=topology).item(), 1e-7)

    def test_chirality_distinguishes_reflection(self):
        args = inputs([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        topology = graph(4)
        topology.update(chirality=torch.tensor([[[0, 1, 2, 3]]]), chirality_sign=torch.tensor([[1.]]))
        loss = physics(chirality_weight=1)
        self.assertEqual(loss(*args, graph=topology).item(), 0)
        mirrored = args[0] * torch.tensor([-1., 1, 1])
        self.assertGreater(loss(mirrored, args[1], args[2], graph=topology).item(), 0)

    def test_degenerate_geometry_has_finite_fp32_loss_and_gradient(self):
        args = inputs([[0, 0, 0]] * 4, dtype=torch.float16)
        topology = graph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        topology.update(angles=torch.tensor([[[0, 1, 2]]]), angle_cos=torch.tensor([[-0.5]]),
                        planarity=torch.tensor([[[0, 1, 2, 3]]]),
                        torsions=torch.tensor([[[0, 1, 2, 3]]]),
                        torsion_periodicity=torch.tensor([[3]]), torsion_phase=torch.tensor([[0.]]),
                        torsion_amplitude=torch.tensor([[1.]]),
                        chirality=torch.tensor([[[0, 1, 2, 3]]]), chirality_sign=torch.tensor([[1.]]))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            total = physics(bond_weight=1, clash_weight=1, angle_weight=1,
                            ring_weight=1, torsion_weight=1, chirality_weight=1)(*args, graph=topology)
        self.assertEqual(total.dtype, torch.float32)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(args[0].grad).all())

    def test_geometric_loss_is_invariant_to_rotation_and_translation(self):
        args = inputs([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0.5, 0.7]])
        topology = graph(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        loss = physics(bond_weight=1, clash_weight=1)
        rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        transformed = args[0] @ rotation.T + torch.tensor([10., -5, 3])
        torch.testing.assert_close(loss(*args, graph=topology),
                                   loss(transformed, args[1], args[2], graph=topology))

    def test_invalid_graph_and_missing_active_targets_fail_clearly(self):
        args = inputs([[0, 0, 0], [1.54, 0, 0], [0, 1, 0]])
        topology = graph(3, [(0, 1, 1)])
        topology['bond_types'][0, 1, 0] = 0
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            physics(bond_weight=1)(*args, graph=topology)
        with self.assertRaisesRegex(ValueError, 'angles'):
            physics(angle_weight=1)(*args, graph=graph(3))
        topology = graph(3)
        topology.update(angles=torch.tensor([[[0, 1, -1]]]), angle_cos=torch.tensor([[0.]]))
        with self.assertRaisesRegex(ValueError, 'partially padded'):
            physics(angle_weight=1)(*args, graph=topology)

    def test_partial_statistics_retain_bond_order_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bonds.json'
            path.write_text(json.dumps({'6-6-2': {'mean': 1.30, 'std': 0.04},
                                        '6-7-2': {'mean': -1, 'std': 0.04}}), encoding='utf-8')
            with self.assertWarnsRegex(UserWarning, 'Invalid'):
                loss = physics(bond_weight=1, bond_stats_path=path)
            self.assertAlmostEqual(loss.bond_mean_by_type[2, 6, 6].item(), 1.30, places=6)
            self.assertAlmostEqual(loss.bond_mean_by_type[1, 6, 6].item(), 1.54, places=6)
            self.assertAlmostEqual(loss.bond_mean_by_type[2, 6, 7].item(), 1.28, places=6)
            path.write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'object'):
                physics(bond_stats_path=path)

    def test_legacy_statistics_do_not_overwrite_double_bonds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bonds.json'
            path.write_text(json.dumps({'6-6': {'mean': 1.50, 'std': 0.08}}), encoding='utf-8')
            with self.assertWarnsRegex(UserWarning, 'single bonds only'):
                loss = physics(bond_stats_path=path)
            self.assertAlmostEqual(loss.bond_mean_by_type[1, 6, 6].item(), 1.50, places=6)
            self.assertAlmostEqual(loss.bond_mean_by_type[2, 6, 6].item(), 1.34, places=6)


if __name__ == '__main__':
    unittest.main()
