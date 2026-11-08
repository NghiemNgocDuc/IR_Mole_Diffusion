"""Non-vacuous invariance and conditioning tests for the coordinate predictor."""

import unittest

import torch

from diffusion.denoiser import EquivariantDenoiser
from diffusion.layers import SinusoidalTimeEmb, StableEGNNLayer
from diffusion.utils import masked_center


class DenoiserTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        self.x = torch.randn(2, 6, 3)
        self.atom_types = torch.tensor([[6, 6, 7, 8, 1, 0], [6, 7, 6, 8, 1, 1]])
        self.mask = self.atom_types != 0
        self.t = torch.tensor([4, 13])
        self.condition = torch.randn(2, 12)
        self.self_cond = torch.randn_like(self.x)
        self.tokens = torch.randn(2, 4, 12)
        self.token_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        self.bonds = torch.zeros(2, 6, 6, dtype=torch.long)
        self.bonds[:, 0, 2] = self.bonds[:, 2, 0] = 1

    def model(self, **kwargs):
        options = dict(cond_dim=12, hidden_dim=24, time_dim=7, n_layers=3,
                       k_neighbors=2, self_cond_dim=8, attention_heads=4)
        options.update(kwargs)
        model = EquivariantDenoiser(**options).eval()
        # Activate every coordinate head: zero initialization would trivially pass.
        with torch.no_grad():
            for layer in model.egnn_layers:
                if layer.update_coords:
                    layer.coord_mlp[-2].weight.normal_(0, 0.08)
                    layer.coord_mlp[-2].bias.normal_(0, 0.08)
            model.output_scale[-1].weight.normal_(0, 0.08)
            model.output_scale[-1].bias.normal_(0, 0.08)
            if model.readout == 'vector':
                for head in (model.vector_mlp, model.radial_mlp):
                    head[-1].weight.normal_(0, 0.08)
                    head[-1].bias.normal_(0, 0.08)
        return model

    def forward(self, model, x=None, atom_types=None, mask=None, self_cond=None,
                tokens=None, token_mask=None, graph=None):
        return model(
            self.x if x is None else x,
            self.atom_types if atom_types is None else atom_types,
            self.t, self.condition,
            self.mask if mask is None else mask,
            self.self_cond if self_cond is None else self_cond,
            condition_tokens=self.tokens if tokens is None else tokens,
            condition_mask=self.token_mask if token_mask is None else token_mask,
            graph=graph,
        )

    def test_active_heads_rotation_reflection_translation_and_centering(self):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        for readout in ('vector', 'residual'):
            for mode in ('pooled', 'node', 'node_edge'):
                with self.subTest(readout=readout, mode=mode):
                    model = self.model(readout=readout, conditioning_mode=mode)
                    output = self.forward(model)
                    self.assertGreater(output.abs().max().item(), 1e-3)
                    torch.testing.assert_close(output.sum(1), torch.zeros(2, 3), atol=2e-6, rtol=0)
                    self.assertEqual(output[~self.mask].abs().sum().item(), 0)
                    for transform in (rotation, rotation @ torch.diag(torch.tensor([-1., 1., 1.]))):
                        actual = self.forward(
                            model, x=self.x @ transform.T,
                            self_cond=self.self_cond @ transform.T,
                        )
                        torch.testing.assert_close(actual, output @ transform.T, atol=2e-5, rtol=2e-5)
                    translation = torch.tensor([7., -3., 11.])
                    translated = self.forward(model, x=self.x + translation,
                                              self_cond=self.self_cond + translation)
                    torch.testing.assert_close(translated, output, atol=2e-5, rtol=2e-5)

    def test_large_coordinates_still_rotate_equivariantly(self):
        model = self.model(conditioning_mode='node_edge')
        x = self.x * 90
        sc = self.self_cond * 90
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        output = self.forward(model, x=x, self_cond=sc)
        actual = self.forward(model, x=x @ rotation.T, self_cond=sc @ rotation.T)
        self.assertGreater(output.abs().max().item(), 1e-3)
        torch.testing.assert_close(actual, output @ rotation.T, atol=1e-3, rtol=3e-5)

    def test_graph_and_atom_permutation_equivariance(self):
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        model = self.model(conditioning_mode='node_edge', graph_conditioning=True)
        output = self.forward(model, graph={'bond_types': self.bonds})
        actual = self.forward(
            model, x=self.x[:, permutation], atom_types=self.atom_types[:, permutation],
            mask=self.mask[:, permutation], self_cond=self.self_cond[:, permutation],
            graph={'bond_types': self.bonds[:, permutation][:, :, permutation]},
        )
        self.assertGreater(output.abs().max().item(), 1e-3)
        torch.testing.assert_close(actual, output[:, permutation], atol=2e-5, rtol=2e-5)

    def test_atom_padding_invariance(self):
        for mode in ('pooled', 'node', 'node_edge'):
            with self.subTest(mode=mode):
                model = self.model(conditioning_mode=mode)
                output = self.forward(model)
                x = torch.cat([self.x, torch.randn(2, 3, 3) * 1e4], dim=1)
                sc = torch.cat([self.self_cond, torch.randn(2, 3, 3) * 1e4], dim=1)
                atoms = torch.cat([self.atom_types, torch.zeros(2, 3, dtype=torch.long)], dim=1)
                mask = torch.cat([self.mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
                actual = self.forward(model, x=x, atom_types=atoms, mask=mask, self_cond=sc)
                torch.testing.assert_close(actual[:, :6], output, atol=2e-5, rtol=2e-5)
                self.assertEqual(actual[:, 6:].abs().sum().item(), 0)

    def test_token_padding_and_empty_spectrum_are_safe(self):
        for mode in ('node', 'node_edge'):
            with self.subTest(mode=mode):
                model = self.model(conditioning_mode=mode)
                output = self.forward(model)
                tokens = torch.cat([self.tokens, torch.randn(2, 5, 12) * 1e4], dim=1)
                mask = torch.cat([self.token_mask, torch.zeros(2, 5, dtype=torch.bool)], dim=1)
                padded = self.forward(model, tokens=tokens, token_mask=mask)
                torch.testing.assert_close(padded, output, atol=2e-5, rtol=2e-5)
                all_masked = self.forward(model, token_mask=torch.zeros_like(self.token_mask))
                changed = self.forward(model, tokens=self.tokens * 1e4,
                                       token_mask=torch.zeros_like(self.token_mask))
                empty = self.forward(model, tokens=self.tokens[:, :0],
                                     token_mask=self.token_mask[:, :0])
                self.assertTrue(torch.isfinite(all_masked).all())
                torch.testing.assert_close(changed, all_masked, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(empty, all_masked, atol=1e-6, rtol=1e-6)
                changed_valid = self.tokens.clone()
                changed_valid[:, 0] += torch.randn(2, 12) * 4
                conditioned = self.forward(model, tokens=changed_valid)
                self.assertGreater((conditioned - output).abs().max().item(), 1e-4)

    def test_same_element_atom_queries_use_geometry(self):
        model = self.model(conditioning_mode='node', n_layers=1)
        x = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 3., 0.]]])
        queries = []
        handle = model.node_attn[0].register_forward_pre_hook(
            lambda module, args: queries.append(args[0].detach().clone())
        )
        try:
            model(x, torch.full((1, 3), 6), torch.tensor([5]), torch.randn(1, 12),
                  condition_tokens=torch.randn(1, 3, 12))
        finally:
            handle.remove()
        self.assertEqual(len(queries), 1)
        self.assertGreater((queries[0] - queries[0][:, :1]).abs().max().item(), 1e-4)

    def test_cutoff_excludes_distant_edges_and_self_edges(self):
        model = self.model(adaptive_cutoff=True, cutoff_radius=0.5, k_neighbors=4)
        x = torch.tensor([[[0., 0., 0.], [0.2, 0., 0.], [3., 0., 0.], [9., 0., 0.]]])
        mask = torch.tensor([[True, True, True, False]])
        indices, valid = model.compute_edges(x, mask, 4, return_mask=True)
        for atom in range(4):
            neighbors = set(indices[0, atom, valid[0, atom]].tolist())
            expected = {1} if atom == 0 else {0} if atom == 1 else set()
            self.assertEqual(neighbors, expected)
            self.assertNotIn(atom, neighbors)

    def test_all_bonds_survive_knn_and_cutoff(self):
        model = self.model(graph_conditioning=True, adaptive_cutoff=True,
                           cutoff_radius=0.5, k_neighbors=1)
        x = torch.tensor([[[0., 0., 0.], [0.1, 0., 0.], [3., 0., 0.], [9., 0., 0.]]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        bonds = torch.zeros(1, 4, 4, dtype=torch.long)
        bonds[0, 0, 2] = bonds[0, 2, 0] = 1
        bonds[0, 0, 3] = bonds[0, 3, 0] = 4
        indices, valid = model.compute_edges(x, mask, 1, return_mask=True,
                                             graph={'bond_types': bonds})
        self.assertEqual(set(indices[0, 0, valid[0, 0]].tolist()), {1, 2, 3})
        for source, destination in ((2, 0), (3, 0)):
            self.assertIn(destination, indices[0, source, valid[0, source]].tolist())

    def test_knn_boundary_ties_are_all_retained(self):
        model = self.model(k_neighbors=1)
        x = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]]])
        indices, valid = model.compute_edges(x, torch.ones(1, 3, dtype=torch.bool), 1,
                                             return_mask=True)
        self.assertEqual(set(indices[0, 0, valid[0, 0]].tolist()), {1, 2})

    def test_single_atom_and_no_valid_atoms(self):
        model = self.model(conditioning_mode='node_edge', graph_conditioning=True)
        for mask in (torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool)):
            x = torch.randn(2, 1, 3)
            output = model(x, torch.full((2, 1), 6), self.t, self.condition, mask,
                           self_cond=x + 10, condition_tokens=self.tokens,
                           condition_mask=self.token_mask,
                           graph={'bond_types': torch.zeros(2, 1, 1, dtype=torch.long)})
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(output.abs().sum().item(), 0)

    def test_vector_readout_is_not_hard_bounded(self):
        model = self.model()
        with torch.no_grad():
            for head in (model.vector_mlp, model.radial_mlp, model.output_scale):
                head[-1].weight.zero_()
                head[-1].bias.zero_()
            model.radial_mlp[-1].bias.fill_(3)
        x = self.x * 20
        output = self.forward(model, x=x)
        torch.testing.assert_close(output, 3 * masked_center(x, self.mask), atol=1e-4, rtol=1e-6)
        self.assertGreater(output.abs().max().item(), 30)

    def test_invalid_neighbors_cannot_affect_layer_messages(self):
        layer = StableEGNNLayer(8).eval()
        with torch.no_grad():
            layer.coord_mlp[-2].weight.normal_()
            layer.coord_mlp[-2].bias.normal_()
        h = torch.randn(1, 2, 8)
        x = torch.randn(1, 2, 3)
        indices = torch.tensor([[[1], [0]]])
        valid = torch.zeros(1, 2, 1, dtype=torch.bool)
        first_h, first_x = layer(h, x, indices, edge_mask=valid)
        # Every message is excluded: arbitrary edge index changes leave outputs fixed.
        second_h, second_x = layer(h, x, torch.zeros_like(indices), edge_mask=valid)
        torch.testing.assert_close(first_h, second_h)
        torch.testing.assert_close(first_x, x)
        torch.testing.assert_close(second_x, x)

    def test_token_gradients_are_finite_and_padding_has_no_gradient(self):
        model = self.model(conditioning_mode='node_edge')
        tokens = self.tokens.clone().requires_grad_(True)
        x = self.x.clone().requires_grad_(True)
        output = self.forward(model, x=x, tokens=tokens)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(tokens.grad[self.token_mask].abs().sum().item(), 1e-5)
        self.assertEqual(tokens.grad[~self.token_mask].abs().sum().item(), 0)
        self.assertEqual(x.grad[~self.mask].abs().sum().item(), 0)

    def test_configuration_and_graph_validation(self):
        for options in (dict(n_layers=0), dict(hidden_dim=1),
                        dict(conditioning_mode='bad'), dict(readout='bad'),
                        dict(conditioning_mode='node', hidden_dim=25),
                        dict(readout='residual', n_layers=1), dict(time_dim=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options)
        model = self.model(graph_conditioning=True)
        with self.assertRaisesRegex(ValueError, 'bond_types'):
            self.forward(model)
        invalid = self.bonds.clone()
        invalid[:, 0, 2] = 4
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            self.forward(model, graph={'bond_types': invalid})
        for dim in (1, 2, 3, 7):
            embedding = SinusoidalTimeEmb(dim)(self.t)
            self.assertEqual(embedding.shape, (2, dim))
            self.assertTrue(torch.isfinite(embedding).all())


if __name__ == '__main__':
    unittest.main()
