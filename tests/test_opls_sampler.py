"""Sampler semantics, topology binding and no-op reproducibility."""
import numpy as np
import pytest
import torch
from diffusion.model import EquivariantDiffusion3D_v4
from diffusion.opls_guidance import OPLSGuidance
from diffusion.opls_bridge import MappedEnergy, bounded_correction
from diffusion.opls_backend import declared_graph, map_export
from diffusion.utils import masked_center, centered_randn_like, sampling_timesteps


def setup(scale=2.5):
    torch.manual_seed(2)
    model = EquivariantDiffusion3D_v4(encoder_spec={'hidden_dim': 16, 'n_layers': 1, 'n_heads': 4, 'dropout': 0.},
        hidden_dim=16, n_layers=1, k_neighbors=4, n_timesteps=8, coordinate_scale=scale).eval()
    spectra = {'packed': torch.tensor([[[.1,.4],[.2,.5]]]), 'mask': torch.ones(1,2,dtype=torch.bool)}
    atoms = torch.tensor([[6,6,8,0]])
    counts = torch.tensor([3])
    bonds = torch.zeros(1,4,4,dtype=torch.long)
    bonds[0,0,1] = bonds[0,1,0] = bonds[0,1,2] = bonds[0,2,1] = 1
    return model, spectra, atoms, counts, {'bond_types': bonds}


class HarmonicClient:
    # Analytic stand-in for testing control flow, never used by the real pilot.
    metadata = {'atom_numbers': [6,6,8], 'bond_types': [[0,1,0],[1,0,1],[0,1,0]]}
    def request(self, op, **kwargs):
        assert op == 'correct'
        bridge = MappedEnergy(lambda x: (float((x*x).sum()), -2*x), [0,1,2])
        x, record = bounded_correction(bridge, kwargs['coordinates'], kwargs['max_step'], preserve_centroid=True)
        return {'coordinates': x.tolist(), 'record': record}


def test_zero_strength_is_bitwise_unchanged_and_gate_is_lazy():
    model, spectra, atoms, counts, graph = setup()
    class Never(HarmonicClient):
        def request(self, *a, **kw):
            pytest.fail('No engine request expected')
    options = dict(n_steps=4, graph=graph)
    expected, _ = model.sample(spectra, atoms, counts, generator=torch.Generator().manual_seed(5), **options)
    for policy in (OPLSGuidance([Never()], max_step_angstrom=0), OPLSGuidance([Never()], min_alpha=1)):
        result, _ = model.sample(spectra, atoms, counts, generator=torch.Generator().manual_seed(5), opls_guidance=policy, **options)
        assert torch.equal(expected, result)
        assert not policy.trace


@pytest.mark.parametrize('sampler,steps', [('ddim',4), ('ddpm',8)])
def test_clean_coordinate_transition_and_self_conditioning(sampler, steps):
    model, spectra, atoms, counts, graph = setup()
    actual_inputs = []
    def predictor(x, atoms, t, condition, mask, graph, previous):
        actual_inputs.append((x.clone(), None if previous is None else previous.clone()))
        return torch.zeros_like(x)
    model._predict = predictor
    policy = OPLSGuidance([HarmonicClient()], min_alpha=.7)
    result, mask = model.sample(spectra, atoms, counts, sampler=sampler, n_steps=steps, graph=graph,
        generator=torch.Generator().manual_seed(9), opls_guidance=policy)
    generator = torch.Generator().manual_seed(9)
    x = centered_randn_like(torch.zeros(1,4,3), mask, generator)
    previous = None
    schedule = sampling_timesteps(8, steps, sampler)
    independent_policy = OPLSGuidance([HarmonicClient()], min_alpha=.7)
    for index, time in enumerate(schedule):
        torch.testing.assert_close(actual_inputs[index][0], x)
        if previous is not None:
            torch.testing.assert_close(actual_inputs[index][1], previous)
        a = model.alphas_cumprod[time]
        x0 = masked_center(x/a.sqrt(), mask)
        physical = x0*2.5
        corrected = independent_policy.correct(physical, mask, alpha=float(a), timestep=time)
        noise = torch.zeros_like(x)
        if not torch.equal(physical, corrected):
            x0 = masked_center(corrected/2.5, mask)
            noise = masked_center((x-a.sqrt()*x0)/(1-a).sqrt(), mask)
        if sampler == 'ddpm':
            x = model.posterior_mean_coef1[time]*x0 + model.posterior_mean_coef2[time]*x
            if time > 0:
                x += model.posterior_variance[time].sqrt()*centered_randn_like(x, mask, generator)
        else:
            ap = model.alphas_cumprod[schedule[index+1]] if index+1 < len(schedule) else torch.tensor(1.)
            x = ap.sqrt()*x0 + (1-ap).sqrt()*noise
        x = masked_center(x, mask)
        previous = x0
    torch.testing.assert_close(result, x*2.5, atol=1e-5, rtol=1e-5)
    assert result[~mask].abs().sum() == 0 and policy.trace
    assert all(t['alpha'] >= .7 and t['energy_after'] < t['energy_before'] for t in policy.trace)


def test_graph_mismatch_and_mixed_guidance_rejected():
    model, spectra, atoms, counts, graph = setup()
    policy = OPLSGuidance([HarmonicClient()])
    with pytest.raises(ValueError, match='physics_guidance=0'):
        model.sample(spectra, atoms, counts, graph=graph, physics_guidance=.01, opls_guidance=policy)
    atoms[0,0] = 7
    with pytest.raises(ValueError, match='atom order'):
        model.sample(spectra, atoms, counts, graph=graph, opls_guidance=policy)


@pytest.mark.parametrize('smiles', ['[Na+]', 'C.C', '[CH3]', '[13CH4]', 'N[C@H](C)O', 'F/C=C/F'])
def test_unsupported_graphs_are_explicit(smiles):
    with pytest.raises(ValueError):
        declared_graph(smiles)


def test_mapping_declared_graph_not_noisy_coordinates(tmp_path):
    import zipfile
    from pathlib import Path
    archive = Path('opls_ethanol_review.zip')
    if not archive.exists():
        pytest.skip('Archived fixture unavailable')
    with zipfile.ZipFile(archive) as z:
        content = z.read(next(n for n in z.namelist() if n.endswith('ETH.openmm.pdb')))
    pdb = tmp_path/'ethanol.pdb'
    pdb.write_bytes(content)
    mapping = map_export(declared_graph('CCO'), pdb)
    assert sorted(mapping) == list(range(9))
    with pytest.raises(ValueError, match='does not match'):
        map_export(declared_graph('COC'), pdb)
