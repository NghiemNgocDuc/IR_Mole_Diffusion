"""Integration checks for graph mapping, learning, sampling, and portability."""
import json
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from diffusion.data import NativeNPZDataset, collate_native, load_native_npz
from diffusion.evaluation import kabsch_rmsd
from diffusion.model import EquivariantDiffusion3D_v4
from diffusion.preprocessing import aligned_bond_types, bond_array, pack_peaks
from diffusion.spectrum import SpectrumStructureModel
from torch.utils.data import DataLoader


def small_batch():
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    graph = torch.zeros(2, 4, 4, dtype=torch.long)
    for row in range(2):
        for atom in range(1, int(mask[row].sum())):
            graph[row, atom-1, atom] = graph[row, atom, atom-1] = 1
    return {"coords": torch.randn(2, 4, 3) * mask.unsqueeze(-1),
        "atom_types": torch.where(mask, torch.full((2, 4), 6), 0),
        "num_atoms": mask.sum(1), "mask": mask, "graph": {"bond_types": graph},
        "spectra": {"packed": torch.tensor([[[.1, .2], [.3, .7], [0., 0.]],
                                             [[.2, .5], [.4, .8], [.6, .3]]]),
                    "mask": torch.tensor([[True, True, False], [True, True, True]])},
        "selfies_tokens": torch.tensor([[1, 3, 2, 0], [1, 3, 3, 2]])}


def tiny_model(**kwargs):
    return EquivariantDiffusion3D_v4(encoder_spec={"hidden_dim": 16, "n_layers": 1,
        "n_heads": 4, "dropout": 0.0}, hidden_dim=16, n_layers=2, k_neighbors=4,
        n_timesteps=12, unfreeze_at_epoch=0, warmup_steps=0, **kwargs)


def test_centered_forward_roundtrip_and_single_atom():
    model, batch = tiny_model(), small_batch()
    coords, mask = batch["coords"], batch["mask"]
    times = torch.tensor([0, 11])
    noisy, noise = model.forward_diffusion(coords, times, mask)
    torch.testing.assert_close(noise.sum(1), torch.zeros(2, 3), atol=2e-6, rtol=0)
    assert noise[~mask].abs().sum() == 0
    expected = coords - (coords.sum(1, keepdim=True) / mask.sum(1)[:, None, None])
    expected = expected * mask.unsqueeze(-1)
    torch.testing.assert_close(model.predict_x0_from_noise(noisy, times, noise) * mask.unsqueeze(-1), expected, atol=3e-4, rtol=1e-4)
    singleton = torch.randn(1, 1, 3)
    x, epsilon = model.forward_diffusion(singleton, torch.tensor([5]))
    assert x.abs().sum() == epsilon.abs().sum() == 0


def test_loss_gradient_and_spectral_mask():
    model, batch = tiny_model(), small_batch()
    total, _, _ = model.loss(batch)
    total.backward()
    assert torch.isfinite(total)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.denoiser.parameters())
    model.eval()
    first = model.condition(batch["spectra"])[0]
    altered = {key: value.clone() for key, value in batch["spectra"].items()}
    altered["packed"][~altered["mask"]] = float("nan")
    torch.testing.assert_close(first, model.condition(altered)[0])


def test_sampling_ddim_ddpm_guidance_and_padding():
    model, batch = tiny_model().eval(), small_batch()
    for sampler, steps, guidance in (("ddim", 4, 0.), ("ddpm", 12, 0.), ("ddim", 4, .01)):
        coords, mask = model.sample(batch["spectra"], batch["atom_types"], batch["num_atoms"],
            n_steps=steps, sampler=sampler, graph=batch["graph"], physics_guidance=guidance)
        assert torch.isfinite(coords).all()
        assert coords[~mask].abs().sum() == 0
        torch.testing.assert_close(coords.sum(1), torch.zeros(2, 3), atol=1e-3, rtol=0)
    with pytest.raises(ValueError, match="DDPM"):
        model.sample(batch["spectra"], batch["atom_types"], batch["num_atoms"], n_steps=4, sampler="ddpm")


def test_verified_mapping_survives_atom_permutation():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    AllChem.MMFFOptimizeMolecule(molecule)
    coords = molecule.GetConformer().GetPositions()
    atoms = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    permutation = np.random.default_rng(5).permutation(len(atoms))
    mapped = aligned_bond_types("CCO", atoms[permutation], coords[permutation])
    np.testing.assert_array_equal(mapped, bond_array(molecule)[np.ix_(permutation, permutation)])
    bad = coords[permutation].copy()
    bad[0] += 100
    with pytest.raises(ValueError, match="connectivity"):
        aligned_bond_types("CCO", atoms[permutation], bad)


def test_rmsd_proper_rotation():
    xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    assert kabsch_rmsd(xyz, xyz @ rotation + 7) < 1e-10
    assert kabsch_rmsd(xyz, xyz * [-1, 1, 1]) > .1


def test_real_optimizer_checkpoint_reload_resume(tmp_path):
    batch = small_batch()
    # A dataset of already-batched values keeps the integration check small.
    loader = DataLoader([batch, batch], batch_size=None)
    model = tiny_model(ema_decay=.9)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, num_sanity_val_steps=0)
    trainer.fit(model, loader, loader)
    checkpoint = tmp_path / "portable.ckpt"
    trainer.save_checkpoint(checkpoint)
    reloaded = EquivariantDiffusion3D_v4.load_from_checkpoint(checkpoint)
    assert reloaded.use_ema_weights()
    assert reloaded.encoder.spec == model.encoder.spec
    assert trainer.global_step == 2
    resumed = EquivariantDiffusion3D_v4.load_from_checkpoint(checkpoint)
    trainer2 = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, num_sanity_val_steps=0)
    trainer2.fit(resumed, loader, loader, ckpt_path=checkpoint)
    assert trainer2.global_step == 4
    legacy = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": model.state_dict()}, legacy)
    with pytest.raises(ValueError, match="Legacy"):
        EquivariantDiffusion3D_v4.load_from_checkpoint(legacy)


def test_spectrum_stage_learns_and_generates_protocol():
    torch.manual_seed(8)
    model = SpectrumStructureModel(vocabulary=["<pad>", "<bos>", "<eos>", "[C]"],
        encoder_spec={"hidden_dim": 16, "n_layers": 1, "n_heads": 4, "dropout": 0.0},
        decoder_layers=1, heads=4, max_length=4, dropout=0., label_smoothing=0.)
    batch = small_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    def loss():
        logits = model(batch["spectra"], batch["selfies_tokens"][:, :-1])
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), batch["selfies_tokens"][:, 1:], ignore_index=0)
    start = float(loss().detach())
    for _ in range(30):
        optimizer.zero_grad()
        value = loss()
        value.backward()
        optimizer.step()
    assert float(loss().detach()) < start * .5
    spectrum = {key: value[:1] for key, value in batch["spectra"].items()}
    candidates = model.eval().generate(spectrum, beam_size=2)
    assert candidates and all("<" not in item["selfies"] for item in candidates)
    from predict_structure import predict
    records, molecules, arrays = predict(model, tiny_model().eval(), spectrum,
        beam_size=2, conformers=1, n_steps=4, refine=False, physics_guidance=0.)
    assert records and molecules and arrays
    assert all(np.isfinite(value).all() for value in arrays.values())


def test_consistent_schedule_and_encoder_unfreeze():
    model = tiny_model()
    torch.testing.assert_close(model.alphas_cumprod, torch.cumprod(1-model.betas, 0), atol=1e-7, rtol=1e-5)
    assert model.alphas_cumprod[-1] >= 9e-6
    model._set_encoder_trainable(False)
    model.train()
    assert not model.encoder.training
    model._set_encoder_trainable(True)
    assert model.encoder.training


def test_spectral_control_stays_within_split_and_changes_identity():
    identifiers = np.array(["a", "a", "b", "c", "test-only"])
    data = {"coords": np.zeros((5, 2, 3)), "molecule_ids": identifiers, "bond_types": None}
    selected = np.array([0, 1, 2, 3])
    dataset = NativeNPZDataset(data, selected, spectrum_control="shuffled", seed=7)
    assert set(dataset.spectrum_indices).issubset(set(selected))
    assert np.all(identifiers[dataset.spectrum_indices] != identifiers[selected])
