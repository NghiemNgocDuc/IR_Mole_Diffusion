from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


__all__ = [
    "QMugsDatasetV2",
    "QMugsDataModuleV2",
    "collate_variable_length",
    "peaklist_pack",
]

_DEFAULT_GRID_MIN = 400.0
_DEFAULT_GRID_MAX = 4000.0
_DEFAULT_GRID_POINTS = 3584


def peaklist_pack(x_original, intensities, max_peaks=81,
                  freq_scale=None, int_scale=None,
                  n_channels=2, return_mask=False, **kwargs):

    x = np.asarray(x_original, dtype=np.float32)
    y = (np.asarray(intensities, dtype=np.float32)
         if intensities is not None else np.ones_like(x))
    y = np.maximum(y, 0.0)

    # Keep only top-intensity peaks if too many
    if len(x) > max_peaks:
        top_idx = np.argsort(y)[-max_peaks:]
        x, y = x[top_idx], y[top_idx]

    # Sort by frequency
    order = np.argsort(x)
    x, y = x[order], y[order]
    k = len(x)

    out = np.zeros((max_peaks, n_channels), dtype=np.float32)
    if k > 0:
        out[:k, 0] = x
        out[:k, 1] = y
        if freq_scale is not None:
            out[:k, 0] /= freq_scale
        if int_scale is not None:
            out[:k, 1] /= int_scale

    if return_mask:
        mask = np.zeros(max_peaks, dtype=bool)
        if k > 0:
            mask[:k] = True
        return out, mask
    return out

class QMugsDatasetV2(Dataset):

    def __init__(
        self,
        data_path: str,
        *,
        mode: str = "diffusion",
        ir_rep: str = "peaklist",
        ir_sigma_cm: float = 4.0,
        grid_size_ir: int = _DEFAULT_GRID_POINTS,
        ir_xmin: float | None = _DEFAULT_GRID_MIN,
        ir_xmax: float | None = _DEFAULT_GRID_MAX,
        ir_max_peaks: int = 81,
        ir_peak_freq_scale: float | None = _DEFAULT_GRID_MAX,
        ir_peak_int_scale: float | None = None,
        norm_ir: str = "max",
        symbol_to_idx_path: str = "./data/symbol_to_idx.json",
        center_coords: bool = True,
        peak_channels: int = 2,
        peaklist_emit_mask: bool = True,
        **extra_kwargs: Any,
    ) -> None:
        if mode not in ("selfies", "diffusion"):
            raise ValueError(f"Unknown mode {mode!r}, expected 'selfies' or 'diffusion'")

        self.mode = mode
        self.ir_rep = ir_rep
        self.ir_max_peaks = ir_max_peaks
        self.ir_peak_freq_scale = ir_peak_freq_scale
        self.ir_peak_int_scale = ir_peak_int_scale
        self.grid_size_ir = grid_size_ir
        self.ir_xmin = ir_xmin
        self.ir_xmax = ir_xmax
        self.norm_ir = norm_ir
        self.center_coords = center_coords
        self.peak_channels = peak_channels
        self.peaklist_emit_mask = peaklist_emit_mask

        # Load npz
        data = np.load(data_path, allow_pickle=True)
        required = {"ir_frequencies", "ir_intensities", "coords",
                    "atom_types", "smiles"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing fields in {data_path}: {sorted(missing)}")

        self.ir_frequencies = data["ir_frequencies"]
        self.ir_intensities = data["ir_intensities"]
        self.coords = data["coords"]
        self.atom_types = data["atom_types"]
        self.smiles = data["smiles"]
        self.selfies = data["selfies"] if "selfies" in data.files else None
        self.decoderin = data["decoderin"] if "decoderin" in data.files else None
        self.decoderout = data["decoderout"] if "decoderout" in data.files else None

        # Vocabulary for selfies mode
        if os.path.exists(symbol_to_idx_path):
            with open(symbol_to_idx_path, "r", encoding="utf-8") as f:
                self.symbol_to_idx = json.load(f)
            self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}
            self.nop_idx = self.symbol_to_idx.get("[nop]", 0)
        else:
            self.symbol_to_idx = {}
            self.idx_to_symbol = {}
            self.nop_idx = 0

        data.close()

        # Pre-compute dense spectra for grid mode
        xmin = ir_xmin if ir_xmin is not None else _DEFAULT_GRID_MIN
        xmax = ir_xmax if ir_xmax is not None else _DEFAULT_GRID_MAX
        self._x_grid = np.linspace(xmin, xmax, grid_size_ir, dtype=np.float64)

        if ir_rep in ("gaussian", "interp"):
            self._spectra = np.stack([
                self._interpolate_spectrum(f, i)
                for f, i in zip(self.ir_frequencies, self.ir_intensities)
            ], axis=0).astype(np.float32)
        else:
            self._spectra = None

    def _interpolate_spectrum(self, frequencies, intensities):
        if len(frequencies) == 0:
            return np.zeros(self.grid_size_ir, dtype=np.float32)
        freqs = np.asarray(frequencies, dtype=np.float64)
        ints = np.asarray(intensities, dtype=np.float64)
        order = np.argsort(freqs)
        freqs, ints = freqs[order], ints[order]
        uniq_freqs, uniq_idx = np.unique(freqs, return_index=True)
        uniq_ints = ints[uniq_idx]
        if len(uniq_freqs) < 2:
            out = np.zeros(self.grid_size_ir, dtype=np.float32)
            if len(uniq_freqs) == 1:
                closest = int(np.argmin(np.abs(self._x_grid - uniq_freqs[0])))
                out[closest] = float(uniq_ints[0])
            return out
        try:
            fn = interp1d(uniq_freqs, uniq_ints, kind="linear",
                          bounds_error=False, fill_value=0.0)
            out = np.maximum(fn(self._x_grid), 0.0).astype(np.float32)
        except Exception:
            out = np.zeros(self.grid_size_ir, dtype=np.float32)
        if self.norm_ir == "max":
            m = out.max()
            if m > 1e-9:
                out /= m
        return out

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if self.mode == "selfies":
            return self._selfies_item(idx)
        return self._diffusion_item(idx)

    def _build_spectrum(self, idx):
        """Build spectrum tensor/dict for one sample."""
        if self._spectra is not None:
            return torch.as_tensor(self._spectra[idx], dtype=torch.float32)

        packed, mask = peaklist_pack(
            np.asarray(self.ir_frequencies[idx], dtype=np.float32),
            np.asarray(self.ir_intensities[idx], dtype=np.float32),
            max_peaks=self.ir_max_peaks,
            freq_scale=self.ir_peak_freq_scale,
            int_scale=self.ir_peak_int_scale,
            n_channels=self.peak_channels,
            return_mask=True,
        )
        return {
            "packed": torch.as_tensor(packed, dtype=torch.float32),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
        }

    def _selfies_item(self, idx):
        spectrum = self._build_spectrum(idx)
        din = torch.as_tensor(
            np.asarray(self.decoderin[idx], dtype=np.int64), dtype=torch.long)
        dout = torch.as_tensor(
            np.asarray(self.decoderout[idx], dtype=np.int64), dtype=torch.long)
        # Atom counts: [C, H, N, O, F, P, S, Cl, Br, I]
        atom_types_arr = np.asarray(self.atom_types[idx], dtype=np.int64)
        atom_counts = self._compute_atom_counts(atom_types_arr)
        return spectrum, din, dout, atom_counts

    def _diffusion_item(self, idx):
        spectrum = self._build_spectrum(idx)
        coords = np.asarray(self.coords[idx], dtype=np.float32).copy()
        if self.center_coords and coords.shape[0] > 0:
            coords -= coords.mean(axis=0, keepdims=True)
        coords_t = torch.as_tensor(coords, dtype=torch.float32)
        atom_types = torch.as_tensor(
            np.asarray(self.atom_types[idx], dtype=np.int64), dtype=torch.long)
        num_atoms = int(atom_types.shape[0])

        if getattr(self, 'emit_smiles', False) and self.smiles is not None:
            return spectrum, coords_t, atom_types, num_atoms, str(self.smiles[idx])
        return spectrum, coords_t, atom_types, num_atoms

    @staticmethod
    def _compute_atom_counts(atom_types):
        ATOM_LIST = [6, 1, 7, 8, 9, 15, 16, 17, 35, 53]  # C H N O F P S Cl Br I
        counts = torch.zeros(10, dtype=torch.float32)
        for i, z in enumerate(ATOM_LIST):
            counts[i] = float((atom_types == z).sum())
        return counts

# Collate functions

def _stack_spectra(spectra_list):
    if isinstance(spectra_list[0], dict):
        packed = torch.stack([s["packed"] for s in spectra_list], dim=0)
        mask = torch.stack([s["mask"] for s in spectra_list], dim=0)
        return {"packed": packed, "mask": mask}
    return torch.stack(spectra_list, dim=0)


def collate_variable_length(batch):
    """Collate for diffusion mode with variable atom counts."""
    has_smiles = len(batch[0]) == 5
    if has_smiles:
        spectra_list, coords_list, atom_types_list, num_atoms_list, smiles_list = zip(*batch)
    else:
        spectra_list, coords_list, atom_types_list, num_atoms_list = zip(*batch)
        smiles_list = None

    spectra = _stack_spectra(list(spectra_list))
    max_atoms = max(len(a) for a in atom_types_list)
    B = len(batch)

    coords_padded = torch.zeros(B, max_atoms, 3, dtype=torch.float32)
    atom_types_padded = torch.zeros(B, max_atoms, dtype=torch.long)
    atom_mask = torch.zeros(B, max_atoms, dtype=torch.bool)
    for i, (c, a, n) in enumerate(zip(coords_list, atom_types_list, num_atoms_list)):
        coords_padded[i, :n, :] = c
        atom_types_padded[i, :n] = a
        atom_mask[i, :n] = True

    num_atoms = torch.tensor(num_atoms_list, dtype=torch.long)
    if has_smiles:
        return spectra, coords_padded, atom_types_padded, num_atoms, atom_mask, list(smiles_list)
    return spectra, coords_padded, atom_types_padded, num_atoms, atom_mask


# DataModule for Lightning training

class QMugsDataModuleV2(pl.LightningDataModule):

    def __init__(self, data_path: str, *, mode: str = "selfies",
                 batch_size: int = 32, num_workers: int = 4,
                 val_split: float = 0.1, **ds_kwargs):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.ds_kwargs = ds_kwargs

    def setup(self, stage=None):
        full_ds = QMugsDatasetV2(
            self.data_path, mode=self.mode, **self.ds_kwargs)
        n = len(full_ds)
        n_val = int(n * self.val_split)
        n_train = n - n_val
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        collate = (collate_variable_length
                   if self.mode == 'diffusion' else None)
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate,
            pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        collate = (collate_variable_length
                   if self.mode == 'diffusion' else None)
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate,
            pin_memory=True,
        )
