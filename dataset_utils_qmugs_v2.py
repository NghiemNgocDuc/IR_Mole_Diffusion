"""Compatibility dataset import. Legacy object-array loading was retired."""
from diffusion.data import NativeNPZDataset, collate_native, load_native_npz


class QMugsDatasetV2(NativeNPZDataset):
    def __init__(self, data_path, mode="diffusion", **kwargs):
        if mode not in ("diffusion", "selfies"):
            raise ValueError("mode must be diffusion or selfies")
        if kwargs:
            raise TypeError("Legacy preprocessing options were retired; use prepare_squirl.py and NativeNPZDataset.")
        super().__init__(load_native_npz(data_path))
        if mode == "selfies" and "selfies_tokens" not in self.data:
            raise ValueError("SELFIES mode requires converted numeric token labels.")


collate_variable_length = collate_native
