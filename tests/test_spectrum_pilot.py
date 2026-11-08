"""Real preparation/three-arm training, final budget, and protected continuation."""
import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import selfies as sf
import torch
from rdkit import Chem

from diffusion.data import file_sha256
from diffusion.spectrum import SpectrumStructureModel
from evaluate_spectrum import main as evaluate
from scripts.run_spectrum_pilot import check_budget, main as pilot, read_json


def test_final_budget_requires_completed_epochs_steps_and_ema():
    checkpoint = {"epoch": 5, "global_step": 7800, "ema_state_dict": {"weight": 1},
        "loops": {"fit_loop": {"epoch_progress": {"total": {"completed": 5}}}}}
    check_budget(checkpoint, 5, 7800)
    for key, value in (("epoch", 4), ("global_step", 7799), ("ema_state_dict", {})):
        altered = copy.deepcopy(checkpoint)
        altered[key] = value
        with pytest.raises(ValueError):
            check_budget(altered, 5, 7800)
    checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"]["completed"] = 4
    with pytest.raises(ValueError):
        check_budget(checkpoint, 5, 7800)


def test_three_arm_cpu_pilot_and_verified_continuation(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        identities = ["C", "N", "O", "CC", "CN", "CO", "CCC", "CCO", "COC", "CF"]
        encodings = [list(sf.split_selfies(sf.encoder(value))) for value in identities]
        vocab = ["<pad>", "<bos>", "<eos>", *sorted({token for tokens in encodings for token in tokens})]
        tokens = np.zeros((10, 5), dtype=np.int64)
        atoms = np.zeros((10, 29), dtype=np.int64)
        counts = []
        for row, (identity, encoding) in enumerate(zip(identities, encodings)):
            values = [1, *[vocab.index(token) for token in encoding], 2]
            tokens[row, :len(values)] = values
            mol = Chem.AddHs(Chem.MolFromSmiles(identity))
            counts.append(mol.GetNumAtoms())
            atoms[row, :counts[-1]] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        data, splits, checkpoint = [tmp_path / name for name in ("tiny.npz", "splits.json", "baseline.ckpt")]
        np.savez(data, coords=np.zeros((10, 29, 3), dtype=np.float32), atom_types=atoms,
            num_atoms=np.array(counts), molecule_ids=np.array(identities), selfies_tokens=tokens, vocabulary=np.array(vocab),
            spectra_packed=np.random.default_rng(5).uniform(.01, .8, (10, 2, 2)).astype(np.float32),
            spectra_mask=np.ones((10, 2), dtype=bool))
        splits.write_text(json.dumps({"schema_version": 1, "sample_count": 10, "data_sha256": file_sha256(data),
            "splits": {"train": list(range(6)), "val": [6, 7, 8], "test": [9]}}))
        model = SpectrumStructureModel(vocabulary=vocab,
            encoder_spec={"hidden_dim": 16, "n_layers": 1, "n_heads": 4, "dropout": 0.},
            decoder_layers=1, heads=4, max_length=5, dropout=0.)
        model.run_metadata = {"data_sha256": file_sha256(data), "split_sha256": file_sha256(splits),
            "preprocessing": None, "spectrum_control": "none", "diagnostic": True}
        model._ema = {name: value.detach().clone() for name, value in model.named_parameters()}
        payload = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}
        model.on_save_checkpoint(payload)
        torch.save(payload, checkpoint)
        original_hash = file_sha256(checkpoint)
        audit = tmp_path / "audit.json"
        evaluate(["--checkpoint", str(checkpoint), "--data", str(data), "--splits", str(splits), "--output", str(audit),
                  "--selection", "random", "--limit", "3", "--beam_size", "5", "--seed", "20261105"])
        output = tmp_path / "pilot"
        # Preserve and skip an interrupted attempt, rather than consuming its checkpoint.
        interrupted = output / "ce_attempt_1/checkpoints"
        interrupted.mkdir(parents=True)
        (interrupted / "last.ckpt").write_bytes(b"interrupted diagnostic attempt")
        arguments = ["--audit", str(audit), "--checkpoint", str(checkpoint), "--data", str(data),
            "--splits", str(splits), "--output", str(output), "--device", "cpu", "--epochs", "1", "--batch_size", "2",
            "--mine_limit", "4", "--decode_widths", "5", "--diagnostic"]
        pilot([*arguments, "--prepare_only"])
        assert (output / "preparation_results.zip").is_file()
        assert (output / "PREPARATION.md").is_file()
        assert not (output / "ce_complete.json").exists()
        assert file_sha256(checkpoint) == original_hash
        mining = read_json(output / "mined_negatives.json")
        assert mining["model_generated_pairs"] > 0 and mining["attempted_training_rows"] == 4
        assert mining["held_out_labels_used"] is False
        pilot(arguments)
        assert (output / "ce_attempt_2/checkpoints/last.ckpt").is_file()
        assert (interrupted / "last.ckpt").read_bytes() == b"interrupted diagnostic attempt"
        assert file_sha256(checkpoint) == original_hash
        comparison = read_json(output / "comparison.json")
        assert comparison["rows"] == 3 and comparison["recommendation"]["status"] == "diagnostic_only"
        assert comparison["weights_automatically_promoted"] is False
        assert set(comparison["arms"]) == {"baseline", "ce", "ranking", "mined"}
        trained_configs = {name: read_json(Path(read_json(output / f"{name}_complete.json")["checkpoint"]).parent.parent / "resolved_config.json")
                           for name in ("ranking", "mined")}
        assert trained_configs["ranking"]["provenance"]["ranking_negatives_sha256"] != trained_configs["mined"]["provenance"]["ranking_negatives_sha256"]
        with np.load(output / "ranking_negatives.npz") as negatives:
            assert set(negatives["negative_rows"]).issubset(set(range(6)))
        with zipfile.ZipFile(output / "pilot_results.zip") as bundle:
            assert bundle.testzip() is None
            assert not any(name.endswith((".ckpt", ".npz")) for name in bundle.namelist())
            assert "comparison.json" in bundle.namelist()
        marker_before = (output / "ce_complete.json").read_bytes()
        pilot(arguments)
        assert (output / "ce_complete.json").read_bytes() == marker_before
        assert not (output / "ce_attempt_3").exists()
        with pytest.raises(SystemExit):
            pilot([*arguments, "--epochs", "2"])
        assert (output / "ce_complete.json").read_bytes() == marker_before
        final = Path(read_json(output / "ce_complete.json")["checkpoint"])
        with final.open("ab") as handle:
            handle.write(b"tampered")
        with pytest.raises(SystemExit):
            pilot(arguments)
        assert file_sha256(checkpoint) == original_hash
    finally:
        torch.set_num_threads(original_threads)
