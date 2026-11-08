"""Exercise real checkpoint loading/decoding and interruption recovery on tiny data."""
import json

import numpy as np
import pytest
import torch

from diffusion.data import file_sha256
from diffusion.spectrum_audit import exclusive_report
from diffusion.spectrum import SpectrumStructureModel
from evaluate_spectrum import main


@pytest.fixture
def evaluation_case(tmp_path):
    torch.manual_seed(17)
    vocab = ["<pad>", "<bos>", "<eos>", "[C]", "[N]", "[O]", "[F]"]
    data = tmp_path / "tiny.npz"
    atoms = np.zeros((8, 29), dtype=np.int64)
    atoms[:, 0] = 6
    # Synthetic numeric inputs exercise plumbing; they are not a chemistry benchmark.
    peaks = np.arange(32, dtype=np.float32).reshape(8, 2, 2) / 32
    np.savez(data, coords=np.zeros((8, 29, 3), dtype=np.float32), atom_types=atoms,
             num_atoms=np.ones(8, dtype=np.int64), spectra_packed=peaks,
             spectra_mask=np.ones((8, 2), dtype=bool),
             molecule_ids=np.array(["C", "N", "O", "F", "CC", "CN", "CO", "CF"]),
             selfies_tokens=np.tile([1, 3, 2, 0], (8, 1)), vocabulary=np.array(vocab))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"schema_version": 1, "sample_count": 8,
        "data_sha256": file_sha256(data), "splits": {"train": [0], "val": [1, 2, 3, 4, 5, 6], "test": [7]}}))
    model = SpectrumStructureModel(vocabulary=vocab,
        encoder_spec={"hidden_dim": 16, "n_layers": 1, "n_heads": 4, "dropout": 0.0},
        decoder_layers=1, heads=4, max_length=4, dropout=0.)
    model.run_metadata = {"data_sha256": file_sha256(data), "split_sha256": file_sha256(splits),
                          "preprocessing": None, "diagnostic": True}
    model._ema = {name: value.detach().clone() for name, value in model.named_parameters()}
    checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "tiny.ckpt"
    torch.save(checkpoint, path)
    arguments = ["--checkpoint", str(path), "--data", str(data), "--splits", str(splits),
                 "--selection", "random", "--limit", "4", "--seed", "123", "--beam_size", "2",
                 "--save_every", "10", "--progress_interval", "0"]
    return arguments, path


@pytest.mark.parametrize("control", ["none", "shuffled"])
def test_interrupted_resume_matches_uninterrupted_and_preserves_selection(evaluation_case, tmp_path, monkeypatch, control):
    arguments, _ = evaluation_case
    arguments += ["--spectrum_control", control]
    baseline = main([*arguments, "--output", str(tmp_path / "full.json")])
    output, selection = tmp_path / "resumed.json", tmp_path / "selection.json"
    resume_args = [*arguments, "--output", str(output), "--selection_manifest", str(selection), "--resume"]
    generate = SpectrumStructureModel.generate
    calls = []
    def interrupted(model, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt("simulated interruption")
        return generate(model, *args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(SpectrumStructureModel, "generate", interrupted)
        with pytest.raises(KeyboardInterrupt):
            main(resume_args)
    partial = json.loads(output.read_text())
    assert partial["complete"] is False and partial["attempted_rows"] == 2
    assert partial["ema_weights"] is True and partial["checkpoint_training_diagnostic"] is True
    resumed = main(resume_args)
    assert resumed["complete"] is True and resumed["rows"] == baseline["rows"]
    assert resumed["provenance"] == baseline["provenance"]
    assert resumed["selection"] == baseline["selection"]
    assert set(resumed["selected_rows"]).issubset({1, 2, 3, 4, 5, 6})
    before = output.read_bytes()
    assert main(resume_args) == resumed  # Idempotent, validated completion.
    assert output.read_bytes() == before
    progress = json.loads(output.with_suffix(".progress").joinpath("progress.json").read_text())
    assert progress["status"] == "completed"


def test_resume_rejects_checkpoint_change_and_existing_output_without_resume(evaluation_case, tmp_path):
    arguments, checkpoint_path = evaluation_case
    output = tmp_path / "report.json"
    main([*arguments, "--output", str(output)])
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main([*arguments, "--output", str(output)])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["run_metadata"]["test_change"] = True
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(SystemExit):
        main([*arguments, "--output", str(output), "--resume"])
    assert output.read_bytes() == before


def test_selection_change_rejected_without_overwriting_manifest(evaluation_case, tmp_path):
    arguments, _ = evaluation_case
    selection = tmp_path / "selection.json"
    main([*arguments, "--selection_manifest", str(selection), "--output", str(tmp_path / "first.json")])
    before = selection.read_bytes()
    with pytest.raises(SystemExit):
        main([*arguments, "--selection_manifest", str(selection), "--seed", "999",
              "--output", str(tmp_path / "other.json")])
    assert selection.read_bytes() == before
    assert not (tmp_path / "other.json").exists()


def test_concurrent_writer_lock_is_released_after_failure(tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError):
        with exclusive_report(output):
            with pytest.raises(ValueError, match="Another evaluator"):
                with exclusive_report(output):
                    pytest.fail("Second writer acquired the same report")
            raise RuntimeError("interrupted writer")
    with exclusive_report(output):
        pass


def test_checkpoint_change_during_load_is_rejected(evaluation_case, tmp_path, monkeypatch):
    arguments, checkpoint_path = evaluation_case
    load = SpectrumStructureModel.load_from_checkpoint
    def changing_checkpoint(*args, **kwargs):
        model = load(*args, **kwargs)
        with checkpoint_path.open("ab") as handle:
            handle.write(b"changed during load")
        return model
    monkeypatch.setattr(SpectrumStructureModel, "load_from_checkpoint", changing_checkpoint)
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        main([*arguments, "--output", str(output)])
    assert not output.exists()
