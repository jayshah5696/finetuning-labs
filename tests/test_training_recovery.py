"""Specify recovery failures before changing checkpoint persistence.

A partial Drive copy must not replace the previous usable checkpoint. A finished
stage must not run again, and a partial generation line must remain recoverable.
"""

import json

import pytest

from finetuning_labs.rewriter.colab_training import (
    finish_stage,
    read_saved_outputs,
    restore_recovery,
    save_recovery,
    stage_finished,
)


def test_corrupt_newer_drive_copy_recovers_previous_complete_checkpoint(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    old = local / "checkpoint-20"
    old.mkdir(parents=True)
    (old / "model.safetensors").write_bytes(b"best weights")
    (old / "optimizer.pt").write_bytes(b"old optimizer")
    save_recovery(local, drive, 20, old, "bound")
    new = local / "checkpoint-40"
    new.mkdir()
    (new / "model.safetensors").write_bytes(b"new weights")
    (new / "optimizer.pt").write_bytes(b"new optimizer")
    save_recovery(local, drive, 40, old, "bound")
    newest = next(p for p in drive.glob("*.json") if json.loads(p.read_text())["step"] == 40)
    (drive / json.loads(newest.read_text())["archive"]).write_bytes(b"interrupted copy")
    restored = tmp_path / "restored"
    assert restore_recovery(drive, restored, "bound") == restored / "checkpoint-20"
    assert (restored / "checkpoint-20/optimizer.pt").read_bytes() == b"old optimizer"


def test_recovery_preserves_best_weights_but_only_latest_optimizer(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    for step in [20, 40]:
        checkpoint = local / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(str(step).encode())
        (checkpoint / "optimizer.pt").write_bytes(str(step).encode())
    save_recovery(local, drive, 40, local / "checkpoint-20", "bound")
    restored = tmp_path / "restored"
    restore_recovery(drive, restored, "bound")
    assert (restored / "checkpoint-40/optimizer.pt").exists()
    assert (restored / "checkpoint-20/model.safetensors").read_bytes() == b"20"
    assert not (restored / "checkpoint-20/optimizer.pt").exists()
    with pytest.raises(ValueError, match="binding"):
        restore_recovery(drive, tmp_path / "foreign", "different")


def test_finished_stage_is_idempotent_and_rejects_changed_results(tmp_path):
    report = tmp_path / "training-result.json"
    report.write_text('{"loss": 1}')
    finish_stage(tmp_path, [report])
    assert stage_finished(tmp_path)
    report.write_text('{"loss": 9}')
    with pytest.raises(ValueError, match="completed"):
        stage_finished(tmp_path)


def test_only_unterminated_last_generation_line_can_be_quarantined(tmp_path):
    path = tmp_path / "outputs.jsonl"
    path.write_bytes(b'{"case_id":"one"}\n{"case_id":')
    assert read_saved_outputs(path) == [{"case_id": "one"}]
    assert path.read_bytes() == b'{"case_id":"one"}\n'
    assert len(list(tmp_path.glob("outputs.partial-*.bin"))) == 1
    path.write_bytes(b'broken\n{"case_id":"one"}\n')
    with pytest.raises(json.JSONDecodeError):
        read_saved_outputs(path)
