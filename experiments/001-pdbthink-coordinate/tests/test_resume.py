import io
import json
import sys
from types import SimpleNamespace

import pytest
from snowball_pdbthink.pins import MODEL_REVISION, SKYRL_REVISION
from snowball_pdbthink.resume import checkpoint_location, verify_checkpoint

URI = (
    "s3://bucket/tmp/ttl=14d/skyrl/marin/checkpoints/bizon/"
    "snowball-pdbthink-epoch/2026.10.08.1/checkpoints/global_step_320"
)


def test_interrupted_export_requires_explicit_stopped_checkpoint_selection():
    from snowball_pdbthink.recover_export import validate_training_state

    validate_training_state("succeeded", selected_intermediate=False, interrupted_training=False)
    validate_training_state("failed", selected_intermediate=True, interrupted_training=True)
    for state, selected, interrupted in [
        ("failed", True, False), ("failed", False, True), ("running", True, True), ("succeeded", True, True),
    ]:
        with pytest.raises(ValueError):
            validate_training_state(state, selected_intermediate=selected, interrupted_training=interrupted)


def test_resume_rejects_foreign_or_terminal_checkpoint_paths():
    assert checkpoint_location(URI)["step"] == 320
    for bad in [URI.replace("320", "876"), URI.replace("320", "0"), URI + "/extra", URI.replace("snowball", "other")]:
        with pytest.raises(ValueError):
            checkpoint_location(bad)


def test_resume_requires_completed_state_and_matching_input_artifacts(monkeypatch):
    inputs = {
        "model": {"tokenizer_revision": MODEL_REVISION}, "train_data": [{"uri": "frozen-data"}],
        "validation_data": [{"uri": "frozen-data", "relative_path": "monitor.parquet"}],
    }
    config = {
        "runtime": {"launcher_commit": SKYRL_REVISION}, "inputs": inputs,
        "artifacts": {"checkpoint_root": URI.rsplit("/", 1)[0]},
        "skyrl": {"trainer": {"max_steps": 876}},
    }
    source = checkpoint_location(URI)["run_root"] + "/resolved-launch.yaml"
    files = {
        source: json.dumps({"config": config}).encode(),
        URI.rsplit("/", 1)[0] + "/latest_ckpt_global_step.txt": b"320",
        URI + "/hf_export_request.json": json.dumps({"step": 320, "checkpoint_path": URI}).encode(),
    }

    class FakePath(str):
        def __truediv__(self, name):
            return FakePath(self + "/" + name)

        @property
        def parent(self):
            return FakePath(self.rsplit("/", 1)[0])

        def open(self, mode):
            return io.BytesIO(files[self]) if "b" in mode else io.StringIO(files[self].decode())

        def exists(self):
            return self in files

        def size(self):
            return len(files[self])

    monkeypatch.setitem(sys.modules, "rigging.filesystem.storage_path", SimpleNamespace(StoragePath=FakePath))
    for name in [
        "trainer_state.pt", "data.pt", "policy/.metadata", "policy/common.pt", "policy/extra_state.pt",
        "policy/metadata.json", "policy/huggingface/config.json", "policy/huggingface/tokenizer.json",
        "policy/huggingface/chat_template.jinja", *(f"policy/__{i}_0.distcp" for i in range(32)),
    ]:
        files[URI + "/" + name] = b"saved state"
    assert verify_checkpoint(URI, expected_inputs=inputs)["global_step"] == 320
    full_validation = {**inputs, "validation_data": [{"uri": "frozen-data", "relative_path": "validation.parquet"}]}
    assert verify_checkpoint(URI, expected_inputs=full_validation)["global_step"] == 320
    with pytest.raises(ValueError, match="artifacts differ"):
        verify_checkpoint(URI, expected_inputs={**inputs, "train_data": [{"uri": "different-data"}]})
    del files[URI + "/data.pt"]
    with pytest.raises(ValueError, match="data.pt"):
        verify_checkpoint(URI)
    files[URI + "/data.pt"] = b"saved state"
    files[URI + "/policy/__31_0.distcp"] = b""
    with pytest.raises(ValueError, match="__31_0.distcp"):
        verify_checkpoint(URI)
    files[URI.rsplit("/", 1)[0] + "/latest_ckpt_global_step.txt"] = b"256"
    with pytest.raises(ValueError, match="completion marker"):
        verify_checkpoint(URI)
