"""Recover a published terminal model after an Iris export-job teardown failure."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import PurePosixPath

from .pins import MODEL_REVISION, SKYRL_REVISION


def validate_export_files(manifest, size_for, read_bytes, expected_metadata):
    """Check publication completeness without re-downloading the large weight shards."""
    if manifest["format_version"] != 1 or manifest["tokenizer_mode"] != "embedded":
        raise ValueError("Expected a native export manifest with an embedded tokenizer")
    files = {}
    metadata = {}
    for item in manifest["files"]:
        name = item["path"]
        if not name or PurePosixPath(name).name != name or name in {".", ".."} or name in files:
            raise ValueError("Unsafe or duplicate export manifest path")
        if item["size"] <= 0 or size_for(name) != item["size"]:
            raise ValueError("Incomplete export file: " + name)
        files[name] = item
        if not name.endswith(".safetensors"):
            metadata[name] = read_bytes(name)
            if hashlib.sha256(metadata[name]).hexdigest() != item["sha256"]:
                raise ValueError("Export metadata checksum mismatch: " + name)
    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    if not required <= metadata.keys():
        raise ValueError("Export lacks required model/tokenizer metadata")
    shards = set(json.loads(metadata["model.safetensors.index.json"])["weight_map"].values())
    if not shards or not shards <= files.keys() or any(not name.endswith(".safetensors") for name in shards):
        raise ValueError("Export index references missing weight shards")
    for name in ("tokenizer.json", "chat_template.jinja"):
        if hashlib.sha256(metadata[name]).hexdigest() != expected_metadata[name]:
            raise ValueError("Export changed the frozen tokenizer/template: " + name)
    return {
        "files_verified_by_size": len(files),
        "weight_shards": len(shards),
        "metadata_verified_by_sha256": len(metadata),
    }


def recover_export(iris, terminal_uri, coordinator, step, prepared_manifest):
    from iris.cluster.types import JobName
    from iris.resources.state import JobState
    from rigging.filesystem.storage_path import StoragePath

    def read_bytes(path):
        with path.open("rb") as source:
            return source.read()

    terminal = StoragePath(terminal_uri)
    resolved_path = terminal.parent / "resolved-launch.yaml"
    resolved_bytes = read_bytes(resolved_path)
    # The pinned launcher serializes this .yaml artifact as JSON.
    config = json.loads(resolved_bytes)["config"]
    if config["runtime"]["launcher_commit"] != SKYRL_REVISION:
        raise ValueError("Recovery requires the pinned MarinSkyRL runtime")
    if config["inputs"]["model"]["tokenizer_revision"] != MODEL_REVISION:
        raise ValueError("Recovery requires the frozen original tokenizer")
    if step <= 0 or config["skyrl"]["trainer"]["max_steps"] != step:
        raise ValueError("Recovery only accepts the predeclared final checkpoint")
    artifacts = config["artifacts"]
    if artifacts["terminal_manifest_uri"] != terminal_uri or artifacts["export_root"] != str(
        terminal.parent / "exports"
    ):
        raise ValueError("Resolved artifacts do not belong to this run")
    training_job = coordinator.rstrip("/") + "/" + config["iris"]["job_name"]
    if iris.job_state(JobName.from_wire(training_job)) != JobState.SUCCEEDED:
        raise ValueError("Recovery requires a successful training job")
    marker = StoragePath(artifacts["checkpoint_root"]) / "latest_ckpt_global_step.txt"
    if int(read_bytes(marker).strip()) != step:
        raise ValueError("Checkpoint marker does not match the predeclared final step")
    root = StoragePath(artifacts["export_root"]) / f"global_step_{step}" / "policy"
    manifest_bytes = read_bytes(root / ".marinskyrl-model-manifest.json")
    manifest = json.loads(manifest_bytes)
    checks = validate_export_files(
        manifest,
        lambda name: (root / name).size(),
        lambda name: read_bytes(root / name),
        prepared_manifest["model_metadata_sha256"],
    )
    model = {
        "policy_export_uri": str(root),
        "global_step": step,
        "tokenizer_uri": config["inputs"]["model"]["tokenizer_uri"],
        "tokenizer_revision": MODEL_REVISION,
        "checkpoint_root": artifacts["checkpoint_root"],
        "terminal_manifest_uri": terminal_uri,
    }
    record = {
        "recovered_at": datetime.now(UTC).isoformat(),
        "reason": "Evaluate a complete native export independently of export-job teardown status",
        "training_job": training_job,
        "training_job_state": "succeeded",
        "coordinator": coordinator,
        "coordinator_state_at_recovery": str(iris.job_state(JobName.from_wire(coordinator))),
        "native_terminal_manifest_present": terminal.exists(),
        "resolved_config_uri": str(resolved_path),
        "resolved_config_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
        "export_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "export_manifest": manifest,
        "checks": checks,
        "weight_verification": "Native publisher SHA256 manifest plus remote sizes; weight bytes not downloaded again",
        "model": model,
    }
    return model, record
