"""Verify a native Snowball checkpoint before allocating a continuation."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .pins import MODEL_REVISION, SKYRL_REVISION


def checkpoint_location(uri):
    match = re.fullmatch(
        r"(s3://[^/]+)/tmp/ttl=\d+d/skyrl/"
        r"(marin/checkpoints/bizon/snowball-pdbthink-(?:pilot|epoch)/([^/]+))/checkpoints/global_step_(\d+)",
        uri,
    )
    if not match or not 0 < int(match[4]) < 876:
        raise ValueError("Expected a native Snowball checkpoint before terminal step 876")
    return {"run_root": match[1] + "/" + match[2], "version": match[3], "step": int(match[4])}


def verify_checkpoint(uri, *, expected_inputs=None):
    from rigging.filesystem.storage_path import StoragePath

    location = checkpoint_location(uri)
    checkpoint = StoragePath(uri)
    source = StoragePath(location["run_root"]) / "resolved-launch.yaml"
    with source.open("rb") as stream:
        source_bytes = stream.read()
    config = json.loads(source_bytes)["config"]
    if config["runtime"]["launcher_commit"] != SKYRL_REVISION:
        raise ValueError("Resume runtime differs from the pinned MarinSkyRL runtime")
    if config["inputs"]["model"]["tokenizer_revision"] != MODEL_REVISION:
        raise ValueError("Resume tokenizer revision differs")
    if expected_inputs is not None:
        # A pilot may use the fixed monitor subset of the same frozen artifact.
        # Training data and all model/data identities must still match exactly.
        def normalized(inputs):
            result = json.loads(json.dumps(inputs))
            for item in result.get("validation_data", []):
                if item["relative_path"] not in {"monitor.parquet", "validation.parquet"}:
                    raise ValueError("Unexpected validation cohort in resume artifacts")
                item["relative_path"] = "validation.parquet"
            return result

        if normalized(config["inputs"]) != normalized(expected_inputs):
            raise ValueError("Resume model or dataset artifacts differ from the continuation")
    if config["artifacts"]["checkpoint_root"] != str(checkpoint.parent):
        raise ValueError("Resume checkpoint does not belong to its source run")
    with (checkpoint.parent / "latest_ckpt_global_step.txt").open("r") as stream:
        latest = int(stream.read().strip())
    if not location["step"] <= latest <= config["skyrl"]["trainer"]["max_steps"]:
        raise ValueError("Resume checkpoint has no matching completion marker")
    with (checkpoint / "hf_export_request.json").open("r") as stream:
        request = json.load(stream)
    if request["step"] != location["step"] or request["checkpoint_path"] != uri:
        raise ValueError("Resume export request identifies a different checkpoint")
    files = [
        "trainer_state.pt", "data.pt", "policy/.metadata", "policy/common.pt",
        "policy/extra_state.pt", "policy/metadata.json",
        "policy/huggingface/config.json", "policy/huggingface/tokenizer.json",
        "policy/huggingface/chat_template.jinja",
        *(f"policy/__{rank}_0.distcp" for rank in range(32)),
    ]
    sizes = {}
    for name in files:
        path = checkpoint / name
        if not path.exists() or path.size() <= 0:
            raise ValueError("Incomplete resume checkpoint: " + name)
        sizes[name] = path.size()
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "checkpoint_uri": uri, "global_step": location["step"],
        "source_run_root": location["run_root"], "source_version": location["version"],
        "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "launcher_commit": SKYRL_REVISION, "tokenizer_revision": MODEL_REVISION,
        "latest_checkpoint_marker": latest, "file_sizes": sizes,
        "verification": "Completion marker, native export request, and nonempty trainer/data/32 policy shards; no weight-byte checksum replay",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-coordinator", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = verify_checkpoint(args.checkpoint)
    record["source_coordinator"] = args.source_coordinator
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "file_sizes"}, indent=2))


if __name__ == "__main__":
    main()
