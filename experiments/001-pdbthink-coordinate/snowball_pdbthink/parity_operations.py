"""Verify source checkpoints and stage the immutable parity experiment inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from rigging.filesystem.storage_path import StoragePath

from .resume import verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "launch-manifest.json").read_text())
    for step in (384, 512):
        cfg = yaml.safe_load((args.directory / f"launch-{step}.yaml").read_text())
        result = verify_checkpoint(cfg["skyrl"]["trainer"]["resume_path"], expected_inputs=cfg["inputs"])
        (args.directory / f"checkpoint-{step}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"verified_checkpoint": step, "files": len(result["file_sizes"])}), flush=True)
    for name in ("frozen-panel.json.gz", "panel-manifest.json", manifest["code_file"], "launch-manifest.json"):
        payload = (args.directory / name).read_bytes()
        remote_name = f"launch-manifest-{manifest['code_sha256']}.json" if name == "launch-manifest.json" else name
        target = StoragePath(manifest["prefix"]) / remote_name
        if target.exists():
            with target.open("rb") as stream:
                if stream.read() != payload:
                    raise ValueError("Refusing to replace an immutable probe input: " + name)
        else:
            with target.open("wb") as stream:
                stream.write(payload)
        print(
            json.dumps({"staged": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}),
            flush=True,
        )


if __name__ == "__main__":
    main()
