"""Launch the replay retry through the same pinned Marin/Iris infrastructure."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from . import adapter
from .parity_install import patch_role_plan
from .pins import SKYRL_REVISION

BOOTSTRAP = r"""
import hashlib, io, os, subprocess, sys, zipfile
from pathlib import Path
from rigging.filesystem.storage_path import StoragePath
with StoragePath(os.environ["SNOWBALL_REPLAY_CODE_URI"]).open("rb") as stream:
    payload = stream.read()
if hashlib.sha256(payload).hexdigest() != os.environ["SNOWBALL_REPLAY_CODE_SHA256"]:
    raise ValueError("Replay overlay checksum mismatch")
root = Path("/app/snowball-replay")
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe replay overlay path")
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
subprocess.run([sys.executable, str(root / "replay_install.py"), "/app/marinskyrl", str(root)], check=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not args.inner:
        from marin.rl.skyrl import _launcher_command

        requirement = f"marinskyrl[cpu] @ git+https://github.com/marin-community/MarinSkyRL.git@{SKYRL_REVISION}"
        command = _launcher_command(requirement, "unused")
        command = command[: command.index("marinskyrl")] + [
            "python",
            "-m",
            "experiments.snowball_pdbthink.replay_launch",
            "--inner",
            "--config",
            str(args.config.resolve()),
        ]
        if args.validate:
            command.append("--validate")
        raise SystemExit(subprocess.call(command))

    from cloud.iris import iris_backend, role_plan
    from cloud.iris.launch_config import load_launch_config
    from cloud.iris.runtime_bundle import validate_bundled_runtime
    from iris.cluster.types import CoschedulingConfig

    raw = yaml.safe_load(args.config.read_text())
    if raw["iris"]["allocation"]["gpu_variant"] == "GB200":
        original_nodes = role_plan._minimum_disaggregated_rollout_nodes

        def replay_nodes(**values):
            if values == {
                "replicas": 2,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 8,
                "gpus_per_node": 4,
                "multiprocessing_backend": False,
            }:
                return 4
            return original_nodes(**values)

        role_plan._minimum_disaggregated_rollout_nodes = replay_nodes
        original_bundle = iris_backend.build_runtime_bundle

        def replay_bundle(expected_launcher_commit):
            workspace = original_bundle(expected_launcher_commit)
            relative = "cloud/iris/role_plan.py"
            path = workspace / relative
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_text(patch_role_plan(path.read_text()))
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            identity_path = workspace / ".marinskyrl-runtime.json"
            identity = json.loads(identity_path.read_text())
            entries = [entry for entry in identity["files"] if entry["path"] == relative]
            if len(entries) != 1 or entries[0]["sha256"] != before:
                raise ValueError("Original runtime bundle identity mismatch")
            entries[0]["sha256"] = after
            identity["snowball_replay_placement_patch"] = {"base_sha256": before, "installed_sha256": after}
            identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
            return workspace

        iris_backend.build_runtime_bundle = replay_bundle
        original_topology = iris_backend._gpu_multinode

        def replay_topology(gpu_variant, gpu_count, replicas):
            if (gpu_variant, gpu_count, replicas) == ("GB200", 4, 12):
                return CoschedulingConfig(group_by="nvlink.domain.preferred")
            return original_topology(gpu_variant, gpu_count, replicas)

        iris_backend._gpu_multinode = replay_topology
    cfg = load_launch_config(args.config)
    if args.validate:
        workspace = iris_backend.build_runtime_bundle(SKYRL_REVISION)
        try:
            validate_bundled_runtime(workspace)
        finally:
            shutil.rmtree(workspace)
        print(json.dumps({"valid": True, "job": cfg.iris.job_name, "runtime_bundle_verified": True}))
        return
    adapter.INSTALL += "\n" + BOOTSTRAP
    raise SystemExit(adapter.launch_with_adapter(args.config))


if __name__ == "__main__":
    main()
