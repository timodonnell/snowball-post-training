"""Launch a fixed-token parity job with the pinned MarinSkyRL runtime."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from . import adapter
from .parity_install import patch_role_plan
from .pins import SKYRL_REVISION

BOOTSTRAP = r"""
import hashlib, io, os, subprocess, sys, zipfile
from pathlib import Path
from rigging.filesystem.storage_path import StoragePath
with StoragePath(os.environ["SNOWBALL_PARITY_CODE_URI"]).open("rb") as stream:
    payload = stream.read()
if hashlib.sha256(payload).hexdigest() != os.environ["SNOWBALL_PARITY_CODE_SHA256"]:
    raise ValueError("Parity instrumentation checksum mismatch")
root = Path("/app/snowball-parity")
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if Path(name).name != name:
            raise ValueError("Unsafe instrumentation path")
        (root / name).write_bytes(archive.read(name))
subprocess.run([sys.executable, str(root / "parity_install.py"), "/app/marinskyrl", str(root), "/app"], check=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.inner:
        from cloud.iris import role_plan

        original_nodes = role_plan._minimum_disaggregated_rollout_nodes

        def diagnostic_nodes(**values):
            # The pinned runtime supports verified TP1/PP1 cross-node EP,
            # while its older launcher still assumes STRICT_PACK per engine.
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

        role_plan._minimum_disaggregated_rollout_nodes = diagnostic_nodes
        from cloud.iris import iris_backend
        from cloud.iris.launch_config import load_launch_config
        from iris.cluster.types import CoschedulingConfig

        cfg = load_launch_config(args.config)
        from .parity_preflight import check_collation

        check_collation(cfg.skyrl, args.config.parent / "frozen-panel.json.gz")
        original_bundle = iris_backend.build_runtime_bundle

        def diagnostic_bundle(expected_launcher_commit):
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
                raise ValueError("Original runtime bundle role-planner identity mismatch")
            entries[0]["sha256"] = after
            identity["snowball_diagnostic_patch"] = {
                "path": relative,
                "base_sha256": before,
                "patched_sha256": after,
                "instrumentation_sha256": cfg.runtime.task_env["SNOWBALL_PARITY_CODE_SHA256"],
            }
            identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
            return workspace

        iris_backend.build_runtime_bundle = diagnostic_bundle
        topology = cfg.runtime.task_env.get("SNOWBALL_PARITY_TOPOLOGY")
        if topology == "nvlink.domain.preferred":
            original_topology = iris_backend._gpu_multinode

            def diagnostic_topology(gpu_variant, gpu_count, replicas):
                if (gpu_variant, gpu_count, replicas) == ("GB200", 4, 12):
                    return CoschedulingConfig(group_by=topology)
                return original_topology(gpu_variant, gpu_count, replicas)

            iris_backend._gpu_multinode = diagnostic_topology
        if args.validate:
            from cloud.iris.runtime_bundle import validate_bundled_runtime

            workspace = diagnostic_bundle(SKYRL_REVISION)
            try:
                validate_bundled_runtime(workspace)
            finally:
                shutil.rmtree(workspace)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "runtime_bundle_verified": True,
                        "job": cfg.iris.job_name,
                        "updates": cfg.skyrl.trainer.mismatch_probe.updates,
                        "checkpoint": cfg.skyrl.trainer.resume_path,
                    }
                )
            )
            return
        adapter.INSTALL += "\n" + BOOTSTRAP
        raise SystemExit(adapter.launch_with_adapter(args.config))
    from marin.rl.skyrl import _launcher_command

    requirement = f"marinskyrl[cpu] @ git+https://github.com/marin-community/MarinSkyRL.git@{SKYRL_REVISION}"
    command = _launcher_command(requirement, "unused")
    command = command[: command.index("marinskyrl")] + [
        "python",
        "-m",
        "experiments.snowball_pdbthink.parity_launch",
        "--inner",
        "--config",
        str(args.config.resolve()),
    ]
    if args.validate:
        command.append("--validate")
    print(json.dumps({"launcher_revision": SKYRL_REVISION, "config": str(args.config)}), flush=True)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
