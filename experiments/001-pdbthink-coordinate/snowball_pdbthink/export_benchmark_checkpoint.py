"""Freeze the best available validation checkpoint and export it during training.

This deliberately does not use terminal-run recovery: the training job may still
be running. The immutable, completed native checkpoint is verified independently.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .pins import SKYRL_REVISION
from .recover_export import validate_export_files
from .resume import verify_checkpoint
from .select_checkpoint import best_checkpoint


def main():
    from marin.rl.skyrl import _launcher_command
    from rigging.filesystem.storage_path import StoragePath

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--resolved", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster", default="cw-rno2a")
    parser.add_argument("--gpu-variant", default="H100")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--memory", default="1800GB")
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify a published model without launching an export"
    )
    parser.add_argument(
        "--allow-cross-rack-export",
        action="store_true",
        help="Use Iris's supported preferred NVLink topology for this export job only",
    )
    args = parser.parse_args()
    if args.nodes * args.gpus_per_node != 32:
        raise ValueError("Export must retain the saved 32-rank policy world size")
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "selection.json"
    config = json.loads(args.resolved.read_text())["config"]
    if not selection_path.exists():
        snapshot = json.loads(args.status.read_text())
        chosen = best_checkpoint(snapshot["validation"])
        selection = {
            "selected_at": datetime.now(UTC).isoformat(),
            "rule": "Highest complete native full-validation task-weighted accuracy available at selection; ties by family macro, then earlier step",
            "scope": "Best available checkpoint; training has not finished. Original benchmark outcomes are not used for selection.",
            "selected": chosen,
            "validation_snapshot": snapshot,
            "resolved_sha256": hashlib.sha256(args.resolved.read_bytes()).hexdigest(),
        }
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    selection = json.loads(selection_path.read_text())
    if selection["resolved_sha256"] != hashlib.sha256(args.resolved.read_bytes()).hexdigest():
        raise ValueError("Frozen launch configuration changed")
    step = selection["selected"]["step"]
    checkpoint = config["artifacts"]["checkpoint_root"] + f"/global_step_{step}"
    verification = verify_checkpoint(checkpoint, expected_inputs=config["inputs"])
    (args.output / "native-checkpoint-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    source = args.output / "export-training-config.json"
    source.write_text(json.dumps(config, indent=2) + "\n")
    root = StoragePath(config["artifacts"]["export_root"]) / f"global_step_{step}" / "policy"
    manifest_path = root / ".marinskyrl-model-manifest.json"
    if not manifest_path.exists():
        if args.verify_only:
            raise FileNotFoundError("Native model publication is not complete")
        job_name = f"snowball-original-benchmark-export-{step}-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        command = _launcher_command(
            f"marinskyrl @ git+https://github.com/marin-community/MarinSkyRL.git@{SKYRL_REVISION}", "unused"
        )
        command = command[: command.index("marinskyrl")] + [
            "python",
            "-m",
            "cloud.iris.export_hf_checkpoint",
            "--ckpt_path",
            config["artifacts"]["checkpoint_root"],
            "--step",
            str(step),
            "--model_path",
            config["skyrl"]["trainer"]["policy"]["model"]["path"],
            "--model-source-uri",
            config["inputs"]["model"]["uri"],
            "--model-source-identity",
            config["inputs"]["model"]["identity"],
            "--num-nodes",
            str(args.nodes),
            "--gpus-per-node",
            str(args.gpus_per_node),
            "--export_path",
            config["artifacts"]["export_root"],
            "--launch-config",
            str(source),
            "--gpu-variant",
            args.gpu_variant,
            "--cluster",
            args.cluster,
            "--cluster-config",
            str(Path(f"lib/iris/config/{args.cluster}.yaml").absolute()),
            "--target-cluster",
            args.cluster,
            "--parent-cluster-config",
            str(Path("lib/iris/config/marin.yaml").absolute()),
            "--cpu",
            "64",
            "--memory",
            args.memory,
            "--disk",
            "1000GB",
            "--job-name",
            job_name,
        ]
        if args.allow_cross_rack_export:
            # The pinned export CLI does not expose Iris's soft NVLink binding.
            # Change only the local submission function; GPU runtime, checkpoint,
            # 32-rank model geometry, and conversion code remain pinned.
            launch = (
                "from cloud.iris import iris_backend as b; "
                "b._gpu_multinode=lambda variant,count,replicas: "
                "b.CoschedulingConfig(group_by='nvlink.domain.preferred'); "
                "from cloud.iris.launch import main; main()"
            )
            wrapper = (
                "from cloud.iris import export_hf_checkpoint as e\n"
                "original=e.build_command\n"
                "def preferred(spec):\n"
                "    command=original(spec)\n"
                f"    return [command[0], '-c', {launch!r}, *command[3:]]\n"
                "e.build_command=preferred\n"
                "e.main()\n"
            )
            position = command.index("cloud.iris.export_hf_checkpoint")
            command[position - 1 : position + 1] = ["-c", wrapper]
        (args.output / "export-command.json").write_text(json.dumps(command, indent=2) + "\n")
        print("Exporting validation-selected checkpoint", step, "job", job_name, flush=True)
        subprocess.run(command, check=True)

    def read_bytes(path):
        with path.open("rb") as source:
            return source.read()

    manifest_bytes = read_bytes(manifest_path)
    manifest = json.loads(manifest_bytes)
    checks = validate_export_files(
        manifest,
        lambda name: (root / name).size(),
        lambda name: read_bytes(root / name),
        json.loads(args.prepared.read_text())["model_metadata_sha256"],
    )
    record = {
        "verified_at": datetime.now(UTC).isoformat(),
        "step": step,
        "model_uri": str(root),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest": manifest,
        "checks": checks,
        "weight_verification": "Native publisher SHA256 manifest and remote sizes; weight bytes not downloaded again",
    }
    (args.output / "export-verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Verified export", str(root), flush=True)


if __name__ == "__main__":
    main()
