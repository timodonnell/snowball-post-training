"""Install the experiment environment alongside the immutable MarinSkyRL runtime.

MarinSkyRL currently has a fixed environment registry and no external environment
package setting. This overlay adds the PDBThink environment and preserves explicit
empty tools in its existing structured-chat transport. It changes no optimizer,
model, rollout scheduler, loss, or checkpoint code.
"""

import argparse
import hashlib
import io
import json
import shlex
import sys
import zipfile
from pathlib import Path

INSTALL = r"""
import hashlib, io, json, site, subprocess, sys, zipfile
from pathlib import Path
from rigging.filesystem.storage_path import StoragePath
uri, expected = sys.argv[1:3]
with StoragePath(uri).open("rb") as stream:
    payload = stream.read()
if hashlib.sha256(payload).hexdigest() != expected:
    raise ValueError("PDBThink adapter checksum mismatch")
root = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/app/marinskyrl")
package = root / "skyrl-gym/skyrl_gym/envs/pdbthink"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe adapter member")
        target = package / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
registry = root / "skyrl-gym/skyrl_gym/envs/__init__.py"
registration = '\nregister(id="pdbthink", entry_point="skyrl_gym.envs.pdbthink.env:PDBThinkEnv")\n'
registry.write_text(registry.read_text() + registration)
transport = root / "skyrl-train/skyrl_train/trajectory_runners/model_clients.py"
source = transport.read_text()
old = '            result.pop("tools", None)'
if source.count(old) != 1:
    raise ValueError("Pinned structured-chat transport changed")
# OpenAI-compatible transport accepts null for an explicitly disabled tool list.
transport.write_text(source.replace(old, '            result["tools"] = None'))
# Runtime packages are wheels; put this job's source overlay ahead of them in
# every fresh interpreter, including Ray workers. Never modify the uv cache.
site_packages = Path(site.getsitepackages()[0])
if not site_packages.is_relative_to(sys.prefix):
    raise ValueError("Expected an isolated runtime virtual environment")
paths = [str(root / "skyrl-gym"), str(root / "skyrl-train")]
(site_packages / "snowball_pdbthink.pth").write_text("import sys; sys.path[:0] = " + repr(paths) + "\n")
check = '''
import json, sys
from pathlib import Path
import skyrl_gym
from skyrl_gym.verification import RolloutEvidence
assert Path(skyrl_gym.__file__).is_relative_to(Path(sys.argv[1]) / "skyrl-gym")
env = skyrl_gym.make("pdbthink", env_config={}, extras={"family":"probe", "path":"probe",
    "reward_model":{"ground_truth":json.dumps({"answer_schema":"integer","gold_answer":{"value":7},"parameters":{}})}})
message = {"role":"assistant","content":chr(10).join(["<thinking>", "FINAL: 7", "</thinking>"])}
env.set_rollout_evidence(RolloutEvidence(messages=[message], metadata={"assistant_message":message},
    stop_reason="stop", generated_token_count=3))
assert env.step("SkyRL's extracted action has no FINAL field")["reward"] == 1.0
print("PDBThink registry and native reward verified in fresh runtime interpreter", flush=True)
'''
subprocess.run([sys.executable, "-c", check, str(root)], check=True)
(root / "pdbthink-adapter.json").write_text(json.dumps({"uri": uri, "sha256": expected}))
print("Installed PDBThink adapter", expected, flush=True)
"""


def build_adapter(prepared):
    root = Path(__file__).parent
    contents = {name: (root / name).read_bytes() for name in ("env.py", "scoring.py", "__init__.py")}
    template = (prepared / "native_chat_template.jinja").read_bytes()
    manifest = json.loads((prepared / "manifest.json").read_text())
    if hashlib.sha256(template).hexdigest() != manifest["model_metadata_sha256"]["chat_template.jinja"]:
        raise ValueError("Native template differs from the verified model metadata")
    contents["native_chat_template.jinja"] = template
    for path in (prepared / "native_verifier").rglob("*.py"):
        contents[str(path.relative_to(prepared))] = path.read_bytes()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
            archive.writestr(info, content)
    target = prepared / "adapter.zip"
    target.write_bytes(payload.getvalue())
    identity = hashlib.sha256(payload.getvalue()).hexdigest()
    (prepared / "adapter.sha256").write_text(identity + "\n")
    return identity


def run_with_adapter(config):
    # Retain Marin's launcher output validation and artifact/catalog conventions.
    import tempfile

    from marin.rl import skyrl
    from rigging.filesystem.storage_path import StoragePath

    from .pins import MODEL_REVISION

    with (StoragePath(config.model.uri) / "source.json").open("r") as stream:
        model_source = json.load(stream)
    if model_source["revision"] != MODEL_REVISION:
        raise ValueError("The model mirror is incomplete or identifies a different checkpoint")

    original = skyrl._launcher_command(config.launcher_requirement, "unused")
    response = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as launch:
            launch.write(config.launch_config_yaml)
            launch.flush()
            command = original[: original.index("marinskyrl")] + [
                "python",
                str(Path(__file__).resolve()),
                "launch",
                "--config",
                launch.name,
            ]
            result = skyrl._run_launcher(command)
        if not result.stdout.strip():
            raise RuntimeError(f"MarinSkyRL exited {result.returncode}: {result.stderr}")
        response = skyrl._SkyRLLaunchResponse.model_validate_json(result.stdout)
        if result.returncode or response.state != "succeeded":
            raise RuntimeError(f"MarinSkyRL failed: {response.failure}\n{result.stderr}")
        if response.iris_job_id is None or (config.export_hf and response.model is None):
            raise ValueError("MarinSkyRL did not return the required job/export metadata")
        model = response.model
        artifact = skyrl.SkyRLRun(
            path=config.output.terminal_manifest_uri,
            hf_model_uri=model.policy_export_uri if model else None,
            global_step=model.global_step if model else None,
            tokenizer_uri=config.model.tokenizer_uri,
            tokenizer_revision=config.model.tokenizer_revision,
            checkpoint_root=config.output.checkpoint_root,
            draft_checkpoint_root=config.draft_checkpoint_root,
            terminal_manifest_uri=config.output.terminal_manifest_uri,
            iris_job_id=response.iris_job_id,
        )
    except Exception:
        skyrl._record_skyrl_run(config, "failed", response)
        raise
    skyrl._record_skyrl_run(config, "succeeded", response)
    return artifact


def launch_with_adapter(config_path):
    import contextlib
    from dataclasses import asdict

    from cloud.iris import iris_backend
    from cloud.iris.launch import execute_launch
    from cloud.iris.launch_config import load_launch_config
    from skyrl_train.config.trajectory_runner_capabilities import (
        TrajectoryRunnerMode,
        validate_trajectory_runner_capabilities,
    )

    config = load_launch_config(config_path)
    validate_trajectory_runner_capabilities(config.skyrl, TrajectoryRunnerMode.SKYRL_GYM)
    settings = config.runtime.task_env
    install = (
        '\n"$IRIS_VENV/bin/python" -c '
        + shlex.quote(INSTALL)
        + " "
        + shlex.quote(settings.PDBTHINK_ADAPTER_URI)
        + " "
        + shlex.quote(settings.PDBTHINK_ADAPTER_SHA256)
        + "\n"
    )

    class PDBThinkBackend(iris_backend.IrisBackend):
        def launch(self, path):
            # Narrow, scoped bootstrap hook; all allocation and runtime setup remain upstream.
            upstream = iris_backend.task_setup_script
            iris_backend.task_setup_script = lambda commit, profile: upstream(commit, profile) + install
            try:
                return super().launch(path)
            finally:
                iris_backend.task_setup_script = upstream

    with contextlib.redirect_stdout(sys.stderr):
        result = execute_launch(config_path, backend=PDBThinkBackend())
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if str(result.state) in {"succeeded", "prepared", "submitted"} else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "launch"])
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        print(build_adapter(args.prepared))
    else:
        sys.exit(launch_with_adapter(args.config))
