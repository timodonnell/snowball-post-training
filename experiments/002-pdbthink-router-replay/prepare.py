"""Build immutable retry specifications and a narrowly scoped runtime overlay."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "experiments/001-pdbthink-coordinate"
SOURCE = ROOT / "data/MarinSkyRL"
OUT = ROOT / "runs/002-replay"
BASE = "5663f6129f53bb6aaa09549fa6cb423571503e94"
PREFIX = "s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/replay/2026.10.21-v2"
PATCHES = {
    "prediction-position-routes.patch": "b1264f778cc69fd0bfd6a0863d0775da7c0b7daa",
    "recomputation-routes.patch": "81ae0ef7033c500207b191a2ae7c3806f6dc78ff",
}


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def replace_once(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(f"Runtime source anchor changed: {path}: {before}")
    path.write_text(text.replace(before, after))


def build_overlay():
    stage = OUT / "runtime"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "-C", str(SOURCE), "archive", BASE])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(stage, filter="data")
    subprocess.run(["git", "init", "-q", str(stage)], check=True)
    changed = set()
    for name in PATCHES:
        patch = HERE / "patches" / name
        for line in patch.read_text().splitlines():
            if line.startswith("+++ b/"):
                changed.add(line[6:])
        subprocess.run(["git", "apply", str(patch)], cwd=stage, check=True)
    test_patch = subprocess.check_output(
        [
            "git",
            "-C",
            str(SOURCE),
            "show",
            PATCHES["prediction-position-routes.patch"],
            "--format=",
            "--",
            "skyrl-train/tests/cpu",
        ]
    )
    subprocess.run(["git", "apply"], input=test_patch, cwd=stage, check=True)
    prefix = "skyrl-train/skyrl_train/"
    worker = prefix + "workers/megatron/megatron_worker.py"
    changed.add(worker)
    path = stage / worker
    replace_once(
        path,
        "from skyrl_train.workers.grug_validation import GrugValidationSnapshot",
        "from skyrl_train.workers.grug_validation import GrugValidationSnapshot\n"
        "from snowball_replay_runtime import frozen_biases, check_biases",
    )
    replace_once(
        path,
        "        self._expert_block_sender = (\n            ExpertBlockSender(self, mpu)\n"
        "            if self.cfg.generator.weight_sync_transport == WeightSyncTransport.EXPERT_BLOCK\n"
        "            else None\n        )",
        "        self._expert_block_sender = ExpertBlockSender(self, mpu)",
    )
    anchor = (
        '    def _ppo_train_impl(self, train_data, timing: PhaseBreakdown) -> "TrainingOutputBatch":\n'
        '        """Train through Megatron Core\'s pipeline scheduler."""\n'
    )
    replace_once(path, anchor, anchor + "        replay_bias_before = frozen_biases(self)\n")
    replace_once(
        path,
        '        output.metadata = {"train_status": status_mean}\n        return output',
        "        check_biases(self, replay_bias_before)\n"
        '        status_mean["frozen_router_bias_unchanged"] = 1.0\n'
        '        output.metadata = {"train_status": status_mean}\n        return output',
    )
    callbacks = prefix + "callbacks/builtin.py"
    changed.add(callbacks)
    replace_once(
        stage / callbacks,
        "from skyrl_train.mismatch_probe.callback import MismatchProbeCallback",
        "from skyrl_train.mismatch_probe.callback import MismatchProbeCallback\n"
        "from snowball_replay_runtime import ReplayAuditCallback",
    )
    replace_once(
        stage / callbacks,
        "        callbacks.append(MismatchProbeCallback(cfg))\n    return callbacks",
        "        callbacks.append(MismatchProbeCallback(cfg))\n"
        "    callbacks.append(ReplayAuditCallback(cfg))\n    return callbacks",
    )
    manifest = {"base_commit": BASE, "upstream_backports": PATCHES, "files": {}}
    bundle = {}
    for name in sorted(changed):
        base_file = subprocess.run(
            ["git", "-C", str(SOURCE), "show", BASE + ":" + name], capture_output=True, check=False
        )
        original = base_file.stdout if base_file.returncode == 0 else None
        installed = (stage / name).read_bytes()
        manifest["files"][name] = {
            "base_sha256": sha(original) if original is not None else None,
            "installed_sha256": sha(installed),
        }
        bundle["overlay/" + name] = installed
    for name in ("replay_install.py", "replay_runtime.py", "parity_runtime.py"):
        bundle[name] = (PREVIOUS / "snowball_pdbthink" / name).read_bytes()
    bundle["overlay-manifest.json"] = json.dumps(manifest, indent=2).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in sorted(bundle.items()):
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0)), contents)
    payload = output.getvalue()
    bundle_name = "overlay-" + sha(payload) + ".zip"
    (OUT / bundle_name).write_bytes(payload)
    (HERE / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Import the exact installed modules in local CPU tests.
    (stage / "skyrl-train/snowball_replay_runtime.py").write_bytes(bundle["replay_runtime.py"])
    (stage / "skyrl-train/snowball_parity_runtime.py").write_bytes(bundle["parity_runtime.py"])
    return bundle_name, sha(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=["H100", "GB200"], default="GB200")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle, digest = build_overlay()
    cfg = yaml.safe_load((PREVIOUS / "epoch-launch.yaml").read_text())
    name = "snowball-pdbthink-replay-20261021-v2"
    cfg["run"].update(id=name, attempt_id=digest[:12], export_hf=False, submission="wait")
    cluster = "cw-us-east-08a" if args.hardware == "GB200" else "cw-rno2a"
    cfg["iris"].update(
        cluster=cluster,
        target_cluster=cluster,
        cluster_config=f"lib/iris/config/{cluster}.yaml",
        job_name=name,
        timeout=345600,
        max_retries=0,
        priority="interactive",
    )
    if args.hardware == "GB200":
        cfg["iris"]["allocation"].update(num_nodes=12, gpus_per_node=4, gpu_variant="GB200", cpu=32, memory="600GB")
    cfg["ray"].update(rendezvous_dir=PREFIX + "/rendezvous", log_dir=PREFIX + "/ray-logs")
    cfg["artifacts"].update(
        checkpoint_root=PREFIX + "/checkpoints",
        export_root=PREFIX + "/exports",
        attempts_root=PREFIX + "/attempts",
        resolved_config_uri=PREFIX + "/resolved-launch.json",
        terminal_manifest_uri=PREFIX + "/terminal.json",
    )
    env = cfg["runtime"]["task_env"]
    env.update(
        SNOWBALL_REPLAY_CODE_URI=PREFIX + "/" + bundle,
        SNOWBALL_REPLAY_CODE_SHA256=digest,
        SNOWBALL_REPLAY_AUDIT_URI=PREFIX + "/audit",
    )
    trainer, generator = cfg["skyrl"]["trainer"], cfg["skyrl"]["generator"]
    trainer.update(
        resume_mode="none",
        resume_path=None,
        max_steps=876,
        eval_before_train=True,
        eval_interval=32,
        ckpt_interval=32,
        hf_save_interval=32,
        restore_dataloader_state=False,
    )
    trainer["policy"]["grug_query_bias_update_mode"] = "frozen"
    trainer["policy"]["megatron_config"].update(moe_router_replay=True)
    trainer["algorithm"].update(
        policy_loss_type="regular",
        off_policy_correction="custom",
        off_policy_correction_rules=[{"kind": "token", "action": "truncate", "high": 4.0}],
    )
    if args.hardware == "GB200":
        trainer["placement"].update(
            policy_num_nodes=8, policy_num_gpus_per_node=4, ref_num_nodes=8, ref_num_gpus_per_node=4
        )
    generator["engine_init_kwargs"].update(enable_return_routed_experts=True, logprobs_mode="processed_logprobs")
    generator["sampling_params"].update(top_k=-1, top_p=1.0, min_p=0.0, repetition_penalty=1.0)
    generator["trajectory_retention"]["publish_timeout_seconds"] = 600
    (HERE / "launch.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT / "launch.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (HERE / "design.json").write_text(
        json.dumps(
            {
                "run": name,
                "remote_root": PREFIX,
                "overlay": bundle,
                "overlay_sha256": digest,
                "starting_checkpoint": cfg["inputs"]["model"],
                "fresh_optimizer": True,
                "training_tasks": 28045,
                "validation_tasks": 1003,
                "validation_interval": 32,
                "optimizer_updates": 876,
                "learning_rate": 1e-6,
                "samples_per_prompt": 4,
                "batch_prompts": 32,
                "rollout_temperature": 1.0,
                "importance_cap": 4,
                "grug_query_bias_update_mode": "frozen",
                "hardware": args.hardware,
                "score_centering": "Deferred: absent from inspected upstream main 82396a7; needs new top-K distribution path",
                "total_router_recall": "Deferred: replaying weights changes router gradients and requires capture support",
                "kernel_alignment": "Deferred: extensive backend/model-specific work; no claim of zero mismatch",
                "sources": ["https://kiddyboots216.github.io/mismatch/", "https://arxiv.org/abs/2609.20807"],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"bundle": str(OUT / bundle), "remote": PREFIX, "launch": str(OUT / "launch.yaml")}))


if __name__ == "__main__":
    main()
