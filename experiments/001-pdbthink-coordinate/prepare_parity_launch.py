"""Build immutable instrumentation and paired zero-update launch specifications."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/001/parity-probe"
PREFIX = "s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/parity/2026.10.16-v8"
SOURCE = ROOT / "data/MarinSkyRL"
FILES = [
    "cloud/iris/role_plan.py",
    "skyrl-train/skyrl_train/callbacks/builtin.py",
    "skyrl-train/skyrl_train/workers/megatron/megatron_worker.py",
    "skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py",
    "skyrl-train/skyrl_train/inference_engines/inference_engine_client.py",
    "skyrl-train/skyrl_train/weight_sync/expert_block/receiver.py",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    hashes = {name: digest((SOURCE / name).read_bytes()) for name in FILES}
    bundle = {
        name: (HERE / "snowball_pdbthink" / name).read_bytes() for name in ("parity_install.py", "parity_runtime.py")
    }
    bundle["source-hashes.json"] = json.dumps(hashes, sort_keys=True).encode()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in sorted(bundle.items()):
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 10, 16, 0, 0, 0)), contents)
    code = payload.getvalue()
    code_name = "instrumentation-" + digest(code) + ".zip"
    (OUT / code_name).write_bytes(code)
    panel = json.loads((OUT / "panel-manifest.json").read_text())
    for step in (384, 512):
        cfg = yaml.safe_load((HERE / "epoch-resume-launch.yaml").read_text())
        name = f"snowball-parity-{step}-20261016-v8"
        root = PREFIX + f"/step-{step}"
        cfg["run"].update(id=name, attempt_id=digest(code)[:12], export_hf=False, submission="wait")
        cfg["iris"].update(
            cluster="cw-us-east-08a",
            target_cluster="cw-us-east-08a",
            cluster_config="lib/iris/config/cw-us-east-08a.yaml",
            job_name=name,
            priority="interactive",
            timeout=21600,
        )
        cfg["iris"]["allocation"].update(num_nodes=12, gpus_per_node=4, gpu_variant="GB200", cpu=32, memory="600GB")
        cfg["ray"].update(rendezvous_dir=root + "/rendezvous", log_dir=root + "/ray-logs")
        cfg["artifacts"].update(
            checkpoint_root=root + "/unused-checkpoints",
            export_root=root + "/unused-exports",
            attempts_root=root + "/attempts",
            resolved_config_uri=root + "/resolved-launch.json",
            terminal_manifest_uri=root + "/terminal.json",
        )
        env = cfg["runtime"]["task_env"]
        env.update(
            SNOWBALL_PARITY_TOPOLOGY="nvlink.domain.preferred",
            SNOWBALL_PARITY_CODE_URI=PREFIX + "/" + code_name,
            SNOWBALL_PARITY_CODE_SHA256=digest(code),
            SNOWBALL_PARITY_PANEL_URI=PREFIX + "/frozen-panel.json.gz",
            SNOWBALL_PARITY_PANEL_SHA256=panel["compressed_sha256"],
        )
        trainer, generator = cfg["skyrl"]["trainer"], cfg["skyrl"]["generator"]
        trainer.update(
            max_steps=step + 1,
            eval_before_train=False,
            eval_interval=0,
            ckpt_interval=0,
            hf_save_interval=0,
            restore_dataloader_state=False,
            logger="console",
        )
        trainer["resume_path"] = (
            "s3://marin-us-east-02a/tmp/ttl=14d/skyrl/marin/checkpoints/bizon/"
            f"snowball-pdbthink-epoch/2026.10.15.1/checkpoints/global_step_{step}"
        )
        trainer["placement"].update(
            policy_num_nodes=8, policy_num_gpus_per_node=4, ref_num_nodes=8, ref_num_gpus_per_node=4
        )
        trainer["policy"]["megatron_config"].update(moe_router_replay=True, check_train_eval_parity=True)
        trainer["mismatch_probe"] = {
            "enabled": True,
            "updates": 0,
            "seed": panel["seed"],
            "prompts": {"count": panel["prompts"], "samples_per_prompt": 2},
            "archive_uri": root + "/probe",
            "reuse_probe": None,
            "extra_trainer_modes": ["router_replay"],
            "rescore_prefix_cache": "off",
        }
        generator["engine_init_kwargs"].update(
            logprobs_mode="processed_logprobs", generation_config="vllm", enable_return_routed_experts=True
        )
        generator["trajectory_retention"]["enabled"] = False
        (OUT / f"launch-{step}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT / "launch-manifest.json").write_text(
        json.dumps(
            {
                "prefix": PREFIX,
                "code_file": code_name,
                "code_sha256": digest(code),
                "source_sha256": hashes,
                "panel": panel["compressed_sha256"],
                "optimizer_updates": 0,
                "gpu_allocation_per_job": "48 CoreWeave GB200, sequential checkpoint jobs",
                "changes_from_original": [
                    "GB200 instead of H100; unchanged TP1/PP2/EP16 learner and two DP8/EP8 rollout replicas",
                    "Read-only route capture, probability replay and weight verification",
                    "Neutral temperature-1 scoring, no minimum-token EOS suppression",
                    "Checkpoint/evaluation/export callbacks disabled; optimizer call raises",
                    "Prefer a common NVLink rack; allow InfiniBand between racks because free capacity is fragmented",
                    "Normal interactive research priority after an opportunistic batch attempt was preempted",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"prefix": PREFIX, "bundle": code_name, "launches": [384, 512]}))


if __name__ == "__main__":
    main()
