"""Build or run the experiment through Marin's artifact and Iris launch path."""

from dataclasses import replace
from pathlib import Path

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from .adapter import run_with_adapter
from .pins import CONTEXT, MODEL, MODEL_REVISION, RESERVE, SEED, SKYRL_REVISION

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=32,
    policy_mini_batch_size=32,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)


def recipe(scale, data_uri, adapter_sha256):
    steps = {"smoke": 2, "pilot": 64, "epoch": 876}[scale]
    return {
        "entrypoint": "standard",
        "context_budget": {"request_window_tokens": CONTEXT, "max_new_tokens_per_turn": RESERVE, "max_turns": 1},
        "environment": {"env_class": "pdbthink"},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "gradient_checkpointing": True,
            "offload_optimizer_during_rollouts": True,
            "algorithm": {"advantage_estimator": "grpo", "use_kl_loss": False, "use_kl_in_reward": False},
            "epochs": 2 if scale == "smoke" else 1,
            "max_steps": steps,
            "update_epochs_per_batch": 1,
            "eval_batch_size": 32,
            "micro_forward_batch_size_per_gpu": 1,
            "eval_before_train": True,
            "eval_interval": 2 if scale == "smoke" else 8,
            "ckpt_interval": 2 if scale == "smoke" else 8,
            "resume_mode": "none",
            "logger": "wandb",
            "project_name": "snowball-pdbthink",
            "hf_hub_repo_id": None,
            "policy": {
                "optimizer_config": {"lr": 1.0e-6, "max_grad_norm": 1.0},
                "megatron_config": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 2,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 8,
                    "expert_tensor_parallel_size": 1,
                    "optimizer_checkpoint_sharding_type": "dp_reshardable",
                    "ddp_config": {
                        "overlap_grad_reduce": True,
                        "overlap_param_gather": True,
                        "grad_reduce_in_fp32": False,
                    },
                },
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "gpu_memory_utilization": 0.80,
            "enforce_eager": True,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "require_exact_chat_transport": True,
            "chat_template_kwargs": {"enable_thinking": True},
            "max_num_seqs": 16,
            "enable_prefix_caching": False,
            "engine_init_kwargs": {
                "default_chat_template_kwargs": {"enable_thinking": True},
            },
            "sampling_params": {"temperature": 1.0, "top_p": 1.0},
            "eval_sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_generate_length": CONTEXT},
            "eval_n_samples_per_prompt": 1,
            "trajectory_retention": {
                "enabled": True,
                "phases": ["train", "eval"],
                "sample_fraction": 1.0,
                "sample_count_per_step": 0,
                "max_bytes_per_step": None,
                "max_bytes_per_run": None,
                "required": True,
            },
        },
        "trajectory_runner": {"rollout_workers": {"num_workers": 4, "cpus_per_worker": 4}},
        "data": {"kind": "parquet", "shuffle": True, "train_data": [], "val_data": []},
        "extra_env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PDBTHINK_ADAPTER_URI": data_uri + "/adapter.zip",
            "PDBTHINK_ADAPTER_SHA256": adapter_sha256,
        },
    }


def build(scale, data_uri, model_uri, adapter_sha256, cluster, wandb_entity):
    if len(adapter_sha256) != 64:
        raise ValueError("Provide the immutable adapter SHA256 from preparation")
    runtime = SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON)
    if runtime.commit != SKYRL_REVISION:
        raise ValueError(f"Wrong MarinSkyRL revision: {runtime.commit}, expected {SKYRL_REVISION}")
    model = ArtifactStep.adopt(
        "models/bizon/snowball-pdbthink-step38",
        "2026.10.07.1",
        model_uri,
        kind=LevanterCheckpoint,
        config={"hf_repo": MODEL, "revision": MODEL_REVISION},
    )
    pool = ArtifactStep.adopt("documents/bizon/snowball-pdbthink", "2026.10.07.1", data_uri, kind=Artifact)
    name = f"checkpoints/bizon/snowball-pdbthink-{scale}"
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(recipe(scale, data_uri, adapter_sha256)),
            runtime=runtime,
            model=ArtifactHfModel(model, MODEL, MODEL_REVISION, relative_path=""),
            train_data=(
                ArtifactDataSource(pool, relative_path="smoke.parquet" if scale == "smoke" else "train.parquet"),
            ),
            validation_data=(ArtifactDataSource(pool, relative_path="monitor.parquet"),),
            topology=SkyRLTopology(num_nodes=5, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=64,
            memory="1800GB",
            disk="1000GB",
            priority="interactive",
            max_retries=0,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            job_timeout_seconds=21600 if scale == "smoke" else 86400,
            wandb_entity=wandb_entity,
        ),
        export_hf=True,
    )
    return replace(step, run=run_with_adapter)


@click.command()
@click.option("--scale", type=click.Choice(["smoke", "pilot", "epoch"]), required=True)
@click.option("--data-uri", required=True)
@click.option("--model-uri", required=True)
@click.option("--adapter-sha256", required=True)
@click.option("--cluster", default="cw-rno2a", show_default=True)
@click.option("--wandb-entity", default="marin-community", show_default=True)
@click.option("--write-launch", type=click.Path(path_type=Path))
@rl_build_options
def main(scale, data_uri, model_uri, adapter_sha256, cluster, wandb_entity, write_launch):
    step = build(scale, data_uri, model_uri, adapter_sha256, cluster, wandb_entity)
    if write_launch:
        prefix = "s3://marin-us-east-02a/marin"
        context = StepContext.for_run(step.path(prefix), prefix, runtime_args=step.runtime_args, deps=step.deps)
        write_launch.write_text(step.build_config(context).launch_config_yaml)
    return step


if __name__ == "__main__":
    main()
