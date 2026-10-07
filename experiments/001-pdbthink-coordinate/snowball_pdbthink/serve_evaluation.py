"""Serve either frozen checkpoint with the same Marin inference configuration."""

import argparse
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig, create_environment
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.inference.config import IrisConfig, ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris import IrisServiceConfig, run_iris_service

LOADER_ENV = {"RUNAI_STREAMER_CONCURRENCY": "4", "RUNAI_STREAMER_S3_REQUEST_TIMEOUT_MS": "120000"}


def submit_server(client, model_uri, name, hours=4):
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512g", disk="200g", target_cluster="cw-rno2a")
    environment = create_environment(workspace=str(Path.cwd()), env_vars=LOADER_ENV)
    service = IrisServiceConfig(
        model=ServedModelConfig(weights=model_uri, dtype="bfloat16", max_model_len=32768, tensor_parallel_size=1),
        engine=VllmEngineConfig(
            launcher=VllmLauncherType.CUDA,
            source=VllmSource.MARIN_FORK,
            max_num_batched_tokens=16384,
            max_num_seqs=16,
            extra_args=(
                "--data-parallel-size",
                "8",
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
                "--default-chat-template-kwargs",
                '{"enable_thinking":true}',
                "--generation-config",
                "vllm",
            ),
        ),
        iris=IrisConfig(
            worker_resources=resources, worker_environment=environment, max_retries_failure=0, max_retries_preemption=0
        ),
        endpoint_name="/serve/" + name,
        timeout_hours=hours,
        controller_proxy_timeout_seconds=3600,
    )
    return client.submit(
        entrypoint=Entrypoint.from_callable(run_iris_service, service),
        name=name,
        resources=convert_resources(resources),
        environment=EnvironmentSpec(env_vars=LOADER_ENV),
        constraints=convert_constraints(resources),
        ports=["http"],
        max_retries_failure=0,
        max_retries_preemption=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hours", type=int, default=4)
    args = parser.parse_args()
    with (
        connect_controller(cluster_name="marin") as connection,
        IrisClient.remote(connection.url, workspace=Path.cwd(), credentials=connection.credentials) as client,
    ):
        job = submit_server(client, args.model_uri, args.name, args.hours)
        print("Submitted evaluation server", job, flush=True)


if __name__ == "__main__":
    main()
