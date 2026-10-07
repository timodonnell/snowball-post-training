"""Evaluate through a scoped Iris capability kept only in process memory."""

import argparse
import asyncio
import time
from pathlib import Path

import httpx
from iris.cli.connect import open_iris_client
from rigging.timing import Duration

from .evaluate import evaluate


def evaluate_endpoint(args):
    with open_iris_client(config_file=args.cluster_config, workspace=Path.cwd()) as iris:
        started = time.monotonic()
        while not iris.list_endpoint_instances(args.endpoint):
            if time.monotonic() - started > args.ready_timeout:
                raise TimeoutError("Iris model endpoint did not become ready")
            print("Waiting for baseline/evaluation server readiness", flush=True)
            time.sleep(30)
        capability = iris.mint_endpoint_token(args.endpoint, ttl=Duration.from_hours(4))
        if not capability.capability_url:
            raise RuntimeError("Iris did not return a reachable scoped capability")
        args.base_url = capability.capability_url.rstrip("/") + "/v1"
        try:
            response = httpx.get(args.base_url + "/models", timeout=60)
            response.raise_for_status()
            args.model = response.json()["data"][0]["id"]
        except httpx.HTTPError as error:
            raise RuntimeError(f"Cannot query Iris model identity: {type(error).__name__}") from None
        print("Model endpoint ready; beginning frozen-cohort evaluation", flush=True)
        asyncio.run(evaluate(args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--cluster-config", type=Path, default=Path("lib/iris/config/marin.yaml"))
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    evaluate_endpoint(parser.parse_args())


if __name__ == "__main__":
    main()
