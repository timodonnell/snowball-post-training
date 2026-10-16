"""Run frozen original-benchmark requests on one owned CoreWeave Iris server."""

import argparse
import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .evaluate import request_completion
from .scoring import tool_events


def validate_response(body, job):
    if len(body["choices"]) != 1:
        raise ValueError("Benchmark requires exactly one completion")
    if body["usage"]["prompt_tokens"] != job["prompt_tokens"]:
        raise ValueError("Prompt token mismatch: " + job["render_id"])
    choice = body["choices"][0]
    if choice["finish_reason"] not in {"stop", "length"}:
        raise ValueError("Unexpected completion termination")
    if choice["finish_reason"] == "length" and body["usage"]["completion_tokens"] != job["body"]["max_tokens"]:
        raise ValueError("Unexpected output budget clamp: " + job["render_id"])
    if tool_events(body):
        raise ValueError("Tool event returned despite disabled tools")


async def collect(base_url, model, plan, output, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=30)) as client:

        async def one(job):
            name = hashlib.sha256(job["render_id"].encode()).hexdigest()[:20] + ".json"
            target = output / name
            if target.exists():
                return
            async with semaphore:
                request = {"model": model, **job["body"]}
                body = await request_completion(client, base_url + "/chat/completions", request, job["render_id"])
                try:
                    validate_response(body, job)
                except ValueError:
                    target.with_suffix(".rejected.json").write_text(json.dumps({"request": request, "response": body}))
                    raise
                choice = body["choices"][0]
                record = {
                    "render_id": job["render_id"],
                    "request": request,
                    "response": body,
                    "saved_at": datetime.now(UTC).isoformat(),
                }
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, indent=2) + "\n")
                temporary.replace(target)
                print(
                    json.dumps(
                        {
                            "render_id": job["render_id"],
                            "tokens": body["usage"],
                            "finish_reason": choice["finish_reason"],
                        }
                    ),
                    flush=True,
                )

        await asyncio.gather(*(one(job) for job in plan["jobs"]))


def main():
    from iris.cli.connect import open_iris_client
    from rigging.timing import Duration

    from .serve_evaluation import submit_server

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--server-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gpus", type=int, choices=[2, 4, 8], default=8)
    parser.add_argument("--gpu-variant", choices=["H100", "GB200"], default="H100")
    parser.add_argument("--target-cluster", default="cw-rno2a")
    parser.add_argument("--export-verification", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.export_verification:
        deadline = time.monotonic() + 7200
        while not args.export_verification.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Selected checkpoint export did not finish within two hours")
            time.sleep(30)
        if json.loads(args.export_verification.read_text())["model_uri"] != args.model_uri:
            raise ValueError("Export verification does not identify requested model")
    contract = {
        "model_uri": args.model_uri,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "concurrency": args.concurrency,
        "server": "serve_evaluation.submit_server",
        "gpus": f"{args.gpus} CoreWeave {args.gpu_variant}",
        "cluster": args.target_cluster,
        "sampling": "Frozen clean-coordinate-v4 requests; one completion, seed 20260929, temperature 0.6",
    }
    path = args.output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Evaluation contract changed")
    path.write_text(json.dumps(contract, indent=2) + "\n")
    with open_iris_client(config_file=Path("lib/iris/config/marin.yaml"), workspace=Path.cwd()) as iris:
        server = submit_server(
            iris,
            args.model_uri,
            args.server_name,
            gpus=args.gpus,
            gpu_variant=args.gpu_variant,
            target_cluster=args.target_cluster,
        )
        (args.output / "server.json").write_text(
            json.dumps({"job": str(server), "started_at": datetime.now(UTC).isoformat()})
        )
        try:
            endpoint = "/serve/" + args.server_name
            deadline = time.monotonic() + 2400
            while not iris.list_endpoint_instances(endpoint):
                from iris.resources.state import is_job_finished

                if is_job_finished(iris.job_state(server.job_id)):
                    raise RuntimeError("Benchmark server terminated before endpoint readiness")
                if time.monotonic() > deadline:
                    raise TimeoutError("Original benchmark server did not become ready")
                time.sleep(30)
            capability = iris.mint_endpoint_token(endpoint, ttl=Duration.from_hours(4))
            base_url = capability.capability_url.rstrip("/") + "/v1"
            try:
                response = httpx.get(base_url + "/models", timeout=60)
                response.raise_for_status()
                model = response.json()["data"][0]["id"]
            except httpx.HTTPError as error:
                raise RuntimeError("Model identity request failed: " + type(error).__name__) from None
            print("Original benchmark server ready", args.server_name, flush=True)
            asyncio.run(collect(base_url, model, plan, args.output, args.concurrency))
            (args.output / "complete.json").write_text(
                json.dumps({"completed": len(plan["jobs"]), "at": datetime.now(UTC).isoformat()})
            )
        finally:
            server.cancel()
            print("Released owned benchmark server", str(server), flush=True)


if __name__ == "__main__":
    main()
