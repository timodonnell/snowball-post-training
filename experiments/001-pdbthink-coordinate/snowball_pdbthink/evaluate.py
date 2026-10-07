"""Resumable tool-free evaluation with the full remaining native context budget."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from .pins import CONTEXT, MODEL_REVISION
from .scoring import load_scorer, score, tool_events


def summarize(rows, expected=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    families = {}
    for family, items in sorted(grouped.items()):
        families[family] = {
            "n": len(items),
            "successes": sum(r["reward"] for r in items),
            "accuracy": sum(r["reward"] for r in items) / len(items),
            "format_error_rate": sum(r["format_error"] for r in items) / len(items),
            "truncation_rate": sum(r["truncated"] for r in items) / len(items),
            "mean_output_tokens": sum(r["output_tokens"] for r in items) / len(items),
        }
    count = len(rows)
    return {
        "completed": count,
        "expected": expected,
        "coverage": count / expected if expected else None,
        "task_weighted_accuracy": sum(r["reward"] for r in rows) / count if count else None,
        "family_macro_accuracy": sum(r["accuracy"] for r in families.values()) / len(families) if families else None,
        "format_errors": sum(r["format_error"] for r in rows),
        "truncations": sum(r["truncated"] for r in rows),
        "tool_violations": sum(r["tool_violation"] for r in rows),
        "mean_output_tokens": sum(r["output_tokens"] for r in rows) / count if count else None,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "families": families,
    }


def request_for(row, model):
    return {
        "model": model,
        "messages": row["prompt"],
        "tools": [],
        "tool_choice": "none",
        "max_tokens": CONTEXT - row["input_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": int(hashlib.sha256(("eval-17:" + row["path"]).encode()).hexdigest()[:8], 16) % (2**31),
        "chat_template_kwargs": {"enable_thinking": True},
    }


async def evaluate(args):
    tasks = pq.read_table(args.tasks).to_pylist()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "model": args.model,
        "model_identity": args.model_identity,
        "tokenizer_revision": MODEL_REVISION,
        "context": CONTEXT,
        "temperature": 0.0,
        "budget": "32768 - exact native prompt tokens",
        "tools": [],
        "tool_choice": "none",
    }
    manifest = args.output / "evaluation.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("Cannot resume evaluation under a different model, cohort, or budget policy")
    manifest.write_text(json.dumps(contract, indent=2) + "\n")
    scorer = load_scorer(args.verifier)
    results = []
    pending = []
    for task in tasks:
        target = args.output / f"{task['path']}.json"
        if target.exists():
            results.append(json.loads(target.read_text())["result"])
        else:
            pending.append(task)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=30)) as client:

        async def one(task):
            async with semaphore:
                request = request_for(task, args.model)
                try:
                    response = await client.post(args.base_url.rstrip("/") + "/chat/completions", json=request)
                    response.raise_for_status()
                except httpx.HTTPError as error:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    # Iris capability URLs contain credentials; never include one in logs.
                    raise RuntimeError(
                        f"Model request failed for {task['path']}: {type(error).__name__}, status={status}"
                    ) from None
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                result = score(
                    scorer,
                    message.get("content") or "",
                    task["reward_model"]["ground_truth"],
                    truncated=choice["finish_reason"] == "length",
                    refusal=bool(message.get("refusal")),
                    tool_violation=tool_events(body),
                )
                if body["usage"]["prompt_tokens"] != task["input_tokens"]:
                    raise ValueError(f"Served prompt token count mismatch: {task['path']}")
                result.update(
                    path=task["path"],
                    family=task["family"],
                    source_group=task["source_group"],
                    prompt_sha256=task["prompt_sha256"],
                    finish_reason=choice["finish_reason"],
                    output_tokens=body["usage"]["completion_tokens"],
                )
                target = args.output / f"{task['path']}.json"
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps({"request": request, "response": body, "result": result}) + "\n")
                temporary.replace(target)
                results.append(result)
                summary = summarize(results, len(tasks))
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(
                    f"{len(results)}/{len(tasks)} accuracy={summary['task_weighted_accuracy']:.4f} macro={summary['family_macro_accuracy']:.4f}",
                    flush=True,
                )

        await asyncio.gather(*(one(task) for task in pending))
    (args.output / "summary.json").write_text(json.dumps(summarize(results, len(tasks)), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
