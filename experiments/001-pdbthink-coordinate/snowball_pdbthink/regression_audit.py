"""Replay retained RL evidence and analyze the scaled run's regression.

The collector reads existing archives only. It never generates new answers,
changes rewards, or changes the running training job.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from .pilot_evidence import answer_state, coordinate_label_audit, group_bootstrap
from .scoring import load_scorer, score, tool_events


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect_evals(args):
    import boto3
    from botocore.config import Config

    client = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
                          config=Config(s3={"addressing_style": "virtual"}))
    bucket, prefix = args.retention_root.removeprefix("s3://").split("/", 1)
    ledger = read_json(args.evidence / "retention-ledger.json")
    manifests = []
    for key, metadata in sorted(ledger["archives"].items()):
        if "phase=eval/" not in key:
            continue
        step = int(re.search(r"step=(\d+)", key)[1])
        if step not in args.steps:
            continue
        path = args.evidence / "eval-archives" / Path(key).name
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            client.download_file(bucket, prefix.rstrip("/") + "/" + key, str(path))
        payload = path.read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == path.stem
        records = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for item in json.loads(archive.read("manifest.json"))["records"]:
                record = json.loads(gzip.decompress(archive.read(item["entry"])))
                extras = record["trajectory"]["environment_extras"]
                response = record["response"]
                messages = record["prompt"]["messages"]
                prompt_hash = hashlib.sha256((messages[0]["content"] + "\n" + messages[1]["content"]).encode()).hexdigest()
                assert prompt_hash == extras["prompt_sha256"]
                records.append({
                    "path": extras["path"], "prompt_sha256": prompt_hash,
                    "messages": [m for m in response["messages"] if m.get("role") == "assistant"],
                    "input_tokens": len(record["prompt"]["token_ids"]), "output_tokens": len(response["token_ids"]),
                    "stop_reason": response["stop_reason"], "reward": record["reward"]["outcome"],
                    "has_tools": tool_events(response["messages"]), "verification_result": record["verification_result"],
                    "exception_type": record["disposition"]["exception_type"],
                })
        write_json(args.evidence / "evals" / str(step) / "native-responses.json", records)
        manifests.append({"key": key, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        print(f"Recovered original API responses for evaluation {step}: {len(records)}", flush=True)
    write_json(args.evidence / "eval-retention-manifest.json", manifests)


def collect_training(args):
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        config=Config(s3={"addressing_style": "virtual"}, max_pool_connections=12),
    )
    columns = ["path", "family", "source_group", "input_tokens", "prompt_sha256", "reward_model"]
    tasks = {r["path"]: r for r in pq.read_table(args.prepared / "train.parquet", columns=columns).to_pylist()}
    scorer = load_scorer(args.prepared / "native_verifier")
    ledger = read_json(args.evidence / "retention-ledger.json")
    prefix = args.retention_root.removeprefix("s3://")
    bucket, prefix = prefix.split("/", 1)
    entries = [
        (key, value) for key, value in sorted(ledger["archives"].items())
        if "phase=train/" in key and int(re.search(r"step=(\d+)", key)[1]) <= args.max_step
    ]
    cache = args.evidence / "training-archives"
    cache.mkdir(exist_ok=True)

    def collect(entry):
        key, metadata = entry
        path = cache / Path(key).name
        if not path.exists():
            client.download_file(bucket, prefix.rstrip("/") + "/" + key, str(path))
        payload = path.read_bytes()
        if len(payload) != metadata["bytes"]:
            raise ValueError(f"Archive length mismatch: {key}")
        assert hashlib.sha256(payload).hexdigest() == path.stem
        rows = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            for item in manifest["records"]:
                record = json.loads(gzip.decompress(archive.read(item["entry"])))
                extras = record["trajectory"]["environment_extras"]
                task = tasks[extras["path"]]
                messages = record["prompt"]["messages"]
                prompt_hash = hashlib.sha256((messages[0]["content"] + "\n" + messages[1]["content"]).encode()).hexdigest()
                assert len(messages) == 2 and prompt_hash == task["prompt_sha256"] == extras["prompt_sha256"]
                assert extras["reward_model"] == task["reward_model"]
                assert len(record["prompt"]["token_ids"]) == task["input_tokens"]
                assert extras["family"] == task["family"] and extras["source_group"] == task["source_group"]
                response = record["response"]
                assistant = [m for m in response["messages"] if m.get("role") == "assistant"]
                result = score(
                    scorer, "".join(m.get("content") or "" for m in assistant),
                    task["reward_model"]["ground_truth"], truncated=response["stop_reason"] == "length",
                    refusal=any(bool(m.get("refusal")) for m in assistant), tool_violation=tool_events(response["messages"]),
                )
                assert result["reward"] == record["reward"]["outcome"] == record["reward"]["shaped"]
                assert record["disposition"]["exception_type"] is None
                assert record["provenance"]["model_version_step"] == record["global_step"] - 1
                rows.append({
                    "id": record["record_id"], "step": record["global_step"], "task": task["path"],
                    "family": task["family"], "group": task["source_group"], "reward": result["reward"],
                    "format_error": result["format_error"], "truncated": result["truncated"],
                    "input_tokens": task["input_tokens"], "output_tokens": len(response["token_ids"]),
                    "state": answer_state(result),
                })
        assert {r["id"] for r in rows} == set(metadata["record_ids"])
        return rows, {"key": key, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    records, manifests = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (rows, manifest) in enumerate(pool.map(collect, entries), 1):
            records.extend(rows)
            manifests.append(manifest)
            if index % 500 == 0:
                print(f"Audited {index}/{len(entries)} archives, {len(records)} responses", flush=True)
    assert len({r["id"] for r in records}) == len(records)
    write_json(args.evidence / "training-records.json", records)
    write_json(args.evidence / "training-download-manifest.json", manifests)
    print(f"Verified {len(records)} training responses from {len(manifests)} archives", flush=True)


def load_evals(args, tasks, scorer):
    cohorts, aggregates = {}, {}
    prompt_hashes = {}
    for step in args.steps:
        records = {}
        directory = args.evidence / "evals" / str(step)
        native = {r["path"]: r for r in read_json(directory / "native-responses.json")}
        aggregate = json.loads((directory / "aggregated_results.jsonl").read_text().splitlines()[-1])
        for path in sorted(directory.glob("*.jsonl")):
            if path.name == "aggregated_results.jsonl":
                continue
            for line in path.read_text().splitlines():
                record = json.loads(line)
                extras = record["env_extras"]
                task_id = extras["path"]
                task = tasks[task_id]
                assert task_id not in records
                for key in ("family", "source_group", "prompt_sha256", "input_tokens", "reward_model"):
                    assert extras[key] == task[key], (step, task_id, key)
                rendered_hash = hashlib.sha256(record["input_prompt"].encode()).hexdigest()
                assert prompt_hashes.setdefault(task_id, rendered_hash) == rendered_hash
                assert record["exception_type"] is None and record["server_error"] is None
                original = native[task_id]
                assert original["exception_type"] is None
                assert original["prompt_sha256"] == task["prompt_sha256"]
                assert original["input_tokens"] == task["input_tokens"]
                assert original["output_tokens"] == len(record["score"])
                messages = original["messages"]
                text = "".join(m.get("content") or "" for m in messages)
                result = score(
                    scorer, text, task["reward_model"]["ground_truth"],
                    truncated=record["stop_reason"] == "length",
                    refusal=any(bool(m.get("refusal")) for m in messages), tool_violation=original["has_tools"],
                )
                assert result["reward"] == sum(record["score"]) == original["reward"], (step, task_id)
                records[task_id] = {
                    "family": task["family"], "group": task["source_group"], "reward": result["reward"],
                    "state": answer_state(result), "tokens": len(record["score"]),
                    "format_error": result["format_error"], "truncated": result["truncated"],
                    "parsed": result["outcome"]["parsed"], "text": text,
                }
        assert records.keys() == tasks.keys()
        for flag, metric in (("reward", "avg_score"), ("format_error", "environment/pdbthink/format_error"),
                             ("truncated", "environment/pdbthink/truncated")):
            actual = statistics.mean(r[flag] for r in records.values())
            assert math.isclose(actual, aggregate[f"eval/all/{metric}"]), (step, flag, actual, aggregate[f"eval/all/{metric}"])
        assert math.isclose(statistics.mean(r["tokens"] for r in records.values()), aggregate["eval/all/response_tokens_mean"])
        cohorts[step], aggregates[step] = records, aggregate
    return cohorts, aggregates


HISTORY_KEYS = [
    "trainer/global_step", "reward/avg_raw_reward", "policy/raw_grad_norm", "policy/policy_entropy",
    "policy/policy_lr", "policy/skipped_steps", "policy/policy_update_steps", "policy/ppo_clip_ratio",
    "policy/ppo_ratio_exact_unit_fraction", "policy/mismatch/pooled/kl_k3", "policy/mismatch/pooled/ess_fraction",
    "policy/mismatch/pooled/frac_outside_0_5_2", "environment/format_error", "environment/truncated",
    "generate/avg_num_tokens", "reward/informative_group_fraction", "timing/step", "async/staleness_max",
    "generate/trajectory_retention/publish_timeouts", "generate/trajectory_retention/write_errors",
    "generate/tis/exact_match_fraction", "generate/tis/alignment_fail_count",
]


def standardized_training_windows(steps, windows, reference_end=448):
    """Hold family proportions fixed, excluding absent families explicitly.

    This controls family composition only, not within-family task difficulty.
    All four responses per prompt stay in the descriptive response average.
    """
    reference = Counter()
    for row in steps:
        if row["step"] <= reference_end:
            reference.update(row["family_counts"])
    if not reference:
        raise ValueError("No reference observations for family standardization")
    summaries = []
    for low, high in windows:
        rows = [r for r in steps if low <= r["step"] <= high]
        if not rows:
            raise ValueError(f"Empty training window: {low}-{high}")
        counts, successes = Counter(), Counter()
        for row in rows:
            counts.update(row["family_counts"])
            successes.update(row["family_successes"])
        n = sum(counts.values())
        summaries.append({
            "first_step": low, "last_step": high, "responses": n,
            "family_counts": dict(sorted(counts.items())), "family_successes": dict(sorted(successes.items())),
            "raw_accuracy": sum(successes.values()) / n,
            "mean_input_tokens": sum(r["mean_input_tokens"] * r["responses"] for r in rows) / n,
            "family_mix_total_variation": sum(
                abs(counts[f] / n - reference[f] / reference.total()) for f in counts.keys() | reference.keys()
            ) / 2,
        })
    common = sorted(set(reference).intersection(*(r["family_counts"] for r in summaries)))
    if not common:
        raise ValueError("No families present in every training window")
    retained = sum(reference[f] for f in common)
    weights = {f: reference[f] / retained for f in common}
    for row in summaries:
        row["fixed_family_mix_accuracy"] = sum(
            weights[f] * row["family_successes"].get(f, 0) / row["family_counts"][f] for f in common
        )
        row["included_response_fraction"] = sum(row["family_counts"][f] for f in common) / row["responses"]
    return {
        "reference_first_step": min(r["step"] for r in steps), "reference_last_step": reference_end,
        "weights": weights, "reference_coverage": retained / reference.total(),
        "excluded_families": sorted(set(reference) - set(common)), "windows": summaries,
        "limitation": "Descriptive family-composition adjustment; does not control within-family difficulty.",
    }


def analyze(args):
    tasks = {r["path"]: r for r in pq.read_table(args.prepared / "validation.parquet").to_pylist()}
    scorer = load_scorer(args.prepared / "native_verifier")
    cohorts, _aggregates = load_evals(args, tasks, scorer)
    history = [
        {k: r[k] for k in HISTORY_KEYS if k in r}
        for r in read_json(args.evidence / "training-history.json")
        if "policy/raw_grad_norm" in r and r["trainer/global_step"] <= args.max_step
    ]
    history.sort(key=lambda r: r["trainer/global_step"])
    assert len({r["trainer/global_step"] for r in history}) == len(history)
    old, new = cohorts[args.before], cohorts[args.after]
    transitions = defaultdict(Counter)
    families, groups, examples = {}, {}, []
    for task_id in sorted(tasks):
        a, b = old[task_id], new[task_id]
        transitions[a["state"]][b["state"]] += 1
        group = groups.setdefault(a["group"], {"n": 0, "baseline": 0, "trained": 0})
        group["n"] += 1
        group["baseline"] += int(a["reward"])
        group["trained"] += int(b["reward"])
    for family in sorted({r["family"] for r in tasks.values()}):
        ids = sorted(k for k in tasks if tasks[k]["family"] == family)
        families[family] = {
            "n": len(ids), "before_correct": sum(old[k]["reward"] for k in ids),
            "after_correct": sum(new[k]["reward"] for k in ids),
            "lost": sum(bool(old[k]["reward"] and not new[k]["reward"]) for k in ids),
            "gained": sum(bool(new[k]["reward"] and not old[k]["reward"]) for k in ids),
            "before_format_errors": sum(old[k]["format_error"] for k in ids),
            "after_format_errors": sum(new[k]["format_error"] for k in ids),
            "before_tokens": statistics.mean(old[k]["tokens"] for k in ids),
            "after_tokens": statistics.mean(new[k]["tokens"] for k in ids),
            "most_common_after_answers": Counter(json.dumps(new[k]["parsed"].get("value")) for k in ids).most_common(3),
        }
        losses = [k for k in ids if old[k]["reward"] and not new[k]["reward"]]
        if losses and family in ("G01", "G03", "P03", "S09"):
            key = losses[0]  # Stable task-ID order, not chosen for a dramatic failure.
            examples.append({
                "task": key, "family": family, "selection": "First lost task in sorted task-ID order within family",
                "gold": json.loads(tasks[key]["reward_model"]["ground_truth"]),
                "before": {k: old[key][k] for k in ("tokens", "state", "parsed", "text")},
                "after": {k: new[key][k] for k in ("tokens", "state", "parsed", "text")},
            })
    intervals = [(321, 384), (385, 448), (449, 480), (481, 512), (513, args.max_step)]
    windows = []
    for low, high in intervals:
        rows = [r for r in history if low <= r["trainer/global_step"] <= high]
        windows.append({"first_step": low, "last_step": high, "updates": len(rows), "means": {
            k: statistics.mean(r[k] for r in rows if k in r) for k in HISTORY_KEYS if k != "trainer/global_step"
            and any(k in r for r in rows)
        }})
    result = {
        "created_at": datetime.now(UTC).isoformat(), "comparison_steps": [args.before, args.after],
        "evaluation_replay": {"records": sum(len(r) for r in cohorts.values()), "reward_mismatches": 0,
                              "same_frozen_tasks_and_prompts": True, "tasks_per_checkpoint": len(tasks)},
        "coordinate_gold_audit": coordinate_label_audit(tasks),
        "validation": [{
            "step": step, "tasks": len(rows), "correct": sum(r["reward"] for r in rows.values()),
            "accuracy": statistics.mean(r["reward"] for r in rows.values()),
            "format_errors": sum(r["format_error"] for r in rows.values()),
            "truncations": sum(r["truncated"] for r in rows.values()),
            "mean_output_tokens": statistics.mean(r["tokens"] for r in rows.values()),
            "families": {f: {"n": len(subset), "accuracy": statistics.mean(r["reward"] for r in subset)}
                         for f in families if (subset := [r for r in rows.values() if r["family"] == f])},
        } for step, rows in cohorts.items()],
        "transitions": dict(transitions), "families": families, "source_groups": groups,
        "source_group_bootstrap": group_bootstrap(groups, 50000, 17),
        "history": history, "training_windows": windows, "examples": examples,
        "resolved_training_configuration": read_json(args.config)["config"]["skyrl"],
        "source_download_manifest": read_json(args.evidence / "download-manifest.json"),
        "original_response_manifest": read_json(args.evidence / "eval-retention-manifest.json"),
        "audit_inputs": [{
            "file": name, "bytes": (args.evidence / name).stat().st_size,
            "sha256": hashlib.sha256((args.evidence / name).read_bytes()).hexdigest(),
        } for name in ("training-history.json", "retention-ledger.json")],
    }
    training_path = args.evidence / "training-records.json"
    if training_path.exists():
        training = [r for r in read_json(training_path) if r["step"] <= args.max_step]
        by_step = defaultdict(list)
        for row in training:
            by_step[row["step"]].append(row)
        steps = []
        for step, rows in sorted(by_step.items()):
            assert len(rows) == 128 and len({r["task"] for r in rows}) == 32
            counts = Counter(r["task"] for r in rows)
            assert set(counts.values()) == {4}
            steps.append({
                "step": step, "responses": len(rows), "prompts": len(counts),
                "mean_reward": statistics.mean(r["reward"] for r in rows),
                "mean_input_tokens": statistics.mean(r["input_tokens"] for r in rows),
                "family_counts": dict(Counter(r["family"] for r in rows)),
                "family_successes": {f: sum(r["reward"] for r in rows if r["family"] == f)
                                     for f in sorted({r["family"] for r in rows})},
            })
        history_by_step = {r["trainer/global_step"]: r for r in history}
        assert set(by_step) == set(history_by_step) == set(range(321, args.max_step + 1))
        assert len({r["task"] for r in training}) == len(steps) * 32
        for row in steps:
            assert math.isclose(row["mean_reward"], history_by_step[row["step"]]["reward/avg_raw_reward"])
        result["training_replay"] = {
            "records": len(training), "archives": len(read_json(args.evidence / "training-download-manifest.json")),
            "reward_mismatches": 0, "prompt_and_tokenization_mismatches": 0, "stale_policy_records": 0,
            "steps": steps,
        }
        result["family_standardization"] = standardized_training_windows(steps, intervals)
        for name in ("training-records.json", "training-download-manifest.json"):
            path = args.evidence / name
            result["audit_inputs"].append({
                "file": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("evaluation_replay", "transitions", "source_group_bootstrap")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect-training", "collect-evals", "analyze"])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=565)
    parser.add_argument("--retention-root")
    parser.add_argument("--steps", type=int, nargs="+", default=[320, 384, 448, 512])
    parser.add_argument("--before", type=int, default=384)
    parser.add_argument("--after", type=int, default=512)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "collect-training":
        collect_training(args)
    elif args.command == "collect-evals":
        collect_evals(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
