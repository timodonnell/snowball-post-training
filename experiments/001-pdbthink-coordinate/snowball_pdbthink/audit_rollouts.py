"""Check retained MarinSkyRL trajectories against the frozen prepared inputs."""

import argparse
import gzip
import json
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .pins import CONTEXT
from .scoring import tool_events


def audit_record(record, tasks):
    extras = record["trajectory"]["environment_extras"]
    task = tasks[extras["path"]]
    if record["prompt"]["messages"] != task["prompt"]:
        raise ValueError(f"Runtime prompt differs from frozen source: {task['path']}")
    prompt_tokens = len(record["prompt"]["token_ids"])
    output_tokens = len(record["response"]["token_ids"])
    if prompt_tokens != task["input_tokens"]:
        raise ValueError(f"Runtime tokenization differs from frozen source: {task['path']}")
    if prompt_tokens + output_tokens > CONTEXT:
        raise ValueError(f"Runtime exceeded native context: {task['path']}")
    if extras["family"] != task["family"] or extras["source_group"] != task["source_group"]:
        raise ValueError(f"Runtime task metadata differs: {task['path']}")
    reward = record["reward"]["outcome"]
    if reward not in (0, 1):
        raise ValueError(f"Non-binary verifier reward: {task['path']}")
    response = record["response"]
    if reward and (response["stop_reason"] == "length" or tool_events(response["messages"])):
        raise ValueError(f"Truncated or tool-using response received reward: {task['path']}")
    if record["disposition"]["exception_type"] is not None:
        raise ValueError(f"Runtime generation exception: {task['path']}")
    return task["family"], reward, output_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {r["path"]: r for r in pq.read_table(args.tasks).to_pylist()}
    counts, successes, phases = Counter(), Counter(), Counter()
    seen = set()
    tokens = 0
    for path in args.archives:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.startswith("records/") or not name.endswith(".json.gz"):
                    continue
                record = json.loads(gzip.decompress(archive.read(name)))
                if record["record_id"] in seen:
                    raise ValueError("Duplicate trajectory record")
                seen.add(record["record_id"])
                family, reward, output_tokens = audit_record(record, tasks)
                counts[family] += 1
                successes[family] += reward
                phases[record["phase"]] += 1
                tokens += output_tokens
    if not seen:
        raise ValueError("No retained trajectory records found")
    result = {
        "records": len(seen),
        "phases": dict(phases),
        "family_counts": dict(counts),
        "family_successes": dict(successes),
        "output_tokens": tokens,
        "exact_prompts_and_token_counts": True,
        "native_context_respected": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
