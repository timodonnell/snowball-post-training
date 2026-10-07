"""Freeze the published context cohort, without reading any teacher traces."""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .pins import CONTEXT, DATASET, DATASET_REVISION, EXPECTED_COUNTS, MODEL, MODEL_REVISION, RESERVE

TOKENIZER = None
COHORT = None


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def init_worker(model_path, cohort):
    global TOKENIZER, COHORT
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    COHORT = cohort


def read_shard(shard):
    rows, native_files = [], {}
    columns = ["path", "family", "source_group", "split", "prompt_sha256", "task_sha256", "task_binary"]
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=columns):
        for row in batch.to_pylist():
            context = COHORT.get(row["path"])
            if context is None:
                continue
            if sha256(row["task_binary"]) != row["task_sha256"]:
                raise ValueError(f"Corrupt task archive: {row['path']}")
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as archive:
                prompt = json.load(archive.extractfile("prompt.json"))
                gold = json.load(archive.extractfile("tests/gold.json"))
                for member in archive.getmembers():
                    if member.isfile() and (
                        member.name.startswith("tests/coordinate_scoring/") or member.name == "tests/verify.py"
                    ):
                        content = archive.extractfile(member).read()
                        name = member.name.removeprefix("tests/")
                        if name in native_files and native_files[name] != content:
                            raise ValueError("Mixed verifier versions")
                        native_files[name] = content
            prompt_hash = sha256((prompt["system_prompt"] + "\n" + prompt["user_prompt"]).encode())
            if prompt_hash != row["prompt_sha256"] or prompt_hash != context["prompt_sha256"]:
                raise ValueError(f"Prompt mismatch: {row['path']}")
            for key in ("split", "family", "source_group"):
                if row[key] != context[key]:
                    raise ValueError(f"Context metadata mismatch: {row['path']} {key}")
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"]},
            ]
            tokens = TOKENIZER.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=True,
                tools=[],
                truncation=False,
            )["input_ids"]
            if len(tokens) != context["input_tokens"] or CONTEXT - len(tokens) != context["available_output_tokens"]:
                raise ValueError(f"Native tokenizer mismatch: {row['path']}")
            # Gold is an environment extra. Only `prompt` is passed to the model.
            rows.append(
                {
                    **{key: row[key] for key in ("path", "family", "source_group", "split", "task_sha256")},
                    "prompt_sha256": prompt_hash,
                    "input_tokens": len(tokens),
                    "available_output_tokens": CONTEXT - len(tokens),
                    "prompt": messages,
                    "env_class": "pdbthink",
                    "data_source": row["family"],
                    "reward_model": {"ground_truth": json.dumps(gold, sort_keys=True)},
                }
            )
    return rows, native_files


def select_monitor(rows, per_family=8):
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    return sorted(
        (
            row
            for family in sorted(families)
            for row in sorted(families[family], key=lambda r: sha256(("monitor-17:" + r["path"]).encode()))[:per_family]
        ),
        key=lambda r: r["path"],
    )


def prepare(dataset, model_path, output, workers=8):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared cohort: {output}")
    release = json.loads((dataset / "manifest.json").read_text())
    if (
        release["version"] != "1.3.0"
        or release["task_set_fingerprint"] != "1e29e23ff33c603d73b2d51af2e6ae3e7ee7efcf66529efd6724fd0570c46fb4"
    ):
        raise ValueError("Expected the pinned v1.3.0 release")
    for name, expected in release["data_hashes"].items():
        if sha256((dataset / name).read_bytes()) != expected:
            raise ValueError(f"Release checksum mismatch: {name}")
    cohort = {
        r["path"]: r
        for r in pq.read_table(dataset / "snowball_context.parquet").to_pylist()
        if r["available_output_tokens"] >= RESERVE
    }
    if dict(Counter(r["split"] for r in cohort.values())) != EXPECTED_COUNTS:
        raise ValueError("Published cohort counts changed")
    output.mkdir(parents=True)
    by_split = defaultdict(list)
    verifier = {}
    shards = sorted((dataset / "data").glob("*.parquet"))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(model_path), cohort)) as pool:
        for index, (rows, files) in enumerate(pool.map(read_shard, shards)):
            for row in rows:
                by_split[row["split"]].append(row)
            for name, content in files.items():
                if name in verifier and verifier[name] != content:
                    raise ValueError("Mixed verifier implementations across shards")
                verifier[name] = content
            if index % 20 == 0:
                print(f"Verified {index + 1}/{len(shards)} shards", flush=True)
    found = [r["path"] for rows in by_split.values() for r in rows]
    if len(found) != len(set(found)) or set(found) != set(cohort):
        raise ValueError("Missing or duplicate cohort task IDs")
    groups = {split: {r["source_group"] for r in rows} for split, rows in by_split.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"Source group leakage: {a}/{b}")
    for name, content in verifier.items():
        path = output / "native_verifier" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    hashes, family_counts = {}, {}
    for split, rows in by_split.items():
        rows.sort(key=lambda r: r["path"])
        path = output / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        hashes[path.name] = sha256(path.read_bytes())
        family_counts[split] = dict(sorted(Counter(r["family"] for r in rows).items()))
    monitor = select_monitor(by_split["validation"])
    pq.write_table(pa.Table.from_pylist(monitor), output / "monitor.parquet", compression="zstd")
    # Smoke covers every available family plus the longest eligible training prompts.
    smoke = select_monitor(by_split["train"], per_family=1)
    smoke = {r["path"]: r for r in smoke + sorted(by_split["train"], key=lambda r: -r["input_tokens"])[:14]}
    smoke_rows = sorted(smoke.values(), key=lambda r: r["path"])
    pq.write_table(pa.Table.from_pylist(smoke_rows), output / "smoke.parquet", compression="zstd")
    for name in ("monitor.parquet", "smoke.parquet"):
        hashes[name] = sha256((output / name).read_bytes())
    write_json(
        output / "cohort.json",
        [
            {
                k: r[k]
                for k in (
                    "path",
                    "split",
                    "family",
                    "source_group",
                    "prompt_sha256",
                    "input_tokens",
                    "available_output_tokens",
                )
            }
            for r in sorted(cohort.values(), key=lambda r: r["path"])
        ],
    )
    shutil.copy2(dataset / "snowball_context.json", output / "source_context.json")
    shutil.copy2(model_path / "chat_template.jinja", output / "native_chat_template.jinja")
    write_json(
        output / "manifest.json",
        {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "counts": EXPECTED_COUNTS,
            "family_counts": family_counts,
            "monitor_count": len(monitor),
            "monitor_family_counts": dict(sorted(Counter(r["family"] for r in monitor).items())),
            "smoke_count": len(smoke_rows),
            "source_group_disjoint": True,
            "context": CONTEXT,
            "cohort_reserve": RESERVE,
            "enable_thinking": True,
            "teacher_traces": False,
            "parquet_sha256": hashes,
            "cohort_sha256": sha256((output / "cohort.json").read_bytes()),
            "native_verifier_sha256": {k: sha256(v) for k, v in sorted(verifier.items())},
            "model_metadata_sha256": {p.name: sha256(p.read_bytes()) for p in model_path.iterdir() if p.is_file()},
        },
    )
    print(json.dumps(json.loads((output / "manifest.json").read_text()), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.dataset, args.model_metadata, args.output, args.workers)


if __name__ == "__main__":
    main()
