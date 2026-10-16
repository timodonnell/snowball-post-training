"""Freeze exact archived validation token sequences for a paired engine audit."""

import gzip
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/001/parity-probe"
SEED = 20261017
RESPONSE_LIMIT = 2048


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    records = {}
    sources = {}
    for path in sorted((ROOT / "runs/001/regression-audit/eval-archives").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".json.gz")]
            first = json.loads(gzip.decompress(archive.read(members[0])))
            step = int(first["global_step"])
            if step not in (384, 512):
                continue
            records[step] = {}
            for name in members:
                raw = gzip.decompress(archive.read(name))
                row = json.loads(raw)
                task = row["trajectory"]["environment_extras"]["path"]
                records[step][task] = (row, digest(raw))
            sources[str(step)] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes())}
    assert set(records) == {384, 512}
    assert records[384].keys() == records[512].keys()
    families = defaultdict(list)
    for task, (row, _) in records[384].items():
        families[row["trajectory"]["environment_extras"]["family"]].append(task)
    rng = random.Random(SEED)
    selected = [task for family in sorted(families) for task in rng.sample(sorted(families[family]), 2)]
    rows = []
    for task in selected:
        for step in (384, 512):
            record, sha = records[step][task]
            prompt = record["prompt"]["token_ids"]
            assert prompt == records[384][task][0]["prompt"]["token_ids"]
            response = record["response"]["token_ids"][:RESPONSE_LIMIT]
            mask = record["response"]["loss_mask"][:RESPONSE_LIMIT]
            assert response and len(response) == len(mask) and all(mask)
            assert len(prompt) + len(response) <= 32768
            rows.append(
                {
                    "sample_id": f"{task}:archive-{step}",
                    "prompt_id": task,
                    "family": record["trajectory"]["environment_extras"]["family"],
                    "source_group": record["trajectory"]["environment_extras"]["source_group"],
                    "archive_step": step,
                    "reported_model_version": record["provenance"]["model_version_step"],
                    "record_sha256": sha,
                    "record_id": record["record_id"],
                    "prompt_token_ids": prompt,
                    "response_token_ids": response,
                    "loss_mask": mask,
                    "original_response_tokens": len(record["response"]["token_ids"]),
                }
            )
    panel = {
        "schema_version": 1,
        "seed": SEED,
        "prompts": len(selected),
        "sequences": len(rows),
        "selection": "Two random validation questions per present family; both archived answers per question",
        "response_limit": RESPONSE_LIMIT,
        "token_policy": "Exact archived IDs; first 2048 response tokens or full response if shorter; full prompt",
        "sources": sources,
        "rows": rows,
    }
    payload = json.dumps(panel, sort_keys=True, separators=(",", ":")).encode()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "frozen-panel.json.gz"
    compressed = gzip.compress(payload, mtime=0)
    if target.exists() and target.read_bytes() != compressed:
        raise ValueError("Refusing to replace the frozen panel")
    target.write_bytes(compressed)
    manifest = {k: v for k, v in panel.items() if k != "rows"}
    manifest.update(
        payload_sha256=digest(payload),
        compressed_sha256=digest(compressed),
        response_tokens=sum(len(row["response_token_ids"]) for row in rows),
        questions=[
            {k: v for k, v in row.items() if k not in ("prompt_token_ids", "response_token_ids", "loss_mask")}
            for row in rows
        ],
    )
    (OUTPUT / "panel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("questions", "sources")}, indent=2))


if __name__ == "__main__":
    main()
