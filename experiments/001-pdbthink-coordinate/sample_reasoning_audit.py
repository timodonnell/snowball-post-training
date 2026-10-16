"""Freeze a category/outcome-stratified, paired sample for qualitative reasoning review."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from snowball_pdbthink.scoring import load_scorer, score

ROOT = Path("runs/001/reasoning-audit")
SEED = 20261016


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    prepared = Path("data/pdbthink-001-v3")
    tasks = {r["path"]: r for r in pq.read_table(prepared / "validation.parquet").to_pylist()}
    native_path = Path("runs/001/regression-audit/evals/384/native-responses.json")
    native = {r["path"]: r for r in json.loads(native_path.read_text())}
    assert tasks.keys() == native.keys()
    scorer = load_scorer(prepared / "native_verifier")
    strata, records, totals = defaultdict(list), {}, defaultdict(Counter)
    for task_id, task in sorted(tasks.items()):
        base_path = Path("runs/001/baseline-validation-v2") / (task_id + ".json")
        base = json.loads(base_path.read_text())
        assert base["request"]["messages"] == task["prompt"]
        assert base["request"]["max_tokens"] == 32768 - task["input_tokens"]
        after = native[task_id]
        assert after["prompt_sha256"] == task["prompt_sha256"]
        btext = base["response"]["choices"][0]["message"]["content"] or ""
        atext = "".join(m.get("content") or "" for m in after["messages"])
        answers = {}
        for step, text, truncated, recorded in (
            (0, btext, base["result"]["truncated"], base["result"]["reward"]),
            (384, atext, after["stop_reason"] == "length", after["reward"]),
        ):
            result = score(scorer, text, task["reward_model"]["ground_truth"], truncated=truncated)
            assert result["reward"] == recorded
            answers[str(step)] = {"text": text, "text_sha256": sha(text.encode()), "result": result}
        state = f"{int(answers['0']['result']['reward'])}{int(answers['384']['result']['reward'])}"
        strata[task["family"], state].append(task_id)
        totals[task["family"]][state] += 1
        records[task_id] = {
            "id": task_id,
            "family": task["family"],
            "group": task["source_group"],
            "transition": state,
            "prompt_sha256": task["prompt_sha256"],
            "prompt": task["prompt"],
            "gold": json.loads(task["reward_model"]["ground_truth"]),
            "answers": answers,
            "source_files": {"baseline": str(base_path), "native": str(native_path)},
            "baseline_record_sha256": sha(base_path.read_bytes()),
        }
    selected, allocation = [], {}
    rng = random.Random(SEED)
    for family in sorted(totals):
        quota = 3 if family == "G04" else 13 if family == "S09" else 7
        keys = sorted(totals[family])
        rng.shuffle(keys)
        counts = Counter()
        for _ in range(quota):
            eligible = [k for k in keys if counts[k] < totals[family][k]]
            k = min(eligible, key=lambda k: counts[k])
            counts[k] += 1
        for state, n in sorted(counts.items()):
            selected.extend(rng.sample(strata[family, state], n))
        allocation[family] = {
            "population": dict(sorted(totals[family].items())),
            "sample": dict(sorted(counts.items())),
        }
    selected.sort()
    assert len(selected) == len(set(selected)) == 100
    manifest = {
        "version": 1,
        "seed": SEED,
        "selection_unit": "question; the same 100 IDs at both steps, 200 responses total",
        "method": "7 questions per family, except all 3 G04 and 13 S09; equal allocation over available paired binary outcome strata within each family, capacity-limited; uniform random sampling without replacement inside each stratum",
        "strata": "family × (step-0 success, step-384 success); 00=both fail, 01=gain, 10=loss, 11=both succeed",
        "scope": "All 14 categories present in the 1003-task validation cohort. I01/S06/S07/S08/T01 have no validation tasks and cannot be assessed here.",
        "limitation": "Outcome-stratified descriptive sample, not representative accuracy estimates. Archived standalone step-0 and native step-384 serving paths differ.",
        "validation_sha256": sha((prepared / "validation.parquet").read_bytes()),
        "native_records_sha256": sha(native_path.read_bytes()),
        "allocation": allocation,
        "selected_ids": selected,
        "selected_response_sha256": {
            i: {s: a["text_sha256"] for s, a in records[i]["answers"].items()} for i in selected
        },
    }
    path = ROOT / "sample-manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        previous.pop("frozen_at", None)
        assert previous == manifest, "Refusing to change frozen sample"
    else:
        path.write_text(json.dumps({"frozen_at": datetime.now(UTC).isoformat(), **manifest}, indent=2) + "\n")
    (ROOT / "sample.json").write_text(json.dumps([records[i] for i in selected], indent=2) + "\n")
    print(json.dumps(allocation, indent=2))
    print("Total response characters", sum(len(a["text"]) for i in selected for a in records[i]["answers"].values()))


if __name__ == "__main__":
    main()
