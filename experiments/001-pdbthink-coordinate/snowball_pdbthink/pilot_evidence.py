"""Audit the pilot's paired validation responses and write plotting data.

No responses are regenerated, repaired, or selected using their gold answers.
The bootstrap resamples paired source groups, retaining all tasks in each draw.
It describes sensitivity to source composition, not RL-run/decoding variability.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .compare import compare, load_evaluation
from .evaluate import request_for
from .scoring import load_scorer, score, tool_events


def answer_state(result):
    if result["reward"]:
        return "correct"
    # Flags overlap. This ordering makes the transition table mutually exclusive.
    if result["format_error"]:
        return "format_error"
    if result["truncated"] or result["refusal"] or result["tool_violation"]:
        return "other_invalid"
    return "well_formed_wrong"


def group_bootstrap(groups, draws, seed):
    values = np.array([[r["n"], r["baseline"], r["trained"]] for r in groups.values()])
    rng = np.random.default_rng(seed)
    totals = values[rng.integers(len(values), size=(draws, len(values)))].sum(axis=1)
    deltas = (totals[:, 2] - totals[:, 1]) / totals[:, 0]
    return {
        "method": "paired source-group percentile bootstrap; task-weighted ratio in each draw",
        "groups": len(values),
        "resamples": draws,
        "seed": seed,
        "delta_95_percent_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "limitation": "Source-composition uncertainty only; not training-seed or serving variability.",
    }


def residue_key(identifier):
    match = re.fullmatch(r"([A-Za-z0-9]):[A-Za-z](-?\d+)", identifier)
    if not match:
        raise ValueError(f"Unexpected protein residue identifier: {identifier}")
    return match[1], int(match[2])


def coordinate_label_audit(tasks):
    """Recompute G01/G03/P03 golds from rendered PDB text without generator code."""
    counts = Counter()
    distance_errors = []
    coordinate_errors = []
    for row in tasks.values():
        family = row["family"]
        if family not in {"G01", "G03", "P03"}:
            continue
        atoms = {}
        residues = defaultdict(list)
        for message in row["prompt"]:
            for line in message["content"].splitlines():
                if line[:6] not in {"ATOM  ", "HETATM"}:
                    continue
                chain, number, atom = line[21], int(line[22:26]), line[12:16].strip()
                xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
                key = chain, number, atom
                if key in atoms:
                    raise ValueError(f"Ambiguous rendered atom: {row['path']} {key}")
                atoms[key] = xyz
                residues[chain, number].append(xyz)
        gold = json.loads(row["reward_model"]["ground_truth"])
        params, expected = gold["parameters"], gold["gold_answer"]["value"]
        if family == "G01":
            points = []
            for name in ("atom1", "atom2"):
                residue, atom = params[name].rsplit(":", 1)
                points.append(atoms[(*residue_key(residue), atom)])
            error = abs(math.dist(*points) - expected)
            distance_errors.append(error)
            # Gold distances are rounded to 0.001 A; answers allow 0.02 A error.
            if error > 0.001:
                raise ValueError(f"Rendered distance disagrees with gold: {row['path']} {error}")
        elif family == "P03":
            actual = atoms[(*residue_key(params["residue"]), params["atom"])]
            error = max(abs(a - b) for a, b in zip(actual, expected, strict=True))
            coordinate_errors.append(error)
            if error > 1e-9:
                raise ValueError(f"Rendered coordinates disagree with gold: {row['path']}")
        else:
            target = residues[residue_key(params["target"])]
            distances = {
                candidate: min(math.dist(a, b) for a in target for b in residues[residue_key(candidate)])
                for candidate in params["candidates"]
            }
            if min(distances, key=distances.get) != expected:
                raise ValueError(f"Rendered nearest residue disagrees with gold: {row['path']}")
        counts[family] += 1
    return {
        "checked": dict(counts),
        "mismatches": 0,
        "max_distance_gold_rounding_error_angstrom": max(distance_errors),
        "max_coordinate_gold_error_angstrom": max(coordinate_errors),
        "scope": "Recomputed gold labels, not an independent proof of model reasoning.",
    }


def analyze(args):
    before_contract, before = load_evaluation(args.baseline)
    after_contract, after = load_evaluation(args.trained)
    comparison = compare(before_contract, before, after_contract, after)
    validation_path = args.prepared / "validation.parquet"
    tasks = {r["path"]: r for r in pq.read_table(validation_path).to_pylist()}
    digest = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    if digest != before_contract["tasks_sha256"] or tasks.keys() != before.keys():
        raise ValueError("Prepared validation cohort differs from evaluated cohort")
    split_metadata = {
        split: pq.read_table(
            args.prepared / f"{split}.parquet", columns=["path", "source_group", "prompt_sha256"]
        ).to_pylist()
        for split in ("train", "validation", "test")
    }
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}/{right}"] = {
            key: len({r[key] for r in split_metadata[left]} & {r[key] for r in split_metadata[right]})
            for key in ("path", "source_group", "prompt_sha256")
        }
    if any(n for values in overlaps.values() for n in values.values()):
        raise ValueError(f"Cohort overlap: {overlaps}")

    scorer = load_scorer(args.prepared / "native_verifier")
    replayed = 0
    for contract, records in ((before_contract, before), (after_contract, after)):
        for task_id, record in records.items():
            task = tasks[task_id]
            for key in ("path", "family", "source_group", "prompt_sha256"):
                if record["result"][key] != task[key]:
                    raise ValueError(f"Result metadata differs from prepared task: {task_id} {key}")
            if record["request"] != request_for(task, contract["model"]):
                raise ValueError(f"Request differs from frozen policy: {task_id}")
            body = record["response"]
            choice = body["choices"][0]
            message = choice["message"]
            replay = score(
                scorer, message.get("content") or "", task["reward_model"]["ground_truth"],
                truncated=choice["finish_reason"] == "length",
                refusal=bool(message.get("refusal")), tool_violation=tool_events(body),
            )
            if any(record["result"][key] != value for key, value in replay.items()):
                raise ValueError(f"Native score replay differs: {task_id}")
            if body["usage"]["prompt_tokens"] != task["input_tokens"]:
                raise ValueError(f"Prompt token count differs: {task_id}")
            replayed += 1

    groups = {}
    transitions = defaultdict(Counter)
    valid_families = defaultdict(lambda: Counter(n=0, baseline=0, trained=0, gained=0, lost=0))
    paired_rows = []
    examples = {}
    for task_id in sorted(before):
        old, new = before[task_id]["result"], after[task_id]["result"]
        group = groups.setdefault(old["source_group"], {"n": 0, "baseline": 0, "trained": 0})
        group["n"] += 1
        group["baseline"] += int(old["reward"])
        group["trained"] += int(new["reward"])
        old_state, new_state = answer_state(old), answer_state(new)
        transitions[old_state][new_state] += 1
        both_valid = all(s in {"correct", "well_formed_wrong"} for s in (old_state, new_state))
        if both_valid:
            family = valid_families[old["family"]]
            family["n"] += 1
            family["baseline"] += int(old["reward"])
            family["trained"] += int(new["reward"])
            family["gained"] += int(new["reward"] > old["reward"])
            family["lost"] += int(old["reward"] > new["reward"])
            if old_state == "well_formed_wrong" and new_state == "correct" and old["family"] in {"G01", "G03", "P03"}:
                examples.setdefault(old["family"], {
                    "path": task_id,
                    "before": old["outcome"]["parsed"]["value"],
                    "after": new["outcome"]["parsed"]["value"],
                    "gold": json.loads(tasks[task_id]["reward_model"]["ground_truth"])["gold_answer"]["value"],
                    "selection": "First task ID with well-formed wrong -> correct in this family; illustrative only.",
                })
        paired_rows.append({
            "path": task_id, "family": old["family"], "source_group": old["source_group"],
            "before_state": old_state, "after_state": new_state,
            "before_tokens": old["output_tokens"], "after_tokens": new["output_tokens"],
        })

    train_categories = defaultdict(Counter)
    for row in pq.read_table(args.prepared / "train.parquet", columns=["family", "reward_model"]).to_pylist():
        gold = json.loads(row["reward_model"]["ground_truth"])
        if gold["answer_schema"] == "category":
            train_categories[row["family"]][gold["gold_answer"]["value"]] += 1
    categorical = {}
    for family, counts in sorted(train_categories.items()):
        cohort = [t for t in tasks.values() if t["family"] == family]
        if not cohort:
            continue
        majority = min(counts, key=lambda label: (-counts[label], label))
        golds = Counter(json.loads(t["reward_model"]["ground_truth"])["gold_answer"]["value"] for t in cohort)
        categorical[family] = {
            "train_gold_counts": dict(counts), "train_majority_label": majority,
            "validation_gold_counts": dict(golds), "n": len(cohort),
            "constant_majority_correct": golds[majority],
            "constant_majority_accuracy": golds[majority] / len(cohort),
            "baseline_correct": comparison["baseline"]["families"][family]["successes"],
            "trained_correct": comparison["trained"]["families"][family]["successes"],
            "trained_parsed_predictions": dict(Counter(
                str(after[t["path"]]["result"]["outcome"]["parsed"]["value"]) for t in cohort
            )),
        }
    valid_totals = {key: sum(c[key] for c in valid_families.values()) for key in ("n", "baseline", "trained", "gained", "lost")}
    return {
        "comparison": comparison,
        "audit": {
            "responses_replayed": replayed, "score_mismatches": 0,
            "requests_match_frozen_policy": True, "split_overlaps": overlaps,
            "coordinate_labels": coordinate_label_audit(tasks),
        },
        "bootstrap": group_bootstrap(groups, args.bootstrap_draws, args.seed),
        "source_groups": groups,
        "source_group_signs": {
            "improved": sum(g["trained"] > g["baseline"] for g in groups.values()),
            "unchanged": sum(g["trained"] == g["baseline"] for g in groups.values()),
            "regressed": sum(g["trained"] < g["baseline"] for g in groups.values()),
        },
        "transitions_by_answer_state": {s: dict(c) for s, c in transitions.items()},
        "both_valid": {"total": valid_totals, "families": dict(valid_families)},
        "both_valid_caveat": "Selected using both model outputs; descriptive, not a causal format-only ablation.",
        "categorical_controls": categorical,
        "illustrative_examples": examples,
        "monitor": json.loads(args.monitor.read_text()),
        "paired_tasks": paired_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("audit", "bootstrap", "source_group_signs", "both_valid")}, indent=2))


if __name__ == "__main__":
    main()
