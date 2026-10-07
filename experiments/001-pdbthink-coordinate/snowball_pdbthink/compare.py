"""Compare complete, paired evaluations under an identical request policy."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .evaluate import summarize


def load_evaluation(directory):
    contract = json.loads((directory / "evaluation.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    records = {}
    for path in sorted(directory.glob("pdbthink-*.json")):
        record = json.loads(path.read_text())
        key = record["result"]["path"]
        if key in records:
            raise ValueError(f"Duplicate evaluated task: {key}")
        records[key] = record
    if len(records) != summary["expected"] or summary["completed"] != summary["expected"]:
        raise ValueError("Only complete evaluations can be compared")
    return contract, records


def compare(baseline_contract, baseline, trained_contract, trained):
    excluded = {"model", "model_identity"}
    if {k: v for k, v in baseline_contract.items() if k not in excluded} != {
        k: v for k, v in trained_contract.items() if k not in excluded
    }:
        raise ValueError("Evaluation cohorts or policies differ")
    if not baseline or baseline.keys() != trained.keys():
        raise ValueError("Evaluation task IDs differ or are empty")
    transitions = Counter()
    family_transitions = {}
    for task, before in baseline.items():
        after = trained[task]
        for key in ("path", "prompt_sha256", "family", "source_group"):
            if before["result"][key] != after["result"][key]:
                raise ValueError(f"Task metadata changed: {task} {key}")
        if {k: v for k, v in before["request"].items() if k != "model"} != {
            k: v for k, v in after["request"].items() if k != "model"
        }:
            raise ValueError(f"Per-task generation request changed: {task}")
        old, new = before["result"]["reward"], after["result"]["reward"]
        label = "both_correct" if old and new else "gained" if new else "lost" if old else "both_incorrect"
        transitions[label] += 1
        family_transitions.setdefault(before["result"]["family"], Counter())[label] += 1
    count = len(baseline)
    before = summarize([r["result"] for r in baseline.values()], count)
    after = summarize([r["result"] for r in trained.values()], count)
    return {
        "baseline_identity": baseline_contract["model_identity"],
        "trained_identity": trained_contract["model_identity"],
        "paired_tasks": count,
        "source_groups": len({r["result"]["source_group"] for r in baseline.values()}),
        "baseline": before,
        "trained": after,
        "task_weighted_accuracy_delta": after["task_weighted_accuracy"] - before["task_weighted_accuracy"],
        "family_macro_accuracy_delta": after["family_macro_accuracy"] - before["family_macro_accuracy"],
        "transitions": dict(transitions),
        "families": {
            family: {
                "n": values["n"],
                "baseline_accuracy": values["accuracy"],
                "trained_accuracy": after["families"][family]["accuracy"],
                "accuracy_delta": after["families"][family]["accuracy"] - values["accuracy"],
                "transitions": dict(family_transitions[family]),
            }
            for family, values in before["families"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(*load_evaluation(args.baseline), *load_evaluation(args.trained))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for family, row in result["families"].items():
        print(
            f"{family} n={row['n']}: {row['baseline_accuracy']:.3%} -> "
            f"{row['trained_accuracy']:.3%} ({row['accuracy_delta']:+.3%})"
        )


if __name__ == "__main__":
    main()
