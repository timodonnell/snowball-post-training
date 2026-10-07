"""Summarize training rewards and fixed validation success from W&B."""

import argparse
import json
from pathlib import Path

import wandb


def summarize_history(history, counts):
    points = []
    for row in history:
        scores = {
            family: row[f"eval/{family}/avg_score"]
            for family in counts
            if row.get(f"eval/{family}/avg_score") is not None
        }
        if not scores:
            continue
        covered = sum(counts[f] for f in scores)
        points.append(
            {
                "step": row.get("trainer/global_step", row.get("global_step", row.get("_step"))),
                "family_accuracy": scores,
                "family_macro_accuracy": sum(scores.values()) / len(scores),
                "task_weighted_accuracy": sum(counts[f] * score for f, score in scores.items()) / covered,
                "covered_tasks": covered,
                "expected_tasks": sum(counts.values()),
            }
        )
    return points


def training_history(history, families):
    # MarinSkyRL escapes uppercase bytes in reward domain names (G01 becomes
    # _source__4701). Evaluation metric names retain the original family ID.
    keys = {family: f"reward/domain/_source__{ord(family[0]):02x}{family[1:]}/avg_raw_reward" for family in families}
    points = []
    for row in history:
        if row.get("reward/avg_raw_reward") is None:
            continue
        points.append(
            {
                "step": row.get("trainer/global_step", row.get("_step")),
                "sample_success_rate": row["reward/avg_raw_reward"],
                "prompt_pass_at_4": row.get("reward/avg_pass_at_4"),
                "family_sample_success_rate": {f: row[k] for f, k in keys.items() if row.get(k) is not None},
                "policy_loss": row.get("policy/policy_loss"),
                "raw_grad_norm": row.get("policy/raw_grad_norm"),
            }
        )
    return points, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    run = wandb.Api().run(args.run)
    # Read full rows: requesting all sparse metric keys can drop otherwise valid history rows.
    history = list(run.scan_history())
    curve = summarize_history(history, manifest["monitor_family_counts"])
    training, keys = training_history(history, manifest["family_counts"]["train"])
    result = {
        "run": args.run,
        "url": run.url,
        "state": run.state,
        "validation": curve,
        "training": training,
        "training_metric_keys": keys,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "url": run.url,
                "state": run.state,
                "latest_validation": curve[-1] if curve else None,
                "latest_training": training[-1] if training else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
