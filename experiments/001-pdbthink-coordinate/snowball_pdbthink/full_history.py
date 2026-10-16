"""Collect the full Snowball training lineage without conflating evaluation protocols."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

RUNS = ["xlyyl24u", "ppc4ryud", "e2s5w3qn"]
METRICS = [
    "reward/avg_raw_reward", "policy/raw_grad_norm", "policy/policy_entropy",
    "policy/mismatch/pooled/kl_k3", "policy/mismatch/pooled/ess_fraction", "generate/avg_num_tokens",
    "policy/ppo_clip_ratio", "policy/ppo_ratio_exact_unit_fraction", "policy/policy_update_steps",
]


def select_training(raw):
    """A logged null is an absent observation, not an optimizer update."""
    return [
        {"step": r["trainer/global_step"], **{k: r[k] for k in METRICS if r.get(k) is not None}}
        for r in raw if r.get("policy/raw_grad_norm") is not None
    ]


def native_evaluations(raw):
    return [{
        "step": r["trainer/global_step"], "accuracy": r["eval/all/avg_score"],
        "format_rate": r["eval/all/environment/pdbthink/format_error"],
        "truncation_rate": r["eval/all/environment/pdbthink/truncated"],
        "mean_output_tokens": r["eval/all/response_tokens_mean"],
        "family_accuracy": {k.split("/")[1]: v for k, v in r.items()
                            if k.startswith("eval/") and k.endswith("/avg_score")
                            and k != "eval/all/avg_score" and v is not None},
    } for r in raw if r.get("eval/all/avg_score") is not None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("runs/001/full-history"))
    parser.add_argument("--results", type=Path, default=Path("experiments/001-pdbthink-coordinate/results"))
    parser.add_argument("--refresh", action="store_true", help="Refresh W&B snapshots instead of using cached raw rows")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    snapshots, sources = {}, []
    for run_id in RUNS:
        path = args.cache / f"{run_id}.json"
        if args.refresh or not path.exists():
            import wandb

            run = wandb.Api().run(f"timodonnell/snowball-pdbthink/{run_id}")
            # Sparse keys must not be requested together: W&B would drop rows.
            path.write_text(json.dumps(list(run.scan_history(page_size=500)), allow_nan=False) + "\n")
        payload = path.read_bytes()
        snapshots[run_id] = json.loads(payload)
        sources.append({"run": f"timodonnell/snowball-pdbthink/{run_id}",
                        "raw_history_sha256": hashlib.sha256(payload).hexdigest(),
                        "snapshot_file_mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()})
    retained, discarded = [], []
    for run_id, low, high in [(RUNS[0], 1, 64), (RUNS[1], 65, 320), (RUNS[2], 321, float("inf"))]:
        for row in select_training(snapshots[run_id]):
            if low <= row["step"] <= high:
                retained.append({"run": run_id, **row})
            elif run_id == RUNS[1] and row["step"] > 320:
                discarded.append({"run": run_id, **row})
    retained.sort(key=lambda r: r["step"])
    assert len({r["step"] for r in retained}) == len(retained)
    last_step = max(r["step"] for r in retained)
    missing = sorted(set(range(1, last_step + 1)) - {r["step"] for r in retained})
    standalone = json.loads((args.results / "pilot-validation-comparison.json").read_text())
    paired_points = []
    for key, step in [("baseline", 0), ("trained", 64)]:
        row = standalone[key]
        paired_points.append({
            "step": step, "accuracy": row["task_weighted_accuracy"],
            "format_rate": row["format_errors"] / row["completed"],
            "truncation_rate": row["truncations"] / row["completed"],
            "mean_output_tokens": row["mean_output_tokens"],
            "family_accuracy": {f: r["accuracy"] for f, r in row["families"].items()},
        })
    monitor = json.loads((args.results / "pilot-monitor.json").read_text())["validation"]
    result = {
        "created_at": datetime.now(UTC).isoformat(), "sources": sources,
        "last_optimizer_update": last_step, "training": retained, "discarded_training": discarded,
        "missing_training_updates": missing,
        "lineage": "Pilot updates 1–64; first continuation 65–320; resumed continuation 321 onward. "
                   "First-attempt updates 321–375 were discarded after the upload failure, then repeated.",
        "missing_note": "W&B did not retain the pilot's final update 64. It completed, but its optimizer diagnostics "
                        "are absent here; no value is interpolated. Update metrics first exist at step 1, not step 0.",
        "standalone_validation": {"tasks": 1003, "points": paired_points,
                                  "source": "pilot-validation-comparison.json", "runtime": "standalone serving"},
        "pilot_monitor": {"tasks": 107, "points": monitor, "source": "pilot-monitor.json"},
        "native_validation": [{"run": run_id, "tasks": 1003, "points": native_evaluations(snapshots[run_id])}
                              for run_id in RUNS[1:]],
        "protocol_note": "Standalone and native full validation have different serving runtimes; "
                         "the pilot native monitor uses a different cohort. Keep separate plotted series.",
    }
    path = args.results / "full-training-history.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"last_update": last_step, "retained_observations": len(retained),
                      "discarded_observations": len(discarded), "missing_updates": missing}))


if __name__ == "__main__":
    main()
