"""Compare completed fixed-token checkpoint diagnostics and plot their evidence."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def array(row, prefix):
    return np.frombuffer(row[prefix], dtype=row[prefix + "_dtype"]).reshape(row[prefix + "_shape"])


def stats(delta):
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Missing or nonfinite token comparison")
    weights = np.exp(delta - delta.max())
    return {
        "tokens": len(delta),
        "mean_abs_log_ratio": float(np.abs(delta).mean()),
        "p99_abs_log_ratio": float(np.quantile(np.abs(delta), 0.99)),
        "k3": float(np.mean(np.expm1(delta) - delta)),
        "share_beyond_2x": float(np.mean(np.abs(delta) > np.log(2))),
        "ess_fraction": float(weights.sum() ** 2 / (len(weights) * np.square(weights).sum())),
    }


def checkpoint(folder, panel):
    complete = json.loads((folder / "complete.json").read_text())
    if complete["optimizer_updates"] != 0:
        raise ValueError("Checkpoint has optimizer updates")
    probes = {row["sample_id"]: row for row in pq.read_table(folder / "probe.parquet").to_pylist()}
    if set(probes) != {row["sample_id"] for row in panel["rows"]}:
        raise ValueError("The archive contains a different sample set")
    scores = pq.read_table(folder / "scores.parquet").to_pylist()
    indexed = {(row["sample_id"], row["scorer"], row["mode"]): row for row in scores}
    if len(indexed) != len(scores):
        raise ValueError("Duplicate sample/scorer/mode rows")
    delta, routing = {}, {}
    for mode in ("native", "repeat", "router_replay", "vllm_repeat"):
        differences, set_agreement, ordered_agreement, valid_rows = [], [], [], []
        by_source, by_family = {}, {}
        for row in panel["rows"]:
            sample = row["sample_id"]
            probe = probes[sample]
            if (
                probe["prompt_token_ids"] != row["prompt_token_ids"]
                or probe["vllm_output_ids"] != row["response_token_ids"]
            ):
                raise ValueError("Frozen token identity differs")
            reference = indexed[sample, "vllm.rescore", ""]
            scorer, label = ("vllm.rescore", "repeat") if mode == "vllm_repeat" else ("trainer", mode)
            candidate = indexed[sample, scorer, label]
            if len(reference["logprobs"]) != len(row["response_token_ids"]) or len(candidate["logprobs"]) != len(
                reference["logprobs"]
            ):
                raise ValueError("Incomplete scores")
            values = np.asarray(candidate["logprobs"], dtype=np.float64) - reference["logprobs"]
            mask = np.asarray(probe["loss_mask"], dtype=bool)
            differences.append(values[mask])
            by_source.setdefault(row["archive_step"], []).append(values[mask])
            by_family.setdefault(row["family"], []).append(values[mask])
            if candidate["expert_choices"] is not None:
                expected, actual = array(probe, "routed_experts"), array(candidate, "expert_choices")
                valid = np.asarray(probe["route_valid_mask"], dtype=bool)
                if expected.shape != actual.shape or np.any(actual[valid] < 0):
                    raise ValueError("Incomplete expert routing observations")
                set_agreement.append((np.sort(expected, axis=-1) == np.sort(actual, axis=-1)).all(axis=-1))
                ordered_agreement.append((expected == actual).all(axis=-1))
                valid_rows.append(valid)
        delta[mode] = np.concatenate(differences)
        result = {
            "overall": stats(delta[mode]),
            "by_archive_step": {str(key): stats(np.concatenate(value)) for key, value in by_source.items()},
            "by_family": {key: stats(np.concatenate(value)) for key, value in by_family.items()},
        }
        if valid_rows:
            valid = np.concatenate(valid_rows)
            unordered, ordered = np.concatenate(set_agreement), np.concatenate(ordered_agreement)
            result["routing"] = {
                "token_layer_rows": int(valid.sum()),
                "same_expert_set_fraction": float(unordered[valid].mean()),
                "same_order_fraction": float(ordered[valid].mean()),
                "set_disagreement_by_layer": (np.sum(~unordered & valid, axis=0) / valid.sum(axis=0)).tolist(),
            }
        routing[mode] = result
    return complete, delta, routing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel_bytes = args.panel.read_bytes()
    panel = json.loads(gzip.decompress(panel_bytes))
    data = {step: checkpoint(args.directory / f"step-{step}", panel) for step in (384, 512)}
    if data[384][0]["probe_hash"] != data[512][0]["probe_hash"]:
        raise ValueError("Checkpoint probe hashes differ")
    if any(values[0]["panel_sha256"] != hashlib.sha256(panel_bytes).hexdigest() for values in data.values()):
        raise ValueError("Downloaded results do not identify the supplied frozen panel")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoints": {str(step): values[2] for step, values in data.items()},
        "panel_sha256": data[384][0]["panel_sha256"],
        "interpretation": "Descriptive fixed-panel mismatch; k3 is not an unbiased current-policy KL estimate.",
        "limitations": [
            "GB200 hardware differs from the original H100 training run.",
            "Family-balanced 28-question panel, both archived responses truncated to at most 2048 tokens.",
            "Router replay forces captured response routes; prompt routes remain native.",
            "Fresh workers test reproducibility after restore, not the historical in-memory synchronization state.",
        ],
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {384: "#2467a5", 512: "#c45227"}
    for step, (_, differences, summary) in data.items():
        for mode, style in (("native", "-"), ("router_replay", "--")):
            absolute = np.sort(np.abs(differences[mode]))
            positions = np.linspace(0, len(absolute) - 1, min(len(absolute), 3000), dtype=int)
            axes[0, 0].plot(
                absolute[positions],
                (positions + 1) / len(absolute),
                style,
                color=colors[step],
                label=f"Step {step}: {mode.replace('_', ' ')}",
            )
            routes = summary[mode]["routing"]["set_disagreement_by_layer"]
            axes[1, 0].plot(np.arange(1, len(routes) + 1), 100 * np.asarray(routes), style, color=colors[step])
    axes[0, 0].set(
        xscale="symlog", xlabel="Absolute log probability ratio (learner / vLLM)", ylabel="Fraction of response tokens"
    )
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set(xlabel="MoE layer", ylabel="Different expert sets (% of token positions)")
    labels = ["Native", "Rebatched", "Router replay", "vLLM repeat"]
    modes = ["native", "repeat", "router_replay", "vllm_repeat"]
    for ax, metric, ylabel, scale in (
        (axes[0, 1], "mean_abs_log_ratio", "Mean absolute log probability ratio", 1),
        (axes[1, 1], "share_beyond_2x", "Tokens with probability ratio beyond ½–2 (%)", 100),
    ):
        for index, step in enumerate((384, 512)):
            ax.bar(
                np.arange(4) + (index - 0.5) * 0.36,
                [scale * data[step][2][mode]["overall"][metric] for mode in modes],
                width=0.36,
                color=colors[step],
                label=f"Step {step}",
            )
        ax.set_xticks(np.arange(4), labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Identical frozen tokens, fresh checkpoint restores, zero optimizer updates", fontsize=14)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"fixed-token-parity.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("mean_abs_log_ratio", "k3"),
        ("Mean absolute log probability ratio", "Mean exp(δ) − 1 − δ"),
        strict=True,
    ):
        values = np.asarray(
            [
                [data[step][2]["native"]["by_archive_step"][str(source)][metric] for source in (384, 512)]
                for step in (384, 512)
            ]
        )
        heatmap = ax.imshow(values, cmap="YlOrRd", vmin=0, aspect="auto")
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f"{values[i, j]:.4g}", ha="center", va="center", color="black")
        ax.set_xticks([0, 1], ["Archive 384", "Archive 512"])
        ax.set_yticks([0, 1], ["Checkpoint 384", "Checkpoint 512"])
        ax.set_xlabel("Source of frozen response text")
        ax.set_title(title, fontsize=11)
        fig.colorbar(heatmap, ax=ax, shrink=0.8)
    fig.suptitle("Does disagreement follow the checkpoint, the response text, or both?", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"checkpoint-by-response-source.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
