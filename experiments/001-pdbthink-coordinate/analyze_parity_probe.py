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
    k3 = np.expm1(delta) - delta
    k3_total = k3.sum()
    return {
        "tokens": len(delta),
        "mean_log_ratio": float(delta.mean()),
        "mean_abs_log_ratio": float(np.abs(delta).mean()),
        "p99_abs_log_ratio": float(np.quantile(np.abs(delta), 0.99)),
        "max_abs_log_ratio": float(np.abs(delta).max()),
        "k3": float(k3.mean()),
        "k3_largest_token_fraction": float(k3.max() / k3_total) if k3_total else 0.0,
        "share_beyond_2x": float(np.mean(np.abs(delta) > np.log(2))),
        "ess_fraction": float(weights.sum() ** 2 / (len(weights) * np.square(weights).sum())),
    }


def verified_weights(folder, step):
    evidence = json.loads((folder / "weight-verification.json").read_text())
    routers = json.loads((folder / "router-weight-verification.json").read_text())
    receivers, policy = evidence["receivers"], evidence["policy"]
    if evidence["global_step"] != step or len(receivers) != 16 or len(policy) != 32:
        raise ValueError("Incomplete or wrong-step weight verification")
    if any(
        row["version"] != step
        or row["mismatched_bytes"] != 0
        or row["compared_bytes"] != row["parameter_bytes"]
        or row["compared_bytes"] <= 0
        for row in receivers
    ):
        raise ValueError("Serving weights did not pass complete byte verification")
    if any(
        row[part]["mismatched_bytes"] != 0 or row[part]["version"] != step
        for row in policy
        for part in ("replay", "replicas")
    ):
        raise ValueError("Learner weight verification failed")
    if len(routers) != 832 or any(row["dtype"] != "float32" or row["unequal"] != 0 for row in routers):
        raise ValueError("Incomplete or unequal FP32 router state")
    return {
        "serving_workers": len(receivers),
        "learner_ranks": len(policy),
        "serving_compared_bytes": sum(row["compared_bytes"] for row in receivers),
        "learner_peer_compared_bytes": sum(row["replicas"]["compared_bytes"] for row in policy),
        "mismatched_bytes": 0,
        "router_tensor_comparisons": len(routers),
        "router_unequal_elements": 0,
    }


def checkpoint(folder, panel):
    marker = "complete.json" if (folder / "complete.json").exists() else "fixed-panel-complete.json"
    complete = json.loads((folder / marker).read_text())
    if complete["optimizer_updates"] != 0:
        raise ValueError("Checkpoint has optimizer updates")
    manifests = pq.read_table(folder / "manifest.parquet").to_pylist()
    if (
        len(manifests) != 1
        or manifests[0]["status"] != "complete"
        or manifests[0]["starting_global_step"] != complete["global_step"]
    ):
        raise ValueError("Primary panel has no matching complete archive manifest")
    probes = {row["sample_id"]: row for row in pq.read_table(folder / "probe.parquet").to_pylist()}
    if set(probes) != {row["sample_id"] for row in panel["rows"]}:
        raise ValueError("The archive contains a different sample set")
    scores = pq.read_table(folder / "scores.parquet").to_pylist()
    if any(
        row["global_step"] != complete["global_step"]
        or row["update"] != 0
        or row["probe_hash"] != complete["probe_hash"]
        for row in scores
    ):
        raise ValueError("Score rows do not match the completed zero-update probe")
    indexed = {(row["sample_id"], row["scorer"], row["mode"]): row for row in scores}
    if len(indexed) != len(scores):
        raise ValueError("Duplicate sample/scorer/mode rows")
    delta, routing = {}, {}
    for mode in ("native", "repeat", "router_replay", "vllm_repeat"):
        differences, set_agreement, ordered_agreement, valid_rows = [], [], [], []
        by_source, by_family, by_question = {}, {}, {}
        for row in panel["rows"]:
            sample = row["sample_id"]
            probe = probes[sample]
            if (
                probe["prompt_token_ids"] != row["prompt_token_ids"]
                or probe["vllm_output_ids"] != row["response_token_ids"]
                or probe["loss_mask"] != list(map(bool, row["loss_mask"]))
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
            by_question.setdefault(row["prompt_id"], []).append(values[mask])
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
            "by_question": {key: stats(np.concatenate(value)) for key, value in by_question.items()},
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
    routing["learner_repeat_vs_native"] = stats(delta["repeat"] - delta["native"])
    routing["weight_verification"] = verified_weights(folder, complete["global_step"])
    routing["decode_control"] = complete.get("decode_control")
    routing["completion_scope"] = marker
    routing["software"] = json.loads(manifests[0]["software_json"])
    return complete, delta, routing


def restore_control(previous, current, panel):
    verified_weights(previous, 384)
    probes, scores = [], []
    for folder in (previous, current):
        probes.append({row["sample_id"]: row for row in pq.read_table(folder / "probe.parquet").to_pylist()})
        selected = [
            row
            for row in pq.read_table(folder / "scores.parquet").to_pylist()
            if row["scorer"] == "vllm.rescore" and row["mode"] == ""
        ]
        indexed = {row["sample_id"]: row for row in selected}
        if (
            len(indexed) != len(panel["rows"])
            or len(selected) != len(indexed)
            or any(row["global_step"] != 384 for row in selected)
        ):
            raise ValueError("Restore control has missing or duplicate scores")
        scores.append(indexed)
    differences, agreement = [], []
    for row in panel["rows"]:
        sample = row["sample_id"]
        for group in probes:
            if (
                group[sample]["prompt_token_ids"] != row["prompt_token_ids"]
                or group[sample]["vllm_output_ids"] != row["response_token_ids"]
            ):
                raise ValueError("Restore control differs from the frozen token panel")
        if any(len(group[sample]["logprobs"]) != len(row["response_token_ids"]) for group in scores):
            raise ValueError("Restore control has incomplete token scores")
        differences.append(np.asarray(scores[1][sample]["logprobs"]) - scores[0][sample]["logprobs"])
        left, right = (array(group[sample], "routed_experts") for group in probes)
        if left.shape != right.shape:
            raise ValueError("Restore control has different route tensor shapes")
        valid = np.asarray(probes[0][sample]["route_valid_mask"], dtype=bool)
        if not np.array_equal(valid, probes[1][sample]["route_valid_mask"]):
            raise ValueError("Restore control has different route validity masks")
        agreement.append((np.sort(left, axis=-1) == np.sort(right, axis=-1)).all(axis=-1)[valid])
    return {
        "checkpoint": 384,
        "comparison": "Second fresh-restore vLLM replay versus first fresh-restore vLLM replay",
        **stats(np.concatenate(differences)),
        "same_expert_set_fraction": float(np.concatenate(agreement).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-control", type=Path)
    args = parser.parse_args()
    panel_bytes = args.panel.read_bytes()
    panel = json.loads(gzip.decompress(panel_bytes))
    data = {step: checkpoint(args.directory / f"step-{step}", panel) for step in (384, 512)}
    if any(values[0]["global_step"] != step for step, values in data.items()):
        raise ValueError("Downloaded checkpoint steps do not match their directories")
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
    if args.restore_control:
        result["vllm_fresh_restore_control"] = restore_control(args.restore_control, args.directory / "step-384", panel)
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
    axes[0, 0].set_xscale("symlog", linthresh=1e-5)
    axes[0, 0].set(xlabel="Absolute log probability ratio (learner / vLLM)", ylabel="Fraction of response tokens")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set(xlabel="MoE layer", ylabel="Different expert sets (% of token positions)")
    labels = ["Native", "Rebatched", "Router replay", "vLLM\nrepeat", "Learner\nrebatch / self"]
    modes = ["native", "repeat", "router_replay", "vllm_repeat", "learner_repeat_vs_native"]
    for ax, metric, ylabel, scale in (
        (axes[0, 1], "mean_abs_log_ratio", "Mean absolute log probability ratio", 1),
        (axes[1, 1], "share_beyond_2x", "Tokens with probability ratio beyond ½–2 (%)", 100),
    ):
        for index, step in enumerate((384, 512)):
            ax.bar(
                np.arange(len(modes)) + (index - 0.5) * 0.36,
                [
                    scale
                    * (data[step][2][mode] if mode == "learner_repeat_vs_native" else data[step][2][mode]["overall"])[
                        metric
                    ]
                    for mode in modes
                ],
                width=0.36,
                color=colors[step],
                label=f"Step {step}",
            )
        ax.set_xticks(np.arange(len(modes)), labels, fontsize=8)
        ax.set_ylabel(ylabel)
        if "vllm_fresh_restore_control" in result:
            ax.axhline(
                scale * result["vllm_fresh_restore_control"][metric],
                linestyle=":",
                color="#555555",
                linewidth=1.4,
                label="384 vLLM: separate fresh restores",
            )
        ax.legend(fontsize=8)
    fig.suptitle("Identical frozen tokens, fresh checkpoint restores, zero optimizer updates", fontsize=14)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"fixed-token-parity.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, title, scale in zip(
        axes,
        ("mean_abs_log_ratio", "share_beyond_2x"),
        ("Mean absolute log probability ratio", "Probability ratios outside ½–2 (%)"),
        (1, 100),
        strict=True,
    ):
        values = np.asarray(
            [
                [scale * data[step][2]["native"]["by_archive_step"][str(source)][metric] for source in (384, 512)]
                for step in (384, 512)
            ]
        )
        heatmap = ax.imshow(values, cmap="YlOrRd", vmin=0, aspect="auto")
        for i, j in np.ndindex(values.shape):
            color = "white" if values[i, j] > 0.65 * values.max() else "black"
            ax.text(j, i, f"{values[i, j]:.4g}", ha="center", va="center", color=color)
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
