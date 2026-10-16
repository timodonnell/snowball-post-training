"""Render the regression audit from committed JSON; needs matplotlib and numpy."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from plot_pilot_evidence import BLUE, DARK, GRAY, GREEN, NAMES, ORANGE, style

RED = "#B4474C"


def save(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def evidence(data, previous, output):
    fig = plt.figure(figsize=(15.5, 12.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.35], hspace=0.43, wspace=0.40)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.16, top=0.88)
    fig.suptitle("Snowball RL: the regression is real", y=0.968, fontsize=22, weight="bold")
    fig.text(0.5, 0.93, "Same 1,003 validation tasks • native evaluator • 4,012 archived responses replayed exactly",
             ha="center", fontsize=12)
    val = data["validation"]
    before, after = data["comparison_steps"]
    left, right = [next(r for r in val if r["step"] == step) for step in (before, after)]
    delta = (right["accuracy"] - left["accuracy"]) * 100

    ax = fig.add_subplot(grid[0, 0])
    old = previous["validation"]
    ax.plot([r["step"] for r in old], [r["task_weighted_accuracy"] * 100 for r in old],
            "o-", color=GRAY, label="Earlier run")
    ax.plot([r["step"] for r in val], [r["accuracy"] * 100 for r in val],
            "o-", color=BLUE, linewidth=2.3, label="Resumed from 320")
    ax.scatter([before, after], [left["accuracy"] * 100, right["accuracy"] * 100],
               color=[GREEN, RED], s=68, zorder=4)
    ax.annotate("Best: 616/1,003\n61.4%", (before, left["accuracy"] * 100),
                xytext=(290, 73), fontsize=11, color=GREEN,
                arrowprops={"arrowstyle": "-", "color": GREEN})
    ax.annotate("334/1,003\n33.3%", (after, right["accuracy"] * 100),
                xytext=(405, 15), fontsize=11, color=RED,
                arrowprops={"arrowstyle": "-", "color": RED})
    ax.text(0.05, 0.29, f"{delta:.1f} percentage points", transform=ax.transAxes,
            weight="bold", color=RED, fontsize=13)
    ax.set(xlim=(45, 535), ylim=(0, 85), xlabel="Cumulative optimizer updates", ylabel="Validation accuracy")
    ax.set_xticks([64, 192, 320, 384, 512])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_title("A  Full validation over training", loc="left", pad=14)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    style(ax)

    ax = fig.add_subplot(grid[0, 1])
    xs = np.arange(len(val))
    ax.bar(xs - 0.17, [100 * r["format_errors"] / r["tasks"] for r in val],
           width=0.32, color=ORANGE, label="Format errors")
    ax.bar(xs + 0.17, [100 * r["truncations"] / r["tasks"] for r in val],
           width=0.32, color=GRAY, label="Truncated")
    ax.set(ylim=(0, 50), ylabel="Fraction of validation answers", xlabel="Checkpoint")
    ax.set_xticks(xs, [r["step"] for r in val])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_title("B  Answers shorten; truncation falls", loc="left", pad=14)
    style(ax)
    twin = ax.twinx()
    lengths = [r["mean_output_tokens"] / 1000 for r in val]
    twin.plot(xs, lengths, "o-", color=BLUE, linewidth=2, label="Mean output tokens")
    twin.set(ylim=(0, 8), ylabel="Output tokens (thousands)")
    twin.spines[["top", "left"]].set_visible(False)
    twin.spines["right"].set_color("#CAD2DA")
    for i, length in enumerate(lengths):
        twin.annotate(f"{length:.1f}k", (i, length), xytext=(0, 9), textcoords="offset points",
                      ha="center", color=BLUE, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="upper left", ncol=1, frameon=False, fontsize=9)

    ax = fig.add_subplot(grid[1, 0])
    families = list(data["families"])
    for y, family in enumerate(families):
        row = data["families"][family]
        a, b = [100 * row[k] / row["n"] for k in ("before_correct", "after_correct")]
        ax.plot([a, b], [y, y], color="#CAD2DA", linewidth=2.6)
        ax.scatter(a, y, color=GREEN, s=48, zorder=3)
        ax.scatter(b, y, color=RED, s=48, zorder=4, marker="D")
    ax.set_yticks(range(len(families)), [f"{f}  {NAMES[f]}  ({data['families'][f]['n']})" for f in families])
    ax.tick_params(axis="y", labelsize=10)
    ax.set(xlim=(-3, 103), ylim=(len(families) - 0.4, -0.7), xlabel="Accuracy • task counts in parentheses")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_title("C  Geometry and rotamers lose most", loc="left", pad=38)
    ax.legend(handles=[Line2D([], [], color=color, marker=marker, linestyle="none", label=f"Step {step}")
                       for color, marker, step in [(GREEN, "o", before), (RED, "D", after)]],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=10)
    style(ax, "x")

    ax = fig.add_subplot(grid[1, 1])
    transitions = data["transitions"]
    states = ["well_formed_wrong", "format_error", "other_invalid"]
    colors = [RED, ORANGE, GRAY]
    for y, lost in [(1, True), (0, False)]:
        offset = 0
        for state, color in zip(states, colors):
            n = transitions["correct"].get(state, 0) if lost else transitions.get(state, {}).get("correct", 0)
            ax.barh(y, n, left=offset, color=color, height=0.43)
            if n >= 20:
                ax.text(offset + n / 2, y, str(n), color="white", ha="center", va="center", weight="bold")
            offset += n
        ax.text(offset + 5, y, str(offset), va="center", weight="bold")
    ax.set_yticks([1, 0], ["Lost correct", "Newly correct"])
    ax.set(xlim=(0, 350), ylim=(-3.5, 1.8))
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("D  Most lost answers are valid but wrong", loc="left", pad=38)
    for y, color, text in zip([-0.8, -1.2, -1.6], colors,
                              ["Well-formed but wrong", "Format error", "Other invalid"]):
        ax.scatter(6, y, color=color, s=60, marker="s")
        ax.text(20, y, text, va="center", fontsize=10)
    ax.text(0, -2.3, "312 losses − 30 gains = 282 fewer correct answers\n"
            "217/312 losses (70%) are well-formed errors.\n"
            "All 20 source groups decline.", va="top", fontsize=11, linespacing=1.6)
    low, high = [100 * n for n in data["source_group_bootstrap"]["delta_95_percent_interval"]]
    fig.text(0.20, 0.055,
             f"Step {before} → {after}: 95% paired source-group bootstrap interval {low:.1f} to {high:.1f} pp. "
             "Source-composition uncertainty only.\n"
             "Native evaluation has a different serving runtime from the original-model/pilot standalone comparison. "
             "Do not combine their absolute scores.\n"
             "Format errors and truncation can overlap. All panels use the frozen native verifier; no new generations.",
             fontsize=9, color="#596979", linespacing=1.5)
    save(fig, output, "epoch-regression-evidence")


def diagnostics(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.3))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.20, top=0.84, hspace=0.48, wspace=0.30)
    fig.suptitle("The policy becomes less stable before accuracy collapses", y=0.965, fontsize=22, weight="bold")
    fig.text(0.5, 0.92,
             "245 updates on 31,360 training responses • faint: each update • dark: trailing 16-update mean",
             ha="center", fontsize=12)
    fig.text(0.5, 0.884,
             "Shading marks steps 448–512, the interval containing the sharp validation drop; it is not an estimated change point.",
             ha="center", fontsize=10, color="#596979")
    panels = [
        ("reward/avg_raw_reward", "A  Training accuracy also falls", "Correct responses", 100, BLUE),
        ("policy/policy_entropy", "B  Policy entropy falls", "Mean token entropy (nats)", 1, GREEN),
        ("policy/raw_grad_norm", "C  Gradient norms rise", "Norm before gradient clipping", 1, ORANGE),
        ("policy/mismatch/pooled/kl_k3", "D  Train / rollout mismatch grows", "Sampled-token k3 mismatch", 1, RED),
        ("policy/mismatch/pooled/ess_fraction", "E  Probability ratios concentrate", "Importance-ratio ESS fraction", 100, RED),
        ("generate/avg_num_tokens", "F  Training responses shorten", "Mean generated tokens", 1, BLUE),
    ]
    for ax, (key, title, label, scale, color) in zip(axes.flat, panels):
        rows = [r for r in data["history"] if key in r]
        x = np.array([r["trainer/global_step"] for r in rows])
        y = np.array([r[key] * scale for r in rows])
        ax.axvspan(448, 512, color=ORANGE, alpha=0.10, linewidth=0)
        ax.plot(x, y, color=color, alpha=0.20, linewidth=0.7)
        ax.plot(x[15:], np.convolve(y, np.ones(16) / 16, mode="valid"), color=color, linewidth=2)
        ax.set(xlim=(320, max(x)), xlabel="Optimizer update", ylabel=label)
        ax.set_xticks([320, 384, 448, 512, 565])
        ax.set_title(title, loc="left", pad=12, fontsize=12)
        ax.set_ylim(bottom=0)
        style(ax)
        if scale == 100:
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        if key == "policy/raw_grad_norm":
            ax.axhline(1, color=DARK, linestyle="--", linewidth=1)
            ax.text(326, 1.03, "Gradient clipping threshold", fontsize=8, color=DARK)
        if key == "reward/avg_raw_reward":
            for row in data["family_standardization"]["windows"]:
                ax.plot([row["first_step"], row["last_step"]], [row["fixed_family_mix_accuracy"] * 100] * 2,
                        color=DARK, linewidth=2.2, linestyle="--")
            ax.legend(handles=[Line2D([], [], color=DARK, linestyle="--", label="Fixed family mix, window mean")],
                      loc="lower left", frameon=False, fontsize=8)
    mix = data["family_standardization"]
    fig.text(0.075, 0.065,
             "PPO clipping activated on 0% of tokens in every audited update; one optimizer pass per batch. "
             "Reference KL, entropy loss and rollout correction were disabled.\n"
             "Mismatch is between Megatron training and vLLM rollout probabilities, not KL to the original model. "
             "ESS is a ratio diagnostic, not independent task count.\n"
             f"Fixed-family weights use steps 321–448, retaining {mix['reference_coverage']:.1%} of reference responses. "
             "Within-family difficulty is uncontrolled. These are observational diagnostics.",
             fontsize=9, color="#596979", linespacing=1.6)
    save(fig, output, "epoch-regression-diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path(__file__).parent / "results/epoch-regression-audit.json")
    parser.add_argument("--previous", type=Path, default=Path(__file__).parent / "results/epoch-interruption.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelcolor": DARK, "text.color": DARK,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    data = json.loads(args.audit.read_text())
    evidence(data, json.loads(args.previous.read_text()), args.output)
    diagnostics(data, args.output)


if __name__ == "__main__":
    main()
