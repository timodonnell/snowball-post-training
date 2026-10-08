"""Render audited pilot evidence; requires matplotlib (no GPU or raw responses)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

BLUE = "#4477AA"
ORANGE = "#D47624"
GREEN = "#228877"
GRAY = "#8B96A4"
DARK = "#253444"

NAMES = {
    "G01": "Atom distance", "G02": "Nearest atom", "G03": "Nearest residue",
    "G04": "Steric clashes", "N01": "Shared contact residue", "P01": "Chain identifiers",
    "P02": "Residue count", "P03": "Coordinate extraction", "S01": "Salt-bridge partner",
    "S02": "Phosphorylated residue", "S03": "Solvent exposure", "S04": "Secondary structure",
    "S05": "Chain fold class", "S09": "Side-chain rotamer",
}


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CAD2DA")
    ax.tick_params(color="#CAD2DA", labelcolor=DARK)
    ax.grid(axis=axis, color="#E7ECF0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot(data, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelcolor": DARK, "text.color": DARK,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    comparison = data["comparison"]
    fig = plt.figure(figsize=(15.4, 12.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.4], hspace=0.43, wspace=0.5)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.89, bottom=0.17)
    fig.suptitle("Snowball RL pilot: a real score gain, with important limits", x=0.51, y=0.97, fontsize=21, weight="bold")
    fig.text(0.51, 0.935, "64 RL updates • preselected final checkpoint • no teacher traces", ha="center", fontsize=13)

    ax = fig.add_subplot(grid[0, 0])
    values = [comparison[k]["task_weighted_accuracy"] * 100 for k in ("baseline", "trained")]
    ax.bar([0, 1], values, color=[BLUE, ORANGE], width=0.52)
    for x, value, key in zip([0, 1], values, ["baseline", "trained"], strict=True):
        count = sum(v["successes"] for v in comparison[key]["families"].values())
        ax.text(x, value + 1, f"{value:.1f}%", ha="center", fontsize=20, weight="bold")
        ax.text(x, value / 2, f"{count:.0f}/1,003", ha="center", va="center", color="white", weight="bold", fontsize=12)
    low, high = [v * 100 for v in data["bootstrap"]["delta_95_percent_interval"]]
    delta = comparison["task_weighted_accuracy_delta"] * 100
    ax.set_title("A  Matched full validation", loc="left", pad=18)
    ax.set_xticks([0, 1], ["Original Snowball", "After 64 updates"])
    ax.set_ylim(0, 58)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel("Correct answers")
    ax.text(0.5, 55, f"+{delta:.1f} percentage points", ha="center", fontsize=14, weight="bold", color=GREEN)
    ax.text(0.5, 50, f"95% group-bootstrap interval: +{low:.1f} to +{high:.1f} pp", ha="center", fontsize=10)
    style(ax)

    ax = fig.add_subplot(grid[0, 1])
    monitor = data["monitor"]["validation"]
    steps = [p["step"] for p in monitor]
    rates = [p["task_weighted_accuracy"] * 100 for p in monitor]
    ax.plot(steps, rates, "o-", color=ORANGE, linewidth=2, markersize=6)
    ax.set_title("B  Learning curve: fixed 107-task monitor", loc="left", pad=18)
    ax.set_ylim(0, 50)
    ax.set_xlim(-3, 69)
    ax.set_xticks(range(0, 65, 8))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("RL updates")
    ax.set_ylabel("Correct answers")
    ax.annotate("21/107\n19.6%", (0, rates[0]), xytext=(3, 7), fontsize=10, ha="left",
                arrowprops={"arrowstyle": "-", "color": GRAY})
    peak = max(monitor, key=lambda p: p["task_weighted_accuracy"])
    ax.annotate("Peak: 36/107 (33.6%)", (peak["step"], peak["task_weighted_accuracy"] * 100),
                xytext=(25, 43), fontsize=10, arrowprops={"arrowstyle": "-", "color": GRAY})
    ax.annotate("Final: 32/107\n29.9%", (64, rates[-1]), xytext=(47, 10), fontsize=10,
                arrowprops={"arrowstyle": "-", "color": GRAY})
    ax.text(0.02, 0.04, "Subset + different serving runtime; compare within each panel.",
            transform=ax.transAxes, fontsize=9, color="#596979")
    style(ax)

    ax = fig.add_subplot(grid[1, 0])
    families = list(comparison["families"])
    for i, family in enumerate(families):
        row = comparison["families"][family]
        old, new = row["baseline_accuracy"] * 100, row["trained_accuracy"] * 100
        ax.plot([old, new], [i, i], color="#C9D1D8", linewidth=2.5, zorder=1)
        ax.scatter(old, i, s=52, color=BLUE, zorder=3)
        ax.scatter(new, i, s=52, color=ORANGE, zorder=4)
        if family in data["categorical_controls"]:
            rate = data["categorical_controls"][family]["constant_majority_accuracy"] * 100
            ax.scatter(rate, i, s=85, marker="|", color=DARK, linewidth=2, zorder=5)
    ax.set_title("C  Gains by task family", loc="left", pad=37)
    ax.set_yticks(range(len(families)), [f"{f}  {NAMES[f]}  ({comparison['families'][f]['n']})" for f in families])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(len(families) - 0.4, -0.7)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Full-validation accuracy • task counts in parentheses")
    handles = [
        Line2D([], [], color=BLUE, marker="o", linestyle="none", label="Original"),
        Line2D([], [], color=ORANGE, marker="o", linestyle="none", label="RL"),
        Line2D([], [], color=DARK, marker="|", markersize=11, markeredgewidth=2, linestyle="none", label="Train-majority constant answer"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.015, 1.01), ncol=3,
              frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1)
    style(ax, "x")

    ax = fig.add_subplot(grid[1, 1])
    transitions = data["transitions_by_answer_state"]
    states = ["well_formed_wrong", "format_error", "other_invalid"]
    labels = ["Well-formed, wrong", "Format error", "Other invalid*"]
    colors = [GREEN, ORANGE, GRAY]
    for y, direction in ((1, "gained"), (0, "lost")):
        left = 0
        for state, color in zip(states, colors, strict=True):
            count = transitions[state].get("correct", 0) if direction == "gained" else transitions["correct"].get(state, 0)
            ax.barh(y, count, left=left, color=color, height=0.48)
            if count >= 15:
                ax.text(left + count / 2, y, str(count), color="white", ha="center", va="center", weight="bold")
            left += count
        ax.text(left + 4, y, str(left), va="center", weight="bold")
    ax.set_title("D  What changed in the paired answers?", loc="left", pad=18)
    ax.set_yticks([1, 0], ["Newly correct", "Newly wrong"])
    ax.set_xlim(0, 205)
    ax.set_ylim(-3.5, 1.7)
    ax.set_xticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        ax.scatter(4, -0.65 - i * 0.3, color=color, s=55, marker="s")
        ax.text(12, -0.65 - i * 0.3, label, va="center", fontsize=11)
    gains = comparison["transitions"]["gained"]
    invalid_gains = sum(transitions[s].get("correct", 0) for s in ("format_error", "other_invalid"))
    valid = data["both_valid"]["total"]
    ax.text(0, -1.9, f"{invalid_gains}/{gains} gains ({invalid_gains / gains:.0%}) began with\nan invalid response.",
            fontsize=13, weight="bold", linespacing=1.5)
    ax.text(0, -2.8, f"Valid before and after ({valid['n']} tasks):\n{valid['gained']} gains − {valid['lost']} losses = +{valid['trained'] - valid['baseline']} correct",
            fontsize=12, linespacing=1.5)
    ax.text(0, -3.4, "Descriptive subset; not a causal formatting ablation.\n*Truncation/refusal/tool failure without a format-error flag.",
            fontsize=9, color="#596979", linespacing=1.6)

    fig.text(0.20, 0.045,
             "Evidence: 1,003 paired validation tasks from 20 held-out source groups; 19/20 groups improved.\n"
             "All 2,006 scores replay exactly. Approximate 95% interval: 50,000 paired group resamples (seed 17).\n"
             "One RL run; no seed-replication interval. Separate paired test was incomplete at this analysis.",
             fontsize=10, linespacing=1.5, color="#596979")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem; saves PNG and PDF")
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.output)


if __name__ == "__main__":
    main()
