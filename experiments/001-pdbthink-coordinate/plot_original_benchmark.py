"""Plot the completed matched original-benchmark comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

BLUE, GREEN, RED, DARK = "#28799e", "#298765", "#b74c53", "#173042"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/001/original-benchmark/comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/001-pdbthink-coordinate/results"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    assert data["n"] == 40
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": DARK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(14.8, 7.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1], wspace=0.38)
    fig.suptitle("Does the RL checkpoint improve the original PDBThink benchmark?", fontsize=20, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.918,
        "Same 40 context-fitting questions · same GB200 inference runtime · checkpoint selected on validation alone",
        ha="center",
        fontsize=10,
    )
    ax = fig.add_subplot(grid[0, 0])
    for x, policy in enumerate(("strict", "boxed")):
        for offset, model, color in ((-0.18, "baseline", BLUE), (0.18, "trained", GREEN)):
            v = data[policy][model]
            ax.bar(
                x + offset,
                100 * v["accuracy"],
                width=0.32,
                color=color,
                label="Original Snowball" if x == 0 and model == "baseline" else "RL step 384" if x == 0 else None,
            )
            ax.text(x + offset, 100 * v["accuracy"] + 2, f"{v['correct']}/40", ha="center", fontsize=11)
        ax.text(x, 93, f"{data[policy]['difference_pp']:+.1f} pp", ha="center", fontsize=13, weight="bold")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Strict\noriginal scoring", "Terminal-box\ncorrection"],
        ylim=(0, 100),
        ylabel="Exact accuracy",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.06, -0.16), frameon=False, fontsize=9)
    ax.set_title("A  Matched accuracy", loc="left", pad=16)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax = fig.add_subplot(grid[0, 1])
    families = list(data["strict"]["families"])
    values = np.array(
        [
            [
                data[p]["families"][f][m] / data[p]["families"][f]["n"]
                for p in ("strict", "boxed")
                for m in ("baseline", "trained")
            ]
            for f in families
        ]
    )
    ax.imshow(values, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    for y, f in enumerate(families):
        for x, (p, m) in enumerate((p, m) for p in ("strict", "boxed") for m in ("baseline", "trained")):
            row = data[p]["families"][f]
            ax.text(
                x,
                y,
                f"{row[m]}/{row['n']}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if values[y, x] < 0.5 else DARK,
            )
    ax.set(
        yticks=range(len(families)),
        yticklabels=families,
        xticks=range(4),
        xticklabels=["Original\nstrict", "RL\nstrict", "Original\nbox", "RL\nbox"],
    )
    ax.axvline(1.5, color="white", linewidth=3)
    ax.tick_params(length=0)
    ax.set_title("B  Correct answers by family", loc="left", pad=16)
    ax = fig.add_subplot(grid[0, 2])
    for x, policy in enumerate(("strict", "boxed")):
        v = data[policy]
        ax.bar(x - 0.17, v["gains"], 0.3, color=GREEN, label="Newly correct" if x == 0 else None)
        ax.bar(x + 0.17, -v["losses"], 0.3, color=RED, label="Newly wrong" if x == 0 else None)
        ax.text(x - 0.17, v["gains"] + 0.7, str(v["gains"]), ha="center")
        ax.text(x + 0.17, -v["losses"] - 1.5, str(v["losses"]), ha="center")
    ax.axhline(0, color=DARK, linewidth=0.8)
    ax.set(xticks=[0, 1], xticklabels=["Strict", "Box corrected"], ylabel="Paired question transitions", ylim=(-3, 10))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.06, -0.16), frameon=False, fontsize=9)
    ax.set_title("C  Gains and losses", loc="left", pad=16)
    s, b = data["strict"], data["boxed"]
    fig.text(
        0.04,
        0.042,
        f"Paired protein-group bootstrap 95% interval for improvement: strict {s['group_bootstrap_95_delta_pp'][0]:+.1f} to {s['group_bootstrap_95_delta_pp'][1]:+.1f} pp; "
        f"box corrected {b['group_bootstrap_95_delta_pp'][0]:+.1f} to {b['group_bootstrap_95_delta_pp'][1]:+.1f} pp.",
        fontsize=9,
    )
    fig.text(
        0.04,
        0.017,
        "40/88 questions fit; 48 are context-excluded. One response per model/question at temperature 0.6. Intervals exclude training-seed and decoding uncertainty.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.065, right=0.98, top=0.83, bottom=0.23)
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.output / f"original-benchmark.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
