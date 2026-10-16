"""Plot the full recorded RL lineage, starting at step zero."""

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
from plot_regression import RED, save

PURPLE = "#8856A7"


def time_axis(ax, last):
    ax.set_xlim(0, last + 6)
    ax.set_xticks([0, 128, 256, 384, 512, last])
    ax.set_xlabel("Optimizer update")
    for step in (64, 320):
        ax.axvline(step, color=GRAY, linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axvspan(448, 512, color=ORANGE, alpha=0.08, linewidth=0)
    style(ax)


def evaluation_series(data, ax, metric, scale=1):
    standalone = data["standalone_validation"]["points"]
    ax.plot([r["step"] for r in standalone], [r[metric] * scale for r in standalone],
            "s--", color=PURPLE, linewidth=2, markersize=6, label="Full validation, standalone (1,003)")
    for index, run in enumerate(data["native_validation"]):
        rows = run["points"]
        ax.plot([r["step"] for r in rows], [r[metric] * scale for r in rows],
                "o-", color=BLUE, linewidth=2.2, markersize=6,
                label="Full validation, native (1,003)" if index == 0 else None)


def evaluation(data, output):
    last = data["last_optimizer_update"]
    native_points = [r for run in data["native_validation"] for r in run["points"]]
    best = max(native_points, key=lambda r: r["accuracy"])
    latest = native_points[-1]
    initial = data["standalone_validation"]["points"][0]
    fig = plt.figure(figsize=(16, 12.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.4], hspace=0.43, wspace=0.32)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.88, bottom=0.17)
    fig.suptitle(f"Snowball RL: the full history from step 0 to {last}", y=0.97, fontsize=22, weight="bold")
    fig.text(0.55, 0.935, "Different colors keep the evaluation protocols separate; they are not one matched accuracy curve.",
             ha="center", fontsize=11)
    fig.text(0.55, 0.908, "Dotted vertical lines: scale-up at 64 and restart at 320 • shaded interval: 448–512",
             ha="center", fontsize=10, color="#596979")

    ax = fig.add_subplot(grid[0, 0])
    evaluation_series(data, ax, "accuracy", 100)
    monitor = data["pilot_monitor"]["points"]
    ax.plot([r["step"] for r in monitor], [r["task_weighted_accuracy"] * 100 for r in monitor],
            ".:", color=GRAY, linewidth=1.2, label="Pilot native monitor (107)")
    ax.annotate(f"Step 0: {initial['accuracy']:.1%}", (0, initial["accuracy"] * 100),
                xytext=(100, 12), color=PURPLE, fontsize=10,
                arrowprops={"arrowstyle": "-", "color": PURPLE})
    ax.annotate(f"Best: {best['accuracy']:.1%}", (best["step"], best["accuracy"] * 100),
                xytext=(best["step"] - 84, 72), color=GREEN, fontsize=11,
                arrowprops={"arrowstyle": "-", "color": GREEN})
    ax.annotate(f"Step {latest['step']}: {latest['accuracy']:.1%}", (latest["step"], latest["accuracy"] * 100),
                xytext=(latest["step"] - 157, 23), color=RED, fontsize=10,
                arrowprops={"arrowstyle": "-", "color": RED})
    ax.set(ylim=(0, 85), ylabel="Correct answers")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_title("A  Accuracy from the original model onward", loc="left", pad=14)
    time_axis(ax, last)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = fig.add_subplot(grid[0, 1])
    evaluation_series(data, ax, "mean_output_tokens", 0.001)
    ax.set(ylim=(0, 6.5), ylabel="Mean output tokens (thousands)")
    ax.set_title("B  Full-validation response length", loc="left", pad=14)
    time_axis(ax, last)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = fig.add_subplot(grid[1, 0])
    standalone = data["standalone_validation"]["points"]
    columns = standalone + [r for run in data["native_validation"] for r in run["points"]]
    families = list(standalone[0]["family_accuracy"])
    values = np.array([[r["family_accuracy"][f] * 100 for r in columns] for f in families])
    im = ax.imshow(values, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    for y in range(len(families)):
        for x in range(len(columns)):
            value = values[y, x]
            ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if value < 50 else DARK)
    ax.axvline(1.5, color="white", linewidth=3)
    ax.axvline(6.5, color="white", linewidth=1.5, linestyle="--")
    labels = [str(r["step"]) for r in columns]
    labels[0], labels[1], labels[2], labels[7] = "0\nS", "64\nS", "64\nN", "320\nR"
    ax.set_xticks(range(len(columns)), labels)
    ax.tick_params(axis="x", labelsize=9, length=0)
    ax.set_yticks(range(len(families)), [f"{f}  {NAMES[f]}" for f in families])
    ax.tick_params(axis="y", labelsize=9, length=0)
    ax.set_title("C  Every family, from step 0 (accuracy %)", loc="left", pad=14)
    ax.set_xlabel("Checkpoint • S: standalone; N: native; R: restart")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025).set_label("Accuracy (%)")

    ax = fig.add_subplot(grid[1, 1])
    for metric, color, label in [("format_rate", ORANGE, "Format error"), ("truncation_rate", GRAY, "Truncated")]:
        ax.plot([r["step"] for r in standalone], [100 * r[metric] for r in standalone],
                "s--", color=color, linewidth=2)
        for run in data["native_validation"]:
            rows = run["points"]
            ax.plot([r["step"] for r in rows], [100 * r[metric] for r in rows], "o-", color=color, linewidth=2)
    ax.set(ylim=(0, 55), ylabel="Fraction of validation answers")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_title("D  Format errors and truncation", loc="left", pad=14)
    time_axis(ax, last)
    ax.legend(handles=[
        Line2D([], [], color=ORANGE, label="Format errors"), Line2D([], [], color=GRAY, label="Truncated"),
        Line2D([], [], color=DARK, marker="s", linestyle="--", label="Standalone serving"),
        Line2D([], [], color=DARK, marker="o", label="Native serving"),
    ], loc="upper right", frameon=False, fontsize=9)
    fig.text(0.19, 0.07,
             "Standalone full validation: step 0 = 230/1,003; step 64 = 378/1,003. "
             "Native full validation begins at step 64 = 401/1,003.\n"
             "The two serving runtimes differ, so no line joins them at step 64. "
             "The 107-task pilot monitor is shown only in panel A.\n"
             "Step 320 was evaluated twice around the restart (576 vs 580 correct). "
             f"Latest full validation is at {latest['step']}; no scores are extrapolated to later updates.\n"
             "Format and truncation errors can overlap. Heatmap cells show rounded percentages; small families have few tasks.",
             fontsize=9, color="#596979", linespacing=1.6)
    save(fig, output, "full-training-evidence")


def smoothed(rows, key, low, high):
    values = {r["step"]: r[key] for r in rows if r.get(key) is not None}
    x = np.arange(low, high + 1)
    y = np.array([values.get(int(step), np.nan) for step in x])
    # A missing update remains a gap, including any averaging window containing it.
    return x, y, x[15:], np.convolve(y, np.ones(16) / 16, mode="valid")


def diagnostics(data, output):
    last = data["last_optimizer_update"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.7))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.20, top=0.83, hspace=0.48, wspace=0.30)
    fig.suptitle("Snowball RL: optimizer and rollout diagnostics from the start", y=0.965, fontsize=21, weight="bold")
    fig.text(0.5, 0.92,
             f"Steps 0–{last} • faint: each update • colored line: trailing 16-update mean • gray dashed: discarded branch",
             ha="center", fontsize=11)
    fig.text(0.5, 0.88,
             "Dotted vertical lines: scale-up at 64 and restart at 320 • shaded interval: 448–512",
             ha="center", fontsize=10, color="#596979")
    panels = [
        ("reward/avg_raw_reward", "A  Training success rate", "Correct responses", 100, BLUE),
        ("policy/policy_entropy", "B  Policy entropy", "Mean token entropy (nats)", 1, GREEN),
        ("policy/raw_grad_norm", "C  Gradient norm", "Norm before gradient clipping", 1, ORANGE),
        ("policy/mismatch/pooled/kl_k3", "D  Train / rollout mismatch", "Sampled-token k3 mismatch", 1, RED),
        ("policy/mismatch/pooled/ess_fraction", "E  Probability-ratio concentration", "Importance-ratio ESS fraction", 100, RED),
        ("generate/avg_num_tokens", "F  Training response length", "Mean generated tokens", 1, BLUE),
    ]
    for ax, (key, title, label, scale, color) in zip(axes.flat, panels):
        x, y, smooth_x, smooth_y = smoothed(data["training"], key, 1, last)
        ax.plot(x, y * scale, color=color, alpha=0.20, linewidth=0.65)
        ax.plot(smooth_x, smooth_y * scale, color=color, linewidth=2)
        discarded = data["discarded_training"]
        if discarded:
            bx, by, sx, sy = smoothed(discarded, key, min(r["step"] for r in discarded), max(r["step"] for r in discarded))
            ax.plot(bx, by * scale, color=GRAY, alpha=0.25, linewidth=0.65)
            ax.plot(sx, sy * scale, color=GRAY, linestyle="--", linewidth=1.4)
        time_axis(ax, last)
        ax.set_ylabel(label)
        ax.set_title(title, loc="left", pad=12, fontsize=12)
        ax.set_ylim(bottom=0)
        if scale == 100:
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        if key == "policy/raw_grad_norm":
            ax.axhline(1, color=DARK, linestyle="--", linewidth=1)
            ax.text(8, 1.04, "Gradient clipping threshold", color=DARK, fontsize=8)
    fig.text(0.075, 0.065,
             "Retained lineage: pilot 1–64 → first continuation 65–320 → resumed continuation 321 onward. "
             "The failed attempt's updates 321–375 were discarded and repeated.\n"
             "Update diagnostics first exist at step 1. W&B is missing pilot update 64; gaps are left unfilled. "
             f"The plotted history contains {len(data['training'])} retained update records.\n"
             "Mismatch compares Megatron training with vLLM rollout probabilities, not the original model. "
             "ESS describes token probability ratios, not independent tasks.",
             fontsize=9, color="#596979", linespacing=1.6)
    save(fig, output, "full-training-diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "results/full-training-history.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold",
        "axes.labelcolor": DARK, "text.color": DARK, "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    data = json.loads(args.data.read_text())
    evaluation(data, args.output)
    diagnostics(data, args.output)


if __name__ == "__main__":
    main()
