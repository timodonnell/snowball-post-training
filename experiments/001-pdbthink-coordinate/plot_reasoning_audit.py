"""Plot archival validation scores beside explicit shortcut controls, and audited angles."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "results" / "reasoning-audit"
NAMES = {
    "G01": "Atom distance",
    "G02": "Nearest nonbonded atom",
    "G03": "Nearest residue",
    "G04": "Worst steric clash",
    "N01": "Shared contact",
    "P01": "Chain IDs",
    "P02": "Residue count",
    "P03": "Coordinate lookup",
    "S01": "Salt bridge",
    "S02": "Phosphorylated residue",
    "S03": "Solvent exposure",
    "S04": "Local secondary structure",
    "S05": "Fold class",
    "S09": "Rotamer",
}
# Last numerical angle discussed in each reviewed step-384 trace, approximate when rounded.
# These are model statements, NOT independently measured angles or proof of valid reasoning.
STATED = [-54, -82, -1.508, -10.3, 50.8, 45.14, -177.326, -179.2, 90.6, -41.6, 48.9, -102.5, 97.5]


def main():
    data = json.loads((ROOT / "statistics.json").read_text())
    pop = data["population"]
    families = sorted(pop)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(families))
    for key, offset, color, label in [
        ("correct0", -0.18, "#9cb5c3", "Step 0: standalone evaluator"),
        ("correct384", 0.18, "#16748b", "Step 384: native training evaluator"),
    ]:
        values = [100 * pop[f][key] / pop[f]["n"] for f in families]
        ax.barh(y + offset, values, height=0.33, color=color, label=label)
        for j, f in enumerate(families):
            ax.text(values[j] + 0.7, j + offset, f"{pop[f][key]}/{pop[f]['n']}", va="center", fontsize=8)
    controls = {"G03": 79 / 108, "S03": 98 / 119, "S04": 68 / 111, "S05": 8 / 12, "S09": 46 / 116}
    for j, f in enumerate(families):
        if f in controls:
            ax.scatter(
                100 * controls[f],
                j,
                marker="D",
                s=37,
                color="#bd5f26",
                zorder=4,
                label="Coordinate proxy or constant-label control" if f == "G03" else None,
            )
    ax.set_yticks(y, [f"{f}  {NAMES[f]}" for f in families])
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Strictly correct answers in the full validation category (%)")
    ax.grid(axis="x", color="#dddddd", alpha=0.55)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[1:] + handles[:1],
        labels[1:] + labels[:1],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.905),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Higher answer accuracy can coexist with shortcuts", fontsize=17, fontweight="bold", x=0.04, ha="left")
    fig.text(
        0.04,
        0.93,
        "All 1,003 validation tasks; counts are not estimated from the 100-question audit sample.",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.035,
        "Controls: G03 = nearest alpha carbon; S03 = always exposed; S04 = always coil;\n"
        "S05 = always alpha helical; S09 = always g−. Evaluator paths differ between checkpoints.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.30, right=0.97, top=0.82, bottom=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"category-controls.{ext}", dpi=180)
    plt.close(fig)

    annotations = json.loads((ROOT / "annotations.json").read_text())
    geometry = json.loads((ROOT / "geometry.json").read_text())
    rots = [r for r in annotations if r["family"] == "S09"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    for lo, hi, color in [(-180, -120, "#eee6d9"), (-120, 0, "#e2eff4"), (0, 120, "#e9f0dd"), (120, 180, "#eee6d9")]:
        ax.axhspan(lo, hi, color=color, alpha=0.5)
    true = [geometry["sample"][r["id"]]["angle"] for r in rots]
    for r, x, yy in zip(rots, true, STATED, strict=True):
        correct = bool(r["steps"]["384"]["reward"])
        ax.scatter(x, yy, color="#16748b" if correct else "#ba4935", s=45)
        offsets = {91: (8, 9), 92: (6, -13), 97: (-20, 10), 93: (22, 15), 94: (38, -1)}
        offset = offsets.get(r["index"], (5, 4))
        ax.annotate(
            str(r["index"]),
            (x, yy),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#777777"} if r["index"] in offsets else None,
        )
    ax.plot([-180, 180], [-180, 180], color="#777777", lw=1, linestyle="--")
    ax.set(
        xlim=(-188, 188),
        ylim=(-188, 188),
        xlabel="Independently computed angle (degrees)",
        ylabel="Angle stated by step 384 (degrees)",
        title="13 audited rotamer responses",
    )
    ax.set_xticks([-180, -120, 0, 120, 180])
    ax.set_yticks([-180, -120, 0, 120, 180])
    ax.text(
        0.04,
        0.96,
        "Blue: correct final bin\nRed: wrong final bin\nNumbers: audit case index",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax = axes[1]
    allrots = geometry["all_s09"]
    xx = [r["angle"] for r in allrots]
    yy = [r["missing_bond_norm_angle"] for r in allrots]
    ax.scatter(xx, yy, color="#16748b", s=18, alpha=0.65)
    ax.plot([-180, 180], [-180, 180], color="#777777", lw=1, linestyle="--")
    for boundary in (-120, 0, 120):
        ax.axvline(boundary, color="#cccccc", lw=0.7)
        ax.axhline(boundary, color="#cccccc", lw=0.7)
    ax.set(
        xlim=(-188, 188),
        ylim=(-188, 188),
        xlabel="Correct dihedral (degrees)",
        ylabel="Dihedral with missing bond norm (degrees)",
        title="Wrong formula, correct input coordinates",
    )
    ax.set_xticks([-180, -120, 0, 120, 180])
    ax.set_yticks([-180, -120, 0, 120, 180])
    ax.text(
        0.04,
        0.96,
        "116/116 coarse bins unchanged\nThis is an offline formula control,\nnot 116 model responses.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.suptitle("Rotamer reward does not certify the angle calculation", fontsize=17, fontweight="bold")
    fig.text(
        0.04,
        0.02,
        "Bins: g− = [−120°, 0°); g+ = [0°, 120°); t = remaining angles. Correct bins can follow incorrect mathematics.\n"
        "Left: outcome-stratified sample, not a population accuracy estimate; stated angles include transcription and arithmetic errors.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.19, wspace=0.32)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"rotamer-audit.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
