"""Prepend the dated sampling correction to the existing results report."""

from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "001-pdbthink-coordinate/results/snowball-rl-report-with-probability-diagnostic.pdf"


def main():
    for name, filename in (("D", "DejaVuSans.ttf"), ("DB", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, f"/usr/share/fonts/truetype/dejavu/{filename}"))
    pdfmetrics.registerFontFamily("D", normal="D", bold="DB")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TextD", fontName="D", fontSize=9, leading=12.5, spaceAfter=10))
    styles.add(ParagraphStyle(name="TitleD", fontName="DB", fontSize=18, leading=23, spaceAfter=12))
    styles.add(ParagraphStyle(name="SmallD", parent=styles["TextD"], fontSize=8, leading=11))
    story = []

    def paragraph(text, style="TextD"):
        story.append(Paragraph(text, styles[style]))

    paragraph("Correction: validation sampling", "TitleD")
    paragraph("Snowball PDBThink RL report · October 21, 2026", "SmallD")
    paragraph(
        "<b>The earlier native in-training validation was not greedy.</b> Its configuration requested "
        "temperature 0, but the pinned MarinSkyRL serving entrypoint supplied training sampling settings "
        "to a shared wrapper. That wrapper overwrote each chat request's temperature with the training "
        "value, 1. Earlier report statements describing that curve as greedy are incorrect. This "
        "correction supersedes those statements in the historical report reproduced after this page."
    )
    rows = [
        ["Measurement", "Effective sampling"],
        ["Previous native training validation", "Temperature 1; top-p 1; top-k disabled"],
        ["Separate full step-zero baseline", "Temperature 0; direct endpoint; 1,003 tasks"],
        ["Matched original PDBThink benchmark", "Temperature 0.6; top-p 0.95; top-k 20; direct endpoint"],
        ["Corrected router-replay retry", "Temperature 0; fixed 1,003-task validation from step 0"],
    ]
    table = Table([[Paragraph(cell, styles["SmallD"]) for cell in row] for row in rows], colWidths=[208, 314])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4eef5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d3dc")),
            ]
        )
    )
    story.extend([table, Spacer(1, 12)])
    paragraph(
        "<b>What changes:</b> comparisons between the separate greedy baseline and native training "
        "validation also differ in sampling temperature. The paired reasoning audit remains a "
        "description of the archived responses, but is not a matched causal estimate of RL. The new "
        "greedy curve must be assessed against its own baseline; the old native curve is not a directly "
        "comparable control. This discovery does not by itself explain the late-training regression."
    )
    paragraph(
        "<b>What remains valid:</b> archived answers and verifier scores are unchanged. The matched "
        "original-benchmark comparison used a separately deployed endpoint and frozen requests, outside "
        "this wrapper, so those results are unaffected. The controlled, identical-token probability "
        "checks still establish that expert selection contributes to learner–rollout disagreement."
    )
    paragraph(
        "<b>Retry changes:</b> honor per-request sampling; request no unused probability logs for greedy "
        "validation; preserve both Snowball stop IDs explicitly; reject validation execution errors or "
        "incomplete coverage. Training uses route replay, importance weights capped at 4, fresh optimizer "
        "state, and frozen quantile biases. Its minimum-token setting is now 0 so the first-token EOS "
        "mask cannot create an unmatched sampling processor. No new RL improvement is claimed here."
    )
    paragraph(
        "<b>Evidence:</b> pinned MarinSkyRL 5663f612, skyrl_train/entrypoints/main_base.py "
        "(create_ray_wrapped_inference_engines_from_config) and inference_engines/vllm/utils.py "
        "(apply_openai_sampling); original resolved launch artifact; retry v3 validation failure and "
        "v4 request-path regression tests. See experiments/002-pdbthink-router-replay/README.md and "
        "results/v3-startup-and-failure.json for the execution record. All 273 final CPU checks passed; "
        "these are not evidence of training improvement.",
        "SmallD",
    )
    erratum = HERE / "results/validation-sampling-erratum.pdf"
    SimpleDocTemplate(
        str(erratum),
        pagesize=(612, 792),
        leftMargin=45,
        rightMargin=45,
        topMargin=38,
        bottomMargin=36,
        title="Snowball validation sampling correction",
    ).build(story)
    with pymupdf.open(erratum) as correction:
        if len(correction) != 1:
            raise ValueError("Expected a one-page correction")
        with pymupdf.open(OLD) as original:
            correction.insert_pdf(original)
        correction.save(HERE / "results/snowball-rl-report-corrected.pdf", garbage=4, deflate=True)
    print(erratum)


if __name__ == "__main__":
    main()
