"""Render measured parity results and their interpretation as a PDF addendum."""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--interpretation", type=Path, required=True)
    parser.add_argument("--base-report", type=Path)
    args = parser.parse_args()
    data = json.loads((args.results / "comparison.json").read_text())
    interpretation = json.loads(args.interpretation.read_text())
    for name, filename in (("D", "DejaVuSans.ttf"), ("DB", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, f"/usr/share/fonts/truetype/dejavu/{filename}"))
    pdfmetrics.registerFontFamily("D", normal="D", bold="DB")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyD", fontName="D", fontSize=9, leading=13, spaceAfter=9))
    styles.add(ParagraphStyle(name="TitleD", fontName="DB", fontSize=20, leading=25, spaceAfter=15))
    styles.add(ParagraphStyle(name="HeadD", fontName="DB", fontSize=12, leading=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallD", parent=styles["BodyD"], fontSize=7.5, leading=10))
    story = []

    def paragraph(text, style="BodyD"):
        story.append(Paragraph(escape(text), styles[style]))

    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles["SmallD"]) for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1)
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4eef5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d3dc")),
                ]
            )
        )
        story.extend([item, Spacer(1, 12)])

    paragraph("Snowball RL: controlled probability check", "TitleD")
    paragraph("Fresh checkpoint restores · identical tokens · zero optimizer updates", "HeadD")
    for text in interpretation["findings"]:
        paragraph(text)
    rows = [["Checkpoint / mode", "Mean |δ|", "Mean k3", "Beyond 2×", "ESS / N"]]
    for step in ("384", "512"):
        for mode, label in (("native", "native"), ("router_replay", "forced response routes")):
            metric = data["checkpoints"][step][mode]["overall"]
            rows.append(
                [
                    f"{step}: {label}",
                    f"{metric['mean_abs_log_ratio']:.5g}",
                    f"{metric['k3']:.5g}",
                    f"{100 * metric['share_beyond_2x']:.3f}%",
                    f"{metric['ess_fraction']:.4g}",
                ]
            )
    table(rows, [180, 75, 75, 75, 75])
    paragraph(
        "The main comparisons are mean |δ| and the fraction beyond 2×. On archived text, very low "
        "reference probabilities can produce extreme ratios; k3 and ESS can consequently be dominated "
        "by a few tokens. These fixed-panel k3 values are not estimates of current-policy KL and must "
        "not be compared numerically with the training-time k3 curve.",
        "SmallD",
    )
    paragraph("Controls", "HeadD")
    for step in ("384", "512"):
        item = data["checkpoints"][step]
        weight = item["weight_verification"]
        paragraph(
            f"Checkpoint {step}: all {weight['serving_workers']} serving workers and "
            f"{weight['learner_ranks']} learner ranks passed verification. "
            f"{weight['serving_compared_bytes']:,} serving bytes and "
            f"{weight['learner_peer_compared_bytes']:,} learner-peer bytes were compared, with zero mismatches. "
            f"All {weight['router_tensor_comparisons']} full-FP32 router weight/bias comparisons agreed exactly. "
            f"Mean |δ| for repeated vLLM scoring: {item['vllm_repeat']['overall']['mean_abs_log_ratio']:.5g}; "
            f"rebatched learner versus original learner: {item['learner_repeat_vs_native']['mean_abs_log_ratio']:.5g}."
        )
    for text in interpretation.get("limitations", []):
        paragraph(text)
    story.append(PageBreak())
    paragraph("Decode and repeatability controls", "TitleD")
    paragraph(
        "These measurements use fresh checkpoint-specific responses, separate from the 56-sequence "
        "fixed panel. All ratios in the table below use the probability recorded during vLLM autoregressive "
        "decoding as the denominator. Prefill rescoring checks numerical differences within vLLM itself."
    )
    rows = [["Checkpoint / comparison", "Mean |δ|", "Mean k3", "Beyond 2×", "ESS / N"]]
    for step in ("384", "512"):
        if data["checkpoints"][step]["decode_control"] is None:
            paragraph(
                f"Checkpoint {step}: no auxiliary decode measurements. Its primary fixed-panel comparison completed and was archived before the auxiliary request failed."
            )
            continue
        for mode, label in (
            ("prefill_vs_decode", "vLLM prefill / decode"),
            ("native", "native learner / decode"),
            ("router_replay", "forced routes / decode"),
        ):
            metric = data["checkpoints"][step]["decode_control"][mode]
            rows.append(
                [
                    f"{step}: {label}",
                    f"{metric['mean_abs_log_ratio']:.5g}",
                    f"{metric['k3']:.5g}",
                    f"{100 * metric['share_beyond_2x']:.3f}%",
                    f"{metric['ess_fraction']:.4g}",
                ]
            )
    table(rows, [180, 75, 75, 75, 75])
    for text in interpretation.get("decode_control", []):
        paragraph(text)
    paragraph(
        "This control covers four questions and short response prefixes. It cannot establish the "
        "magnitude of decode/prefill differences over complete, long reasoning traces. It does not "
        "provide a controlled checkpoint comparison with fixed response tokens."
    )
    if "vllm_fresh_restore_control" in data:
        paragraph("Separate fresh restores", "HeadD")
        metric = data["vllm_fresh_restore_control"]
        paragraph(
            "An additional checkpoint-384 control compares vLLM scores saved in two separate fresh "
            "restores, both passing weight verification. The first attempt committed its vLLM scores "
            "before failing to archive the learner scores. On identical tokens, the two vLLM runs have "
            f"mean |δ| {metric['mean_abs_log_ratio']:.5g}, k3 {metric['k3']:.5g}, "
            f"{100 * metric['share_beyond_2x']:.3f}% of ratios beyond 2×, and "
            f"{100 * metric['same_expert_set_fraction']:.3f}% expert-set agreement."
        )
    story.append(PageBreak())
    paragraph("Methods and definitions", "TitleD")
    paragraph(
        "The fixed panel contains 28 randomly selected validation questions, two in each of 14 available "
        "families (seed 20261017). Each question contributes its archive-384 and archive-512 response: "
        "56 sequences and 84,274 scored response tokens. Original prompt token IDs are retained. Responses "
        "are limited to their first 2,048 tokens, or their full length when shorter. Archive 384 contributes "
        "52,985 tokens and archive 512 contributes 31,289. This is a family-balanced diagnostic panel, "
        "not an accuracy evaluation or a representative sample of training tokens. The source archives "
        "report model versions 383 and 511, respectively, because evaluation precedes the next update. "
        "The checkpoints used to rescore both sources are completed updates 384 and 512. Historical "
        "archive probabilities are not used."
    )
    paragraph(
        "Each checkpoint is restored into fresh MarinSkyRL workers using 48 CoreWeave GB200 GPUs: "
        "32 learner GPUs with tensor parallelism 1, pipeline parallelism 2 and expert parallelism 16; "
        "two rollout replicas with data/expert parallelism 8 each. The original training used H100 GPUs. "
        "The learner and rollout engines retain their native implementations. vLLM uses FlashAttention 4 "
        "and the Triton MoE backend in both jobs. Model weights are BF16; "
        "router state is checked separately in FP32. Native broadcast installs serving weights. An "
        "independent expert-block scratch replay then verifies the installed bytes without reinstalling "
        "them, followed by full-FP32 readback of all router weights and biases."
    )
    paragraph(
        "vLLM scores the fixed text using prompt log probabilities, plus a final-token candidate score. "
        "Temperature is 1, top-p is 1, top-k is disabled, and prefix caching is disabled. Its expert choices "
        "are captured on that same forward. Megatron then scores the exact tokens in three modes: native "
        "(microbatch 1), reordered/rebatched (microbatch 2), and forced response-router replay. Replay "
        "sets the response input-token expert choices to vLLM's choices; prompt routes and expert combine "
        "weights remain native. The final response token has no captured input route because it was not "
        "fed back to the model. vLLM scoring is repeated after the learner forwards. No optimizer update "
        "occurs; the training-update method is guarded to raise if called."
    )
    paragraph(
        "An auxiliary decode control generates eight responses of at most 256 tokens to each of four "
        "questions (G01, G03, P03, S09). It compares probabilities recorded during autoregressive decoding "
        "against vLLM prefill rescoring and the three learner modes. These 32 short responses differ "
        "from the common fixed-token panel. A shared request seed is used, so repeated prompts are not "
        "independent samples. The completed auxiliary control's coverage is listed separately from the "
        "primary comparison."
    )
    paragraph(
        "Definitions: δ = log p_learner(token | prefix) − log p_vLLM(token | prefix), using natural logs. "
        "Mean |δ| measures the typical absolute log-probability difference. k3 = exp(δ) − 1 − δ, averaged "
        "over scored tokens; on this fixed panel it is a descriptive mismatch statistic, not an unbiased "
        "current-policy KL estimate. Beyond 2× means |δ| > log 2, so the probability ratio is below 1/2 "
        "or above 2. ESS/N = (sum r)² / (N sum r²), where r = exp(δ); a few large ratios lower it. Expert-set "
        "disagreement counts input positions/layers where the unordered top-four expert IDs differ."
    )
    paragraph("Frozen-panel SHA-256: " + data["panel_sha256"], "SmallD")
    paragraph(
        "MarinSkyRL commit: 5663f6129f53bb6aaa09549fa6cb423571503e94. "
        "vLLM build: marin.39e62869693c.cu132; PyTorch 2.13.0+cu132. "
        "Primary scoring definitions are identical between the two runs; only the auxiliary request "
        "changed for API compatibility. The accompanying evidence-manifest.json records archive "
        "locations, checkpoint identities, source hashes and verification artifacts.",
        "SmallD",
    )
    paragraph(
        "The following two pages contain the fixed-token distributions and checkpoint-by-text comparison.", "SmallD"
    )
    text_path = args.results / "diagnostic-text.pdf"
    SimpleDocTemplate(str(text_path), rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36).build(story)
    output = args.results / "probability-diagnostic-report.pdf"
    with pymupdf.open() as document:
        for path in (
            text_path,
            args.results / "fixed-token-parity.pdf",
            args.results / "checkpoint-by-response-source.pdf",
        ):
            with pymupdf.open(path) as section:
                document.insert_pdf(section)
        document.save(output)
    text_path.unlink()
    if args.base_report:
        combined = args.results.parent / "snowball-rl-report-with-probability-diagnostic.pdf"
        with pymupdf.open(args.base_report) as document, pymupdf.open(output) as appendix:
            first_page = len(document) + 1
            outline = document.get_toc()
            document.insert_pdf(appendix)
            document.set_toc(outline + [[1, "Controlled probability diagnostic", first_page]])
            document.save(combined)
        print(combined)
    print(output)


if __name__ == "__main__":
    main()
