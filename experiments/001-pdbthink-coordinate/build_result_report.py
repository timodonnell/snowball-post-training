"""Build the self-contained Snowball RL result report from completed evidence.

Requires reportlab, Pillow and PyMuPDF (available in the local PDBThink environment).
Never substitutes planned benchmark results for completed measurements.
"""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BLUE = colors.HexColor("#185a78")
DARK = colors.HexColor("#173042")
GRAY = colors.HexColor("#536775")
NAMES = {
    "G01": "Atom distance",
    "G02": "Nearest nonbonded atom",
    "G03": "Nearest residue",
    "G04": "Worst steric clash",
    "I01": "Interface contacts",
    "N01": "Shared contact residue",
    "P01": "Chain inventory",
    "P02": "Residue count",
    "P03": "Atom coordinates",
    "S01": "Salt bridge",
    "S02": "Phosphorylated residue",
    "S03": "Solvent exposure",
    "S04": "Local secondary structure",
    "S05": "Fold class",
    "S06": "Ligand contacts",
    "S07": "Metal coordination",
    "S08": "Disulfide partner",
    "S09": "Side-chain rotamer",
    "T01": "Two-state comparison",
}


def read(name):
    return json.loads((RESULTS / name).read_text())


def pct(x):
    return f"{100 * x:.2f}%"


class ReportDocument(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "HeadR":
            key = f"section-page-{self.page}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0)


def build(args):
    benchmark = json.loads((args.benchmark / "comparison.json").read_text())
    plan = json.loads((args.benchmark / "plan.json").read_text())
    export = json.loads((args.benchmark / "export-verification.json").read_text())
    for model in ("baseline", "trained"):
        assert json.loads((args.benchmark / model / "complete.json").read_text())["completed"] == 40
    assert benchmark["n"] == 40 and export["step"] == 384
    history = read("full-training-history.json")
    assert read("pilot-validation-comparison.json")["paired_tasks"] == 1003
    assert read("pilot-test-comparison.json")["paired_tasks"] == 1370
    cohort = json.loads((ROOT / "cohort_manifest.json").read_text())
    runtime = json.loads((args.benchmark / "runtime-provenance.json").read_text())
    selection = benchmark["selection"]
    strict, boxed = benchmark["strict"], benchmark["boxed"]
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (("D", "DejaVuSans.ttf"), ("DB", "DejaVuSans-Bold.ttf"), ("DI", "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("D", normal="D", bold="DB", italic="DI", boldItalic="DB")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyR", fontName="D", fontSize=9.3, leading=14.3, spaceAfter=9, textColor=DARK))
    styles.add(ParagraphStyle(name="SmallR", parent=styles["BodyR"], fontSize=7.5, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name="CellR", parent=styles["BodyR"], fontSize=7.3, leading=10, spaceAfter=0))
    styles.add(ParagraphStyle(name="HeadR", fontName="DB", fontSize=21, leading=26, textColor=BLUE, spaceAfter=16))
    styles.add(
        ParagraphStyle(name="SubR", fontName="DB", fontSize=12, leading=17, textColor=BLUE, spaceBefore=8, spaceAfter=8)
    )
    styles.add(ParagraphStyle(name="TitleR", fontName="DB", fontSize=31, leading=38, textColor=BLUE, spaceAfter=20))
    styles.add(ParagraphStyle(name="CaptionR", parent=styles["SmallR"], textColor=GRAY, spaceBefore=8))
    story = []
    vector_figures = []

    def p(text, style="BodyR"):
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        p(text, "HeadR")

    def sub(text):
        p(text, "SubR")

    def page(title):
        story.append(PageBreak())
        heading(title)

    def table(rows, widths=None, small=False):
        cells = [[Paragraph(escape(str(c)), styles["CellR"]) for c in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, BLUE),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#dbe3e8")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5 if small else 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if small else 7),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])

    def figure(name, title, caption, landscape=False):
        if landscape:
            story.extend([NextPageTemplate("landscape"), PageBreak()])
        else:
            story.append(PageBreak())
        heading(title)
        path = RESULTS / name
        vector_figures.append(path.with_suffix(".pdf"))
        width, height = ImageReader(str(path)).getSize()
        available_w, available_h = (744, 390) if landscape else (504, 565)
        scale = min(available_w / width, available_h / height)
        story.append(Image(str(path), width=width * scale, height=height * scale, hAlign="CENTER"))
        p(caption, "CaptionR")
        if landscape:
            story.append(NextPageTemplate("portrait"))

    def page_decoration(canvas, doc):
        w, _h = canvas._pagesize
        canvas.setStrokeColor(colors.HexColor("#dbe3e8"))
        canvas.line(40, 36, w - 40, 36)
        canvas.setFont("D", 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(40, 23, "Snowball × PDBThink | RL results and methods | 16 October 2026")
        canvas.drawRightString(w - 40, 23, str(doc.page))

    doc = ReportDocument(
        str(args.output),
        pagesize=(595.28, 841.89),
        title="Snowball RL on PDBThink: Results and Methods",
        author="Snowball post-training experiment",
        leftMargin=44,
        rightMargin=44,
        topMargin=42,
        bottomMargin=48,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="portrait",
                frames=Frame(44, 48, 507.28, 751.89, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                onPage=page_decoration,
                pagesize=(595.28, 841.89),
            ),
            PageTemplate(
                id="landscape",
                frames=Frame(44, 48, 753.89, 505.28, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                onPage=page_decoration,
                pagesize=(841.89, 595.28),
            ),
        ]
    )

    p("SNOWBALL POST-TRAINING / EXPERIMENT 001", "SmallR")
    story.append(Spacer(1, 20))
    p("Reinforcement learning<br/>for molecular coordinate reasoning", "TitleR")
    p("Results, methods, original-benchmark evaluation, and regression analysis", "SubR")
    p("16 October 2026 · CoreWeave H100 training / GB200 evaluation · Marin / MarinSkyRL / Iris", "SmallR")
    sub("Main findings")
    p(
        f"On the original PDBThink coordinate benchmark’s 40 context-fitting questions, the matched rerun changes strict exact accuracy from <b>{strict['baseline']['correct']}/40 ({pct(strict['baseline']['accuracy'])})</b> to <b>{strict['trained']['correct']}/40 ({pct(strict['trained']['accuracy'])})</b>: <b>{strict['difference_pp']:+.2f} percentage points</b>. Both models use the same frozen prompts, sampling settings, scoring code, and standalone serving runtime."
    )
    p(
        f"With the pre-existing, gold-independent terminal-box scoring correction applied to both models, accuracy changes from <b>{boxed['baseline']['correct']}/40</b> to <b>{boxed['trained']['correct']}/40</b> ({boxed['difference_pp']:+.2f} points). This sensitivity analysis helps separate gains in answer formatting from gains under a less format-sensitive score."
    )
    p(
        "<b>The transfer result is encouraging but imprecise.</b> Both paired source-group 95% intervals include zero. Box normalization reduces the net advantage from six questions to three; truncations rise from 11 to 18. This is evidence of a higher observed score on this panel, not a precise estimate of a broad reasoning improvement."
    )
    p(
        "The 64-update pilot also improved a separate 1,370-task held-out test from <b>291 correct (21.24%) to 433 (31.61%)</b>, a 10.36-point gain. On the 1,003-task development validation set, native evaluation peaked at <b>616 correct (61.42%) at step 384</b>, then regressed to 334 (33.30%) at step 512. These are distinct evaluation protocols, not points on one matched before/after curve."
    )
    p(
        "The regression is a real degradation in answer content as well as format. Replayed evidence is consistent with unstable RL optimization and growing disagreement between training and rollout probabilities. It does not identify a unique initiating bug. A fresh restore/parity experiment and controlled optimizer ablations remain necessary for a causal conclusion."
    )
    sub("Scope and cutoff")
    p(
        f"Checkpoint <b>384</b> was selected using validation alone at {escape(selection['selected_at'])}, before original-benchmark outcomes. It is the best checkpoint available at that time, not a claim about an already completed 876-update run. Training remains active; the full-history snapshot in this report reaches update <b>{history['last_optimizer_update']}</b>, with full validation through step 576. The later 398/1,003 score (39.68%) remains below the selected peak. No original-benchmark score is used to select another checkpoint."
    )

    page("1. Original PDBThink benchmark")
    p(
        "We evaluated the existing <b>clean-coordinate-v4.0</b> coordinate benchmark: 88 frozen questions spanning 19 task families. This is the original benchmark’s clean, explicitly tool-free revision, not a newly sampled subset of the RL dataset. The version replaced two earlier singleton-chain inventory questions with multichain questions; the other 86 prompts were retained. Its frozen fingerprint is recorded in the reproducibility appendix."
    )
    p(
        "Exactly <b>40/88 questions (45.45%)</b>, covering 18 families and 28 benchmark protein-group IDs, fit Snowball’s native context. The other 48 prompts already exhaust 32,768 tokens and are excluded for both models. T01 has no eligible question. The 40-question membership and every request body were checked against the benchmark’s existing Snowball plan; only model identity changes. No prompt was cropped, shortened, rewritten, or supplemented with labels."
    )
    p(
        "An older Snowball benchmark run selected 25 questions by reserving 8,192 output tokens. The clean benchmark instead uses the full remaining context and admits 40. This report reproduces that later clean protocol; the 25-to-40 change is benchmark coverage, not a change in the 1,003-task RL validation set."
    )
    table(
        [
            ["Score on the same 40 questions", "Original", "RL step 384", "Change"],
            [
                "Strict exact accuracy",
                f"{strict['baseline']['correct']}/40 ({pct(strict['baseline']['accuracy'])})",
                f"{strict['trained']['correct']}/40 ({pct(strict['trained']['accuracy'])})",
                f"{strict['difference_pp']:+.2f} pp",
            ],
            [
                "Strict family-macro exact accuracy",
                pct(strict["baseline"]["family_macro"]),
                pct(strict["trained"]["family_macro"]),
                f"{strict['family_macro_difference_pp']:+.2f} pp",
            ],
            [
                "Terminal-box-corrected exact accuracy",
                f"{boxed['baseline']['correct']}/40 ({pct(boxed['baseline']['accuracy'])})",
                f"{boxed['trained']['correct']}/40 ({pct(boxed['trained']['accuracy'])})",
                f"{boxed['difference_pp']:+.2f} pp",
            ],
            [
                "Box-corrected family macro",
                pct(boxed["baseline"]["family_macro"]),
                pct(boxed["trained"]["family_macro"]),
                f"{boxed['family_macro_difference_pp']:+.2f} pp",
            ],
            [
                "Strict format errors",
                strict["baseline"]["format_error"],
                strict["trained"]["format_error"],
                "Flags can overlap truncation",
            ],
            [
                "Native-context truncations",
                strict["baseline"]["truncated"],
                strict["trained"]["truncated"],
                "Zero reward",
            ],
        ],
        [222, 94, 94, 97],
    )
    p(
        f"The strict comparison has <b>{strict['gains']} newly correct</b> answers and <b>{strict['losses']} newly wrong</b> answers. The approximate paired protein-group bootstrap 95% interval for the task-weighted difference is <b>{strict['group_bootstrap_95_delta_pp'][0]:+.2f} to {strict['group_bootstrap_95_delta_pp'][1]:+.2f} points</b>. Under box correction it is {boxed['group_bootstrap_95_delta_pp'][0]:+.2f} to {boxed['group_bootstrap_95_delta_pp'][1]:+.2f} points, with {boxed['gains']} gains and {boxed['losses']} losses."
    )
    p(
        "These intervals resample the 28 benchmark protein groups with replacement 50,000 times (NumPy seed 17), retaining each model’s paired responses and all questions in each group. They describe sensitivity to source composition. One sampled answer per model/question cannot quantify decoding variance; one training run cannot quantify training-seed variance. This small panel provides a direct transfer test, with limited precision for individual families."
    )
    p(
        "The headline follows the clean benchmark’s exact-correct criterion; partial set-overlap credit is not counted as success. The box correction is the already defined <b>snowball-boxed-v1</b> policy: unwrap only a balanced terminal \\boxed{…}, normalize escaped spaces, and add FINAL: if absent. It does not repair numbers, choose among guesses using the gold answer, or relax truncation, refusal, or numerical-tolerance rules. The strict and corrected scores are always shown for both models."
    )
    lengths = benchmark["response_statistics"]
    p(
        f"Mean generated length is {lengths['baseline']['mean_output_tokens']:,.0f} tokens for the original model and {lengths['trained']['mean_output_tokens']:,.0f} for step 384. Maximum budgets are matched; actual generated-token counts need not be equal. These counts include thinking and answer text and exclude prompt tokens."
    )

    page("1a. What changed on the original benchmark?")
    p(
        "Under strict scoring, seven questions become correct and one becomes wrong. Five of the seven gains were previously format errors, one was flagged as a refusal, and one was a well-formed wrong answer. The existing box correction recovers three of those five baseline format errors as correct without RL. Consequently, the net improvement falls from six questions under strict scoring to three after applying the same normalization to both models."
    )
    p(
        "After box correction, four questions become correct: three previously well-formed wrong answers and one previously refused answer. The one lost answer is a solvent-exposure question whose trained response reaches the context limit. Among the 16 questions with valid responses under both models after normalization, correct answers rise from 7 to 10. This subset is selected using the observed outputs and is descriptive, not a controlled format-only intervention."
    )
    table(
        [
            ["Well-formed wrong → correct after box normalization", "Original answer", "RL answer", "Gold"],
            ["G01 / fb_7t9r: atom distance", "8.22 Å", "8.81 Å", "8.812 Å; tolerance 0.02"],
            ["G03 / fb_7t9r: nearest residue", "A:C29", "A:K38", "A:K38"],
            ["S09 / fb_8aga: side-chain rotamer", "g+", "t", "t"],
        ],
        [245, 78, 78, 106],
    )
    p(
        "The G01 and G03 gains concern the same protein group, fb_7t9r. They are correlated evidence, not two independent protein-level replications. The benchmark’s small, uneven family denominators also make its family-macro gain sensitive to single questions: G01 and G03 each contain only one eligible question."
    )
    p(
        "Correct final answers do not certify correct reasoning or clean surrounding output. The three responses above include invented CID/RID metadata and malformed surrounding JSON. The G03 explanation describes a Cα-based comparison even though the question asks for minimum heavy-atom distance. The unchanged original parser can extract the correct FINAL value, so these count as benchmark successes. They are not evidence that every stated intermediate computation or invented metadata field is valid. Here, “strict” means the original scoring policy without the additional terminal-box correction; it does not mean a new, stricter whole-response validator."
    )
    p(
        "Trained exact accuracy is still zero in 11 of the 18 eligible families. Overall strict format-error flags increase from 18 to 20; refusal flags fall from 8 to 0; truncations increase from 11 to 18. The flags can overlap. Mean generated length rises from 3,196 to 5,369 tokens, about 68% more output tokens under matched maximum budgets. The useful answer improvements therefore coexist with substantial failure rates and higher realized generation cost."
    )

    figure(
        "original-benchmark.png",
        "2. Transfer to the original benchmark",
        "Matched inference on 40 frozen questions; both models use temperature 0.6 and the full remaining native context. Strict scoring is primary; terminal-box normalization is a fixed sensitivity analysis. Family denominators are small and unequal. The footer intervals resample benchmark protein groups and exclude training/decoding uncertainty.",
        landscape=True,
    )

    page("3. Exact benchmark inference method")
    table(
        [
            ["Setting", "Value / rationale"],
            ["Baseline", "Original Snowball Step38, revision cfc1d845dae89b067cdc7250d0164abefa5a69cf"],
            ["RL model", "One validation-selected native checkpoint: global_step_384; native Megatron-to-HF export"],
            [
                "Serving",
                "Same Marin standalone vLLM configuration for both; 4 CoreWeave GB200 per server, BF16, TP1 / DP4 with expert parallelism",
            ],
            [
                "Sampling",
                "One response per question; temperature 0.6, top-p 0.95, top-k 20; seed 20260929; thinking enabled",
            ],
            ["Stop tokens", "128001 and 128009, as in the frozen original Snowball benchmark requests"],
            [
                "Context / output budget",
                "32,768 native tokens; max_tokens = 32768 − exact chat-template prompt tokens; remaining budgets range from 1,191 to 20,136",
            ],
            [
                "Tool policy",
                "tools=null and tool_choice=none on every request; raw responses audited for tool events; no tool execution",
            ],
            [
                "Concurrency",
                "4 requests in both new evaluations; historical deployment used 2. Historical scores are not used as the paired baseline.",
            ],
            [
                "Verifier",
                "Existing benchmark scorer and question parameters; source file hashes frozen before inference; strict and pre-existing boxed policy both retained",
            ],
            [
                "Failures / resumption",
                "Only transient transport/service errors retried, at most three attempts with identical request; model errors never retried. Each completed raw response saved atomically.",
            ],
        ],
        [112, 395],
    )
    p(
        "The runner checks served prompt-token counts against locally computed counts and rejects an unexpected output-budget clamp. The checkpoint export is checked against its native publication manifest, including every file’s size and metadata hashes; tokenizer and chat template must match the frozen baseline. Native publisher weight hashes are preserved, but the large weight shards are not downloaded a second time solely for checksum replay."
    )
    p(
        "As a scorer cross-check, all 40 historical original-benchmark responses were replayed under both scoring policies: 80 decisions, zero disagreements with the published clean report. That historical deployment scored 5/40 strictly and 10/40 after terminal-box normalization. These historical values are context only; the RL improvement is computed against the fresh baseline on the same GB200 serving stack as the trained model."
    )
    p(
        "The original benchmark uses sampled decoding (temperature 0.6), while RL development validation uses greedy decoding (temperature 0). Its full-context membership also differs from the RL cohort’s 8,192-token reserve rule. Therefore the original benchmark should be interpreted as its own matched endpoint comparison. Absolute scores need not match either native RL validation or the pilot’s standalone held-out evaluation."
    )

    page("4. Model, data, and split construction")
    p(
        "The starting model is <b>open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38</b>, a mixture-of-experts model named for approximately 67 billion total and 2 billion active parameters. “Step38” identifies the upstream checkpoint; step 0 in this report is the start of our additional RL, not an untrained language model. We use the pinned native tokenizer and thinking chat template throughout."
    )
    p(
        "Training uses <b>open-athena/pdbthink-coordinate-tasks v1.3.0</b>, a 100,000-task release generated from 2,671 source entries and 1,867 source groups. Published splits are retained. The release audit reports zero overlap with configured benchmark sources by entry ID, exact protein sequence, and RCSB 30% sequence-cluster membership, and zero duplicate prompts or semantic tasks. These safeguards concern this training dataset; they do not prove absence from the foundation model’s pretraining."
    )
    p(
        "For this experiment, every selected prompt is retokenized with Snowball’s exact native template, including thinking-mode setup. Eligibility requires at most <b>24,576 input tokens</b>, leaving at least 8,192 of the 32,768-token context for an answer. This produces 28,045 training tasks, 1,003 validation tasks, and 1,370 test tasks. The filter changes family and class frequencies substantially; these are context-eligible subsets, not representative samples of every released task."
    )
    table(
        [
            ["Cohort", "Tasks", "Groups", "Use"],
            ["Training", "28,045", "438", "Sampled prompts used for policy updates"],
            ["Full validation", "1,003", "20", "Development and checkpoint selection"],
            ["Pilot monitor", "107", "20", "Fixed subset of validation, at most 8 per family"],
            ["Held-out test", "1,370", "29", "Paired pilot test; final selected-model test is a separate follow-up"],
            [
                "Original benchmark",
                "40 / 88",
                "28 eligible benchmark IDs",
                "External frozen transfer panel; separate context rule",
            ],
        ],
        [110, 65, 100, 232],
    )
    p(
        "Release source groups connect entries sharing exact protein sequences, RCSB sequence clusters, or two-state partners; entire connected groups remain in one split. Thus many task rows are correlated observations from the same structure or related source. The benchmark’s protein_group_id units are its own grouping identifiers, not the 20 validation-source groups."
    )
    p(
        "Prompts contain sanitized, rigidly rotated and translated coordinates, with source-identifying annotations removed. Molecular questions are answered from the displayed coordinates under deterministic geometric definitions. Gold answers are verifier metadata and are never included in model-visible messages. <b>No GLM responses, other teacher traces, supervised trace fitting, or distillation were used.</b>"
    )

    page("5. Cohort composition by task family")
    rows = [["Family / task", "Train", "Validation", "Test", "Original eligible"]]
    for f, name in NAMES.items():
        rows.append(
            [
                f + "  " + name,
                cohort["family_counts"]["train"].get(f, 0),
                cohort["family_counts"]["validation"].get(f, 0),
                cohort["family_counts"]["test"].get(f, 0),
                plan["selected_by_family"].get(f, 0),
            ]
        )
    table(rows, [235, 62, 72, 62, 76], small=True)
    p(
        "Validation covers 14 families. I01, S06, S07, S08, and T01 are absent, so the validation peak says nothing directly about those tasks. The original benchmark adds eligible I01/S06/S07/S08 questions, but still no T01. Several families have only one to four original-benchmark questions; a single success can shift their family accuracy dramatically."
    )
    p(
        "The 107-question pilot monitor uses exactly eight tasks per validation family except G04, which has only three. It is a fixed subset, not a new random sample every evaluation. The full validation set contains the monitor tasks and never changes membership during the recorded long run."
    )

    page("6. How the RL updates were computed")
    p(
        "We used <b>group-relative policy optimization (GRPO)</b> through Marin’s pinned MarinSkyRL launcher, with Megatron training and vLLM response generation. Each update samples 32 training prompts and generates four independent answers per prompt, for 128 responses. Rollouts use temperature 1.0, top-p 1.0, top-k −1 (disabled), min-p 0, repetition penalty 1, thinking mode, and an 8,192-token answer cap. There is one optimizer pass over each batch."
    )
    p(
        "The frozen native PDBThink verifier reads the model’s original response text and the task’s hidden gold metadata. Reward R is 1 only for a correctly formatted, exactly correct answer under the task’s numerical or set-matching rules; it is otherwise 0. For example, numeric correctness uses the task tolerance rather than literal string identity. Truncation, refusal, tool calls, and oversized outputs receive zero reward. Partial-credit diagnostics do not enter the reward."
    )
    sub("Relative advantages")
    p(
        "Within each prompt’s group of four responses, MarinSkyRL computes the mean reward and sample standard deviation. Each response receives advantage Aᵢ = (Rᵢ − mean(R)) / (std(R) + 10⁻⁶). That scalar is broadcast over its generated-token mask. Groups whose four rewards are identical have zero centered advantage and provide no outcome-reward gradient. No additional whole-batch advantage normalization is enabled, and no learned critic is used."
    )
    sub("Policy objective and optimizer")
    p(
        "The configured loss is regular PPO with token-mean reduction: minimize the negative mean of min(rₜAₜ, clip(rₜ, 0.8, 1.2)Aₜ) over response tokens, where rₜ is the exponential difference between the current action log probability and the stored old action log probability. Prompt tokens are masked from the loss. Thinking tokens have weight 1.0; longer responses contribute more token positions under token-mean reduction."
    )
    table(
        [
            ["Optimization setting", "Resolved value"],
            ["AdamW learning rate", "1 × 10⁻⁶; constant schedule; 0 warmup steps"],
            ["Adam coefficients / decay", "β₁=0.9, β₂=0.999; weight decay 0.01"],
            ["Gradient norm limit", "1.0; diagnostic raw norm is measured before clipping"],
            ["Batch / microbatch", "32 prompt groups × 4 answers; per-GPU microbatch 1; one optimizer pass"],
            ["PPO ratio clip interval", "[0.8, 1.2]; see the effective-clipping caveat below"],
            [
                "Reference KL / entropy / off-policy correction",
                "All disabled; unused positive default coefficients do not make these losses active",
            ],
            ["Random seed", "17 for the training recipe"],
            ["Training extent", "Pilot 64 updates; continuation targets 876 cumulative updates"],
        ],
        [209, 298],
    )
    p(
        "In this pinned implementation, old action probabilities are recomputed by the learner before the update. With one pass and no intervening parameter update, the recorded ratio is exactly one and PPO clipping is inactive. This is expected behavior of this recipe, not evidence that the clip function is broken. It also means that this PPO term does not bound the actual Adam parameter step or the mismatch to the rollout policy. Gradient clipping remains a separate active mechanism."
    )

    page("7. Infrastructure and checkpoint lineage")
    p(
        "All training used CoreWeave H100 GPUs managed by Marin’s Iris infrastructure. The 64-update pilot and long continuation each allocate <b>48 GPUs</b>: four 8-GPU policy workers and two 8-GPU rollout workers. Policy geometry is tensor parallelism 1, pipeline parallelism 2, and expert parallelism 16 (TP1/PP2/EP16). Each rollout worker uses data parallelism 8 and expert parallelism 8. The earlier two-update smoke used only one rollout worker (40 GPUs total)."
    )
    p(
        "Policy workers use BF16, gradient checkpointing, per-GPU microbatch size one, 256-token chunks for log-probability calculation, and optimizer-state offload during rollout. Gradient reduction is not forced to FP32. MoE router replay, train/eval parity checks, and data-parallel weight-consistency checks are disabled in the resolved recipe. Each training node requests 64 CPU cores, 1,800 GB host RAM, and 1,000 GB disk."
    )
    p(
        "LoRA adapters are disabled (rank 0); this is policy-weight training, with the Grug query-bias update mode frozen as specified in the resolved configuration. Checkpoint conversion and both original-benchmark inference jobs use CoreWeave GB200 capacity because the H100 pool was saturated. The first H100 baseline attempt was preempted before it returned answers. This hardware change affects neither the training lineage nor which checkpoint was selected."
    )
    table(
        [
            ["Stage", "Updates / outcome", "Meaning for lineage"],
            ["Pilot, 7–8 October", "1–64 completed", "2,048 prompts at most; exported step 64"],
            [
                "First long attempt, 8–10 October",
                "65–375; failed on required rollout archive upload",
                "Last complete checkpoint 320; updates 321–375 discarded",
            ],
            [
                "Resumed run, 15 October onward",
                "Restored policy, optimizer, scheduler, and dataloader at 320",
                "Repeats updates 321 onward; target 876",
            ],
            [
                "Recovery change",
                "Archive-upload timeout 120 → 600 seconds",
                "No change to model, data, optimization, or evaluation recipe",
            ],
            [
                "Selected checkpoint",
                "384; 616/1,003 native validation",
                "Selection frozen before original-benchmark inference",
            ],
            [
                "Current report snapshot",
                f"Through update {history['last_optimizer_update']}",
                "Training still active; not a completed 876-update report",
            ],
        ],
        [116, 214, 177],
    )
    p(
        "The first attempt stopped on an infrastructure publication timeout, not on a detected model-quality threshold. Resumption from 320 drops the unsaved 55-update tail. The full-history plots show that discarded tail separately and do not treat it as part of the selected model’s lineage. W&B did not retain the pilot’s final optimizer-diagnostic row at 64; training completion is independently verified, but no missing metric is fabricated or interpolated."
    )
    p(
        "Every 64-step full-validation checkpoint in the long run is banked by a pending native HF export request, which protects it from ordinary rolling checkpoint retention. Selection maximizes full-validation task-weighted accuracy; ties use family-macro accuracy, then the earlier step. Export restores the native 32-rank policy geometry on 32 CoreWeave GB200 GPUs (8 nodes × 4), with the same TP1/PP2/EP16 layout and converts it to a complete Hugging Face model. The original-benchmark servers use 4 GB200s each and release their owned resources after evaluation."
    )
    p(
        "At the 876-update target, 876 × 32 = 28,032 prompt slots are processed in the retained one-pass schedule; the final incomplete batch of 13 training tasks is omitted. The pilot is the initial segment of that lineage, not an extra full epoch. Repeated updates after the failed attempt consume additional compute but are not extra retained updates."
    )

    page("8. Validation sets and protocol changes")
    table(
        [
            ["Measurement / phase", "Tasks", "When", "Sampling / budget"],
            [
                "Pilot native monitor",
                "107 fixed validation tasks",
                "Before training; steps 8,16,…,64",
                "Temperature 0; one answer; full remaining native context",
            ],
            [
                "Standalone full validation",
                "Same 1,003 frozen tasks",
                "Original model (0) and pilot checkpoint (64)",
                "Temperature 0; one answer; full remaining context",
            ],
            [
                "First long-run native full validation",
                "Same 1,003 frozen tasks",
                "64 before continuation; 128,192,256,320",
                "Temperature 0; one answer; native training engine",
            ],
            [
                "Resumed native full validation",
                "Same 1,003 frozen tasks",
                "320 before updates; 384,448,512,576 in this report",
                "Unchanged task membership and native evaluation recipe",
            ],
            [
                "Pilot held-out test",
                "1,370 tasks from 29 groups",
                "Paired original and preselected pilot step 64",
                "Standalone; temperature 0; full remaining context",
            ],
            [
                "Original benchmark",
                "40 eligible of 88",
                "Original and validation-selected step 384",
                "Original protocol: temperature 0.6; one answer; full remaining context",
            ],
        ],
        [139, 95, 135, 138],
    )
    p(
        "The principal cohort change is at the pilot-to-long-run boundary: native monitoring expands from 107 to all 1,003 validation tasks. The standalone 1,003-task baseline/pilot comparison is a separate measurement. From the first long-run evaluation at step 64 onward, the native validation membership does not change. The step-320 restart repeats the same set; 576/1,003 before interruption versus 580/1,003 immediately after restore differs by four answers before any new RL update. Those four answers are not evidence of learning."
    )
    p(
        "Standalone serving and the native training evaluator use different execution paths and vLLM fork revisions. At step 64, standalone full validation is 378/1,003 (37.69%) while native full validation is 401/1,003 (39.98%). The mean output length also differs materially. Consequently, subtracting the standalone step-0 score of 22.93% from the native step-384 score of 61.42% is not a controlled estimate of RL improvement."
    )
    p(
        "All evaluation budgets use 32,768 minus exact native prompt length. The 8,192-token reserve determines RL cohort membership, and the 8,192-token training cap bounds rollout generation; neither is a fixed validation output cap. The original benchmark uses a separate full-context eligibility rule, so it includes prompts with less than 8,192 tokens remaining."
    )
    p(
        "The training monitor, full validation, and test each have a role: monitor and full validation are repeatedly observed development evidence; the final test is not used for checkpoint selection. The original benchmark is a separate transfer evaluation requested after observing the learning curve. Results have been inspected for the pilot test already, so it should not be described as forever untouched or as an independent model-selection set."
    )

    figure(
        "pilot-evidence.png",
        "9. What the pilot established",
        "Pilot evidence snapshot: full standalone validation compares the identical 1,003 prompts at step 0 and step 64. Group-bootstrap uncertainty pertains to source composition. The completed 1,370-task paired test is reported in the following section.",
        landscape=True,
    )

    page("10. Pilot results and their interpretation")
    table(
        [
            ["Paired measurement", "Original", "Pilot step 64", "Difference"],
            ["Full validation (1,003)", "230 / 1,003; 22.93%", "378 / 1,003; 37.69%", "+14.76 pp"],
            ["Validation family macro", "23.44%", "35.25%", "+11.81 pp"],
            ["Held-out test (1,370)", "291 / 1,370; 21.24%", "433 / 1,370; 31.61%", "+10.36 pp"],
            ["Native monitor (107)", "21 / 107; 19.63%", "32 / 107; 29.91%", "+10.28 pp"],
        ],
        [169, 118, 122, 98],
    )
    p(
        "The validation comparison has 186 newly correct answers and 38 newly wrong answers; the test has 192 gains and 50 losses. Nineteen of the 20 validation source groups improve. The paired source-group bootstrap interval for validation improvement is +10.86 to +17.94 percentage points (50,000 resamples, seed 17). All 2,006 validation responses replay exactly through the native verifier, and 336 G01/G03/P03 coordinate-derived gold answers were independently checked without disagreement."
    )
    p(
        "However, <b>133 of the 186 validation gains (71.5%)</b> were previously invalid outputs: 109 format errors and 24 other invalid responses. Only 53 gains came from previously well-formed wrong answers. Among 412 tasks valid under both models, correct answers rise from 220 to 245. That output-selected subset is descriptive, not a causal estimate of what would happen if formatting alone were repaired."
    )
    table(
        [
            ["Categorical family", "Fixed training-majority answer", "Control", "Original", "Pilot"],
            ["S03 exposure", "solvent-exposed", "98/119", "70/119", "90/119"],
            ["S04 secondary structure", "coil", "68/111", "55/111", "69/111"],
            ["S05 fold class", "predominantly alpha helical", "8/12", "5/12", "4/12"],
            ["S09 rotamer", "g−", "46/116", "11/116", "24/116"],
        ],
        [133, 173, 67, 67, 67],
    )
    p(
        "The constant labels are chosen from training data, not validation labels. The pilot predicts coil on 110/111 S04 tasks; it exceeds that trivial control by only one correct answer. Three other categorical families remain below their constant controls. G04, N01, S01 and S02 remain at zero, and G02 is only 1/120. These results support useful benchmark adaptation, with narrower evidence of improved geometric computation or general structural understanding."
    )
    p(
        "Pilot validation format errors decline from 419 to 256, while truncations rise from 138 to 189 and mean generated length rises from about 3,898 to 4,580 tokens. The same maximum token allowance does not imply equal actual inference compute. The improvement is real under the chosen evaluation protocol, but it is not a claim of equal-compute gains or broad mastery of every family."
    )

    figure(
        "full-training-evidence.png",
        "11. Full recorded learning curve from step 0",
        "Step 0 means the starting Snowball model. Gray: fixed 107-task native pilot monitor. Purple/dashed: paired 1,003-task standalone validation. Blue/solid: 1,003-task native training validation, with a separate segment at restart. Heatmap labels S=standalone, N=native, R=restart. Every plotted time axis begins at zero; missing optimizer observations are not invented.",
        landscape=True,
    )
    figure(
        "full-training-diagnostics.png",
        "12. Full recorded optimizer diagnostics",
        "Faint traces are raw per-update values; the emphasized curve is a trailing 16-update mean. The discarded first-attempt tail at steps 321–375 is shown separately in gray. Vertical dotted lines mark the pilot/continuation boundary at 64 and restore at 320. Shading highlights 448–512, the interval bracketing the validation collapse; it is not an estimated exact change point. Definitions follow in the glossary.",
        landscape=True,
    )

    page("13. The regression is an answer-quality loss")
    table(
        [
            ["Native validation", "Correct / 1,003", "Format errors", "Truncated", "Mean output tokens"],
            ["320 after restore", "580 (57.83%)", 189, 120, "4,063"],
            ["384", "616 (61.42%)", 209, 127, "5,220"],
            ["448", "591 (58.92%)", 187, 82, "4,397"],
            ["512", "334 (33.30%)", 297, 3, "1,400"],
            ["576", "398 (39.68%)", 345, 18, "1,920"],
        ],
        [109, 112, 86, 75, 125],
    )
    p(
        "From step 384 to 512, accuracy falls <b>28.12 percentage points</b>. There are 312 losses and 30 gains, with 304 tasks correct under both checkpoints and 385 wrong under both. All 20 validation source groups decline. A paired source-group bootstrap interval for the change is −33.05 to −24.07 points. This is much larger than the observed four-answer restart variation at step 320, though no single interval captures all serving or training randomness."
    )
    p(
        "Of the 312 lost correct answers, <b>217 become well-formed wrong answers</b> and 95 become format errors. Thus formatting cannot explain the whole collapse. Truncations become rare, while answers become much shorter. This is not a failure driven by running out of answer budget."
    )
    table(
        [
            ["Family", "Step 384", "Step 512", "Observed failure"],
            ["G01 atom distance", "100/110", "16/110", "Large numerical reasoning loss; no format errors at 512"],
            ["G03 nearest residue", "78/108", "12/108", "Candidate selection degrades"],
            ["P03 coordinates", "114/118", "74/118", "Previously reliable extraction degrades"],
            ["S09 rotamer", "85/116", "17/116", "All 54 valid outputs choose t; 62 format-invalid"],
        ],
        [128, 75, 75, 229],
    )
    p(
        "A deterministic example, selected as the first lost G01 task by ID, illustrates the content change. Coordinates are (−0.792, −17.922, −10.053) and (2.386, 0.974, −7.489); gold distance is 19.332 Å. Step 384 computes the differences (3.178, 18.896, 2.564) and answers 19.33. Step 512 extracts the coordinates but mixes axes in a two-dimensional expression and answers 17.32. This example illustrates an audited transition; it is not an estimate of how frequently that particular error occurs."
    )

    figure(
        "epoch-regression-evidence.png",
        "14. Detailed regression evidence",
        "Audit window: native full validation at 320/384/448/512, with paired transitions and source-group uncertainty for 384→512. The original saved audit covers those checkpoints; the full-history figure separately includes the later 576 evaluation. Each comparison uses the same frozen 1,003 tasks.",
        landscape=True,
    )
    figure(
        "epoch-regression-diagnostics.png",
        "15. Diagnostics around the collapse",
        "The audit’s response replay and fixed-family-weight analysis cover resumed updates 321–565. Family standardization holds the reference family mixture fixed; it cannot remove changes in task difficulty within a family. All measured training and validation rewards were replayed from original responses. See the glossary for k3 mismatch, ESS, gradient norm, and entropy.",
        landscape=True,
    )

    page("16. What caused the regression?")
    p(
        "The strongest supported conclusion is <b>policy degradation during continued RL, associated with a loss of optimization stability</b>. It is supported by a fixed validation set, paired raw-answer losses, falling sampled training reward after family adjustment, increasing gradient norms, and growing training-versus-rollout probability mismatch. The evidence is observational: it identifies a credible mechanism and rules out several simpler explanations, but does not establish a unique trigger."
    )
    sub("Checks that passed")
    p(
        "All 4,012 native validation responses from checkpoints 320/384/448/512 reproduce their recorded rewards and diagnostics when replayed using original API response content. All 31,360 sampled training responses in 7,840 four-response prompts from updates 321–565 also replay exactly. Prompt hashes, native tokenization, family/source metadata, and gold metadata agree. The audit checks 7,844 native archive hashes and independently recomputes 336 coordinate-derived gold answers. It finds no reward mismatch, stale-policy record, token-alignment failure, skipped optimizer step, or upload timeout in that audited interval."
    )
    p(
        "Holding the family mixture fixed to updates 321–448 gives sampled success rates of 58.02% at 321–384, 60.81% at 385–448, 58.58% at 449–480, 44.99% at 481–512, and 37.24% at 513–565. Common families cover 99.85% of reference prompts; I01/S06 are excluded because they are absent from at least one window. This reduces the family-composition explanation but does not control within-family difficulty."
    )
    sub("A concrete stability weakness")
    p(
        "The learner recomputes old action log probabilities with its current weights, and there is one optimizer pass. Every one of the 245 audited updates has PPO clip fraction 0 and exact-unit ratio fraction 1. The configured PPO ratio term therefore does not constrain these updates against the actual rollout probabilities. There is also no reference-KL loss, no entropy bonus, and no off-policy correction. The active gradient norm limit is a separate safeguard, not a replacement for those constraints."
    )
    p(
        "Comparing updates 321–384 with 545–565, mean token entropy falls from 0.14138 to 0.05278 nats, raw gradient norm rises from 0.19749 to 1.12197, k3 training/rollout mismatch rises from 0.007837 to 0.106081, and ESS fraction falls from 0.96285 to 0.36206. The wider history shows entropy falling during successful learning too, so entropy decline alone is not specific evidence of collapse. The later mismatch and gradient-norm changes are more diagnostic."
    )
    sub("What remains unresolved")
    p(
        "Zero nominal policy-version staleness does not prove numerical parity between Megatron and vLLM. Possible contributors include MoE routing, precision/backend differences, or weight synchronization; the disabled parity/router-replay checks leave these unresolved. A fresh standalone evaluation of the selected checkpoint tests transfer after restore, but does not by itself isolate the training/rollout mismatch or explain checkpoint 512."
    )
    p(
        "The next causal experiment should restore 384 and 512 in fresh workers and check weight and log-probability parity, then compare matched continuations from 384 under the current recipe and behavior-referenced clipping, with learning-rate changes tested separately. These ablations have not been run. It would be premature to claim that a smaller learning rate, a specific clipping option, or a synchronization fix has already solved the regression."
    )

    page("17. Reading the plots: accuracy and cohorts")
    table(
        [
            ["Term", "Precise meaning"],
            [
                "Step / optimizer update",
                "One policy update after a batch of 32 prompts × 4 sampled responses. Step 0 is the supplied Snowball checkpoint before this experiment’s RL.",
            ],
            [
                "Success rate / accuracy / reward",
                "Fraction of responses with binary reward 1: valid and fully correct under the native verifier. A numeric answer may match within a specified tolerance. Partial credit is excluded.",
            ],
            [
                "Training reward",
                "Mean reward on that update’s sampled training responses at temperature 1 and an 8,192-token cap. Prompts change with batches; this is not held-out accuracy.",
            ],
            [
                "Pass@1 / pass@4",
                "Pass@1 is success for one response. Training pass@4 is the fraction of prompt groups with at least one success among four samples; it is not the mean success over all four.",
            ],
            [
                "Task-weighted / micro accuracy",
                "Total correct divided by total questions. Common families and larger source groups contribute more tasks.",
            ],
            [
                "Family-macro accuracy",
                "Compute accuracy within each included family, then average those family accuracies equally. Excluded/absent families are not silently assigned zero.",
            ],
            [
                "Percentage points (pp)",
                "Difference between percentages: 37.69% −22.93%=14.76 points; not a 14.76% relative increase.",
            ],
            [
                "Validation / monitor / test",
                "Validation guides development and checkpoint selection. Monitor is a fixed 107-task validation subset. The 1,370-task test is a separate grouped split. The original 88-question benchmark is another dataset.",
            ],
            [
                "Source group",
                "A unit of related structures/tasks kept together by sequence/cluster/two-state relationships in the RL release; benchmark protein-group IDs use the original benchmark grouping.",
            ],
            [
                "Bootstrap interval",
                "A percentile interval from resampling entire paired source groups, retaining their tasks. It quantifies source-composition sensitivity, not variation across training seeds or inference stacks.",
            ],
            [
                "Gains / losses",
                "Gains are wrong→correct paired transitions; losses are correct→wrong. Correct under both and wrong under both complete the transition table.",
            ],
            [
                "Fixed family mix",
                "Reweights per-family training success to a common reference mixture. It adjusts family proportions, not task difficulty within a family.",
            ],
            [
                "S / N / R on heatmap",
                "S=standalone serving; N=native training evaluator; R=the repeated native evaluation immediately after restart.",
            ],
            [
                "Dotted lines / shaded band",
                "Dotted lines mark 64 and 320 phase boundaries; 448–512 shading brackets the observed validation drop, not its exact onset.",
            ],
        ],
        [140, 367],
        small=True,
    )

    page("18. Reading the plots: optimizer and output")
    table(
        [
            ["Term", "Precise meaning and limitation"],
            [
                "Token / output length",
                "A tokenizer unit. Output-token counts include generated thinking and final-answer text, not input coordinates. The model’s context limit covers input plus output.",
            ],
            [
                "Format error",
                "The scorer cannot parse an answer under the requested schema. A well-formed wrong answer parses but fails correctness. A format error and truncation may overlap.",
            ],
            [
                "Truncation",
                "Generation reaches its allowed token budget. It receives zero reward even if a partial answer looks correct. A shorter average response is not automatically an improvement.",
            ],
            [
                "Policy entropy (nats)",
                "Average −Σ p log(p) of the learner’s next-token distribution over scored response positions. Natural logarithms give nats. Lower entropy means a more concentrated distribution, not greater correctness.",
            ],
            [
                "Raw gradient norm",
                "Euclidean norm of the policy gradient before clipping. The configured limit is 1.0. This is not the magnitude of the Adam parameter update, which also depends on optimizer state.",
            ],
            [
                "PPO clipping / clip fraction",
                "Clips the probability-ratio surrogate when it changes the objective in the relevant advantage direction. Different from gradient clipping. In this one-pass recipe the recorded ratios equal 1, so clip fraction is 0.",
            ],
            [
                "Training/rollout ratio r",
                "r=exp(log p_training −log p_rollout) for the sampled token. It compares learner and generator probabilities of the same observed token; nominal policy versions can match while numerical probabilities differ.",
            ],
            [
                "k3 mismatch",
                "Mean r−1−log(r), with the pinned implementation’s numerical safeguards, over valid sampled tokens. A nonnegative KL-style mismatch diagnostic. It is not a percentage, an accuracy, or KL distance from the original Snowball model.",
            ],
            [
                "ESS fraction",
                "(Σr)² / (NΣr²), the importance-weight effective-sample-size ratio. Near 1 means weights are nearly uniform, not necessarily exactly 1. It is not the number of independent biological tasks.",
            ],
            [
                "Token alignment / TIS diagnostics",
                "Alignment diagnostics confirm that learner and generator logs refer to matched tokens. Their presence under a generate/tis metric name does not mean importance-sampling correction is enabled.",
            ],
            [
                "Rolling mean",
                "A trailing 16-update average used only to make trends readable; faint raw traces retain per-update variation. It is not a confidence interval.",
            ],
            [
                "BF16 / TP / PP / EP / DP",
                "BF16 is the 16-bit floating-point format. Tensor, pipeline, expert, and data parallelism respectively partition tensor operations, layers, MoE experts, and data processing across devices.",
            ],
            [
                "GRPO / PPO / KL",
                "Group-relative policy optimization; proximal policy optimization; Kullback–Leibler divergence. The active recipe uses GRPO advantages and regular PPO loss, with reference-model KL disabled.",
            ],
        ],
        [140, 367],
        small=True,
    )

    page("19. Molecular task glossary")
    table(
        [
            ["Family", "What the question asks"],
            [
                "P01 / P02 / P03",
                "List polymer chain IDs; count residues in a named chain; extract a named atom’s displayed (x,y,z) coordinates.",
            ],
            ["G01", "Euclidean distance between two named atoms, in ångströms (Å; 10⁻¹⁰ m)."],
            [
                "G02",
                "Nearest eligible heavy atom, excluding covalent neighbors one or two bonds away. Heavy atoms exclude hydrogen.",
            ],
            [
                "G03",
                "Closest candidate residue using minimum heavy-atom distance under the task definition; not merely Cα-to-Cα distance.",
            ],
            [
                "G04",
                "Identify the residue pair containing the most severe atom-level steric clash. Overlap is r₁+r₂−0.40−distance, using specified Bondi radii and exclusions for bonded neighbors, metals, and SG–SG pairs.",
            ],
            [
                "N01",
                "Identify the unique residue contacting both named anchor residues. Contact means at least one heavy-atom pair lies within the specified cutoff (4 Å in the audited prompts).",
            ],
            ["I01", "Residue contacts at a specified chain interface, using the frozen distance and counting rules."],
            [
                "S01",
                "Salt-bridge partner: relevant Lys/Arg nitrogen to Asp/Glu oxygen distance at most 4 Å, according to the prompt’s named atoms.",
            ],
            ["S02", "Identify a phosphorylated serine, threonine, or tyrosine in the supplied structure."],
            [
                "S03",
                "Classify solvent exposure using relative solvent-accessible surface area: buried≤0.20; exposed≥0.40 in the audited tasks.",
            ],
            [
                "S04 / S05",
                "Local secondary structure (helix/strand/coil); global chain fold class (predominantly alpha helical, predominantly beta sheet, or mixed).",
            ],
            [
                "S06 / S07 / S08",
                "Ligand-contact residues; metal-coordinating residues; the disulfide-bonded partner of a named cysteine. Frozen task-specific geometric rules determine the gold answer.",
            ],
            [
                "S09",
                "Classify the χ1 side-chain dihedral (N–CA–CB–CG), normalized to [−180°,180°): g+ for 0≤χ1<120; t for χ1≥120 or χ1<−120; g− for −120≤χ1<0.",
            ],
            [
                "T01",
                "Compare two supplied structural states under a fixed geometric question. No T01 task fits the selected RL cohort or this model’s original-benchmark subset.",
            ],
        ],
        [92, 415],
        small=True,
    )
    p(
        "A residue is an amino-acid unit in a polymer chain; atom IDs refer to specific atoms within those residues. PDB is the coordinate-record format supplied to the model. Rotations and translations preserve distances and angles but change absolute coordinate values. The sequence-prediction tracks F01/F02 are separate benchmarks and are not mixed into any coordinate-reasoning headline here."
    )

    page("20. Original benchmark: all paired outcomes")
    p(
        "The following table enumerates every eligible question. B=original baseline; RL=step 384; ✓=exactly correct, ×=incorrect. “Box” applies the same pre-existing terminal-box normalization to both outputs. Full prompts, raw responses, verifier details, and hashes are retained in the evidence bundle. Rows are ordered by frozen render ID, not by outcome."
    )
    rows = [["Frozen render ID (representation / rotation suffix omitted)", "B strict", "RL strict", "B box", "RL box"]]
    for row in benchmark["results"]:
        rows.append(
            [
                row["render_id"].split("::")[0],
                *["✓" if row[m][p_]["correct"] else "×" for p_ in ("strict", "boxed") for m in ("baseline", "trained")],
            ]
        )
    table(rows, [291, 54, 54, 54, 54], small=True)

    page("21. Interpretation and limits")
    for label, metric in (("strict", strict), ("box-corrected", boxed)):
        low, high = metric["group_bootstrap_95_delta_pp"]
        if low <= 0 <= high:
            p(
                f"For the {label} comparison, the source-group interval includes zero. The observed point estimate should therefore be treated as imprecise evidence of transfer on this small panel, not as a definitive positive improvement."
            )
        elif low > 0:
            p(
                f"For the {label} comparison, the source-group interval is above zero. The improvement is robust to this specific source-composition resampling, while uncertainty across decoding and training runs remains unmeasured."
            )
        else:
            p(
                f"For the {label} comparison, the source-group interval is below zero. This comparison indicates negative transfer on the measured panel despite the model’s higher native development-validation score."
            )
    p(
        f"The original-benchmark rerun provides a direct measurement of transfer for the validation-selected checkpoint: strict task-weighted change <b>{strict['difference_pp']:+.2f} points</b>, and box-corrected change <b>{boxed['difference_pp']:+.2f} points</b>. The family-macro changes are {strict['family_macro_difference_pp']:+.2f} and {boxed['family_macro_difference_pp']:+.2f} points respectively. These matched comparisons are the relevant evidence for this benchmark; the 61.42% native validation peak is not substituted for a benchmark result."
    )
    p(
        "The strongest established result from the pilot is a reproducible improvement in measured task performance, including a separate grouped test. The interpretation is narrower than a claim that the model has learned general molecular reasoning. Answer-format adaptation and class-frequency learning explain a material part of the pilot gain, while some audited valid-answer transitions demonstrate content improvements. The original benchmark adds a useful transfer check, but its 40 eligible questions and tiny family denominators cannot establish broad mastery."
    )
    p(
        "Selection on repeatedly observed validation makes the peak development score optimistic as an estimate of future performance. The original benchmark was not used to select the checkpoint, but only one trained checkpoint, one training seed, and one sampled response per benchmark question were evaluated. The benchmark’s 28 protein-group bootstrap is not a substitute for repetitions across decoding seeds or independent training runs."
    )
    p(
        "Context filtering favors smaller structures and changes answer distributions. Forty of 88 benchmark questions fit; 48 remain unanswered under this model’s native limit. Excluded questions are not counted as incorrect in the conditional accuracy, and that accuracy must not be presented as full 88-question coverage. The learned model has not acquired a larger context window through this RL procedure."
    )
    p(
        "The regression audit strongly supports an actual loss of policy quality and exposes an inadequately constrained effective update recipe. It does not yet distinguish optimizer dynamics from numerical/backend/weight-sync contributors. A successful independent evaluation of 384 supports retaining that checkpoint; it does not justify continuing indefinitely under the same unstable recipe or claim that a proposed corrective ablation has already worked."
    )
    p(
        "Training remains in progress toward the previously authorized 876-update limit. A separate existing follow-up selects the final best validation checkpoint after completion and runs full standalone validation plus the 1,370-task paired test. Those future results are not included or anticipated here. This report’s benchmark selection remains the frozen step 384 comparison and should be versioned if later experiments are added."
    )

    page("22. Reproducibility and evidence index")
    preserved = read("original-benchmark-manifest.json")
    p(
        f"The completed original-benchmark evidence bundle is <b>{preserved['bytes']:,} bytes</b> and contains all 80 raw API responses, frozen requests, checkpoint selection, export verification, source snapshot, and per-file hashes. Durable URI: {escape(preserved['uri'])}. SHA256: {preserved['sha256']}.",
        "SmallR",
    )
    table(
        [
            ["Artifact", "Pinned identity / location"],
            ["Base model revision", "cfc1d845dae89b067cdc7250d0164abefa5a69cf"],
            ["Dataset release / revision", "v1.3.0 / 3734406cb97b1702844319f9a5d860cbbf8fe660"],
            ["Marin commit", "44c95a7914864fdab8b74d597052e30cbc92b8cc"],
            ["MarinSkyRL commit", "5663f6129f53bb6aaa09549fa6cb423571503e94"],
            ["Standalone benchmark vLLM commit", runtime["vllm_source_commit"]],
            ["Benchmark aarch64 vLLM wheel SHA256", runtime["vllm_wheel"]["sha256"]],
            ["Adapter SHA256", "6fa472524ddea6e042905f87615f83a07fd5fee6339bd0d9f48cae5956fbc969"],
            ["Original benchmark fingerprint", plan["dataset_fingerprint"]],
            ["Native chat-template SHA256", plan["chat_template_sha256"]],
            ["Best-checkpoint export", export["model_uri"]],
            ["Export manifest SHA256", export["manifest_sha256"]],
            [
                "Benchmark source snapshot",
                "File-by-file SHA256 values in original-benchmark plan; preserves local source state as well as git HEAD",
            ],
            ["Full-history cutoff", history["created_at"] + f"; last update {history['last_optimizer_update']}"],
        ],
        [150, 357],
        small=True,
    )
    p(
        "Repository: <link href='https://github.com/timodonnell/snowball-post-training'>github.com/timodonnell/snowball-post-training</link>. All experiment sources and compact evidence are under experiments/001-pdbthink-coordinate. The PDF is generated by build_result_report.py; original_benchmark.py freezes requests and scores completed responses; run_original_benchmark.py owns the inference servers; export_benchmark_checkpoint.py freezes validation selection and verifies native export.",
        "SmallR",
    )
    p(
        "Evidence files: pilot-validation-comparison.json; pilot-test-comparison.json; pilot-evidence.json; epoch-regression-audit.json; full-training-history.json; epoch-resume-resolved-launch.json; cohort_manifest.json; original-benchmark-comparison.json; original-benchmark-manifest.json. Figure sources: plot_pilot_evidence.py, plot_regression.py, plot_full_training.py, and plot_original_benchmark.py. Raw evidence is preserved separately with manifests and SHA256 checksums.",
        "SmallR",
    )
    p(
        "W&amp;B histories: <link href='https://wandb.ai/timodonnell/snowball-pdbthink/runs/xlyyl24u'>pilot xlyyl24u</link>; <link href='https://wandb.ai/timodonnell/snowball-pdbthink/runs/ppc4ryud'>first continuation ppc4ryud</link>; <link href='https://wandb.ai/timodonnell/snowball-pdbthink/runs/e2s5w3qn'>resumed continuation e2s5w3qn</link>. Native stored evaluation records are authoritative for checkpoint selection; external custom summary writes during training can be overwritten by the active W&amp;B writer.",
        "SmallR",
    )
    p(
        "The regression evidence bundle is preserved at s3://marin-us-east-02a/marin/checkpoints/bizon/snowball-pdbthink-epoch/2026.10.15.1/exports/analysis/regression-through565-cb8cf1f493c6cbd7.tar (1,277,900,800 bytes; SHA256 cb8cf1f493c6cbd757e5b7dc6acdf19e9f6ccd2f1783c1e1c1a0bf5062a141b8). This report relies on saved experiment records and pinned implementation source, not on an unverified assumption about the current upstream defaults.",
        "SmallR",
    )
    page("23. Exact validation membership identifiers")
    p(
        "Full validation is the immutable validation.parquet in the prepared input artifact. Monitor membership is the immutable monitor.parquet subset of that same validation split. The complete per-task ID and prompt-hash membership is in cohort.json, with the checksum below. These identifiers distinguish a change in model/serving behavior from a change in the evaluated questions."
    )
    table(
        [
            ["Frozen artifact", "SHA256"],
            ["cohort.json", cohort["cohort_sha256"]],
            *[
                [name, cohort["parquet_sha256"][name]]
                for name in ("validation.parquet", "monitor.parquet", "test.parquet")
            ],
        ],
        [135, 372],
        small=True,
    )
    groups = read("epoch-regression-audit.json")["source_groups"]
    table(
        [
            ["Validation source-group ID", "Tasks", "Correct 384", "Correct 512"],
            *[[name, values["n"], values["baseline"], values["trained"]] for name, values in sorted(groups.items())],
        ],
        [294, 61, 76, 76],
        small=True,
    )
    p(
        "Durable input artifact: s3://marin-us-east-02a/marin/bizon/snowball-pdbthink/inputs/2026.10.07-v3. Neither the 1,003-task validation membership nor its 20 source groups changed between the long-run checkpoints. The only native cohort expansion was 107→1,003 tasks at the pilot/continuation boundary.",
        "SmallR",
    )
    doc.build(story)
    # Retain the layout measured from PNGs, but embed the original vector plots
    # so axis labels remain searchable and sharp at any zoom level.
    with pymupdf.open(args.output) as document:
        figure_pages = [page for page in document if page.get_images()]
        assert len(figure_pages) == len(vector_figures)
        for pdf_page, vector_path in zip(figure_pages, vector_figures, strict=True):
            images = pdf_page.get_images()
            assert len(images) == 1
            xref = images[0][0]
            rectangles = pdf_page.get_image_rects(xref)
            assert len(rectangles) == 1
            pdf_page.delete_image(xref)
            with pymupdf.open(vector_path) as source:
                assert len(source) == 1
                pdf_page.show_pdf_page(rectangles[0], source, 0)
        temporary = args.output.with_suffix(".vector.pdf")
        document.save(temporary, garbage=4, deflate=True)
    temporary.replace(args.output)
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("runs/001/original-benchmark"))
    parser.add_argument("--output", type=Path, default=RESULTS / "snowball-rl-result-report.pdf")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
