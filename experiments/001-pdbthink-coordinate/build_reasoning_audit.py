"""Build the PDF supplement, combined report, and self-contained 200-response audit viewer.

Use the local PDBThink Python environment (ReportLab, Pillow, PyMuPDF).
Scientific plots are produced separately by plot_reasoning_audit.py.
"""

import gzip
import hashlib
import html
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "reasoning-audit"
NAMES = {
    "G01": "Atom distance",
    "G02": "Nearest nonbonded atom",
    "G03": "Nearest residue",
    "G04": "Worst steric clash",
    "N01": "Shared contact residue",
    "P01": "Chain identifiers",
    "P02": "Residue counting",
    "P03": "Coordinate lookup",
    "S01": "Salt-bridge partner",
    "S02": "Phosphorylated residue",
    "S03": "Solvent exposure",
    "S04": "Local secondary structure",
    "S05": "Overall fold class",
    "S09": "Sidechain rotamer",
}
FINDINGS = {
    "G01": (
        "Stronger atom lookup plus actual Euclidean distance calculation.",
        "The model uses coordinate differences, squares, sums and a square root. This is substantive task-relevant behavior, already present in some baseline answers and substantially more reliable at step384. In case1, the baseline uses D73/L79 instead of D80/L94; step384 selects the requested atoms and obtains23.86 Å. Case3 now finishes the computation instead of exhausting its output budget.",
        "Residual failures include wrong residue/atom lookup, arithmetic and formatting. Case6 substitutes a different oxygen and invents a serine CG. Case2 correctly computes12.20 Å but never supplies a parseable FINAL field. Case3 gives21.72 for a gold21.708 and passes the0.02 Å tolerance despite a square-root error.",
    ),
    "G02": (
        "Some coordinate-based search and bond-exclusion reasoning; no robust general solution.",
        "All four step384 successes in the entire120-question category are included because the success stratum is so small. They compute distances to alpha carbons and selected sidechain atoms, then sometimes recover the correct allowed atom in the target residue. Cases10–13 show progress from sequence guesses, missing-target claims or unfinished loops.",
        "The searches are incomplete, and the covalent-bond graph is unreliable. Case8 selects directly bonded GLU8 CB even though it must be excluded. Case9 uses an incorrect arginine connectivity and selects CB instead of CG. Three-letter or otherwise invalid atom identifiers are common. Four successes out of120 is weak evidence of general nearest-nonbonded-atom competence.",
    ),
    "G03": (
        "A clear coordinate shortcut: nearest alpha carbon substitutes for minimum heavy-atom distance.",
        "All seven sampled step384 traces explicitly compare CA–CA distances. Their answers all match the independently computed CA proxy. On the full108-question category,104 predictions match that proxy. The proxy itself scores79/108; the model scores78/108. This strongly supports a useful but incomplete geometric heuristic.",
        "The proxy is wrong on29 questions. The model follows its wrong answer on27 of those29, and solves only one. In case15, C192 is closer by CA distance (5.010 vs6.253 Å), but D191 is closer by the required heavy-atom minimum (2.976 vs3.298 Å). Cases16 and18 show the same structural failure, with additional coordinate-copying mistakes.",
    ),
    "G04": (
        "No demonstrated success at finding the worst steric clash.",
        "The entire validation category has only three questions, and all three are reviewed. Step384 makes more numerical attempts, including distance and radius arithmetic, but no response computes the required maximum positive atom-pair overlap reliably.",
        "It substitutes CA distances or residue-average radii, assumes unmeasured contact distances, mishandles bonded exclusions and omits relevant nonpolymer atoms. Cases21 and23 report the least-negative overlap despite the requirement for a positive clash. Case22 assumes2.8 Å contacts and averages residue radii. All three final identifiers are also invalid.",
    ),
    "N01": (
        "More arithmetic, but the wrong contact definition and no demonstrated solution.",
        "Baseline answers largely speculate about sequence midpoints or get stuck listing residues. All seven step384 samples attempt CA-distance screens at the4 Å cutoff, whereas the task requires any heavy-atom pair within4 Å to each anchor.",
        "All seven trained responses truncate without a final answer; full-category performance remains0/72. A CA pair farther than4 Å cannot rule out a sidechain contact. Misnumbered residues and repeated rescanning compound this invalid screening rule. Numerical effort alone is not evidence that the contact-intersection task was learned.",
    ),
    "P01": (
        "Formatting and completion improve; the sampled chain-identification knowledge was already present.",
        "Both checkpoints identify the right chain in all seven sampled questions. The four scored gains are output-contract repairs: boxed/inline FINAL is not parsed, or a trailing “Final Answer” overrides G with ANSWER. Step384 gives a clean final field. These examples are single-chain structures.",
        "No failure remains at step384 in the32-question validation category. That does not establish more complex multichain parsing, nor does this sample show new molecular geometry knowledge.",
    ),
    "P02": (
        "More consistent enumeration in some cases, with substantial counting and interpretation errors.",
        "Case41 recovers the59-residue count after the baseline omits two positions. Case42 enumerates60 residues correctly where the baseline lists60 but claims83. Both checkpoints can count the38-residue example in case43.",
        "Case39 regresses from78 to150 by mistaking the last author residue number for the count. Case38 counts22 then wrongly excludes TPO as a nonprotein triphosphate. Case40 invents an extra entry. Case44 gives the correct29 but continues generating a repeated sequence until the length cap, so native reward is zero.",
    ),
    "P03": (
        "Strong evidence of improved retrieval and faithful copying of supplied coordinates.",
        "The three sampled gains repair wrong coordinate content, including incorrect signs/components and wrong residue lookup. Case49 changes x from5.668 to the supplied−6.841; case50 finds the requested R349 NH1 rather than another coordinate triple. The full category improves from27/118 to114/118.",
        "Lookup is still imperfect. Case51 returns the SE atom of MSE1 instead of MSE33 at both checkpoints. Case47 locates the correct residue context but substitutes a wrong CG2 triple. These tasks test reading the PDB, not deriving a physical quantity.",
    ),
    "S01": (
        "Some real charged-atom distance calculations, especially for lysine partners.",
        "Cases54,56 and58 compute the specified oxygen–nitrogen distances and return the correct partners. Independent checks recover E860 OE2–K889 NZ=3.174 Å, E9 OE2–K13 NZ=2.863 Å and D6 OD1–K3 NZ=3.210 Å. Case55 computes the correct R170–E162 contact but uses the wrong residue identifier format.",
        "Arginine searches often stop at NE or use CA/carboxylate-carbon proxies, missing NH1/NH2. Cases52,53 and57 all have a qualifying NH1 contact even though every NE-to-partner-oxygen distance exceeds4 Å. Those step384 traces truncate. This explains a concrete limit of the apparent salt-bridge procedure.",
    ),
    "S02": (
        "No scored improvement, and several sampled answers lose correct biochemical identification.",
        "The baseline often recognizes SEP as phosphoserine or TPO as phosphothreonine but fails to convert to the requested parent one-letter residue identifier, e.g. Y:S26. Step384 still recognizes TPO in cases62 and64, but omits the residue number.",
        "In cases60,63 and65, step384 instead calls SEP/TPO a separate small molecule, sulfate or nucleotide and guesses an ordinary Ser/Thr. Case59 ignores phosphate atoms and guesses a common serine site. All seven sampled responses and all nine validation tasks fail. This category illustrates why a zero-to-zero score can hide semantic regressions.",
    ),
    "S03": (
        "A near-constant solvent-exposed answer; no demonstrated solvent-accessibility calculation.",
        "All seven sampled step384 answers assume a typical accessible area or invent a relative SASA value, without computing occlusion from surrounding atoms. Across all119 tasks,118 parsed answers are solvent-exposed and one is unparsed; none is buried.",
        "Always answering solvent-exposed scores98/119, slightly above the model’s97/119. All21 buried questions fail. Cases68 and69 regress from correct burial heuristics to exposure. The score increase is therefore not evidence that the model learned to calculate solvent accessibility.",
    ),
    "S04": (
        "A near-constant coil answer; no demonstrated DSSP or backbone-geometry computation.",
        "All seven sampled step384 answers say coil. They refer to imagined DSSP runs, generic sequence propensities or invented phi/psi values. No actual tool is available or called. Across111 tasks,110 parsed predictions are coil and one is unparsed.",
        "Always answering coil scores68/111, exactly the model’s score. It solves none of the33 helix or10 strand questions. Cases73 and74 lose correct helix guesses. The apparent gain reflects class selection and output reliability, not demonstrated secondary-structure assignment.",
    ),
    "S05": (
        "An alpha-helical class preference and sequence-propensity rhetoric, without structural calculation.",
        "The sampled responses do not calculate hydrogen bonds, backbone angles or helix/strand fractions. At step384,10 of12 full-category predictions are predominantly alpha helical and two are mixed. All eight correct answers are alpha-helical chains.",
        "Always answering alpha helical also scores8/12. The single beta-sheet and all three mixed chains fail. Cases81 and85 show wrong helix guesses; case80 remains wrong mixed. One sampled gain (case82) merely formats the baseline’s already correct helix guess.",
    ),
    "S09": (
        "A more consistent explicit torsion-calculation template, but not a validated dihedral algorithm.",
        "All13 sampled step384 responses attempt coordinate subtraction, cross products and angle calculation. Nine start with the correct four coordinates; four have clear extraction errors. Seven receive reward1, but none survives all checks of coordinates, formula, arithmetic and bin mapping. This is more than an always-g− rule, yet less than reliable rotamer computation.",
        "A recurring formula omits the middle-bond norm; there are also wrong atom selections, sign transcription errors, lost atan2 quadrants and positive-angle-to-g− bin mistakes. Correct category rewards frequently tolerate an incorrect angle. The dedicated rotamer section shows exactly why.",
    ),
}


def readable(text):
    # Add spaces in prose written beside literal numbers without altering identifiers.
    import re

    text = re.sub(
        r"\b(step|case|cases|all|of|from|to|about|than|for|with|only|every|those|the|scores|versus|its|these|within|enumerates|counts|correct|true|October)(?=\d)",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    return text


class AuditDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "HeadA":
            key = f"audit-page-{self.page}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0)


def main():
    sample_path = Path("runs/001/reasoning-audit/sample.json")
    sample = json.loads(sample_path.read_text())
    annotations = json.loads((OUT / "annotations.json").read_text())
    manifest = json.loads((OUT / "sample-manifest.json").read_text())
    stats = json.loads((OUT / "statistics.json").read_text())
    geo = json.loads((OUT / "geometry.json").read_text())
    assert len(sample) == len(annotations) == 100
    for r, a in zip(sample, annotations, strict=True):
        assert r["id"] == a["id"]
        for step in ("0", "384"):
            assert hashlib.sha256(r["answers"][step]["text"].encode()).hexdigest() == a["steps"][step]["text_sha256"]
    (OUT / "sample.json.gz").write_bytes(gzip.compress(sample_path.read_bytes(), mtime=0))
    (OUT / "category-findings.json").write_text(
        json.dumps({f: [readable(t) for t in v] for f, v in FINDINGS.items()}, indent=2) + "\n"
    )
    build_html(sample, annotations, manifest, stats)
    build_pdf(sample, annotations, manifest, stats, geo)
    paths = [
        "sample.json.gz",
        "sample-manifest.json",
        "annotations.json",
        "review-notes.json",
        "statistics.json",
        "geometry.json",
        "category-findings.json",
        "rollout-audit.html",
        "rollout-audit.pdf",
        "category-controls.png",
        "category-controls.pdf",
        "rotamer-audit.png",
        "rotamer-audit.pdf",
    ]
    hashes = {
        name: {"sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest(), "bytes": (OUT / name).stat().st_size}
        for name in paths
    }
    (OUT / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "files": hashes,
                "date": "2026-10-16",
                "review": "Codex qualitative review; not an independent blinded human or multi-rater study",
                "review_scope": "All 100 paired questions reviewed through deduplicated text views and targeted passages; full raw text preserved. Selected computations independently checked, not every arithmetic line.",
                "source_code_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in (
                        "sample_reasoning_audit.py",
                        "analyze_reasoning_audit.py",
                        "plot_reasoning_audit.py",
                        "build_reasoning_audit.py",
                    )
                },
                "combined_report_sha256": hashlib.sha256(
                    (ROOT / "results" / "snowball-rl-result-report-with-rollout-audit.pdf").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print("Built", OUT / "rollout-audit.pdf")


def build_html(sample, annotations, manifest, stats):
    e = html.escape
    rows = []
    for f in sorted(NAMES):
        p = stats["population"][f]
        finding = "<br>".join(e(readable(x)) for x in FINDINGS[f])
        rows.append(
            f"<tr><td>{f} {NAMES[f]}</td><td>{p['correct0']}/{p['n']} → {p['correct384']}/{p['n']}</td><td>{finding}</td></tr>"
        )
    cases = []
    for r, a in zip(sample, annotations, strict=True):
        answers = []
        for step in ("0", "384"):
            result = r["answers"][step]["result"]
            parsed = result["outcome"]["parsed"]["value"]
            answers.append(
                f"<div><h3>Step {step} · reward {result['reward']:g}</h3>"
                f"<p><b>Review:</b> {e(readable(a['review']['step' + step]))}</p>"
                f"<p>Parsed: <code>{e(json.dumps(parsed, ensure_ascii=False))}</code>; "
                f"truncated: {result['truncated']}; refusal flag: {result['refusal']}</p>"
                f"<details><summary>Full verbatim response ({len(r['answers'][step]['text']):,} characters)</summary>"
                f"<pre>{e(r['answers'][step]['text'])}</pre></details>"
                f"<details><summary>Response SHA256</summary><code>{a['steps'][step]['text_sha256']}</code></details></div>"
            )
        prompt = "\n\n".join(m["role"].upper() + "\n" + m["content"] for m in r["prompt"])
        cases.append(
            f'<article id="case-{a["index"]}" data-family="{r["family"]}" data-state="{r["transition"]}">'
            f"<h2>Case {a['index']} · {e(r['id'])}</h2><p>Outcome stratum: <b>{r['transition']}</b>; "
            f"gold: <code>{e(json.dumps(r['gold']['gold_answer']['value'], ensure_ascii=False))}</code>; group: {e(str(r['group']))}</p>"
            f'<div class="pair">{"".join(answers)}</div><details><summary>Full frozen prompt and gold metadata</summary>'
            f"<pre>{e(prompt)}</pre><pre>{e(json.dumps(r['gold'], indent=2))}</pre></details></article>"
        )
    doc = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Snowball rollout reasoning audit</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#193443;margin:32px auto;max-width:1400px;padding:0 20px}
h1,h2,h3{color:#155d77}h2{overflow-wrap:anywhere;font-size:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:25px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:14px;font-size:13px;max-height:700px;overflow:auto}
code{overflow-wrap:anywhere}article{border-top:2px solid #cbdce3;margin:35px 0;padding:15px 0}summary{cursor:pointer;color:#155d77}
table{border-collapse:collapse}td,th{border-bottom:1px solid #ccd9de;padding:12px;vertical-align:top}td:first-child{min-width:150px}
td:nth-child(2){min-width:115px}.filters{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccd9de;padding:12px;z-index:2}
select,input{padding:6px;margin-right:14px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>What did Snowball learn?</h1><p>100 paired validation questions, 200 archived responses · Steps0 and384 · 16 October2026</p>
<p><b>Conclusion:</b> Better coordinate retrieval and distance arithmetic coexist with strong shortcuts, invalid geometry and output-contract effects.
Correct final answers do not certify the reasoning.</p>
<p>Sample seed20261016. Seven questions per present family, except all three G04 and13 S09.
Within each family, allocation balances available paired success/failure strata, then samples uniformly without replacement.
00=both fail;01=gain;10=loss;11=both succeed. Sample success fractions are intentionally unrepresentative.</p>
<p>Step0 uses archived standalone inference; step384 uses native training evaluation. Prompts and scoring match; runtimes differ.
This is a descriptive audit, not a controlled causal attribution of every difference to RL. Review used deduplicated views plus targeted passages and independent geometry checks;
not every arithmetic line was verified. Full unabridged text is below.</p>
<details><summary>Per-category findings and full-population scores</summary><table><thead><tr><th>Category</th><th>Full cohort,0→384</th><th>Interpretation</th></tr></thead><tbody>"""
    doc += "".join(rows) + "</tbody></table></details>"
    doc += '<div class="filters"><label>Category <select id="family"><option value="">All</option>'
    doc += "".join(f"<option>{f}</option>" for f in sorted(NAMES))
    doc += '</select></label><label>Outcome <select id="state"><option value="">All</option><option>00</option><option>01</option><option>10</option><option>11</option></select></label><label>Find ID / case <input id="query"></label><span id="count"></span></div>'
    doc = readable(doc)
    doc += "".join(cases)
    doc += """<script>
const controls=['family','state','query'].map(x=>document.getElementById(x));
function filter(){let n=0;for(const a of document.querySelectorAll('article')){
 const keep=(!controls[0].value||a.dataset.family===controls[0].value)&&(!controls[1].value||a.dataset.state===controls[1].value)&&
 (!controls[2].value||a.querySelector('h2').textContent.toLowerCase().includes(controls[2].value.toLowerCase()));
 a.hidden=!keep;if(keep)n++;}document.getElementById('count').textContent=n+' paired questions';}
controls.forEach(x=>x.addEventListener('input',filter));filter();
</script></html>"""
    (OUT / "rollout-audit.html").write_text(doc)


def build_pdf(sample, annotations, manifest, stats, geo):
    for name, file in [("D", "DejaVuSans.ttf"), ("DB", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, "/usr/share/fonts/truetype/dejavu/" + file))
    pdfmetrics.registerFontFamily("D", normal="D", bold="DB", italic="D", boldItalic="DB")
    styles = getSampleStyleSheet()
    for name, size, leading, color in [
        ("BodyA", 9.4, 14.5, "#193443"),
        ("SmallA", 8, 11.5, "#193443"),
        ("HeadA", 22, 28, "#155d77"),
        ("SubA", 13, 18, "#155d77"),
        ("TitleA", 32, 39, "#155d77"),
    ]:
        styles.add(
            ParagraphStyle(
                name=name,
                fontName="DB" if name in ("HeadA", "SubA", "TitleA") else "D",
                fontSize=size,
                leading=leading,
                spaceAfter=10,
                textColor=colors.HexColor(color),
            )
        )
    story = []

    def p(text, style="BodyA"):
        story.append(Paragraph(readable(text) if style == "BodyA" else text, styles[style]))

    def page(title):
        story.append(PageBreak())
        p(title, "HeadA")

    def table(rows, widths):
        t = Table(
            [[Paragraph(escape(str(c)), styles["SmallA"]) for c in row] for row in rows], colWidths=widths, repeatRows=1
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccdce3")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])

    def fig(name, width=510):
        from PIL import Image as PILImage

        with PILImage.open(OUT / name) as im:
            w, h = im.size
        story.append(Image(str(OUT / name), width=width, height=width * h / w))

    p("What did Snowball<br/>actually learn?", "TitleA")
    p("A paired rollout audit of validation at steps 0 and 384", "SubA")
    p("16 October 2026 · 100 questions · 200 archived responses · 14 task categories", "SmallA")
    p(
        "<b>The improvement is mixed.</b> There is concrete evidence of better coordinate retrieval, distance arithmetic and some salt-bridge calculations. But several higher-scoring categories are explained by class defaults or an easier geometric proxy. The rotamer responses now attempt dihedral calculations, yet their mathematical correctness is much weaker than their final-bin reward suggests."
    )
    p(
        "<b>The strongest shortcut results:</b> step 384 says solvent-exposed in all 118 parseable S03 responses and coil in all 110 parseable S04 responses. In G03, 104 of 108 answers agree with the nearest-alpha-carbon proxy, although the task asks for minimum heavy-atom distance."
    )
    p(
        "<b>Rotamers:</b> all 13 reviewed step-384 traces attempt numerical geometry; none is verified correct end to end. Seven still earn reward 1. A recurring missing factor changes angles but leaves all 116 validation gold bins unchanged when applied to the correct coordinates. This particular error is invisible to the category-only reward on these tasks."
    )
    p(
        "This supplement qualifies the molecular-reasoning interpretation of the original result report; it does not change its measured scores. The matched original-benchmark result remains 5→11 of 40 strictly, or 8→11 after the existing box correction, with wide uncertainty. The present archival validation audit uses different inference paths at the two checkpoints and must not replace that matched comparison."
    )
    p(
        "The accompanying HTML viewer contains every full prompt and response beside its review. The compressed JSON contains the same frozen sample for machine inspection. No new model inference or GPU job was required for this audit.",
        "SmallA",
    )

    page("Sampling, scope and interpretation")
    p(
        "I selected <b>100 question IDs before inspecting their reasoning</b>, then reviewed both archived answers: 100 step-0 rollouts and 100 step-384 rollouts. Pairing the questions lets us compare how each answer changes. Python random.Random uses seed <b>20261016</b>; the manifest was frozen at "
        + escape(manifest["frozen_at"])
        + "."
    )
    p(
        "Sampling is random within strata but deliberately disproportionate. Each of the 14 present families receives seven questions, except G04, which has only three and is reviewed in full, and S09, which receives 13 because the user specifically requested rotamer analysis. Within a family, quota is allocated as evenly as possible across the available joint success states, subject to each stratum’s capacity; ties use a seeded shuffle. IDs are sampled uniformly without replacement inside each stratum."
    )
    p(
        "The strata are <b>00</b> (both fail), <b>01</b> (step-0 fail, step-384 succeed), <b>10</b> (step-0 succeed, step-384 fail), and <b>11</b> (both succeed). For example, S09 contributes three 00, three 01, three 10 and four 11 questions. Its sample therefore scores 7/13 at each checkpoint by construction, even though full-category accuracy increases. G02 includes all four step-384 successes. <b>Do not estimate population accuracy from this sample.</b> All reported population scores and controls use the full category."
    )
    p(
        "The underlying validation cohort is the same frozen 1,003-question, 20-source-group set; the sample spans 17 groups. Questions sharing a structure are correlated. I01 (interface contacts), S06 (ligand contacts), S07 (metal coordination), S08 (disulfide partner) and T01 (two-state comparison) have no eligible validation questions and cannot be assessed here."
    )
    p(
        "Step 0 means the original Snowball checkpoint before this project’s additional RL, not an untrained model. Step 384 is the validation-selected checkpoint requested by the user. The original-model answers come from baseline-validation-v2 standalone serving; step-384 answers come from the native training-evaluator archive. Both use the same frozen prompts, temperature 0, one answer and the remaining 32,768-token context. Their serving/execution paths differ. Thus observed differences are descriptive evidence, not a clean causal decomposition into training versus runtime effects."
    )

    page("What was checked")
    p(
        "The sampler checks question membership, exact baseline request messages, baseline output budgets and native prompt hashes. Replaying the frozen native verifier reproduces all 2,006 recorded rewards. It saves the 100 selected IDs, all response SHA256 hashes, input/native-archive hashes and full prompts. Strict success includes the native requirements for formatting, truncation and refusal; a semantically correct but unparseable answer can score zero."
    )
    p(
        "All 200 responses were inspected through text views that remove exact duplicate lines, show beginning/middle/end sections of long responses, and support targeted examination of computational passages. Repeated trailing invented metadata is retained in the raw artifact; the normal review view focuses on the substantive answer before the first JSON metadata block. Truncation and parsing failures were checked separately, including post-answer repetition. This was a qualitative Codex review, not a blinded independent-human or multiple-rater study. <b>Not every arithmetic line of every long trace was independently recomputed.</b>"
    )
    p(
        "Independent numerical checks parse fixed-column coordinates directly from the displayed prompts. They recompute all 116 S09 gold dihedrals using two equivalent formulas, all 108 G03 heavy-atom minima and their CA-only controls, all seven sampled G01 distances, all seven sampled P03 coordinate answers, and the true charged-atom contact for all seven sampled S01 questions. All recomputed gold answers agree. Unit checks cover known signed torsions, proper rotation/translation/scale invariance, a bin-changing counterexample to the wrong formula, and signed coordinate parsing."
    )
    p(
        "The review distinguishes <b>correct final label</b>, <b>valid observable method</b>, <b>shortcut</b>, <b>format repair</b> and <b>unfinished generation</b>. A model’s prose is evidence of what it wrote, not proof of the internal mechanism that caused its answer. Claims about computation refer to visible operations; mentions of DSSP, Python or a database are not evidence that a tool ran. No tools were available to these model rollouts."
    )
    families = sorted(NAMES)
    allocation_rows = [["Family", "Full N", "Sample 00 / 01 / 10 / 11"] * 2]
    for left, right in zip(families[:7], families[7:], strict=True):
        row = []
        for f in (left, right):
            row.extend(
                [
                    f,
                    stats["population"][f]["n"],
                    " / ".join(str(manifest["allocation"][f]["sample"].get(k, 0)) for k in ("00", "01", "10", "11")),
                ]
            )
        allocation_rows.append(row)
    table(allocation_rows, [40, 45, 170, 40, 45, 170])

    page("Full-cohort scores and simple controls")
    fig("category-controls.png")
    p(
        "Bars: full-category strict accuracy at each archived checkpoint, with numerator/denominator labels. Diamonds: explicitly defined offline controls. These are conditional accuracies on context-eligible validation questions; the sample is used for qualitative interpretation only. The different inference paths prevent treating every bar difference as an isolated RL effect.",
        "SmallA",
    )
    p(
        "The controls require no learned molecular-reasoning policy: G03 selects the nearest CA; S03 always says exposed; S04 always says coil; S05 always says alpha helical; S09 always says g−. The S09 constant is clearly worse than the model. The S03/S04/S05 controls equal or exceed the model, so their score gains do not establish structural computation.",
        "SmallA",
    )

    groups = [
        ("P01", "P02"),
        ("P03", "G01"),
        ("G02", "G03"),
        ("G04", "N01"),
        ("S01", "S02"),
        ("S03", "S04"),
        ("S05", "S09"),
    ]
    for pair in groups:
        page("Category findings: " + " and ".join(pair))
        for family in pair:
            pop = stats["population"][family]
            p(f"{family} · {NAMES[family]}", "SubA")
            p(
                f"Full validation: <b>{pop['correct0']}/{pop['n']} → {pop['correct384']}/{pop['n']}</b>. "
                + escape(readable(FINDINGS[family][0]))
            )
            p(escape(readable(FINDINGS[family][1])))
            p("<b>How it fails:</b> " + escape(readable(FINDINGS[family][2])))

    page("Rotamers: calculation or trick?")
    p(
        "The answer is a mixture: <b>the visible step-384 behavior is an explicit calculation template, not simply a constant rotamer guess, but it is not a reliable dihedral algorithm.</b> It extracts N, CA, CB and a residue-specific fourth atom (e.g. CG or OG1), forms bond vectors and cross products, estimates an angle, and assigns one of three bins. The full category has 46 g−, 37 trans and 33 g+ labels; step384 scores85/116 versus46/116 for always g−."
    )
    p(
        "Let A=N, B=CA, C=CB and D=the fourth atom. Define u=B−A, v=C−B, w=D−C, n₀=u×v and n₁=v×w. With the benchmark’s signed convention, a correct formula is:"
    )
    p("<b>χ₁ = atan2(‖v‖ (n₀ · w), n₀ · n₁)</b>, converted to degrees and wrapped to [−180°,180°).")
    p(
        "A recurring trace formula is <b>atan2(n₀·w, n₀·n₁)</b>, missing the factor ‖v‖. The numerator and denominator then have different length dimensions, and changing coordinate scale changes the computed angle. This is not fixed by merely using atan2. Other traces subsequently replace terms, lose the quadrant, or make arithmetic errors."
    )
    p(
        "Yet applying this wrong formula to the correct four coordinates for every S09 validation question preserves <b>116/116 bins</b>. That is an offline formula control, not a claim that the model calculated all116 angles correctly. The benchmark asks for broad bins: g− is −120°≤χ&lt;0°, g+ is0°≤χ&lt;120°, and trans is the rest. The observed geometries do not expose this formula error at those boundaries. A synthetic115° dihedral with a1.53-length middle bond crosses from g+ to trans under the wrong formula, so the shortcut is not generally valid."
    )
    p(
        "In the13 reviewed step384 traces, four have clearly wrong input coordinates, two map a positive computed angle to g−, and one explicitly loses the atan2 quadrant. These counts are review findings in a biased sample, with overlapping error types. None of the13 passes all stages of an end-to-end computation; nevertheless seven final bins are rewarded. At step0, some correct bins likewise follow invented coordinates or a guess. Correctness of the baseline label is not a trustworthy procedural control either."
    )
    p(
        "This is consistent with reward favoring a reusable, often successful template that does not need exact intermediate mathematics. It does not establish deliberate exploitation or prove that the written rationale causes the final answer. Stronger causal evidence would require controlled interventions on coordinates and comparison under matched inference settings."
    )

    page("Rotamer diagnostic plots")
    fig("rotamer-audit.png")
    p(
        "Left: independently recomputed angle versus the last numerical angle discussed in the sampled step384 trace; annotations are zero-based audit case indices. Blue means correct final bin, red wrong final bin. Angles are periodic, and the numbers reflect the model’s own inconsistent calculations, not a clean implemented formula. Right: the isolated missing-norm formula on correct input coordinates for all116 questions. This separates a formula defect from input-copying and later arithmetic defects.",
        "SmallA",
    )
    table(
        [
            ["Case", "True angle / bin", "Step384 statement", "Why the score is misleading or fails"],
            ["87", "−64.851° / g−", "about−54° / g−", "Right coordinates; missing norm; correct category."],
            ["89", "+177.692° / t", "−1.508° / g−", "atan(y/x) used in place of atan2; quadrant lost."],
            ["90", "−177.982° / t", "about−10.3° / g−", "Lys NZ substituted for backbone N."],
            ["91", "+61.758° / g+", "about+50.8° / g−", "Positive bin mislabeled g− after coordinate arithmetic."],
            [
                "92",
                "+57.067° / g+",
                "about+45.14° / g−",
                "Positive-angle sign/bin mistake, despite correct input points.",
            ],
            ["94", "−178.782° / t", "about−179.2° / t", "Missing norm, but correct quadrant/wrapping and category."],
        ],
        [38, 117, 124, 231],
    )

    page("Implications for the RL result")
    p(
        "<b>What is supported:</b> the policy became much better at reading requested coordinates and completing simple distance calculations in this validation distribution. Some salt-bridge responses also use the right charged-atom geometry. Those are meaningful improvements. The separate matched original-benchmark result is still encouraging, although its uncertainty and limited coverage remain as reported."
    )
    p(
        "<b>What is not supported:</b> a broad claim that the model learned reliable molecular structure analysis. Exposure and local secondary-structure gains are explained by constant labels. Fold classification reaches only the majority-class score. Nearest-residue success mostly follows a CA-only approximation. Rotamer categories often pass while the intermediate geometry is wrong. The hardest searches remain mostly or entirely unsolved."
    )
    p(
        "These findings explain why aggregate validation accuracy is an incomplete progress measure, especially after the context filter changes class frequencies. They qualify the step384 peak. They do <b>not</b> by themselves identify the cause of the later384→512 optimization regression, which requires the separate training/runtime audit in the main report."
    )
    p(
        "A useful next experiment would keep the existing benchmark frozen and add a separate diagnostic set: balanced buried/exposed and helix/strand/coil labels; nearest-residue cases where CA and heavy-atom rankings disagree; rotamers near±120° and0°; atom-name distractors such as N versus NZ; and rigidly rotated copies plus chemically valid torsion changes. A continuous-angle diagnostic would reveal errors hidden by coarse bins. No such new model experiment is claimed in this report."
    )
    p(
        "Selection and calibration matter: step384 was chosen using validation, this sample is conditioned on outcomes, several questions share a structure, and inference paths differ. A stronger learning claim should reproduce the behavior on fresh source groups under matched serving, with repeated decoding where appropriate. The current audit provides concrete hypotheses and failure examples for that work, not an independent test-set estimate."
    )
    p(
        "Reproducibility files: sample-manifest.json (seed, allocation, selected IDs and hashes), sample.json.gz (full prompts and200 raw responses), annotations.json and review-notes.json (all100 case reviews), statistics.json (full-category counts), geometry.json (independent checks), and rollout-audit.html (filterable paired viewer). Code: sample_reasoning_audit.py, analyze_reasoning_audit.py, plot_reasoning_audit.py and build_reasoning_audit.py. All paths are relative to this experiment; artifacts are in results/reasoning-audit/.",
        "SmallA",
    )

    page("Appendix: every reviewed pair")
    p(
        "Indices are stable, zero-based positions in the frozen sample. Outcome states use the native strict scores. Notes describe visible reasoning, including cases where the correct answer content is not accepted by the scorer. Full text and prompt are available under the same index in the HTML viewer.",
        "SmallA",
    )
    for a in annotations:
        start = len(story)
        p(f"Case {a['index']} · {a['family']} · state {a['transition']}", "SubA")
        p(
            escape(a["id"]) + "<br/>Gold: " + escape(json.dumps(a["gold"]["gold_answer"]["value"], ensure_ascii=False)),
            "SmallA",
        )
        for step in ("0", "384"):
            p(f"<b>Step {step}:</b> " + escape(readable(a["review"]["step" + step])), "SmallA")
        entry = story[start:]
        del story[start:]
        story.append(KeepTogether(entry))

    def footer(canvas, doc):
        canvas.setFont("D", 7)
        canvas.setFillColor(colors.HexColor("#536775"))
        canvas.drawString(40, 24, "Snowball × PDBThink | Paired rollout reasoning audit | 16 October 2026")
        canvas.drawRightString(572, 24, str(doc.page))

    path = OUT / "rollout-audit.pdf"
    AuditDocument(
        str(path),
        pagesize=(612, 792),
        leftMargin=50,
        rightMargin=50,
        topMargin=42,
        bottomMargin=46,
        title="Snowball RL: paired validation rollout audit",
        author="Snowball post-training experiment",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    # Bundle the original detailed methods/results with this explicitly dated supplement.
    combined = pymupdf.open(ROOT / "results" / "snowball-rl-result-report.pdf")
    with pymupdf.open(path) as supplement:
        offset = len(combined)
        combined.insert_pdf(supplement)
        combined.set_toc(
            combined.get_toc()
            + [[1, "Rollout reasoning audit supplement", offset + 1]]
            + [[level + 1, title, page + offset] for level, title, page in supplement.get_toc()]
        )
    combined.save(ROOT / "results" / "snowball-rl-result-report-with-rollout-audit.pdf", garbage=4, deflate=True)
    print("Supplement pages", len(pymupdf.open(path)), "combined pages", len(combined))
    combined.close()


if __name__ == "__main__":
    main()
