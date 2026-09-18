"""
generate_dc5_presentation.py
────────────────────────────
Builds the DC 5 progress-review slide deck for the multi-model Cross-Project
Bug Localization (CPL) Phase 1 study (BLAZE + TRANP-CNN + COOBA), mirroring
DC_5_Report/main.tex. Also writes a speaker-notes .md.

It reuses the visual style of Scripts/analysis/generate_presentation.py but is
self-contained. All numbers are the corrected multi-model results
(results/obj1_experimental_results_corrected.csv / journal_draft/main.tex).
The DC 4.5 deck (results/presentation/CPL_Phase1_Results.pptx) is left untouched.

Output
──────
  DC_5_Report/DC5_Presentation.pptx
  DC_5_Report/DC5_SPEAKER_NOTES.md

Run
───
    source obj1/bin/activate
    python Scripts/analysis/generate_dc5_presentation.py
"""

import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = "/home/cs21d002_eashaan/PhD/Objective1"
OUT_DIR    = os.path.join(ROOT, "DC_5_Report")
IMAGES_DIR = os.path.join(ROOT, "DC_5_Report")          # corrected figs live here
PPTX_PATH  = os.path.join(OUT_DIR, "DC5_Presentation.pptx")
NOTES_PATH = os.path.join(OUT_DIR, "DC5_SPEAKER_NOTES.md")

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK   = RGBColor(0x1A, 0x23, 0x3A)
C_BLUE   = RGBColor(0x21, 0x96, 0xF3)
C_GREEN  = RGBColor(0x4C, 0xAF, 0x50)
C_PINK   = RGBColor(0xE9, 0x1E, 0x63)
C_ORANGE = RGBColor(0xFF, 0x98, 0x00)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT  = RGBColor(0xF5, 0xF5, 0xF5)
C_GRAY   = RGBColor(0x90, 0x90, 0x90)
C_ACCENT = RGBColor(0x7C, 0x4D, 0xFF)

os.makedirs(OUT_DIR, exist_ok=True)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H
blank_layout = prs.slide_layouts[6]

FOOT = "DC 5  |  Cross-Project Bug Localization — Multi-Model Phase 1  |  A Eashaan Rao (CS21D002)"


# ── Helpers ──────────────────────────────────────────────────────────────────
def add_rect(slide, l, t, w, h, fill=None, line=None, line_w=Pt(0)):
    shape = slide.shapes.add_shape(1, l, t, w, h)
    shape.line.width = line_w
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h, size=18, bold=False, color=C_DARK,
             align=PP_ALIGN.LEFT, wrap=True, italic=False):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = wrap
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return box


def add_bullets(slide, items, l, t, w, h, size=15, color=C_DARK,
                bullet="▸ ", space=6):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_before = Pt(space)
        lead = "" if item.startswith("   ") else bullet
        r = p.add_run()
        r.text = lead + item.strip()
        r.font.size = Pt(size)
        r.font.color.rgb = color
    return box


def header_bar(slide, title, subtitle=None):
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.15), fill=C_DARK)
    add_text(slide, title, Inches(0.35), Inches(0.12), Inches(12.6), Inches(0.62),
             size=25, bold=True, color=C_WHITE)
    if subtitle:
        add_text(slide, subtitle, Inches(0.35), Inches(0.74), Inches(12.6), Inches(0.38),
                 size=13, color=C_LIGHT, italic=True)
    add_rect(slide, Inches(0), Inches(1.15), SLIDE_W, Inches(0.05), fill=C_BLUE)


def footer(slide, text=FOOT):
    add_rect(slide, Inches(0), Inches(7.15), SLIDE_W, Inches(0.35), fill=C_DARK)
    add_text(slide, text, Inches(0.3), Inches(7.18), Inches(12.7), Inches(0.3),
             size=9, color=C_GRAY)


def table_box(slide, headers, rows, l, t, w, h, header_fill=C_DARK,
              row_fills=None, fsize=12):
    n_cols = len(headers)
    n_rows = len(rows)
    col_w = w / n_cols
    row_h = h / (n_rows + 1)
    for ci, hdr in enumerate(headers):
        add_rect(slide, l + ci * col_w, t, col_w - Inches(0.02), row_h, fill=header_fill)
        add_text(slide, hdr, l + ci * col_w + Inches(0.04), t + Inches(0.03),
                 col_w - Inches(0.08), row_h - Inches(0.04),
                 size=fsize, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    for ri, row in enumerate(rows):
        fill = (row_fills[ri] if row_fills and ri < len(row_fills)
                else (C_LIGHT if ri % 2 == 0 else C_WHITE))
        for ci, cell in enumerate(row):
            add_rect(slide, l + ci * col_w, t + (ri + 1) * row_h,
                     col_w - Inches(0.02), row_h - Inches(0.02),
                     fill=fill, line=C_GRAY, line_w=Pt(0.5))
            add_text(slide, str(cell), l + ci * col_w + Inches(0.04),
                     t + (ri + 1) * row_h + Inches(0.02),
                     col_w - Inches(0.08), row_h - Inches(0.05),
                     size=fsize, color=C_DARK, align=PP_ALIGN.CENTER)


def img(slide, name, l, t, w, h):
    path = os.path.join(IMAGES_DIR, name)
    if os.path.exists(path):
        slide.shapes.add_picture(path, l, t, width=w, height=h)
    else:
        add_rect(slide, l, t, w, h, fill=C_LIGHT, line=C_GRAY, line_w=Pt(1))
        add_text(slide, f"[missing: {name}]", l + Inches(0.1), t + h / 2,
                 w - Inches(0.2), Inches(0.4), size=11, color=C_GRAY,
                 align=PP_ALIGN.CENTER)


def slide():
    return prs.slides.add_slide(blank_layout)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
add_rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, fill=C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.18), SLIDE_H, fill=C_BLUE)
add_rect(s, Inches(0), Inches(5.05), SLIDE_W, Inches(0.06), fill=C_PINK)
add_text(s, "Cross-Project Bug Localization", Inches(0.6), Inches(1.15),
         Inches(12.0), Inches(1.0), size=38, bold=True, color=C_WHITE)
add_text(s, "Multi-Model Phase 1 — DC 5 Progress Review", Inches(0.6), Inches(2.15),
         Inches(12.0), Inches(0.7), size=23, color=C_BLUE)
add_text(s, "Does a bug localizer trained on one project transfer to another —\nand does the answer depend on the model architecture?",
         Inches(0.6), Inches(3.0), Inches(11.5), Inches(1.0), size=17,
         color=C_LIGHT, italic=True)
add_text(s, "3 architecturally distinct models  •  13 Python projects  •  63 structured pairs  •  4 scenarios  •  5 metrics  •  756 runs",
         Inches(0.6), Inches(5.25), Inches(12.0), Inches(0.5), size=13, color=C_GRAY)
add_text(s, "BLAZE (embedding)  ·  TRANP-CNN (CNN reranker)  ·  COOBA (GNN + adversarial)",
         Inches(0.6), Inches(5.75), Inches(12.0), Inches(0.4), size=12, color=C_GRAY)
add_text(s, "A Eashaan Rao  (CS21D002)   ·   Guide: Dr. Sridhar Chimalakonda   ·   IIT Tirupati",
         Inches(0.6), Inches(6.4), Inches(12.0), Inches(0.4), size=12, color=C_GRAY)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Progress since last DC
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Progress Since the Previous DC",
           "From a single-model pilot to a controlled three-architecture study")
footer(s)
add_rect(s, Inches(0.4), Inches(1.45), Inches(6.15), Inches(5.4),
         fill=C_LIGHT, line=C_GRAY, line_w=Pt(0.5))
add_rect(s, Inches(6.75), Inches(1.45), Inches(6.15), Inches(5.4),
         fill=RGBColor(0xE8, 0xF1, 0xFB), line=C_BLUE, line_w=Pt(1))
add_text(s, "Before review (DC 4.5)", Inches(0.6), Inches(1.6), Inches(5.7),
         Inches(0.45), size=17, bold=True, color=C_DARK)
add_bullets(s, [
    "One model: TRANP-CNN only",
    "14 Python projects, 87 ad-hoc source→target pairs, 345 runs",
    "4 research questions (incl. FAISS-ceiling diagnostic)",
    "Findings: CP-transfer improves ranking over WP-small; target LoC dominates; retrieval not the bottleneck",
], Inches(0.6), Inches(2.15), Inches(5.75), Inches(4.4), size=14)
add_text(s, "During review (this DC)", Inches(6.95), Inches(1.6), Inches(5.7),
         Inches(0.45), size=17, bold=True, color=C_BLUE)
add_bullets(s, [
    "Three architecturally distinct models: BLAZE, TRANP-CNN, COOBA — one per paradigm",
    "13 Python projects, structured 63-pair set (Groups A/B/C), 756 runs",
    "3 research questions; identical corpus / splits / candidates / baselines",
    "MRR/MAP denominator correction applied across the pipeline",
    "Full inferential battery: Wilcoxon, Cohen's d, TOST, Spearman, Kendall τ",
    "JSS manuscript drafted",
], Inches(6.95), Inches(2.15), Inches(5.75), Inches(4.6), size=14)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Study setup
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Study Setup: Three Models, One Shared Substrate",
           "Any difference in CPL benefit is attributable to architecture, not to the experimental setup")
footer(s)
models = [
    ("BLAZE", "Embedding dual-encoder;\nre-ranks by max cosine\nsimilarity over frozen\nBGE embeddings + adapters", C_GREEN),
    ("TRANP-CNN", "CNN reranker;\nparallel convolutions over\n(bug, file) pairs;\nmargin ranking loss", C_BLUE),
    ("COOBA", "GNN over AST graphs +\nBiLSTM; gradient-reversal\nadversarial domain\nadaptation (core CPL step)", C_PINK),
]
for i, (name, body, col) in enumerate(models):
    x = Inches(0.4 + i * 4.25)
    add_rect(s, x, Inches(1.4), Inches(3.95), Inches(2.7), fill=col)
    add_text(s, name, x + Inches(0.1), Inches(1.5), Inches(3.75), Inches(0.5),
             size=19, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_text(s, body, x + Inches(0.18), Inches(2.1), Inches(3.6), Inches(1.9),
             size=12, color=C_WHITE)
add_rect(s, Inches(0.4), Inches(4.35), Inches(12.5), Inches(2.4),
         fill=C_LIGHT, line=C_GRAY, line_w=Pt(0.5))
add_bullets(s, [
    "Corpus: 13 Python projects (complete set with ≥100 verified bug reports), 35K–572K LoC, 4 domains; from SWE-bench (11) + BeetleBox (2).",
    "Pairs: structured 63-pair set — Group A (30) DS×DS backbone; Group B (14) jupyterlab ↔ non-DS; Group C (19) strategic cross-domain, ≥3 sources/target.",
    "Pipeline: BAAI/bge-code-v1 embeddings → FAISS IndexFlatIP top-300 → each model re-ranks. Raw FAISS MRR = 0.029 (recall, not precision).",
    "Scenarios: WP-small (20% target), WP-large (80% target), CP-cold-start (100% source, 0% target), CP-transfer (100% source + 20% target). Test set fixed: 20% target, seed 42.",
    "63 pairs × 4 scenarios × 3 models = 756 runs; ~5–6 months wall-clock on NVIDIA L40S GPUs.",
], Inches(0.6), Inches(4.5), Inches(12.1), Inches(2.15), size=12.5, space=4)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Metrics + correction
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Evaluation Metrics & the Denominator Correction")
footer(s)
add_bullets(s, [
    "Top-K Recall (K = 1, 5, 10): fraction of bugs whose ground-truth file is in the top K — retrieval breadth.",
    "MRR: mean of 1/rank of the first correct file — ranking precision.",
    "MAP: average precision across all relevant files (bugs may have several) — ranking precision.",
    "Top-K and MRR/MAP can diverge: high Top-10 with low MRR = the correct file sits near rank 9, not rank 1.",
], Inches(0.5), Inches(1.45), Inches(12.4), Inches(2.5), size=15)
add_rect(s, Inches(0.5), Inches(4.2), Inches(12.4), Inches(2.6),
         fill=RGBColor(0xFF, 0xF3, 0xE0), line=C_ORANGE, line_w=Pt(1.2))
add_text(s, "Denominator correction (applied this review period)", Inches(0.7),
         Inches(4.35), Inches(12.0), Inches(0.45), size=16, bold=True, color=C_DARK)
add_bullets(s, [
    "Previously MRR/MAP were averaged only over test bugs that reached the reranker with a resolvable ground truth; Top-K used all test bugs.",
    "Now MRR/MAP are averaged over ALL held-out test reports (an unretrieved ground truth scores 0). Top-K is unchanged.",
    "Effect: TRANP-CNN CP-transfer MRR 0.380 → 0.214; BLAZE 0.392 → 0.388; COOBA unaffected. No retraining (per-bug ranks were archived).",
], Inches(0.7), Inches(4.85), Inches(12.0), Inches(1.9), size=13, space=4)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Overall results
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Overall Results: Mean Performance Across 63 Pairs",
           "Corrected MRR; CP-transfer is the best CPL scenario for every model")
footer(s)
table_box(s, ["Model", "WP-small", "WP-large", "CP-cold-start", "CP-transfer"], [
    ["BLAZE",     "0.254", "0.466", "0.259", "0.388"],
    ["COOBA",     "0.138", "0.192", "0.027", "0.156"],
    ["TRANP-CNN", "0.185", "0.219", "0.038", "0.214"],
], Inches(0.7), Inches(1.5), Inches(12.0), Inches(2.5), fsize=14)
add_text(s, "Values are mean MRR. FAISS retrieval baseline (all models): MRR = 0.029.",
         Inches(0.7), Inches(4.1), Inches(12.0), Inches(0.35), size=12, italic=True, color=C_GRAY)
add_bullets(s, [
    "Within-project ordering holds: WP-large > CP-transfer ≥ WP-small > CP-cold-start (COOBA cold-start collapses to the FAISS baseline).",
    "BLAZE CP-transfer 0.388 vs WP-small 0.254  —  +0.134 MRR (53% relative gain).",
    "TRANP-CNN CP-transfer 0.214 is statistically equivalent to full within-project training (WP-large 0.219; TOST p < 0.001).",
    "COOBA gains only +0.018 MRR.",
], Inches(0.7), Inches(4.5), Inches(12.0), Inches(2.3), size=13.5)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — RQ1 significance / three-tier profile
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "RQ1: CPL vs Within-Project Training Under Label Scarcity",
           "A three-tier CPL profile emerges under identical conditions")
footer(s)
table_box(s, ["Model", "Win %", "Δ MRR (mean)", "Cohen's d", "Sig."], [
    ["BLAZE",     "90.5%", "+0.134", "1.20  (large)",  "★★★"],
    ["TRANP-CNN", "68.3%", "+0.029", "0.65  (medium)", "★★★"],
    ["COOBA",     "60.3%", "+0.018", "0.27  (small)",  "★  (fails Bonferroni)"],
], Inches(0.55), Inches(1.4), Inches(6.7), Inches(2.3), fsize=11.5)
add_bullets(s, [
    "CP-transfer vs WP-small, MRR, one-sided Wilcoxon; n = 63; α/5 = 0.010.",
    "\"Practically beneficial\" (p<0.010 on MRR & MAP AND d≥0.5): BLAZE and TRANP-CNN qualify; COOBA does not.",
    "BLAZE losses (6 pairs) are within run-to-run noise; COOBA wins and losses are near-symmetric.",
], Inches(0.55), Inches(3.9), Inches(6.8), Inches(3.0), size=12.5)
img(s, "fig_gain_distribution.png", Inches(7.5), Inches(1.4), Inches(5.5), Inches(5.3))

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — RQ1 cold-start
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "RQ1: The Zero-Label Cold-Start Finding",
           "Cross-project transfer without any target annotations — architecture-dependent")
footer(s)
add_bullets(s, [
    "BLAZE CP-cold-start MRR 0.259 ≈ WP-small 0.254: statistically equivalent within ±0.05 MRR (TOST p = 0.006); no metric differs.",
    "Viable (cold-start MRR ≥ 0.20) in 73% of pairs (46/63) — a source-trained BLAZE model can be deployed with zero target labels.",
    "COOBA collapses to MRR 0.027 (the FAISS baseline); TRANP-CNN to 0.038 — viable in 0% and ~10% of pairs.",
    "The dichotomy is architectural: BLAZE's embedding similarity generalizes; learned scalar scores need target-side calibration.",
], Inches(0.5), Inches(1.4), Inches(6.6), Inches(5.3), size=13.5)
img(s, "fig_coldstart_heatmap.png", Inches(7.35), Inches(1.4), Inches(5.6), Inches(5.3))

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — RQ2 target size
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "RQ2: Target Codebase Size Moderates CPL Gain",
           "Gain concentrates in small targets; only TRANP-CNN small beats the within-project ceiling")
footer(s)
table_box(s, ["Model", "Small\n<100K", "Medium\n100–300K", "Large\n>300K"], [
    ["BLAZE  ΔMRR",     "+0.173", "+0.110", "+0.107"],
    ["TRANP-CNN  ΔMRR", "+0.045", "+0.017", "+0.021"],
    ["COOBA  ΔMRR",     "+0.025", "+0.017", "+0.006"],
], Inches(0.55), Inches(1.45), Inches(6.6), Inches(2.4), fsize=11.5)
add_bullets(s, [
    "TRANP-CNN small-target CP-transfer MRR 0.351 > WP-large 0.345 — the only size group where CPL beats the ceiling.",
    "BLAZE stays positive in every size band; COOBA medium/large centred at ~0.",
], Inches(0.55), Inches(4.05), Inches(6.7), Inches(2.7), size=13)
img(s, "fig_gain_by_size.png", Inches(7.4), Inches(1.4), Inches(5.6), Inches(5.3))

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — RQ2 negative transfer + domain
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "RQ2: Negative Transfer & Domain Alignment",
           "A clean three-model risk spectrum")
footer(s)
add_bullets(s, [
    "Negative transfer (CP-transfer MRR below WP-small by >0.01):",
    "   BLAZE 3.2% (2/63)  ·  TRANP-CNN 15.9% (10/63)  ·  COOBA 31.7% (20/63)",
    "Strongest predictor of gain = target label volume (fewer existing bugs → more gain):",
    "   BLAZE ρ = −0.77,  TRANP-CNN ρ = −0.61  (both p < 0.001)",
    "Domain-gap accuracy is non-significant for all three models (|ρ| < 0.20).",
    "Domain alignment: TRANP-CNN most sensitive (24.5 pp cross-domain win-rate drop);",
    "   BLAZE moderate (14.5 pp, still 78.8% cross-domain win); COOBA none (−2 pp).",
], Inches(0.5), Inches(1.4), Inches(6.7), Inches(5.3), size=13)
img(s, "fig_feature_corr.png", Inches(7.4), Inches(1.4), Inches(5.6), Inches(2.65))
img(s, "fig_cross_domain.png", Inches(7.4), Inches(4.15), Inches(5.6), Inches(2.65))

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — RQ3 cross-model consistency
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "RQ3: How Consistent Are CPL Outcomes Across Models?",
           "A CPL result on one architecture does not transfer to another")
footer(s)
table_box(s, ["Model pair", "Direction agreement", "Spearman ρ"], [
    ["BLAZE vs COOBA",      "41%", "0.080  (n.s.)"],
    ["BLAZE vs TRANP-CNN",  "62%", "0.582  (p < 0.001)"],
    ["COOBA vs TRANP-CNN",  "44%", "0.216  (n.s.)"],
], Inches(0.55), Inches(1.45), Inches(6.7), Inches(2.3), fsize=11.5)
add_bullets(s, [
    "Only BLAZE and TRANP-CNN gains are correlated.",
    "Source quality is not a useful selection signal: source WP-large MRR vs achieved CP-transfer MRR is ρ ≈ 0 (BLAZE), −0.21 (TRANP-CNN), and a significant negative −0.30 (COOBA).",
], Inches(0.55), Inches(3.95), Inches(6.7), Inches(2.8), size=12.5)
img(s, "fig_cross_model_scatter.png", Inches(7.4), Inches(1.6), Inches(5.6), Inches(4.6))

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — RQ3 source selection
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "RQ3: Can Metadata Guide Source Selection?",
           "No heuristic ranks sources reliably — but a 'good enough' source recovers most of the oracle")
footer(s)
table_box(s, ["Model", "Hit@1 most-bugs", "Hit@1 domain-gap", "% of oracle MRR"], [
    ["BLAZE",     "15.4%", "46.2%", "88.6–92.7%"],
    ["COOBA",     "38.5%", "23.1%", "78.8–79.2%"],
    ["TRANP-CNN", "30.8%", "23.1%", "86.7–89.8%"],
], Inches(0.55), Inches(1.45), Inches(6.8), Inches(2.3), fsize=11)
add_bullets(s, [
    "No heuristic reaches Hit@1 > 46% or Kendall τ > 0.2 for any model.",
    "Yet the selected source still recovers 79–93% of oracle CP-transfer MRR — the cost of an imperfect choice is modest.",
    "Motivates a learned source-selection meta-model (Objective 3).",
], Inches(0.55), Inches(3.95), Inches(6.8), Inches(2.8), size=12.5)
img(s, "fig_ss_quality_blaze.png", Inches(7.5), Inches(1.5), Inches(5.4), Inches(4.7))

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Illustrative cases
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Illustrative Cases", "Per-scenario MRR: WPS / WPL / CPC / CPT")
footer(s)
add_text(s, "Case 1 — BLAZE cold-start viability:  prefect → lightning-ai  (within-domain DS/ML)",
         Inches(0.5), Inches(1.3), Inches(12.3), Inches(0.35), size=13, bold=True, color=C_DARK)
table_box(s, ["Model", "WPS", "WPL", "CPC", "CPT"], [
    ["BLAZE", "0.322", "0.542", "0.472", "0.496"],
    ["TRANP-CNN", "0.344", "0.399", "0.061", "0.366"],
    ["COOBA", "0.361", "0.416", "0.046", "0.364"],
], Inches(0.5), Inches(1.65), Inches(6.4), Inches(1.55), fsize=10.5)
add_text(s, "BLAZE cold-start (0.472) beats its own WP-small (0.322) with zero target labels.",
         Inches(7.1), Inches(2.0), Inches(5.8), Inches(1.0), size=11.5, italic=True, color=C_GRAY)
add_text(s, "Case 2 — Architecture divergence on one pair:  ansible → jupyterlab  (cross-domain)",
         Inches(0.5), Inches(3.35), Inches(12.3), Inches(0.35), size=13, bold=True, color=C_DARK)
table_box(s, ["Model", "WPS", "WPL", "CPC", "CPT"], [
    ["BLAZE", "0.021", "0.487", "0.280", "0.377"],
    ["COOBA", "0.170", "0.184", "0.057", "0.089"],
    ["TRANP-CNN", "0.327", "0.380", "0.065", "0.400"],
], Inches(0.5), Inches(3.7), Inches(6.4), Inches(1.55), fsize=10.5)
add_text(s, "Same source: one of BLAZE's best gains (+0.355) and one of COOBA's worst (−0.081).",
         Inches(7.1), Inches(4.05), Inches(5.8), Inches(1.0), size=11.5, italic=True, color=C_GRAY)
add_text(s, "Case 3 — CP-transfer beats full within-project training:  jupyterlab → scikit-learn",
         Inches(0.5), Inches(5.4), Inches(12.3), Inches(0.35), size=13, bold=True, color=C_DARK)
table_box(s, ["Model", "WPS", "WPL", "CPC", "CPT"], [
    ["BLAZE", "0.236", "0.510", "0.141", "0.380"],
    ["COOBA", "0.016", "0.036", "0.005", "0.031"],
    ["TRANP-CNN", "0.006", "0.039", "0.005", "0.095"],
], Inches(0.5), Inches(5.75), Inches(6.4), Inches(1.3), fsize=10.5)
add_text(s, "TRANP-CNN CPT 0.095 > WP-large 0.039: joint training beats 4× more target-only data.",
         Inches(7.1), Inches(6.0), Inches(5.8), Inches(1.0), size=11.5, italic=True, color=C_GRAY)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Contributions
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Contributions of the Multi-Model Study")
footer(s)
add_bullets(s, [
    "First matched-budget, three-architecture CPL benchmark for bug localization — identical corpus, splits, candidate pool, and baselines.",
    "CPL benefit is architecture-dependent: large effect for BLAZE (d = 1.20), medium for TRANP-CNN (d = 0.65), marginal for COOBA (d = 0.27).",
    "Zero-label cold-start is viable for an embedding model (BLAZE) in 73% of pairs, but not for the scalar-ranking models.",
    "Quantified negative-transfer rates per paradigm (3.2% / 15.9% / 31.7%) and identified target label volume as the dominant predictor of gain.",
    "Established bounds on data-driven source selection: no heuristic ranks sources reliably, but a good-enough source recovers 79–93% of oracle.",
    "Low cross-model agreement (41–62%): CPL findings from one tool should not be generalized to another.",
], Inches(0.5), Inches(1.45), Inches(12.4), Inches(5.3), size=14)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Limitations & threats
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Limitations & Threats to Validity")
footer(s)
add_bullets(s, [
    "Re-implementation fidelity: TRANP-CNN and COOBA were re-implemented on a shared BGE backbone; three deliberate COOBA departures — its weak/negative result is provisional.",
    "Shared FAISS Recall@300 varies 31–100% (median ~70%); three targets < 50% — conflates retrieval limits with architecture on large targets (within-model CPT vs WPS unaffected).",
    "BLAZE was re-run restricted to the shared FAISS top-300 shortlist (previously it re-ranked the full file snapshot), putting all three models on identical footing.",
    "Single random seed: identically configured re-runs differ by 0.05–0.07 MRR (up to 0.25), the same order as mean CPL gains; median-run re-test preserves all RQ1 conclusions.",
    "Python-only, 13 popular OSS projects; domain labels single-rater; BGE likely pre-trained on these repos (star-count proxy: no significant correlation with cold-start MRR).",
], Inches(0.5), Inches(1.45), Inches(12.4), Inches(5.3), size=13.5)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Objective 2 + next steps
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
header_bar(s, "Objective 2 (Polyglot Bug Localization) & Next Steps")
footer(s)
add_text(s, "Objective 2 — \"Mono-fix or Poly-fix?\"  (descriptive core complete)",
         Inches(0.5), Inches(1.35), Inches(12.3), Inches(0.4), size=15, bold=True, color=C_DARK)
add_bullets(s, [
    "Unit = the fix, labelled from its fix-commit file extensions. Two failure kinds: architectural (no parser for YAML/JSON/SQL) vs semantic (readable but poorly ranked).",
    "No retrieval stage — BM25 / BLAZE / FLIM rank the whole pre-fix snapshot; within-project only; BeetleBox, 6 projects, 3,594 bugs (2,561 mono / 1,033 poly).",
    "RQ1: 27.4% of fixes are Poly-fix (range 8.2–41.9%).  RQ2: FLIM falls back to whole-file text on YAML/XML/SQL/JSON; COOBA's parser returns a 1-node graph.",
    "RQ4 (baseline): BM25 does NOT struggle more on Poly-fix — MRR 0.219 (poly) vs 0.181 (mono); truncation plausible (19.5% of failure-set files start beyond 512 tokens).",
], Inches(0.5), Inches(1.75), Inches(12.4), Inches(2.6), size=12.5, space=4)
add_text(s, "Next steps", Inches(0.5), Inches(4.55), Inches(12.3), Inches(0.4),
         size=15, bold=True, color=C_BLUE)
add_bullets(s, [
    "Finalize & submit the multi-model CPL manuscript (JSS/EMSE); extend Phase 1 to Java/Kotlin/JS to test language-invariance of the architecture profile.",
    "Objective 2: complete BLAZE + FLIM full-snapshot runs (~1–1.5 GPU-days) and the 50-bug agreement check; then the inversion test (RQ4). Venue: FSE 2027 → ICPC 2027 → IST.",
    "Objective 3: adaptive, meta-learning CPL pipeline informed by target-viability conditions and the failure of naive source selection.",
], Inches(0.5), Inches(4.95), Inches(12.4), Inches(2.0), size=12.5, space=4)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — Key takeaways
# ═══════════════════════════════════════════════════════════════════════════════
s = slide()
add_rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, fill=C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.18), SLIDE_H, fill=C_PINK)
add_text(s, "Key Takeaways", Inches(0.6), Inches(0.5), Inches(11), Inches(0.8),
         size=32, bold=True, color=C_WHITE)
cards = [
    "CPL beats limited within-project training — but the size of the benefit is set by the model architecture, not by the paradigm.",
    "BLAZE (embedding): large, low-risk gains; zero-label cold-start viable in 73% of pairs.",
    "TRANP-CNN (CNN): medium gains; CP-transfer matches full within-project training; ~1 in 6 pairs see negative transfer.",
    "COOBA (GNN): marginal, near-symmetric risk; cold-start fails; ~1 in 3 pairs see negative transfer.",
    "Predict gain from target label volume, not domain similarity. Source selection: any 'good enough' source recovers 79–93% of oracle.",
    "Cross-model agreement is only 41–62% — do not generalize a CPL result across architectures.",
]
add_bullets(s, cards, Inches(0.7), Inches(1.6), Inches(12.0), Inches(5.2),
            size=16, color=C_WHITE, bullet="▪  ", space=12)
add_text(s, "Thank you — questions welcome.", Inches(0.7), Inches(6.7),
         Inches(11), Inches(0.5), size=14, italic=True, color=C_GRAY)

# ── Save ─────────────────────────────────────────────────────────────────────
prs.save(PPTX_PATH)
print(f"✓ Saved: {PPTX_PATH}  ({len(list(prs.slides))} slides)")


# ═══════════════════════════════════════════════════════════════════════════════
# SPEAKER NOTES
# ═══════════════════════════════════════════════════════════════════════════════
notes = """# DC 5 — Speaker Notes
Multi-model Cross-Project Bug Localization (CPL), Phase 1. Deck: `DC5_Presentation.pptx`.
Numbers are the corrected multi-model results (`results/obj1_experimental_results_corrected.csv`,
`journal_draft/main.tex`). Mirrors `DC_5_Report/main.tex`.

## Slide 1 — Title
Frame the review: the previous DC reported a single-model (TRANP-CNN) pilot. This DC extends
that to three architecturally distinct SOTA models under identical conditions, so the question
becomes not just "does CPL work?" but "does the answer depend on the model?".

## Slide 2 — Progress since the previous DC
Left = what was presented at DC 4.5 (one model, 87 ad-hoc pairs, 345 runs, 4 RQs). Right = the
work done during this review period: three models one-per-paradigm, a principled 63-pair
design, 756 runs, an MRR/MAP denominator correction, the full statistical battery, and a
drafted JSS manuscript. Everything downstream in the deck is the "right column".

## Slide 3 — Study setup
One model per paradigm: BLAZE (embedding dual-encoder), TRANP-CNN (CNN reranker), COOBA (GNN +
adversarial adaptation). The point of the shared substrate — same corpus, same splits, same
FAISS top-300 candidate pool, same baselines — is that any difference in CPL benefit is
attributable to architecture. 13 Python projects (the complete set passing the ≥100-bug
filter), 63 structured pairs in three groups, 756 runs.

## Slide 4 — Metrics & the denominator correction
Explain the five metrics and why Top-K and MRR/MAP can diverge. Then the correction: MRR/MAP
were previously averaged only over bugs that reached the reranker with a resolvable ground
truth; now averaged over all test bugs (unretrieved GT scores 0). It lowered TRANP-CNN
CP-transfer MRR from 0.380 to 0.214 and BLAZE from 0.392 to 0.388; COOBA was never affected.
No retraining — per-bug ranks were archived.

## Slide 5 — Overall results
Read the table by column. The within-project ordering holds for every model. BLAZE gains most
in absolute terms (+0.134 MRR); TRANP-CNN's CP-transfer is statistically equivalent to
WP-large (full within-project training); COOBA barely moves and its cold-start is at the FAISS
floor.

## Slide 6 — RQ1: significance and the three-tier profile
CP-transfer vs WP-small, one-sided Wilcoxon, n = 63, Bonferroni α/5 = 0.010. BLAZE: large
effect, significant on all five metrics. TRANP-CNN: medium effect, significant on
MRR/MAP/Top-5. COOBA: small effect, nominally significant but nothing survives Bonferroni.
Under the post-hoc "practically beneficial" bar (p<0.010 on MRR & MAP and d≥0.5), BLAZE and
TRANP-CNN qualify, COOBA does not. The gain-distribution figure shows the shape: BLAZE strongly
right-skewed, COOBA near-symmetric.

## Slide 7 — RQ1: the cold-start finding
The practically important result. A source-trained BLAZE model applied with zero target labels
is statistically equivalent to training on 20% of the target's own data (TOST within ±0.05
MRR), and is viable (MRR ≥ 0.20) in 73% of pairs. The two scalar-ranking models fail
cold-start — COOBA at the FAISS baseline, TRANP-CNN barely above it. The dichotomy is
architectural: embedding similarity generalizes to unseen targets; learned scalar scores need
target-side calibration.

## Slide 8 — RQ2: target codebase size
CPL gain concentrates in small targets for every model. The standout: TRANP-CNN CP-transfer on
small targets (MRR 0.351) exceeds WP-large (0.345) — the only size group in the whole study
where cross-project transfer beats the within-project ceiling. COOBA's medium and large groups
sit at roughly zero gain.

## Slide 9 — RQ2: negative transfer & domain alignment
A clean three-model spectrum of negative-transfer risk: 3.2% / 15.9% / 31.7%. The strongest
predictor of gain is target label volume — and it is negative: the fewer bugs a project
already has, the more it gains (BLAZE ρ = −0.77, TRANP-CNN −0.61). Domain-gap accuracy predicts
nothing (|ρ| < 0.20). TRANP-CNN is the most domain-sensitive model; COOBA shows essentially no
domain effect.

## Slide 10 — RQ3: cross-model consistency
Direction agreement is 41–62%. Only BLAZE and TRANP-CNN gains are correlated (ρ = 0.582). So a
CPL result measured on one architecture does not transfer to another. Also: picking the source
on which a model is already best is useless (BLAZE) or counter-productive (COOBA, significant
negative ρ = −0.30).

## Slide 11 — RQ3: source selection heuristics
Neither the most-bugs nor the domain-gap heuristic ranks candidate sources reliably (Hit@1
≤ 46%, τ < 0.2 for all models). But the practical cost is modest: a "good enough" source still
recovers 79–93% of the oracle's CP-transfer MRR. This is the motivation for a learned
source-selection meta-model in Objective 3.

## Slide 12 — Illustrative cases
Case 1 (prefect→lightning): BLAZE cold-start beats its own WP-small with zero labels; the
scalar models collapse. Case 2 (ansible→jupyterlab): the same source is one of BLAZE's best
gains and one of COOBA's worst — no model-agnostic verdict is possible. Case 3
(jupyterlab→scikit-learn): TRANP-CNN CP-transfer beats WP-large — joint source+20%-target
training beats 4× more target-only data.

## Slide 13 — Contributions
The headline is the first matched-budget three-architecture CPL benchmark, and the finding
that CPL benefit is architecture-dependent. Plus the cold-start viability result, per-paradigm
negative-transfer rates, source-selection bounds, and the low cross-model agreement.

## Slide 14 — Limitations & threats
Be candid: re-implementation fidelity (COOBA's negative result is provisional); the shared
Recall@300 ceiling on large targets; BLAZE's full-snapshot evaluation (a candidate-restricted
re-run is planned); single-seed run-to-run variance; Python-only corpus, single-rater domain
labels, likely pre-training exposure.

## Slide 15 — Objective 2 & next steps
Objective 2's descriptive core is done: 27.4% Poly-fix prevalence; FLIM and COOBA have a real
architectural limit on non-code files; classical BM25 does not struggle more on Poly-fix
(0.219 vs 0.181 MRR). Next: finish the BLAZE + FLIM full-snapshot runs and the agreement
check, then the inversion test that gates the venue (FSE 2027 → ICPC 2027 → IST). Objective 1
next steps: submit the manuscript, extend to other languages. Objective 3: the adaptive
meta-learning CPL pipeline.

## Slide 16 — Key takeaways
Six cards. The one-line version: CPL beats limited within-project training, but how much
depends on the architecture — large and low-risk for an embedding model, medium for a CNN
reranker, marginal and risky for a GNN. Predict gain from how little labelled data the target
has, not from domain similarity. And never assume a CPL result generalizes from one model to
another.

## Common DC / viva questions
**Why one model per paradigm rather than several per family?** The design isolates the
architecture bundle (input representation + ranking mechanism + adaptation mechanism). With one
per paradigm this is not a within-paradigm replication — stated as a limitation — but two
architecturally distinct models (BLAZE, TRANP-CNN) both showing benefit argues against it being
one model's inductive bias.

**Isn't BLAZE favoured because it re-scores its own retrieval signal?** Yes — the shared FAISS
pool is built from the same BGE embeddings BLAZE re-ranks with (via MaxSim), unlike TRANP-CNN
and COOBA. This is flagged in threats; the planned candidate-restricted BLAZE re-run addresses
it.

**Why is COOBA so weak?** Two candidate explanations: representation quality (AST + GloVe vs
BGE) and model age. Three observations favour the evaluation-mechanism account (COOBA needs
target representations during adversarial training, so cold-start fails; it overfits source AST
structure, so good within-project sources transfer worse). It remains an interpretation.

**Why Wilcoxon / Spearman / Kendall rather than parametric tests?** Paired observations across a
corpus spanning orders of magnitude in size and bug count; improvement distributions are not
normal; ranks are the robust choice.

**Is the medium TRANP-CNN effect reliable per deployment?** It is detectable at n = 63, but
single-seed run-to-run spread reaches 0.25 MRR, so individual-pair reproducibility should not
be assumed from the aggregate effect size — re-testing against each target's median WP-small
preserves the conclusion.
"""

with open(NOTES_PATH, "w") as f:
    f.write(notes)
print(f"✓ Saved: {NOTES_PATH}")
print("✓ Done.")
