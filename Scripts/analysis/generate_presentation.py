"""
generate_presentation.py
─────────────────────────
Generates a .pptx slide deck for the TRANP-CNN Phase 1 CPL results.
Also writes a detailed speaker-notes document as a .md file.

Output
──────
  results/presentation/CPL_Phase1_Results.pptx
  results/presentation/SPEAKER_NOTES.md

Run
───
    python Scripts/analysis/generate_presentation.py
"""

import os
import textwrap
from copy import deepcopy

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/presentation"
SHOWCASE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/results/showcase_pairs"
IMAGES_DIR   = "/home/cs21d002_eashaan/PhD/Objective1/results/images"

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_DARK   = RGBColor(0x1A, 0x23, 0x3A)   # dark navy
C_BLUE   = RGBColor(0x21, 0x96, 0xF3)   # material blue
C_GREEN  = RGBColor(0x4C, 0xAF, 0x50)   # material green
C_PINK   = RGBColor(0xE9, 0x1E, 0x63)   # CP-transfer pink
C_ORANGE = RGBColor(0xFF, 0x98, 0x00)   # cold-start orange
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT  = RGBColor(0xF5, 0xF5, 0xF5)
C_GRAY   = RGBColor(0x90, 0x90, 0x90)
C_ACCENT = RGBColor(0x7C, 0x4D, 0xFF)   # purple accent

os.makedirs(OUT_DIR, exist_ok=True)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]   # completely blank


# ── Low-level helpers ──────────────────────────────────────────────────────────

def rgb(r, g, b):
    return RGBColor(r, g, b)


def add_rect(slide, l, t, w, h, fill=None, line=None, line_w=Pt(0)):
    shape = slide.shapes.add_shape(1, l, t, w, h)   # MSO_SHAPE_TYPE.RECTANGLE = 1
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


def add_text(slide, text, l, t, w, h,
             size=18, bold=False, color=C_DARK, align=PP_ALIGN.LEFT,
             wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_bullet_box(slide, items, l, t, w, h,
                   size=16, color=C_DARK, indent_char="▸ ",
                   line_spacing=1.2):
    """Add a multi-line bullet text box."""
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = indent_char + item
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return txBox


def header_bar(slide, title, subtitle=None):
    """Dark navy header bar with white title."""
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.15), fill=C_DARK)
    add_text(slide, title,
             Inches(0.35), Inches(0.12), Inches(12.0), Inches(0.65),
             size=26, bold=True, color=C_WHITE)
    if subtitle:
        add_text(slide, subtitle,
                 Inches(0.35), Inches(0.75), Inches(12.0), Inches(0.38),
                 size=14, color=C_LIGHT, italic=True)


def accent_line(slide, y=Inches(1.15)):
    add_rect(slide, Inches(0), y, SLIDE_W, Inches(0.05), fill=C_BLUE)


def footer(slide, text="TRANP-CNN Phase 1 — CPL Results  |  PhD Study"):
    add_rect(slide, Inches(0), Inches(7.15), SLIDE_W, Inches(0.35), fill=C_DARK)
    add_text(slide, text,
             Inches(0.3), Inches(7.17), Inches(10), Inches(0.3),
             size=10, color=C_GRAY)


def table_box(slide, headers, rows, l, t, w, h,
              header_fill=C_DARK, row_fills=None):
    """Draw a simple table as coloured rectangles."""
    n_cols = len(headers)
    n_rows = len(rows)
    col_w  = w / n_cols
    row_h  = h / (n_rows + 1)

    # Header row
    for ci, hdr in enumerate(headers):
        add_rect(slide, l + ci*col_w, t, col_w - Inches(0.02), row_h,
                 fill=header_fill)
        add_text(slide, hdr,
                 l + ci*col_w + Inches(0.05), t + Inches(0.04),
                 col_w - Inches(0.1), row_h - Inches(0.04),
                 size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # Data rows
    for ri, row in enumerate(rows):
        fill = (row_fills[ri] if row_fills and ri < len(row_fills)
                else (C_LIGHT if ri % 2 == 0 else C_WHITE))
        for ci, cell in enumerate(row):
            add_rect(slide, l + ci*col_w, t + (ri+1)*row_h,
                     col_w - Inches(0.02), row_h - Inches(0.02),
                     fill=fill,
                     line=C_GRAY, line_w=Pt(0.5))
            add_text(slide, str(cell),
                     l + ci*col_w + Inches(0.05),
                     t + (ri+1)*row_h + Inches(0.03),
                     col_w - Inches(0.1), row_h - Inches(0.06),
                     size=12, color=C_DARK, align=PP_ALIGN.CENTER)


def add_image_safe(slide, path, l, t, w, h):
    if os.path.exists(path):
        slide.shapes.add_picture(path, l, t, w, h)
    else:
        add_rect(slide, l, t, w, h, fill=C_LIGHT, line=C_GRAY, line_w=Pt(1))
        add_text(slide, f"[Image not found]\n{os.path.basename(path)}",
                 l + Inches(0.1), t + Inches(0.2), w - Inches(0.2), h - Inches(0.4),
                 size=11, color=C_GRAY, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)

add_rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, fill=C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.18), SLIDE_H, fill=C_BLUE)
add_rect(s, Inches(0), Inches(5.2), SLIDE_W, Inches(0.06), fill=C_PINK)

add_text(s, "Cross-Project Bug Localisation",
         Inches(0.5), Inches(1.2), Inches(12.0), Inches(1.1),
         size=38, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
add_text(s, "TRANP-CNN Phase 1 — Experimental Results",
         Inches(0.5), Inches(2.3), Inches(11.0), Inches(0.7),
         size=24, color=C_BLUE, align=PP_ALIGN.LEFT)
add_text(s, "Can a bug localisation model trained on one project\ntransfer to another?",
         Inches(0.5), Inches(3.1), Inches(10.0), Inches(0.9),
         size=18, color=C_LIGHT, italic=True, align=PP_ALIGN.LEFT)

add_text(s, "20 Projects  •  94 Source→Target Pairs  •  4 Scenarios  •  5 Metrics",
         Inches(0.5), Inches(5.4), Inches(11.0), Inches(0.5),
         size=14, color=C_GRAY, align=PP_ALIGN.LEFT)
add_text(s, "PhD Study  |  TRANP-CNN Reranking Model  |  Python & Java Codebases",
         Inches(0.5), Inches(5.9), Inches(11.0), Inches(0.4),
         size=13, color=C_GRAY, align=PP_ALIGN.LEFT)

footer(s, "")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Research Problem & Setup
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Research Problem & Experimental Setup",
           "What are we testing and how?")
accent_line(s)
footer(s)

add_bullet_box(s, [
    "Bug localisation: given a bug report, rank source files by likelihood of containing the bug",
    "Challenge: collecting labelled bug data (bug report → file mappings) is expensive",
    "CPL hypothesis: a model trained on project A can transfer to project B with limited target data",
], Inches(0.4), Inches(1.3), Inches(6.0), Inches(1.8), size=15)

# Scenario table
table_box(s,
    headers=["Scenario", "Training Data", "Research Question"],
    rows=[
        ["WP-small",       "20% target bugs only",          "Baseline: limited within-project"],
        ["WP-large",       "80% target bugs only",          "Upper bound: maximum within-project"],
        ["CP-cold-start",  "100% source, 0% target",        "Can zero-shot transfer work?"],
        ["CP-transfer",    "100% source + 20% target",      "Does CPL beat WP-small?"],
    ],
    l=Inches(0.4), t=Inches(3.2), w=Inches(12.5), h=Inches(2.6),
    row_fills=[
        rgb(0xE3,0xF2,0xFD),  # WP-small
        rgb(0xE8,0xF5,0xE9),  # WP-large
        rgb(0xFF,0xF3,0xE0),  # cold-start
        rgb(0xFC,0xE4,0xEC),  # CP-transfer
    ]
)
add_text(s, "Pipeline: BAAI/bge-code-v1 embeddings → FAISS top-300 retrieval → TRANP-CNN reranking",
         Inches(0.4), Inches(6.0), Inches(12.0), Inches(0.4),
         size=13, color=C_ACCENT, italic=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Metrics Explained
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Evaluation Metrics", "Five complementary lenses on performance")
accent_line(s)
footer(s)

metrics_data = [
    ("Top-1 / Top-5 / Top-10", "Recall@K",
     "Fraction of bugs where the GT file appears\nin rank position ≤ K.\nTop-10 = 0.80 means 80% of bugs\nfound in top 10 candidates."),
    ("MAP", "Mean Average\nPrecision",
     "Precision averaged over all rank positions\nwhere a relevant file appears.\nHigher = better ranked, not just retrieved.\nRange: (0, 1]."),
    ("MRR", "Mean Reciprocal\nRank",
     "Mean of 1/rank across all bugs.\nMRR=0.5 → GT file at rank 2 on average.\nMRR=0.25 → rank 4 on average.\nRange: (0, 1]."),
]

for i, (name, short, desc) in enumerate(metrics_data):
    x = Inches(0.3 + i * 4.35)
    add_rect(s, x, Inches(1.3), Inches(4.1), Inches(4.8),
             fill=C_LIGHT, line=C_GRAY, line_w=Pt(0.8))
    add_rect(s, x, Inches(1.3), Inches(4.1), Inches(0.55),
             fill=[C_BLUE, C_GREEN, C_PINK][i])
    add_text(s, name, x + Inches(0.1), Inches(1.33),
             Inches(3.9), Inches(0.5),
             size=14, bold=True, color=C_WHITE)
    add_text(s, short, x + Inches(0.15), Inches(1.95),
             Inches(3.8), Inches(0.55),
             size=13, bold=True, color=C_DARK)
    add_text(s, desc, x + Inches(0.15), Inches(2.5),
             Inches(3.8), Inches(3.0),
             size=12, color=C_DARK)

add_text(s,
    "Key insight: Top-K recall and MAP/MRR can disagree. "
    "A model may retrieve the file (high Top-10) but rank it 9th, not 1st (low MRR). "
    "We report all five metrics throughout.",
    Inches(0.4), Inches(6.35), Inches(12.5), Inches(0.55),
    size=13, color=C_ACCENT, italic=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Overall Results
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Overall Results — All 5 Metrics",
           "Mean across 92–94 source→target pairs")
accent_line(s)
footer(s)

table_box(s,
    headers=["Scenario", "Top-1", "Top-5", "Top-10", "MAP", "MRR"],
    rows=[
        ["WP-large",      "0.081", "0.200", "0.258", "0.218", "0.248"],
        ["CP-transfer",   "0.064", "0.189", "0.251", "0.197", "0.226"],
        ["WP-small",      "0.056", "0.167", "0.223", "0.176", "0.202"],
        ["CP-cold-start", "0.005", "0.021", "0.040", "0.038", "0.042"],
    ],
    l=Inches(0.4), t=Inches(1.3), w=Inches(12.5), h=Inches(2.8),
    row_fills=[
        rgb(0xE8,0xF5,0xE9),
        rgb(0xFC,0xE4,0xEC),
        rgb(0xE3,0xF2,0xFD),
        rgb(0xFF,0xF3,0xE0),
    ]
)

add_bullet_box(s, [
    "CP-transfer consistently sits between WP-small and WP-large across ALL metrics",
    "Top-K gap to WP-large is small (0.007–0.017) — recall is nearly matched",
    "MAP/MRR gap is larger (0.021–0.022) — ranking precision still lags",
    "CP-cold-start is very low — zero-shot transfer without fine-tuning is limited",
], Inches(0.4), Inches(4.35), Inches(12.2), Inches(2.2), size=15)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Finding 1: CP-transfer vs WP-small (Wilcoxon)
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Finding 1: CPL Significantly Improves Ranking Quality",
           "CP-transfer vs WP-small — Wilcoxon signed-rank test")
accent_line(s)
footer(s)

# Left: stat test explanation box
add_rect(s, Inches(0.3), Inches(1.3), Inches(5.4), Inches(4.5),
         fill=rgb(0xE8,0xEA,0xF0), line=C_ACCENT, line_w=Pt(1.2))
add_text(s, "📐  Wilcoxon Signed-Rank Test",
         Inches(0.45), Inches(1.38), Inches(5.1), Inches(0.45),
         size=14, bold=True, color=C_ACCENT)
add_text(s,
    "A non-parametric paired statistical test.\n\n"
    "Use case: we have 92 paired observations\n"
    "(CP-transfer MRR, WP-small MRR) for the\n"
    "same project pairs. We ask: is CP-transfer\n"
    "systematically higher?\n\n"
    "Null hypothesis H₀: no difference.\n"
    "Alternative H₁: CP-transfer > WP-small.\n\n"
    "p < 0.05 → reject H₀ → difference is real.\n"
    "★ ★ ★ = p < 0.001  ★ ★ = p < 0.01  ★ = p < 0.05  ns = not significant",
    Inches(0.45), Inches(1.85), Inches(5.1), Inches(3.8),
    size=12, color=C_DARK)

# Right: results table
table_box(s,
    headers=["Metric", "Win Rate", "Mean Gain", "p-value"],
    rows=[
        ["Top-1",  "37.0%",  "+0.007", "0.118  (ns)"],
        ["Top-5",  "43.5%",  "+0.021", "0.007  ★★"],
        ["Top-10", "47.8%",  "+0.026", "0.001  ★★"],
        ["MAP",    "64.1%",  "+0.020", "0.006  ★★"],
        ["MRR",    "67.4%",  "+0.022", "0.005  ★★"],
    ],
    l=Inches(6.0), t=Inches(1.3), w=Inches(6.9), h=Inches(3.0),
    row_fills=[
        C_LIGHT, C_LIGHT,
        rgb(0xE8,0xF5,0xE9), rgb(0xE8,0xF5,0xE9), rgb(0xE8,0xF5,0xE9),
    ]
)
add_bullet_box(s, [
    "MAP and MRR: CP-transfer wins significantly (p < 0.01) in 64–67% of pairs",
    "Top-1 improvement is NOT significant — CPL improves ranking, not pinpoint precision",
    "Interpretation: cross-project pre-training gives a better relative ordering of files",
], Inches(6.0), Inches(4.5), Inches(6.9), Inches(2.3), size=14)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Finding 2: CP-transfer vs WP-large
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Finding 2: CP-transfer Matches WP-large on Recall",
           "Top-K and MAP/MRR tell different stories")
accent_line(s)
footer(s)

table_box(s,
    headers=["Metric", "CP-transfer ≥ WP-large", "Within 10% of WP-large", "Mean gap"],
    rows=[
        ["Top-1",  "54.3% (50/92)", "62.0% (57/92)", "−0.018"],
        ["Top-5",  "52.2% (48/92)", "57.6% (53/92)", "−0.012"],
        ["Top-10", "53.3% (49/92)", "63.0% (58/92)", "−0.010"],
        ["MAP",    "30.4% (28/92)", "41.3% (38/92)", "−0.023"],
        ["MRR",    "32.6% (30/92)", "47.8% (44/92)", "−0.025"],
    ],
    l=Inches(0.4), t=Inches(1.3), w=Inches(12.5), h=Inches(3.0),
    row_fills=[
        rgb(0xE8,0xF5,0xE9), rgb(0xE8,0xF5,0xE9), rgb(0xE8,0xF5,0xE9),
        rgb(0xFF,0xF3,0xE0), rgb(0xFF,0xF3,0xE0),
    ]
)

add_bullet_box(s, [
    "For RECALL (Top-K): CP-transfer matches or beats WP-large in 52–54% of pairs",
    "For RANKING (MAP/MRR): WP-large retains advantage (~70% of pairs)",
    "Practical implication: for tools showing a shortlist of 10 files, CP-transfer ≈ WP-large",
    "CP-transfer uses the same 20% target data as WP-small — no extra annotation cost",
], Inches(0.4), Inches(4.55), Inches(12.2), Inches(2.15), size=15)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Finding 3: CPL gain for data-scarce targets
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Finding 3: CPL Benefit is 3–4× Larger for Data-Scarce Targets",
           "CPL is most valuable exactly where it is most needed")
accent_line(s)
footer(s)

table_box(s,
    headers=["Target group", "Top-5 gain", "Top-10 gain", "MAP gain", "MRR gain"],
    rows=[
        ["Few bugs (≤228 bugs, n=48)", "+0.034", "+0.043", "+0.029", "+0.032"],
        ["Many bugs (>228 bugs, n=46)", "+0.007", "+0.007", "+0.011", "+0.012"],
        ["Ratio (few / many)",          "4.9×",   "6.1×",   "2.6×",   "2.7×"],
    ],
    l=Inches(0.4), t=Inches(1.3), w=Inches(12.5), h=Inches(2.3),
    row_fills=[
        rgb(0xFC,0xE4,0xEC),
        rgb(0xE3,0xF2,0xFD),
        rgb(0xFF,0xF3,0xE0),
    ]
)

add_bullet_box(s, [
    "Split at median target bug count (228 bugs) — roughly equal group sizes",
    "CPL gain = CP-transfer MRR − WP-small MRR for the same pair",
    "For data-scarce targets: mean Top-10 gain = +0.043 (significant improvement)",
    "For data-rich targets: gain drops to +0.007 — WP-large is already adequate",
    "Deployment recommendation: prefer CPL when target has < 200 historical bugs",
], Inches(0.4), Inches(3.8), Inches(12.2), Inches(2.85), size=15)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Finding 4: Spearman Correlation / Target Properties
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Finding 4: Target Codebase Size is the Dominant Predictor",
           "Spearman correlation analysis across all 5 metrics")
accent_line(s)
footer(s)

# Left: Spearman explanation
add_rect(s, Inches(0.3), Inches(1.3), Inches(5.3), Inches(4.6),
         fill=rgb(0xE8,0xEA,0xF0), line=C_ACCENT, line_w=Pt(1.2))
add_text(s, "📐  Spearman Rank Correlation (ρ)",
         Inches(0.45), Inches(1.38), Inches(5.0), Inches(0.45),
         size=14, bold=True, color=C_ACCENT)
add_text(s,
    "Measures how monotonically two variables\n"
    "are related — without assuming linearity.\n\n"
    "ρ = +1.0 → perfect positive relationship\n"
    "ρ = −1.0 → perfect negative relationship\n"
    "ρ =  0.0 → no relationship\n\n"
    "Significance: p < 0.05 needed to claim the\n"
    "correlation is not due to chance.\n\n"
    "Example: ρ(tgt_LoC, Top-10) = −0.855***\n"
    "means: as target codebase gets larger,\n"
    "Top-10 recall drops very consistently.",
    Inches(0.45), Inches(1.85), Inches(5.0), Inches(3.9),
    size=12, color=C_DARK)

# Right: table
table_box(s,
    headers=["Feature", "Top-10 ρ", "MAP ρ", "MRR ρ", "Sig."],
    rows=[
        ["tgt_LoC (target size)",       "−0.855", "−0.652", "−0.667", "★★★"],
        ["tgt_bug_verbosity",           "+0.333", "+0.383", "+0.297", "★★/★★★"],
        ["tgt_n_bugs",                  "+0.134", "+0.311", "+0.319", "ns / ★★"],
        ["src_LoC (source size)",       "+0.274", "+0.252", "+0.259", "★★ / ★"],
        ["src_n_bugs (source bugs)",    "−0.045", "+0.016", "+0.070", "ns all"],
        ["domain_gap",                  "+0.116", "+0.012", "+0.038", "ns all"],
    ],
    l=Inches(5.85), t=Inches(1.3), w=Inches(7.1), h=Inches(3.1),
)

add_bullet_box(s, [
    "tgt_LoC dominates: ρ = −0.855 for Top-10 — large codebase = harder localisation",
    "Source features (bugs, LoC) are weak / non-significant for MRR and MAP",
    "Domain gap has NO significant predictive power — dissimilar projects still transfer",
], Inches(5.85), Inches(4.6), Inches(7.1), Inches(2.0), size=14)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Finding 5: FAISS Ceiling
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Finding 5: FAISS Retrieval is Not the Bottleneck",
           "pct_unreachable = 0% — every failure is a model ranking failure")
accent_line(s)
footer(s)

add_rect(s, Inches(3.5), Inches(1.35), Inches(6.3), Inches(1.3),
         fill=rgb(0xE8,0xF5,0xE9), line=C_GREEN, line_w=Pt(2))
add_text(s, "pct_unreachable = 0.00%  across all 94 pairs",
         Inches(3.65), Inches(1.5), Inches(6.0), Inches(0.6),
         size=20, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)
add_text(s, "Every ground-truth file was present in FAISS top-300 candidates",
         Inches(3.65), Inches(2.1), Inches(6.0), Inches(0.45),
         size=13, color=C_DARK, align=PP_ALIGN.CENTER, italic=True)

add_bullet_box(s, [
    "Pipeline: FAISS retrieves top-300 files → TRANP-CNN reranks them",
    "If GT file is NOT in top-300: model cannot fix it (unreachable)",
    "If GT file IS in top-300 but ranked below top-10: that is a model ranking failure",
    "Our result: 0% unreachable → ALL failures are ranking failures, not retrieval failures",
    "",
    "Implication: improving the reranking architecture directly improves all metrics",
    "Increasing TOP_K_CANDIDATES beyond 300 will not help — embeddings are already sufficient",
    "This validates the research direction: better reranker (transformer, GNN) = better results",
], Inches(0.4), Inches(2.75), Inches(12.2), Inches(3.8), size=15)

img_path = os.path.join(IMAGES_DIR, "mod3_stacked_outcomes.png")
add_image_safe(s, img_path, Inches(9.5), Inches(1.35), Inches(3.5), Inches(2.8))


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Finding 6: Commutativity (Pair A)
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Finding 6: CPL is Not Commutative — Direction Matters",
           "Showcase Pair A: matplotlib ↔ jupyterlab")
accent_line(s)
footer(s)

table_box(s,
    headers=["Direction", "WP-small", "WP-large", "CP-cold-start", "CP-transfer"],
    rows=[
        ["matplotlib → jupyterlab", "0.438", "0.554", "0.045", "0.614 ★"],
        ["jupyterlab → matplotlib", "0.372", "0.294", "0.320", "0.236"],
    ],
    l=Inches(0.4), t=Inches(1.3), w=Inches(12.5), h=Inches(1.8),
    row_fills=[rgb(0xE8,0xF5,0xE9), rgb(0xFC,0xE4,0xEC)]
)

img_path = os.path.join(SHOWCASE_DIR,
    "pair_A_matplotlib_jupyterlab", "commutativity_contrast.png")
add_image_safe(s, img_path, Inches(0.3), Inches(3.25), Inches(7.8), Inches(3.7))

add_bullet_box(s, [
    "Same domain gap (0.972), same two projects — just swapped",
    "Forward: jupyterlab = 39K LoC target → easy to rank → CPL wins",
    "Reverse: matplotlib = 249K LoC target → correct file buried → CPL hurts",
    "WP-large even loses to WP-small for jupyterlab→matplotlib",
    "Rule: TARGET codebase size determines difficulty, not source",
], Inches(8.2), Inches(3.3), Inches(4.9), Inches(3.5), size=14)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Showcase Pair B: Best Performer
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Showcase Pair B: numpy → jupyterlab  (Best CPL Performer)",
           "Top-10 recall = 100% for WP-large and CP-transfer; 33× improvement over FAISS")
accent_line(s)
footer(s)

table_box(s,
    headers=["Scenario", "Top-1", "Top-5", "Top-10", "MAP", "MRR"],
    rows=[
        ["WP-large",      "0.565", "1.000", "1.000", "—", "0.724"],
        ["CP-transfer",   "0.478", "1.000", "1.000", "—", "0.666"],
        ["WP-small",      "0.217", "0.870", "1.000", "—", "0.464"],
        ["CP-cold-start", "0.130", "0.348", "0.565", "—", "0.258"],
        ["FAISS baseline","—",     "—",     "—",     "—", "0.022"],
    ],
    l=Inches(0.4), t=Inches(1.3), w=Inches(6.5), h=Inches(2.9),
    row_fills=[
        rgb(0xE8,0xF5,0xE9), rgb(0xFC,0xE4,0xEC),
        rgb(0xE3,0xF2,0xFD), rgb(0xFF,0xF3,0xE0),
        rgb(0xF5,0xF5,0xF5),
    ]
)

img_path = os.path.join(SHOWCASE_DIR,
    "pair_B_numpy_jupyterlab", "all_metrics_comparison.png")
add_image_safe(s, img_path, Inches(7.0), Inches(1.25), Inches(6.1), Inches(5.7))

add_bullet_box(s, [
    "CP-transfer MRR 0.666 ≈ WP-large 0.724 (gap = 0.058, ~8%)",
    "FAISS MRR = 0.022 → model achieves 33× improvement",
    "jupyterlab: 39K LoC, 197 bugs — ideal CPL target",
    "Zero-shot (cold-start): MRR = 0.258 — non-trivial transfer",
], Inches(0.4), Inches(4.4), Inches(6.5), Inches(2.4), size=14)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Showcase Pair C: Worst Performer
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Showcase Pair C: numpy → scipy  (Hard Target Ceiling)",
           "438K LoC codebase — model improves ranks but cannot break top-10")
accent_line(s)
footer(s)

table_box(s,
    headers=["Scenario", "Top-1", "Top-5", "Top-10", "MRR"],
    rows=[
        ["WP-large",      "0.000", "0.000", "0.000", "0.029"],
        ["CP-transfer",   "0.000", "0.000", "0.000", "0.024"],
        ["WP-small",      "0.000", "0.000", "0.000", "0.004"],
        ["CP-cold-start", "0.000", "0.000", "0.000", "0.006"],
    ],
    l=Inches(0.4), t=Inches(1.3), w=Inches(6.0), h=Inches(2.5),
    row_fills=[
        rgb(0xFF,0xCC,0xBC), rgb(0xFF,0xCC,0xBC),
        rgb(0xFF,0xCC,0xBC), rgb(0xFF,0xCC,0xBC),
    ]
)

img_path = os.path.join(SHOWCASE_DIR,
    "pair_C_numpy_scipy", "rank_displacement_boxplot.png")
add_image_safe(s, img_path, Inches(6.5), Inches(1.25), Inches(6.6), Inches(3.2))

add_bullet_box(s, [
    "scipy: 438,217 LoC — largest codebase in our dataset",
    "pct_unreachable = 0% — GT file IS in FAISS top-300",
    "Mean rank displacement = +131 (WP-large) — model IS improving ranks",
    "But: moving from rank 167 to rank 36 is still outside top-10",
    "This is a TARGET VIABILITY failure, not a source selection failure",
    "No source project can fix a 438K LoC target — architecture must improve",
], Inches(0.4), Inches(4.05), Inches(12.2), Inches(2.85), size=14)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Source Project Selection (Kendall tau)
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Source Project Selection — Can We Pick the Best Source?",
           "Kendall τ analysis across 12 targets with ≥3 source candidates")
accent_line(s)
footer(s)

# Left: Kendall tau explanation
add_rect(s, Inches(0.3), Inches(1.3), Inches(5.3), Inches(4.3),
         fill=rgb(0xE8,0xEA,0xF0), line=C_ACCENT, line_w=Pt(1.2))
add_text(s, "📐  Kendall Rank Correlation (τ)",
         Inches(0.45), Inches(1.38), Inches(5.0), Inches(0.45),
         size=14, bold=True, color=C_ACCENT)
add_text(s,
    "Measures whether one ranking agrees with\n"
    "another. For each target project:\n\n"
    "• Sort sources by heuristic score\n"
    "• Sort sources by actual CP-transfer MRR\n"
    "• τ = agreement between these two orderings\n\n"
    "τ = +1 → heuristic perfectly matches oracle\n"
    "τ = −1 → heuristic is perfectly reversed\n"
    "τ = 0  → heuristic is random\n\n"
    "We also use Hit@1: did the top-ranked source\n"
    "by heuristic turn out to be the best source\n"
    "by actual performance?",
    Inches(0.45), Inches(1.85), Inches(5.0), Inches(3.7),
    size=12, color=C_DARK)

# Right: results
table_box(s,
    headers=["Strategy", "Hit@1 (MRR)", "Hit@1 (MAP)", "Mean τ (MRR)"],
    rows=[
        ["Oracle (upper bound)",  "100%",  "100%",  "+1.00"],
        ["Most-bugs heuristic",   "41.7%", "41.7%", "+0.253"],
        ["Random (lower bound)",  "~15%",  "~15%",  "~0.00"],
    ],
    l=Inches(5.85), t=Inches(1.3), w=Inches(7.1), h=Inches(2.0),
    row_fills=[
        rgb(0xE8,0xF5,0xE9),
        rgb(0xFC,0xE4,0xEC),
        rgb(0xF5,0xF5,0xF5),
    ]
)
add_bullet_box(s, [
    "Most-bugs = pick the source with the most bug reports",
    "Hit@1 = 41.7%: correct in fewer than half of cases",
    "τ = +0.253: positive ordering signal — larger sources tend to be better",
    "Captures ~53% of the gap between random and oracle selection",
    "Source selection is PARTIALLY solved — not an algorithm, a useful heuristic",
], Inches(5.85), Inches(3.5), Inches(7.1), Inches(3.0), size=14)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Decision Framework
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Practical Decision Framework for CPL Deployment",
           "Two-phase strategy: target viability first, then source selection")
accent_line(s)
footer(s)

# Phase 1 box
add_rect(s, Inches(0.3), Inches(1.3), Inches(6.1), Inches(4.5),
         fill=rgb(0xE3,0xF2,0xFD), line=C_BLUE, line_w=Pt(1.5))
add_text(s, "Phase 1 — Target Viability Check",
         Inches(0.45), Inches(1.38), Inches(5.8), Inches(0.5),
         size=16, bold=True, color=C_BLUE)
add_bullet_box(s, [
    "Target LoC > 200K?",
    "→ CPL will underperform regardless of source (ρ = −0.855)",
    "",
    "Bug report verbosity very low (< 30 words)?",
    "→ Insufficient signal for any model",
    "",
    "Target has > 300 historical bugs?",
    "→ WP-large may be more reliable; CPL gain smaller",
    "",
    "✓  If target passes: proceed to source selection",
], Inches(0.45), Inches(1.95), Inches(5.8), Inches(3.6),
   size=13, indent_char="")

# Phase 2 box
add_rect(s, Inches(6.7), Inches(1.3), Inches(6.3), Inches(4.5),
         fill=rgb(0xFC,0xE4,0xEC), line=C_PINK, line_w=Pt(1.5))
add_text(s, "Phase 2 — Source Selection Heuristics",
         Inches(6.85), Inches(1.38), Inches(6.0), Inches(0.5),
         size=16, bold=True, color=C_PINK)
add_bullet_box(s, [
    "S1: Default — pick source with MOST bug reports",
    "      (Hit@1 = 41.7%, Kendall τ = +0.25)",
    "",
    "S2: Among tied sources — prefer larger codebase",
    "      (src_LoC has weak positive signal: ρ ≈ +0.27)",
    "",
    "S3: DO NOT use domain gap as filter",
    "      (ρ < 0.12, non-significant for all metrics)",
    "",
    "S4: DO NOT use bug/code similarity as ranking signal",
    "      (no significant correlation in our dataset)",
], Inches(6.85), Inches(1.95), Inches(6.0), Inches(3.6),
   size=13, indent_char="")

add_text(s,
    "This is a set of strategies, not an algorithm. "
    "Hit@1 = 41.7% means the heuristic fails 58% of the time. "
    "Learned source selection is an open research problem.",
    Inches(0.4), Inches(6.0), Inches(12.5), Inches(0.6),
    size=13, color=C_ACCENT, italic=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Research Contributions & Implications
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Research Contributions & Implications",
           "What this study establishes for the CPL literature")
accent_line(s)
footer(s)

table_box(s,
    headers=["Contribution", "Evidence", "Metric support"],
    rows=[
        ["CPL improves ranking over WP-small",        "67.4% win rate, p=0.005",       "MRR, MAP"],
        ["CPL achieves WP-large recall",               "53% pairs Top-10 match",        "Top-10"],
        ["CPL most valuable for data-scarce targets",  "3–4× gain, few-bug targets",    "Top-10, MRR"],
        ["Target LoC is dominant predictor",           "ρ = −0.855 for Top-10",         "All metrics"],
        ["Source selection: most-bugs heuristic",      "Hit@1=41.7%, τ=+0.25",          "MRR, MAP"],
        ["Architecture is the bottleneck",             "pct_unreachable = 0%",          "Diagnostic"],
    ],
    l=Inches(0.4), t=Inches(1.3), w=Inches(12.5), h=Inches(3.4),
)

add_bullet_box(s, [
    "For new projects (< 200 historical bugs): CPL is the recommended default strategy",
    "Domain gap filtering is NOT needed — dissimilar projects can still transfer well",
    "Better reranking architecture is the clearest path to higher MRR/MAP",
    "Source selection remains an open problem — 58% of the time the heuristic misses",
], Inches(0.4), Inches(4.9), Inches(12.2), Inches(1.8), size=15)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — Limitations & Future Work
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
header_bar(s, "Limitations & Future Work",
           "Honest framing + what comes next")
accent_line(s)
footer(s)

add_text(s, "Limitations", Inches(0.4), Inches(1.3), Inches(5.9), Inches(0.45),
         size=18, bold=True, color=C_DARK)
add_bullet_box(s, [
    "Mean MRR 0.20–0.25 is low vs within-project SOTA (> 0.5) — but correct comparison is our FAISS baseline, not cross-study",
    "Results include Python and Java; generalisation to C++, JS, polyglot unknown",
    "Large targets (scipy, 438K LoC) dominate the low-performance tail — stratified reporting is more informative than overall means",
    "CP-cold-start (MAP=0.038) is not practically useful — zero-shot still needs fine-tuning",
    "Source selection Hit@1 = 41.7% — heuristic works in fewer than half of cases",
], Inches(0.4), Inches(1.8), Inches(5.9), Inches(3.8), size=13)

add_text(s, "Future Work", Inches(6.7), Inches(1.3), Inches(5.9), Inches(0.45),
         size=18, bold=True, color=C_DARK)
add_bullet_box(s, [
    "Learned source selection meta-model: train on pair outcomes to predict source quality for unseen targets",
    "Multi-source ensembling: combine all sources rather than picking one — may outperform selection",
    "Transformer/GNN reranker: replace TRANP-CNN CNN architecture to reduce the ranking gap vs WP-large",
    "COOBA comparison: run same Phase 1 analysis for COOBA (GNN-based) and compare transfer behaviour",
    "Large-codebase architecture: hierarchical or retrieval-augmented reranker for > 200K LoC targets",
], Inches(6.7), Inches(1.8), Inches(5.9), Inches(3.8), size=13)

add_rect(s, Inches(0.3), Inches(5.85), Inches(12.7), Inches(0.8),
         fill=rgb(0xE8,0xF5,0xE9), line=C_GREEN, line_w=Pt(1))
add_text(s,
    "Core message: CPL is the right strategy for data-limited targets. "
    "The bottleneck is now the reranking architecture, not the retrieval or the transfer mechanism.",
    Inches(0.5), Inches(5.95), Inches(12.3), Inches(0.6),
    size=14, bold=True, color=rgb(0x1B,0x5E,0x20))


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 17 — Summary / One-Page Takeaway
# ═══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank_layout)
add_rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, fill=C_DARK)
add_rect(s, Inches(0), Inches(0), Inches(0.18), SLIDE_H, fill=C_PINK)
add_rect(s, Inches(0), Inches(1.3), SLIDE_W, Inches(0.05), fill=C_BLUE)

add_text(s, "Key Takeaways", Inches(0.5), Inches(0.15), Inches(10), Inches(0.8),
         size=30, bold=True, color=C_WHITE)

takeaways = [
    ("1", "CP-transfer beats WP-small", "67.4% of pairs (p=0.005) — ranking quality improves significantly", C_PINK),
    ("2", "Top-K recall matches WP-large", "53% of pairs on Top-10 — same recall, 4× less target data", C_BLUE),
    ("3", "Most valuable when data is scarce", "3–4× gain for targets with < 228 bugs", C_GREEN),
    ("4", "Target size governs performance", "tgt_LoC ρ = −0.855 for Top-10 — largest predictor by far", C_ORANGE),
    ("5", "Source selection = heuristic, not algorithm", "Most-bugs heuristic: Hit@1 = 41.7%, τ = +0.25", C_ACCENT),
    ("6", "Architecture is the remaining challenge", "pct_unreachable = 0% — improve reranker, not retrieval", C_GREEN),
]

for i, (num, title, body, color) in enumerate(takeaways):
    row = i // 2
    col = i % 2
    x = Inches(0.35 + col * 6.4)
    y = Inches(1.5 + row * 1.85)
    add_rect(s, x, y, Inches(6.1), Inches(1.65), fill=rgb(0x25,0x2D,0x3D),
             line=color, line_w=Pt(2))
    add_rect(s, x, y, Inches(0.55), Inches(1.65), fill=color)
    add_text(s, num, x + Inches(0.1), y + Inches(0.5), Inches(0.35), Inches(0.65),
             size=22, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_text(s, title, x + Inches(0.65), y + Inches(0.1), Inches(5.3), Inches(0.5),
             size=14, bold=True, color=color)
    add_text(s, body, x + Inches(0.65), y + Inches(0.6), Inches(5.3), Inches(0.9),
             size=12, color=C_LIGHT)

footer(s, "")


# ── Save ──────────────────────────────────────────────────────────────────────
pptx_path = os.path.join(OUT_DIR, "CPL_Phase1_Results.pptx")
prs.save(pptx_path)
print(f"✓ Saved: {pptx_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SPEAKER NOTES
# ═══════════════════════════════════════════════════════════════════════════════

notes = """# CPL Phase 1 — Speaker Notes
## Detailed what-to-say for each slide

---

## Slide 1 — Title

**What to say:**
"Today I'm presenting Phase 1 results of my PhD study on Cross-Project Bug Localisation, abbreviated as CPL.
The central question I'm investigating is: can a bug localisation model trained on one project transfer to another? Instead of labelling thousands of bugs in your own project, can you borrow a pre-trained model from a related project and get useful results?

This is practically important because labelling bug report-to-file mappings is expensive and time-consuming. Many real-world projects simply don't have enough historical data to train a reliable bug localiser from scratch.

I'll walk through the setup, the results across five metrics, six key findings, three illustrative showcase pairs, and what this means for source project selection."

---

## Slide 2 — Research Problem & Setup

**What to say:**
"Let me explain the experimental design. I evaluate four scenarios for each source→target project pair.

WP-small represents the baseline: train only on 20% of the target project's bug data. This simulates a project with limited historical information.

WP-large is the upper bound: train on 80% of target data. This simulates a project with abundant labelled data.

CP-cold-start is the hardest case: train entirely on the source project, apply directly to target with zero target data. This is fully zero-shot cross-project transfer.

CP-transfer is the main CPL hypothesis: train on 100% of source data, then fine-tune on 20% of target data — matching the data budget of WP-small exactly.

The core question is: does CP-transfer beat WP-small? If yes, cross-project knowledge helps when target data is limited.

The pipeline: I use BAAI/bge-code-v1 code embeddings to retrieve the top-300 candidate source files via FAISS, then TRANP-CNN re-ranks those 300 candidates using a trained CNN model."

---

## Slide 3 — Metrics Explained

**What to say:**
"Before going to results, I want to be precise about the five metrics I report, because they each capture something different.

Top-K recall asks: is the ground-truth file somewhere in the top K ranked positions? Top-10 = 0.80 means 80% of bugs had the correct file in the top 10. This is the most lenient metric — it gives credit even if the correct file is ranked 9th.

MAP, or Mean Average Precision, is stricter. It measures precision at every rank position where a relevant file appears. A file ranked 1st contributes more to MAP than one ranked 9th. MAP penalises models that retrieve the file but bury it low in the ranking.

MRR, Mean Reciprocal Rank, is the mean of 1 divided by the rank of the first correct answer. MRR = 0.5 means the correct file is at rank 2 on average. MRR = 0.25 means rank 4 on average.

Here's the important implication: Top-K recall and MAP/MRR can give different signals. A model might have high Top-10 recall but low MRR if it consistently puts the correct file at rank 8 or 9. Throughout my analysis I report all five metrics, and I'll point out where they agree and where they diverge."

---

## Slide 4 — Overall Results

**What to say:**
"Here are the mean results across all 92 to 94 source-target pairs.

The pattern is consistent across all five metrics: WP-large is best, then CP-transfer, then WP-small, then CP-cold-start.

The key observation is how small the gap between CP-transfer and WP-large is on Top-K metrics. On Top-10, the gap is only 0.007. On Top-5, it's 0.011. This means CP-transfer retrieves the correct file at nearly the same rate as WP-large.

For MAP and MRR, WP-large has a larger advantage — about 0.02. So WP-large ranks the correct file higher within the retrieved candidates.

CP-cold-start is very low across the board — MAP 0.038, MRR 0.042. Zero-shot transfer without any fine-tuning on the target project has limited practical value in its current form.

The headline finding is that CP-transfer, which uses only 20% of target data — the same budget as WP-small — substantially outperforms WP-small and nearly matches WP-large on recall."

---

## Slide 5 — Finding 1: Wilcoxon Test

**What to say:**
"Let me now go into the statistical significance of the CP-transfer versus WP-small comparison.

First, a note on the statistical test. I use the Wilcoxon signed-rank test because I have paired observations — for each project pair, I have both a CP-transfer result and a WP-small result. This is a non-parametric test, meaning it doesn't assume the data is normally distributed. It works by ranking the differences between paired observations and testing whether positive differences dominate.

I test the one-sided alternative: CP-transfer is greater than WP-small. The null hypothesis is that there's no systematic difference.

The results: for MAP and MRR, I reject the null hypothesis at p < 0.01. CP-transfer is significantly better. The win rates are 64.1% and 67.4% respectively. For Top-5 and Top-10, also significant at p < 0.01.

Critically, Top-1 is NOT significant — p = 0.118. CPL does not help the model put the correct file at rank 1 more often. It helps the model rank it higher in general, which improves MRR and MAP. This is an important nuance.

The interpretation: cross-project pre-training provides a better learned prior for ranking candidate files. The model has seen more diverse bug-file patterns from the source project, which generalises to better relative ordering on the target project."

---

## Slide 6 — Finding 2: CP-transfer vs WP-large

**What to say:**
"Now let me compare CP-transfer against WP-large — the resource-intensive within-project baseline.

The table shows two comparison windows: CP-transfer greater than or equal to WP-large (they're at least tied), and CP-transfer within 10% of WP-large.

For recall metrics: CP-transfer matches or beats WP-large in 52 to 54% of pairs. For Top-10, 63% of pairs are within 10% of WP-large.

For MAP and MRR: WP-large wins in roughly 70% of pairs. The mean gap is 0.023 for MAP and 0.025 for MRR.

What does this mean practically? For use-cases where you're building a developer tool that shows a shortlist of 10 candidate files, CP-transfer is broadly equivalent to WP-large — and it uses 4 times less target training data.

For tasks requiring tight rank-1 precision — say, an automated patch suggestion tool that acts only on the top-ranked file — WP-large retains a meaningful edge.

The practical CPL deployment recommendation is: if your project needs precise top-of-list localisation and you have abundant labelled data, use WP-large. If you're in a data-limited scenario and you need a shortlist, CP-transfer is essentially equivalent."

---

## Slide 7 — Finding 3: Data-Scarce Targets

**What to say:**
"This finding is what I consider the strongest motivating argument for CPL as a practical research contribution.

I split the 94 project pairs by the target project's bug count, using the median of 228 bugs as the threshold. Projects below the median have limited historical data — this is the scenario CPL is designed for.

For these data-scarce targets, the CPL gain — defined as CP-transfer minus WP-small — is 3 to 6 times larger than for data-rich targets. On Top-10 recall, the gain is +0.043 for few-bug targets versus +0.007 for many-bug targets. On MRR, +0.032 versus +0.012.

The implication is direct: if you have a project with fewer than 200 historical bugs, using CPL is significantly better than training from scratch on that limited data. The benefit diminishes as the target project accumulates more bug data, which makes intuitive sense — at some point, within-project data is sufficient.

This motivates CPL as the recommended default strategy for new projects, projects that have recently changed technology stacks, or niche projects with sparse bug histories."

---

## Slide 8 — Finding 4: Spearman Correlations

**What to say:**
"I used Spearman rank correlation to understand which features of the source and target projects predict CPL performance. Let me first explain Spearman correlation for context.

Spearman's ρ measures the monotonic relationship between two variables. Unlike Pearson, it works on ranks rather than raw values, making it robust to outliers and non-linear relationships. ρ = +1 means as one variable increases, the other always increases. ρ = −1 means they're inversely related. ρ = 0 means no systematic relationship.

In our case, each observation is a source→target pair. I'm asking: do project-level features predict how well CP-transfer performs on that pair?

The standout finding is tgt_LoC, the target codebase size. ρ = −0.855 for Top-10 recall. This is an extremely strong negative correlation — as the target codebase gets larger, Top-10 recall drops very consistently across all 94 pairs. For MAP and MRR, ρ is −0.65 and −0.67, also very strong.

Bug report verbosity is the second-strongest feature at ρ = +0.383 for MAP — more detailed bug reports help the model match bug descriptions to source files.

Critically, source-side features are weak or non-significant. Source bug count has ρ = 0.07 for MRR. Domain gap — the logistic regression accuracy I use as a dissimilarity measure between projects — has ρ = 0.038 for MRR, which is not significant.

The practical conclusion: when deciding whether to deploy CPL on a project, look at the target codebase size and bug report quality. Source selection matters much less than these target properties."

---

## Slide 9 — Finding 5: FAISS Ceiling

**What to say:**
"This is a diagnostic finding that shapes the entire research agenda going forward.

The pipeline works in two stages: FAISS retrieves the top-300 candidate files using embedding similarity, then TRANP-CNN reranks those 300 candidates.

If the ground-truth file is NOT in the top-300 retrieved by FAISS, the model cannot possibly find it — it can only rerank what it's given. This is what I call an 'unreachable' case.

The finding is that pct_unreachable equals exactly 0% across all 94 pairs. Every single ground-truth file was retrieved by FAISS in its top-300 candidates.

This is extremely informative because it means every single failure in my results — every case where the correct file was not in the model's top-10 — is a model ranking failure, not a retrieval failure. FAISS with BAAI/bge-code-v1 embeddings is already good enough to find the correct file; the challenge is ranking it high enough.

The implication for future work is clear: improving the reranking model architecture — replacing the CNN with a transformer, using contrastive learning, or using a GNN — will directly improve all metrics. There's no point in increasing the retrieval pool beyond 300. The FAISS embeddings are not the bottleneck."

---

## Slide 10 — Finding 6: Commutativity (Pair A)

**What to say:**
"This showcase pair illustrates one of the most counterintuitive findings: CPL is not commutative. The same two projects in opposite directions can give very different results.

Matplotlib-to-jupyterlab: CP-transfer achieves MRR 0.614, which is actually the best scenario — even better than WP-small's 0.438 and WP-large's 0.554.

Jupyterlab-to-matplotlib: CP-transfer drops to MRR 0.236, actually worse than WP-small at 0.372.

What explains this? Both pairs have identical domain gap — 0.972 — and they use the exact same two projects, just swapped.

The explanation is entirely target codebase size. When jupyterlab is the target — 39,000 lines of code — it's a compact, well-structured codebase where the correct file can rise to the top-10 easily. When matplotlib is the target — 249,000 lines of code — the correct file is competing against far more candidates and is much harder to rank in the top-10.

This finding also shows that WP-large can be worse than WP-small in some cases. For jupyterlab-to-matplotlib, WP-large gives MRR 0.294 while WP-small gives 0.372. This suggests overfitting: with more training data on matplotlib's specific bug patterns, the model loses generalisability on the test bugs.

The general rule: the target project's codebase size determines the difficulty of localisation, not the source. Commutativity breaks along the tgt_LoC axis."

---

## Slide 11 — Showcase Pair B: Best Performer

**What to say:**
"Numpy-to-jupyterlab is our best-performing pair and provides the strongest evidence for CPL feasibility.

WP-large achieves MRR 0.724 and Top-5 recall of 100% — every single jupyterlab bug is found in the top-5. CP-transfer reaches MRR 0.666 with the same 100% Top-5 recall.

The gap between CP-transfer at 0.666 and WP-large at 0.724 is only 0.058, or about 8%. This is achieved with the same 20% target data budget as WP-small, which only gets MRR 0.464.

The FAISS baseline for this pair is MRR 0.022. The model achieves 33 times that. This is not a subtle improvement — the model is doing substantial work to reorganise the ranking from near-random to highly accurate.

The zero-shot baseline, CP-cold-start, reaches MRR 0.258 using only numpy training data and zero jupyterlab examples. That's non-trivial — over 25% reciprocal rank purely from cross-project transfer before any fine-tuning.

Why does this pair work so well? Jupyterlab is a 39,000-line Python project with 197 well-documented bug reports. Small, focused codebase plus verbose bug reports plus a rich source project like numpy — this is the ideal CPL scenario."

---

## Slide 12 — Showcase Pair C: Worst Performer

**What to say:**
"Numpy-to-scipy is the opposite extreme. Every scenario gets zero Top-1, Top-5, and Top-10 recall. MRR peaks at 0.029 for WP-large.

Before concluding this is a model failure, let me diagnose it carefully.

Scipy has 438,000 lines of code — the largest codebase in our dataset. pct_unreachable is 0% — the correct file is in FAISS's top-300. The problem is not retrieval.

Looking at rank displacement: the model IS improving the ranking. WP-large has a mean rank displacement of +131 — the model pushes the correct file up 131 positions on average. But the file starts at around rank 167 — so after improvement it lands around rank 36, which is still outside top-10.

What would it need? The model would need to displace the correct file by 157 positions to break into top-10. At 131 positions average improvement, it's close but not there yet.

The key point is this: no source project selection will fix this. I tested 11 different source projects for scipy, and the oracle MRR — the best possible source — is still only 0.024. This is a target viability failure. The codebase is simply too large for the current architecture.

This motivates the architectural future work: hierarchical rerankers, better localisation at scale, or retrieval augmentation specifically for large codebases."

---

## Slide 13 — Source Selection (Kendall τ)

**What to say:**
"Given that we have multiple potential source projects, can we predict which one will produce the best CP-transfer results?

I evaluate this using Kendall's tau rank correlation. For each target project with at least 3 source candidates, I rank the sources by a heuristic score and compare that ranking to the actual oracle ranking — the ordering by achieved CP-transfer MRR.

Kendall's tau ranges from -1 to +1. τ = 1 means the heuristic perfectly identifies the best source. τ = 0 means the heuristic is random. Negative τ means the heuristic is worse than random.

The most-bugs heuristic — simply picking the source with the most bug reports — achieves Hit@1 of 41.7% and Kendall τ of +0.253 across 12 targets.

This is better than I initially expected. The heuristic correctly identifies the best source in 5 out of 12 cases and has a positive ordering signal — larger sources genuinely tend to produce better transfer results more often. It captures about 53% of the gap between random and oracle selection.

However, Hit@1 of 41.7% means the heuristic fails 58% of the time. This is a partially solved problem, not a fully solved one. Source selection requires a learned approach — a meta-model trained on historical pair outcomes — to close the remaining gap.

The two negative results I want to highlight: domain gap is NOT a useful filter for source selection, and bug report similarity is NOT a useful proxy. Neither shows significant correlation with achieved performance. These intuitive proxies fail empirically."

---

## Slide 14 — Decision Framework

**What to say:**
"Based on all the findings, I propose a two-phase decision framework for practitioners deploying CPL.

Phase 1 is target viability. Before selecting a source project, check whether the target is a suitable CPL candidate at all.

If the target codebase exceeds 200,000 lines of code — flag it as high-risk. Our Spearman correlation ρ = −0.855 shows this is the dominant factor. If the target is very large, even the best source will produce mediocre results.

If bug reports are very terse — less than 30 words on average — the model has insufficient textual signal to match bug descriptions to files.

If the target already has over 300 historical bugs, WP-large may actually be more reliable than CP-transfer, since the CPL gain shrinks for data-rich targets.

Phase 2 is source selection. If the target passes the viability check, apply these heuristics in order:

First, default to the source with the most bug reports. This achieves 41.7% Hit@1 and is the best simple heuristic available.

Second, among similar-sized sources, prefer larger codebases — src_LoC has a weak positive correlation.

Third, ignore domain gap — it has no significant predictive power for any metric.

Fourth, ignore bug similarity metrics — they showed no reliable correlation in our dataset.

I want to be clear that this is a set of heuristics, not an algorithm. The most-bugs heuristic fails 58% of the time. Learned source selection using a meta-model is an open research problem."

---

## Slide 15 — Contributions & Implications

**What to say:**
"Let me summarise the research contributions and what they mean for the field.

First: CPL significantly improves ranking quality over WP-small with p = 0.005. This is the primary feasibility claim. The literature already established that CPL is possible; we now have a quantified, statistically validated improvement with matched data budgets.

Second: CP-transfer achieves WP-large-level recall in over half of pairs. This is relevant for practitioners who care about building shortlists rather than exact rank-1 localisation.

Third: the gain is 3 to 4 times larger for data-scarce targets. This positions CPL as the recommended strategy for new projects, not as a general-purpose improvement over well-resourced within-project training.

Fourth: target LoC is the dominant predictor at ρ = −0.855. This means that when we evaluate CPL, we must stratify by target codebase size. A study that averages across scipy and jupyterlab will understate the effectiveness of CPL for small targets and overstate it for large ones.

Fifth: the most-bugs heuristic achieves Hit@1 = 41.7% for source selection. This establishes an empirical baseline for future learned source selection methods to beat.

Sixth: pct_unreachable = 0% frames the research agenda. Future work should focus on the reranking model, not the retrieval step."

---

## Slide 16 — Limitations & Future Work

**What to say:**
"I want to be transparent about the limitations.

The mean MRR values of 0.20 to 0.25 look low compared to within-project SOTA papers that report MRR above 0.5. But this comparison is unfair — those papers train and test on the same project, which is fundamentally easier. Our correct comparison is our own FAISS baseline and WP-small, not cross-study numbers.

The results cover Python and Java projects. I expect the qualitative findings to hold across languages — code embeddings are language-aware — but this hasn't been empirically validated.

Large-codebase targets like scipy dominate the low-performance tail and pull down averages. Stratified reporting by target size gives a more honest picture.

For future work, the highest priority is a learned source selection meta-model. Our heuristic captures 53% of the oracle gap — a learned approach could close that further.

Multi-source ensembling is another natural direction: instead of picking the single best source, combine signal from multiple sources. This is analogous to ensemble methods in other transfer learning settings.

The most impactful architectural improvement would be replacing TRANP-CNN's CNN reranker with a transformer-based model. Given that all GT files are retrievable and all failures are ranking failures, a stronger reranker will directly lift all metrics."

---

## Slide 17 — Summary Takeaways

**What to say:**
"To summarise the six key takeaways:

One: CP-transfer beats WP-small in 67.4% of pairs on MRR with p = 0.005. The improvement is statistically significant and consistent.

Two: CP-transfer matches WP-large on Top-10 recall in 53% of pairs. For shortlist-based applications, CP-transfer is essentially equivalent to WP-large with 4 times less labelled data.

Three: CPL is 3 to 4 times more valuable for data-scarce targets — projects with fewer than 228 bugs see substantially larger gains.

Four: target codebase size is the single strongest predictor of achievable performance, with ρ = −0.855 for Top-10. Check the target before deploying CPL.

Five: source selection is partially solved. The most-bugs heuristic achieves 41.7% Hit@1 and positive Kendall τ of +0.25. Domain gap and similarity metrics are not useful for source selection.

Six: the FAISS ceiling is not the bottleneck — pct_unreachable = 0%. The remaining challenge is the reranking architecture, and improving it will directly translate to better results.

Thank you — I'm happy to take questions."

---

## Common PhD Viva / Presentation Questions

**Q: Why is CP-cold-start so low?**
A: Zero-shot cross-project transfer means the model has never seen the target project's code structure, naming conventions, or bug patterns. Fine-tuning on even 20% of target data dramatically changes the loss landscape. CP-cold-start serves as a floor — it shows that some cross-project signal exists, but fine-tuning is necessary to unlock it.

**Q: Why not use more FAISS candidates — say, top-500?**
A: pct_unreachable = 0% shows that top-300 already captures all GT files. Increasing to 500 would not improve recall — the model would just have 200 more non-relevant candidates to rank around, which could dilute the reranking signal.

**Q: Couldn't the difference between CP-transfer and WP-small just be random noise?**
A: That's exactly what the Wilcoxon signed-rank test addresses. With 92 paired observations and p = 0.005 for MRR, the probability of observing this pattern by chance is 0.5%. The test is conservative and non-parametric, so it doesn't assume any particular data distribution.

**Q: Why did you choose Spearman correlation over Pearson?**
A: Pearson assumes a linear relationship and is sensitive to outliers. Our data has projects spanning a very wide range of sizes — from 2,000 to 438,000 LoC. Spearman operates on ranks, which is more robust in this heterogeneous setting. It asks 'is there a monotonic relationship?' rather than 'is there a linear relationship?', which is the more appropriate question here.

**Q: Domain gap of 0.97 — isn't that extreme? Doesn't that mean projects are completely different?**
A: A domain gap of 0.97 means a logistic regression classifier can distinguish embeddings from the two projects with 97% accuracy — yes, they are structurally very different. But the key finding is that domain gap does not predict CPL performance (ρ < 0.12, non-significant). Dissimilar projects can still transfer well because TRANP-CNN learns structural patterns — the relationship between bug description tokens and source file tokens — which are universal across programming contexts.

**Q: What's the difference between MAP and MRR in practice?**
A: MRR gives full credit for the first correct answer only. MAP gives partial credit at every rank position where a relevant file appears. For bug localisation, bugs typically have one ground-truth file, so MAP and MRR behave similarly — but MAP is slightly more standard in information retrieval literature and penalises models that put the correct file just above the threshold more harshly.
"""

notes_path = os.path.join(OUT_DIR, "SPEAKER_NOTES.md")
with open(notes_path, "w") as f:
    f.write(notes)
print(f"✓ Saved: {notes_path}")
print(f"\n✓ Done. Output in {OUT_DIR}/")
