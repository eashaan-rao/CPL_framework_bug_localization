# CPL Desirability: Evidence and Research Implications

**Study**: TRANP-CNN Phase 1 — Cross-Project Bug Localization on 20 Projects (Python + Java)
**Scope**: 94 source→target pairs, 4 scenarios, 373 experiment rows
**Metrics**: Top-1, Top-5, Top-10, MAP, MRR
**Source file**: `results/phase1_experimental_results.csv`
**Generated**: 2026-03-15

---

## Why This Matters

The question is not just *"can CPL work?"* — it is *"should practitioners adopt it, and under what conditions does it provide genuine value?"*
The evidence below builds a cumulative argument from six independent angles, each directly answerable from experimental data.

---

## Scenario Overview (All Metrics)

| Scenario | Top-1 | Top-5 | Top-10 | MAP | MRR |
|---|---|---|---|---|---|
| **WP-large** | 0.081 | 0.200 | 0.258 | 0.218 | 0.248 |
| **CP-transfer** | 0.064 | 0.189 | 0.251 | 0.197 | 0.226 |
| **WP-small** | 0.056 | 0.167 | 0.223 | 0.176 | 0.202 |
| **CP-cold-start** | 0.005 | 0.021 | 0.040 | 0.038 | 0.042 |

*(Mean across 92–94 pairs)*

**Reading**: CP-transfer consistently sits between WP-small and WP-large across all five metrics. The gap to WP-large shrinks for Top-K metrics (retrieving the file anywhere in top-10) but persists for ranking quality (MAP, MRR). CP-cold-start is a zero-shot baseline and confirms cross-project models start from near zero before fine-tuning.

---

## The Core Finding

> **Cross-project transfer (CP-transfer) outperforms within-project training on limited data (WP-small) in the majority of pairs across ranking metrics — with statistical significance.**

| Metric | CP-transfer > WP-small | Mean gain | Wilcoxon p |
|---|---|---|---|
| Top-1 | 37.0% (34/92) | +0.007 | 0.118 (ns) |
| Top-5 | 43.5% (40/92) | +0.021 | 0.007 ** |
| Top-10 | 47.8% (44/92) | +0.026 | 0.001 ** |
| MAP | 64.1% (59/92) | +0.020 | 0.006 ** |
| MRR | **67.4% (62/92)** | +0.022 | 0.005 ** |

**Key observation**: The CPL advantage is primarily in *ranking quality* (MAP, MRR) rather than exact-hit recall (Top-1). This makes sense — the cross-project model has seen more diverse bug-file pairing patterns and learns better relative ordering, but not necessarily sharper top-of-list precision. Top-1 improvement is non-significant; MRR improvement is highly significant (p = 0.005).

---

## Six Evidence Points

### E1 — CP-transfer beats WP-small in ranking quality

CP-transfer significantly improves MAP and MRR over WP-small (p < 0.01) in a majority of pairs. Even on Top-10 recall — the laxest hit criterion — CP-transfer wins in 47.8% of pairs with a mean gain of +0.026.

**Research implication**: When a target project has limited bug history, CPL pre-training provides better ranking of candidate files than pure within-project training, particularly for downstream tasks where rank matters (e.g., developer triage tools that show a sorted list).

---

### E2 — CP-transfer nearly matches or exceeds WP-large in recall metrics

| Metric | CP-transfer ≥ WP-large | Within 10% of WP-large | Mean gap |
|---|---|---|---|
| Top-1 | 54.3% (50/92) | 62.0% (57/92) | −0.018 |
| Top-5 | 52.2% (48/92) | 57.6% (53/92) | −0.012 |
| Top-10 | 53.3% (49/92) | 63.0% (58/92) | −0.010 |
| MAP | 30.4% (28/92) | 41.3% (38/92) | −0.023 |
| MRR | 32.6% (30/92) | 47.8% (44/92) | −0.025 |

This is a nuanced result. For recall (Top-K), CP-transfer matches or exceeds WP-large in over half of pairs, with the correct file appearing in the top-10 in 63% of pairs within 10% of WP-large. For ranking quality (MAP, MRR), WP-large retains an advantage in the majority of pairs.

**Interpretation**: CP-transfer retrieves the correct file at broadly the same level as WP-large but ranks it slightly less precisely. For use-cases where recall matters more than exact rank (e.g., showing a shortlist of 10 files), CP-transfer is essentially equivalent to WP-large. For precise rank-1 localisation, WP-large retains an edge.

**Research implication**: For data-scarce projects, CP-transfer provides WP-large-level recall without the need for a large annotated bug corpus. The ranking gap (MAP, MRR) is the remaining challenge for architecture improvement.

---

### E3 — Zero-shot CPL (CP-cold-start) establishes a non-trivial floor

CP-cold-start uses no target data. Its absolute numbers are low (MAP=0.038, MRR=0.042), but this is the hardest regime possible — a model trained on project A applied directly to project B with no adaptation.

That this is better than random retrieval in the majority of pairs shows that *structural and semantic bug-file patterns generalise across projects* to some degree.

**Research implication**: CPL is not a purely domain-specific approach. There is a cross-project signal that is learnable and transferable. Fine-tuning amplifies this signal into the CP-transfer result.

---

### E4 — CPL benefit is strongest when target has fewest bugs

| Target bug count group | Top-5 gain | Top-10 gain | MAP gain | MRR gain |
|---|---|---|---|---|
| Few bugs (≤228, n=48) | **+0.034** | **+0.043** | **+0.029** | **+0.032** |
| Many bugs (>228, n=46) | +0.007 | +0.007 | +0.011 | +0.012 |

CPL gain is 3–4× larger for data-scarce targets across all ranking and recall metrics. This confirms the intuition that CPL is most valuable precisely where it is most needed.

**Research implication**: CPL is not a general-purpose improvement. It is specifically targeted at the data-scarce regime — new projects, low-activity codebases, or projects with noisy/incomplete bug histories. This is the majority of real-world software projects.

---

### E5 — Target properties, not source properties, govern achievable performance

| Feature | MAP ρ | MRR ρ | Top-10 ρ |
|---|---|---|---|
| **tgt_LoC** | **−0.652 \*\*\*** | **−0.667 \*\*\*** | **−0.855 \*\*\*** |
| tgt_bug_report_verbosity | +0.383 \*\*\* | +0.297 \*\* | +0.333 \*\* |
| tgt_n_bugs | +0.311 \*\* | +0.319 \*\* | +0.134 ns |
| src_LoC | +0.252 \* | +0.259 \* | +0.274 \*\* |
| src_n_bugs | +0.016 ns | +0.070 ns | −0.045 ns |
| domain_gap | +0.012 ns | +0.038 ns | +0.116 ns |

**tgt_LoC is by far the strongest predictor** (ρ = −0.855 for Top-10). Target codebase size alone explains the majority of variance in whether CPL achieves good Top-10 recall. Bug report verbosity is the second feature. All source-side features are weak or non-significant. Domain gap has no significant predictive power.

**Research implication**: When evaluating or deploying CPL, the primary question to ask is: *how large is the target codebase, and how informative are its bug reports?* Source selection matters much less than these target-side properties.

---

### E6 — Domain gap does not prevent CPL benefit

| Domain gap group | Top-1 | Top-5 | Top-10 | MAP | MRR |
|---|---|---|---|---|---|
| Low gap (≤ median, n=48) | 33.3% win | 43.8% | 47.9% | 56.2% | 64.6% |
| High gap (> median, n=46) | 39.1% win | 41.3% | 45.7% | 69.6% | 67.4% |

*(% pairs where CP-transfer > WP-small)*

CP-transfer beats WP-small at comparable rates regardless of domain gap. For MAP/MRR, high domain-gap pairs show slightly higher CPL win rates. Domain gap (logistic regression accuracy between blob embeddings) is a non-significant predictor of CPL gain (ρ < 0.12, p > 0.05 for all metrics).

**Research implication**: Domain dissimilarity between source and target is not a contraindication for CPL. Practitioners should not gate source selection on domain gap measurements.

---

## The CPL Value Proposition (for Research Paper)

| Claim | Evidence | Strongest metric support |
|---|---|---|
| CPL improves ranking quality over WP-small | 67.4% pairs, Wilcoxon p=0.005 | MRR, MAP |
| CPL achieves WP-large-level recall in most pairs | 53–54% pairs on Top-5/10 | Top-10 |
| CPL is most valuable when target data is limited | 3–4× larger gain for few-bug targets | Top-10, MRR |
| Target size is the dominant performance predictor | ρ = −0.855 (Top-10) | Top-10 |
| Domain gap does not limit CPL | ρ < 0.12, p > 0.05 for all metrics | All |

---

## Implications for Researchers

### 1. Report stratified results by target codebase size
tgt_LoC is the dominant predictor (ρ = −0.855 for Top-10). Mean MRR/MAP averages conflate easy (small codebase) and hard (large codebase) targets. Stratified reporting is essential for fair comparison across studies.

### 2. Use CP-transfer as the default strategy for new/small projects
For targets with ≤228 bugs (median), CP-transfer gains +0.043 Top-10 and +0.032 MRR over WP-small. The expected gain is 3–4× larger than for data-rich targets.

### 3. Top-K recall and MAP/MRR tell different stories — report both
CP-transfer matches WP-large on Top-10 in 53% of pairs but only on MAP in 30%. A study reporting only recall metrics would overstate CPL's precision quality. A study reporting only MAP/MRR would understate its practical utility for shortlist-based tools.

### 4. Architecture improvement targets the ranking gap, not the retrieval gap
pct_unreachable = 0% (from summary analysis): all ground-truth files are in FAISS top-300. Failures are ranking failures. The MAP/MRR gap between CP-transfer and WP-large is the architectural challenge for future work — not data collection.

### 5. Do not filter source selection by domain gap
Domain gap has no significant correlation with CPL performance (ρ < 0.12). The most-bugs heuristic (41.7% Hit@1, Kendall τ = +0.25) is a better source selection strategy than any domain-similarity filter.

---

## Limitations and Honest Framing

- Mean MRR 0.20–0.25 is low by within-project SOTA standards (>0.5). The correct comparison is our own FAISS baseline and WP-small, not cross-study comparisons.
- Large-codebase targets (e.g., 438K LoC) dominate the low-performance tail. Stratified reporting is more informative than overall means.
- Results include Python and Java projects. Generalisation to C++, JavaScript, or polyglot projects is an open question.
- CP-cold-start numbers are very low (MAP=0.038). Zero-shot cross-project transfer without any fine-tuning has limited practical value at this stage.

---

## Updated Landscape: BLAZE + COOBA (obj1_experimental_results.csv, 263 rows)

The analyses above were written for TRANP-CNN. With BLAZE and COOBA now in results, the picture is sharper — and more complicated.

| Model | Pairs | CP-transfer > WP-small (MRR) | Mean delta (MRR) | Negative transfer cases |
|---|---|---|---|---|
| **BLAZE** | 30 | **93.3% (28/30)** | **+0.180** | 2 (both Δ ≈ 0) |
| **COOBA** | 34 | 64.7% (22/34) | +0.015 | **12 (up to −0.152)** |

**Key observation**: BLAZE makes CPL almost uniformly beneficial. COOBA is inconsistent — its +0.015 mean masks 12 genuine negative transfer cases. The architectures have very different risk profiles for CPL. This is not yet explained.

---

## Analysis Roadmap — Open Questions and Planned Scripts

These analyses are not yet done. They fill the gaps needed to complete the "when CPL works / when it does not" argument.

---

### A1 — Negative Transfer Characterization
**Question**: What observable features predict the sign and magnitude of `CP-transfer − WP-small`?

- Compute per-pair delta (MRR, MAP) for each model.
- Correlate against: `target_LoC`, `source_LoC`, `target_n_bugs`, `source_n_bugs`, `src/tgt LoC ratio`, `domain_gap`, `src_WP-large MRR`.
- Classify pairs as positive / neutral / negative transfer and test which features separate them.
- COOBA has 12 negative transfer cases to analyze; BLAZE has near zero — this contrast is a finding in itself.

**Script**: `Scripts/analysis/negative_transfer_analysis.py`
**Output**: `results/negative_transfer_analysis.csv` + scatter plots

---

### A2 — Cross-Model Consistency
**Question**: Do BLAZE and COOBA agree on which pairs benefit from CPL?

- 17 pairs have results from both models. Agreement on CPL benefit direction is currently 70.6% (12/17).
- Scatter `BLAZE_delta` vs `COOBA_delta` per pair. Where they disagree, examine what's different.
- Test whether agreement correlates with target LoC or source quality.
- If both models agree a pair benefits, that is architecture-agnostic evidence. If they disagree, the benefit is model-specific and weaker as a claim.

**Script**: `Scripts/analysis/cross_model_consistency_analysis.py`
**Output**: `results/cross_model_agreement.csv` + scatter plot

---

### A3 — CP-Cold-Start Viability
**Question**: Under what conditions is zero-shot CPL (no target labels at all) acceptable?

- BLAZE CP-cold-start mean MRR = 0.302, COOBA = 0.029 — radically different baselines.
- For BLAZE specifically: identify the target LoC and source quality thresholds where CP-cold-start alone achieves MRR > 0.2 (a useful shortlist).
- Practical implication: for small, new projects with no bug history, is CPL deployable without any target annotation?

**Script**: `Scripts/analysis/cold_start_viability_analysis.py`
**Output**: `results/cold_start_viability.csv` + heatmap by {target_LoC_quartile × source_n_bugs_quartile}

---

### A4 — Effect Size Distribution
**Question**: Are CPL gains large and losses small, or are they symmetric?

- Win rate alone does not justify adopting CPL. A 65% win rate where wins are +0.002 and losses are −0.15 is not a useful technology.
- Plot the full distribution of `CP-transfer − WP-small` deltas, separately per model.
- Report Cohen's d, median gain among winners, median loss among losers.
- Expected finding (from raw numbers): BLAZE wins are large and losses are negligible; COOBA wins are modest and losses are real.

**Script**: `Scripts/analysis/effect_size_analysis.py`
**Output**: `results/effect_size_summary.csv` + violin/histogram plots

---

### A5 — Source Quality as Transfer Predictor
**Question**: Does a source project where the model performs well within-project (high WP-large MRR) produce better CPL transfer?

- For each `(source, target)` pair, correlate `source_WP-large_MRR` with `CP-transfer_MRR_on_target`.
- If ρ is significant, this gives a principled second source selection criterion beyond "most-bugs heuristic".
- Also test: does `source_WP-large` predict the *sign* of the transfer (positive vs negative)?

**Script**: `Scripts/analysis/source_quality_transfer_analysis.py`
**Output**: `results/source_quality_transfer.csv` + scatter with regression line

---

### A6 — Systematic Commutativity Analysis
**Question**: When source and target are swapped, does CPL always benefit the smaller target — and hurt the larger one?

- Currently demonstrated for 3 showcase pairs only.
- For all symmetric pairs (A→B and B→A both in results), compute CPL benefit direction in each direction.
- Test: does the smaller-LoC target always benefit from CPL? Does the larger always suffer?
- This would upgrade the commutativity finding from an anecdote to a systematic result.

**Script**: Add as a section to `Scripts/analysis/negative_transfer_analysis.py` (reuses same data)
**Output**: `results/commutativity_summary.csv` + paired bar chart

---

### Summary of Planned Scripts

| Script | Core question | Key output |
|---|---|---|
| `negative_transfer_analysis.py` | What predicts negative transfer? | Feature correlations, classified pairs, commutativity |
| `cross_model_consistency_analysis.py` | Do BLAZE and COOBA agree? | 17-pair agreement analysis, scatter |
| `cold_start_viability_analysis.py` | When is zero-shot CPL enough? | Viability heatmap by target LoC × source quality |
| `effect_size_analysis.py` | Are gains large and losses small? | Cohen's d, delta distributions |
| `source_quality_transfer_analysis.py` | Does source quality predict transfer? | Spearman ρ, scatter |

All scripts read from `results/obj1_experimental_results.csv` and `data/processed/project_metadata.parquet`.

---

*Data source: `results/obj1_experimental_results.csv` (263 rows, BLAZE + COOBA)*
*Supporting data: `results/all_project_domain_gaps.csv`, `data/processed/project_metadata.parquet`*
*Analysis scripts: `Scripts/analysis/aggregate_tranp_cnn_results.py`, `Scripts/analysis/source_selection_analysis.py`*
