# TODO — BLAZE FAISS-top-300 restriction correction

Status as of 2026-09-17: **CSV correction, analysis recompute, and main.tex edits
complete.** Remaining work is human review + PDF compile (no LaTeX toolchain
available in the agent environment that did this pass).

## What the bug was

`src/blaze/pipeline.py`'s `evaluate()` ranked the **entire file snapshot**
instead of being restricted to the same FAISS top-300 candidate pool that
TRANP-CNN and COOBA use, even though the paper's text describes all three
models as sharing one candidate pool. This was a known, previously-deferred
issue — see the "Separate issue — NOT a recompute, needs an author decision"
section of `TODO_metric_correction.md` (the prior MRR/MAP-denominator
correction pass), which explicitly left it open pending an author decision.
Fixed in commit `10afdc9` ("Restrict BLAZE evaluation to the shared FAISS
top-300 candidate pool").

## What this pass did

1. Re-ran BLAZE across all 63 source→target pairs × 4 scenarios (WP-small,
   WP-large, CP-cold-start, CP-transfer) with the fix applied
   (`Scripts/blaze_faiss_restricted_sweep*.py`, four separate sweeps).
2. Replaced BLAZE's 252 rows in the three canonical result files with the
   FAISS-restricted numbers, leaving every COOBA/TRANP-CNN row and all
   formatting byte-identical (verified via sorted diff + line-ending checks):
   - `results/paper_results_complete_corrected.csv`
   - `results/obj1_experimental_results.csv`
   - `results/obj1_experimental_results_corrected.csv`
3. Re-ran all 9 downstream analysis scripts against the corrected data and
   regenerated every derived CSV and figure referenced by `main.tex`
   (`main_results_table.py`, `journal_figures.py`, `effect_size_analysis.py`,
   `cross_model_consistency_analysis.py`, `source_quality_transfer_analysis.py`,
   `cold_start_viability_analysis.py`, `negative_transfer_analysis.py`,
   `source_selection_analysis.py`, `equivalence_and_robustness_tests.py`).
   Confirmed via file mtimes and manual mapping that every figure actually
   `\includegraphics`'d in `main.tex` was refreshed (some paper-facing figure
   filenames differ from the scripts' own output names, e.g.
   `figs/fig_coldstart_heatmap.png` ← `results/images/cs_viability_heatmap_BLAZE.png`;
   `fig_agreement_scatter.png` turned out to be an unused orphan file —
   the real in-use file is `fig_cross_model_scatter.png`).
4. Re-selected two of the three worked case studies in `sec:cases`, since the
   old picks no longer serve their narrative role now that BLAZE is properly
   restricted (author-approved re-selection, not a unilateral change):
   - **Case 1** (cold-start viability): `prefect→lightning-ai` → **`lightning-ai→pydata/xarray`**
     (cleaner, larger margin; both other models collapse harder; no
     run-to-run-variance caveat needed since WP-small sits at the target's
     5-run median).
   - **Case 2** (architecture divergence): `ansible→jupyterlab` → **`wagtail→docker/compose`**
     (the old pair's BLAZE gain collapsed from +0.355 to +0.021 once
     properly restricted — jupyterlab's 47% Recall@300 was exactly what had
     inflated the old number; the new pair is genuinely cross-domain, has
     99.8% Recall@300, and needs no variance caveat).
   - **Case 3** (jupyterlab→scikit-learn) kept as-is; its narrative is about
     TRANP-CNN, which the bug never affected — only its BLAZE row was
     refreshed.
5. Applied ~40 line-level edits to `main.tex` covering every location either
   Explore agent flagged: the dataset-table caveat sentence (removed — now
   false, since BLAZE genuinely shares the pool), Table 7 (`tab:scenario_means`),
   Table 8 (`tab:wilcoxon`, including Cohen's d 95% CIs and rank-biserial r),
   Table 9 (`tab:source_selection`), `tab:model_agreement`, `tab:crossdomain`,
   `tab:stratified`, all three case studies, the cold-start viability heatmap
   caption (recomputed all 16 grid cells), the architecture/mechanism
   discussion, the threats section's run-to-run-variance numbers, and the
   abstract/highlights/introduction/conclusion headline figures.

## Headline numbers, before → after

| Metric | Before | After |
|---|---|---|
| BLAZE CP-transfer MRR (mean) | 0.388 | 0.298 |
| BLAZE Cohen's d (CPT vs WPS, MRR) | 1.20 | 1.16 |
| BLAZE win rate (CPT vs WPS, MRR) | 90.5% | 95.2% |
| BLAZE cold-start viability (MRR≥0.20) | 73.0% (46/63) | 46.0% (29/63) |
| BLAZE negative transfer rate | 3.2% (2/63) | 1.6% (1/63) |
| BLAZE within-domain vs cross-domain gap | 14.5pp | 2.1pp (nearly gone) |
| Cross-model agreement range | 41%–62% | 44%–62% |

The FAISS-restriction fix mostly *lowered* BLAZE's absolute numbers (it can no
longer score files outside the shared candidate pool) but, counter-intuitively,
*narrowed* its domain-alignment sensitivity — cross-domain and within-domain
performance are now much closer together than before.

## New items surfaced by this pass (author should verify)

- **BLAZE's domain-gap correlation is now significant** ($\rho = -0.29$,
  $p = 0.023$), where it previously wasn't ($|\rho| < 0.12$). This is a direct,
  high-confidence consequence of the fix (validated by reproducing TRANP-CNN's
  and the loc-ratio's unchanged numbers with the same script first). Both
  affected paragraphs (`sec:results` negative-transfer subsection and
  "Source-Target Domain Alignment" subsection) were updated to state this for
  BLAZE while leaving COOBA's and TRANP-CNN's own domain-gap numbers untouched.

## Known items intentionally left untouched

- **TRANP-CNN's domain-gap correlation** ($\rho = 0.11$, $p = 0.39$ in the
  paper) does not match an ad hoc reproduction using the same script
  (`negative_transfer_analysis.py`, which gives $\rho = -0.142$) — this is
  the *same* pre-existing, unrelated discrepancy the prior TODO already
  flagged and declined to fix without finding the original methodology. Not
  touched here either, since it predates and is independent of the BLAZE fix.
- **COOBA's source-selection oracle-recovery percentages** in
  `tab:source_selection`: a fresh run of `source_selection_analysis.py` gives
  73.8%/84.8% where the paper says 78.8%/79.2% — again the same category of
  pre-existing drift the prior TODO already verified and left alone for COOBA.
  BLAZE's own oracle-recovery numbers in the same table *were* updated, since
  those changes are attributable to this fix.
- **Three secondary results docs** (`results/ANALYSIS_RESULTS_EXPLANATION.md`,
  `results/ANALYSIS_RUNBOOK.md`, `results/PAPER_RUN_PLAN.md`) contain BLAZE
  numbers that don't match even the *pre*-FAISS-fix paper (e.g. WP-large MRR
  0.471 there vs. 0.466 in the old paper) — they were already stale from an
  earlier intermediate run, independent of this fix. Patching them to the new
  numbers would layer fabricated precision on an already-inconsistent
  document; they need a dedicated refresh pass, not a patch.
- `results/CPL_DESIRABILITY.md` is a legacy document for an entirely earlier
  study design (TRANP-CNN only, 20 projects, 94 pairs) and doesn't describe
  the current 63-pair 3-model study at all; left as a historical artifact.
- `results/SUMMARY.md`, `results/DETAILED_ANALYSIS.md`,
  `results/SOURCE_PROJECT_SELECTION.md` contain no BLAZE mentions (legacy
  TRANP-CNN-only docs) — not applicable.

## Remaining before submission

1. **Compile `main.tex` and proofread the diff.** No LaTeX toolchain was
   available in the environment that made these edits — structural checks
   (brace/dollar/environment balance) pass, but the PDF has not been rendered.
2. Decide whether the three stale secondary docs above are worth a dedicated
   refresh, or can be deleted/archived now that the paper itself is the
   authoritative source.
3. Spot-check the two "new items surfaced" findings above in context — they
   are new, real findings (not just corrections) and deserve a careful read
   to make sure the added sentences read naturally.
