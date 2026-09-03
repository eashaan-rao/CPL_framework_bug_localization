# TODO — MRR/MAP denominator correction

Status as of 2026-09-03: **all recompute and main.tex edits complete.** Remaining
work is human review + PDF compile (no LaTeX toolchain available in the agent
environment that did this pass).

## What the bug was

The per-model `evaluate()` code that produced the submitted results averaged
**MRR and MAP only over the test bug reports that reached the reranking stage
with a resolvable ground-truth file**, while **Top-K was averaged over all test
reports**. For TRANP-CNN that "reachable" subset is the bugs whose ground truth
was retrieved into the FAISS top-300; for BLAZE it is the bugs whose ground-truth
blob resolves in the commit snapshot. Result: MRR/MAP were inflated relative to
Top-K, and several TRANP-CNN MRR values in Table 7 exceeded the maximum their own
Top-1/5/10 permit.

- Fixed in code pre-existing this pass: `src/tranp_cnn/pipeline.py` (commit `b06497a`),
  `src/blaze/pipeline.py` (commit `bb933dc`). COOBA's `evaluate_cooba` was never affected.

## What this pass did

1. Built `results/obj1_experimental_results_corrected.csv` — the file
   `equivalence_and_robustness_tests.py` actually reads (997 rows; a superset of
   `paper_results_complete.csv`'s 756). Verified the 241 extra rows are all COOBA
   (unaffected by the bug) and the 756 shared rows were byte-identical to the old
   stale file before correction, then transplanted corrected MRR/MAP for the 756
   matched rows from `paper_results_complete_corrected.csv`.
2. Repointed `RESULTS_CSV` in all 8 analysis scripts (the 6 originally listed
   below, plus `equivalence_and_robustness_tests.py` -> the new corrected file
   above, and `cross_model_consistency_analysis.py` / `source_selection_analysis.py`
   which also silently pointed at the stale CSV).
3. Added a paired bootstrap 95% CI for Cohen's d to
   `equivalence_and_robustness_tests.py` (`bootstrap_ci_dz`, 10000 resamples) —
   this did not exist in any repo script before.
4. Generalized `cross_model_consistency_analysis.py` with a new
   `build_pairwise_table()` covering all three model-pair combinations
   (BLAZE-COOBA, BLAZE-TRANP-CNN, COOBA-TRANP-CNN) — previously it only computed
   BLAZE-vs-COOBA, but the paper's `tab:model_agreement` needs all three and no
   script produced that table.
5. Reran all 8 scripts; regenerated every derived CSV in `results/` and all 8
   figures in `journal_draft/figs/`.
6. Applied ~155 line-level edits to `main.tex`, covering not just the items
   below but every downstream consequence of TRANP-CNN's MRR shift: cross-domain
   sensitivity (Table `tab:crossdomain`), the 3-way cross-model table
   (`tab:model_agreement`), the stratified-by-size table (`tab:stratified`), the
   three worked case studies (Section `sec:cases`), and the run-to-run-variance /
   median-re-test numbers in the threats section (`sec:threats`). All three
   worked case-study pairs were identified (prefect->lightning, ansible->jupyterlab,
   jupyterlab->scikit-learn) and their per-scenario MRR tables + prose updated
   with exact figures pulled from the corrected data.

Full corrected Table 7, Table 8 (`tab:wilcoxon`, all 15 rows incl. new 95% CI
column), the abstract, contributions, and every other identified location are
now internally consistent with `results/paper_results_complete_corrected.csv`
and `results/obj1_experimental_results_corrected.csv`.

## Known items intentionally left untouched

- **COOBA's source-selection oracle-recovery percentages** (Section on source
  ranking heuristics, "78.8%--79.2%" for COOBA): verified this number was
  *already* stale relative to `source_selection_analysis.py`'s output using the
  **old, uncorrected** CSV too (i.e., pre-existing drift unrelated to the metric
  bug — COOBA's numbers are provably byte-identical old vs. new). Left as-is
  since fixing it is a different, unrelated correction.
- **Domain-gap correlation figures** in the "Source-Target Domain Alignment"
  subsection (`|ρ| < 0.12` for BLAZE/COOBA, `ρ = 0.11` for TRANP-CNN): an ad hoc
  recomputation gave meaningfully different numbers (BLAZE -0.15, COOBA -0.20),
  which may reflect a different methodology/subset than what produced the
  original text. Not confident enough to edit without finding the original
  computation; flagged for manual follow-up.
- **BLAZE full-corpus vs. FAISS top-300 issue** (below) — untouched, per its own
  note; this is a separate author decision, not a recompute.

## Remaining before submission

1. **Compile `main1.tex`/`main.tex` and proofread the diff.** No LaTeX toolchain
   was available in the environment that made these edits — structural checks
   (brace/dollar/environment balance) pass, but the PDF has not been rendered.
2. Spot-check the two "intentionally left untouched" items above.
3. Re-verify the two flagged reworded claims read naturally in context:
   - TRANP-CNN CP-transfer vs. WP-large TOST equivalence now holds even at the
     tightest tested margin (δ=0.02, p<0.001) — a *stronger* result than before,
     reworded accordingly in Section 4.1.1.
   - Case 2 (ansible->jupyterlab): BLAZE's gain there is no longer the single
     largest across all 63 pairs (prefect->scikit-learn is now larger, +0.393)
     — reworded to "one of its largest."

## Separate issue — NOT a recompute, needs an author decision

BLAZE's `evaluate()` ranks the **entire file snapshot, with no FAISS top-300
truncation** (`src/blaze/pipeline.py:357` and `:449-452` — `blaze_shas` is the
full tree). The paper describes BLAZE as reranking the shared FAISS top-300
(~L432, L476, L686) and frames all three models as sharing one candidate pool
(~L357, L487). This is why BLAZE's Top-10 (0.705) exceeds the mean Recall@300
(~65%) and why BLAZE beats 47.0% on the jupyterlab target — it scores
ground-truth files FAISS never retrieved.

Options:
1. Re-run BLAZE restricted to the FAISS top-300 (retrain/re-eval), or
2. Document that BLAZE was evaluated full-corpus, soften the "same candidate
   pool" claim, and note that part of BLAZE's absolute advantage over TRANP-CNN
   and COOBA is a wider candidate set.
