# TODO — WP-small matched-budget correction

Status as of 2026-10-10: **re-run, merge, analysis recompute, figure refresh and
main.tex edits complete.** Remaining work is human review + PDF compile (no LaTeX
toolchain in the agent environment, same as both prior passes) and the Zenodo
re-version.

This is the third correction pass on this paper, after `TODO_metric_correction.md`
(MRR/MAP denominator) and `TODO_blaze_restriction_correction.md` (BLAZE FAISS pool).

## What the bug was

The scenario-generation code drew WP-small's training share as **12.5% of the 80%
train pool** (`train_size=0.125`), i.e. **~10% of each target's bug corpus**, while
CP-transfer drew **25% of the same pool** (~20% of the corpus). The paper's primary
comparison — RQ1, CP-transfer vs WP-small, and every CPL-gain-derived result built
on it — was therefore **not budget-matched**: the baseline was trained on half the
target labels the CPL condition got.

Fixed in `Scripts/run_expts_ph1.py`, `Scripts/blaze_faiss_restricted_sweep_wp_small.py`,
and `src/tranp_cnn/pipeline.py` (`TARGET_TRAIN_SIZE`). The other three scenarios were
unaffected and were not re-run.

## What this pass did

1. **Re-ran WP-small at the corrected 20% budget, in triplicate.** WP-small depends
   only on the target project, so this is 3 runs × 13 targets × 3 models = 117 runs
   covering all 63 pairs, not 3 × 63 × 3. Driver:
   `Scripts/run_wp_small_matched_budget.py` (`WP_SMALL_REP` / `WP_SMALL_REVERSE` env
   vars for concurrent replicate sweeps, plus OOM retry). Outputs:
   `results/wp_small_matched_budget{,_rep2,_rep3}.csv`.
2. **Point estimate = mean of the 3 reps** (author decision). The choice matters most
   for COOBA, whose MRR gain is +0.009 under the median and −0.0005 under the mean.
   `Scripts/analysis/merge_wp_small_matched_budget.py` writes the per-(model,target)
   summary (median/min/max/mean/std) to `results/wp_small_matched_budget_summary.csv`
   and patches the mean into the three primary results files
   (189 / 250 / 250 WP-small rows).
3. **Rebased the derivation.** An earlier partial pass (commit `f9a0a44`) had already
   merged **rep1-only** values and `results/pre_budget_fix_backup/` held that
   intermediate state, so any before/after diff against it measured "rep1 vs final
   estimate", not "10% vs 20% budget". Both the primary files and the backup are now
   derived from commit `233ddd4`, the genuine pre-correction baseline; see
   `results/pre_budget_fix_backup/README.md`. Verified the non-WP-small rows
   (567 / 747 / 747) are byte-identical to that baseline.
4. **Fixed two now-degenerate analyses.** The merge broadcasts one value into every
   WP-small row sharing a target, so the per-pair "repeats" in the primary files have
   zero spread by construction — which silently collapsed the noise-floor robustness
   check (this was the open issue flagged in commit `5ca1cfa`).
   `negative_transfer_analysis.py` and `equivalence_and_robustness_tests.py` now read
   the noise floor from `wp_small_matched_budget_summary.csv` and the raw rep files
   instead. Their now-redundant "median re-test" sections were repurposed into
   **per-replicate sensitivity** checks (does the conclusion hold under each single
   rep, rather than only under their mean).
5. **Reran all 9 WP-small-dependent analysis scripts** plus `mixed_effects_check.py`,
   and refreshed the six hand-copied paper figures (`journal_figures.py` writes its
   three directly into `journal_draft/figs/`). All nine in-use figures now postdate
   the patch. `Study_Design.png` needed no redraw — it names the scenarios without
   stating a target-label budget.
6. **Deleted the duplicate** `Scripts/merge_wp_small_matched_budget.py` (rep1-only,
   from the partial pass), superseded by the one in `Scripts/analysis/`.
7. **Rewrote main.tex end to end** against the corrected numbers — see below.

## Headline numbers, before → after

Before = the original 10%-budget results the draft was written against.

| Metric | Before | After |
|---|---|---|
| BLAZE Cohen's d (CPT vs WPS, MRR) | 1.16 | 0.55 |
| BLAZE win rate (MRR) | 95.2% | 60.3% |
| BLAZE negative transfer | 1.6% | 27.0% |
| TRANP-CNN Cohen's d (MRR) | 0.65 | 0.67 |
| TRANP-CNN win rate (MRR) | 68.3% | 71.4% |
| TRANP-CNN negative transfer | 15.9% | 15.9% |
| COOBA Cohen's d (MRR) | 0.27 | −0.01 |
| COOBA win rate (MRR) | 60.3% | 47.6% |
| COOBA negative transfer | 31.7% | 31.7% |
| Cross-model agreement range | 44–54% | 38–56% |
| WP-small noise floor (mean spread, max) | 0.05–0.07, 0.25 | 0.03–0.05, 0.15 |

**BLAZE absorbed almost all of the correction.** Its WP-small gained most from the
fair budget (+0.058 MRR), which is what cut its effect size and raised its
negative-transfer rate. TRANP-CNN is essentially unchanged — its WP-small barely moved
(0.185 → 0.182). COOBA lost what little effect it had.

## Narrative consequences (not just renumbering)

- **"2 of 3 architectures benefit" survives**, and BLAZE moves from *borderline*
  (sitting exactly on the d = 0.5 cutoff) to clearly meeting the practically-beneficial
  criterion. But the three models no longer span a spectrum: TRANP-CNN (0.67) and
  BLAZE (0.55) are statistically indistinguishable, and COOBA (−0.01) has no effect.
  The paper now reads as a two-group result.
- **The draft was internally contradictory before this pass** and is no longer. Two
  incompatible negative-transfer sets coexisted (30.2/9.5/30.2 in the abstract,
  contributions and summary boxes, from the rep1 partial migration; 1.6/15.9/31.7 in
  §4.2.2 and the conclusion, from the original data), and §4.2.2 asserted the
  jupyterlab/AST-overfitting story in one paragraph and retracted it in the next.
- **Negative transfer is now common for every architecture** (15.9 / 27.0 / 31.7%) and
  survives every noise-calibrated band (6.3 / 7.9 / 9.5% under the strictest), so it is
  no longer attributable to the classification threshold. But COOBA's
  jupyterlab-as-target cases specifically do not survive, so the AST-overfitting
  reading was dropped rather than asserted.
- **COOBA's scenario ordering breaks**: its CP-transfer (0.156) and WP-small (0.157)
  are indistinguishable, so "WP-large > CP-transfer > WP-small > CP-cold-start holds
  cleanly across all five metrics" is now stated for BLAZE and TRANP-CNN only.
- **Domain alignment helps BLAZE only** (24.5pp). For TRANP-CNN (−3.6pp) and COOBA
  (−12.4pp) the within/cross contrast is mildly reversed.
- **COOBA is uncorrelated with both other models** at the pair level (ρ = 0.10 and
  0.15, both n.s.); only BLAZE↔TRANP-CNN remains significant (ρ = 0.505).
- **Case 3 was re-selected** (author-approved, per the precedent in
  `TODO_blaze_restriction_correction.md`): `jupyterlab → scikit-learn` rested on
  scikit-learn's WP-small having collapsed to 0.002, which the corrected budget lifts
  to 0.046, and BLAZE now reads negative there against both baselines. Replaced with
  `prefect → jupyterlab`, where two of three architectures clear their own WP-large
  ceiling. `numpy → xarray` was the numerically stronger candidate (TRANP-CNN +0.121,
  all three clear the ceiling) but was rejected to keep the three cases on distinct
  targets.
- **The paper now discloses the correction**, which it had stopped doing: a third entry
  in the §3.7 metrics footnote, and the 3-replicate design in §3.6 and `tab:scenarios`
  (which gains a "Runs" column).

## New items surfaced by this pass (author should verify)

- **The mixed-effects result flipped direction.** Previously TRANP-CNN's gain was
  significantly higher than BLAZE's in the pooled model (+0.020, p = 0.012) and COOBA's
  did not differ (−0.011, p = 0.17). Now TRANP-CNN does not differ from BLAZE (−0.003,
  p = 0.70) and COOBA is clearly lower (−0.036, p < 0.001). The clustered analysis now
  tells the same two-group story as the effect sizes, which is a cleaner result, but it
  is a reversal and worth a careful read.
- **BLAZE's clustering caveat survives but means something different.** It still fails
  the per-target test (p = 0.073, 8/13), but its per-model point estimate (+0.022) is
  now essentially TRANP-CNN's (+0.024, p = 0.016), so the difference is power at 13
  clusters, not effect magnitude. The abstract's "not significant once pairs are
  clustered" clause is still true but should not be read as BLAZE being weaker.
- **COOBA Top-1 and MRR now have nominally negative effect sizes** (d = −0.08, −0.01).
  Worth deciding whether the paper should say more about COOBA plausibly being *harmed*
  by CPL rather than merely not helped.

## Known items intentionally left untouched

- **COOBA's source-selection oracle-recovery percentages** (paper 78.8%/79.2% vs script
  73.8%/84.8%) and **TRANP-CNN's domain-gap ρ** (paper 0.11 vs script −0.142). Both are
  pre-existing drift, verified and deliberately left alone by the two prior passes;
  unchanged here for the same reason.
- `sec:pipeline` and `sec:statmethods` are labelled but never `\ref`'d — harmless,
  pre-existing.
- The three case-study tabulars are still unlabelled, so they cannot be cross-referenced.
- Stale secondary docs describing earlier study designs: `results/SUMMARY.md`,
  `DETAILED_ANALYSIS.md`, `ANALYSIS_RUNBOOK.md`, `ANALYSIS_RESULTS_EXPLANATION.md`,
  `CPL_DESIRABILITY.md`, `PAPER_RUN_PLAN.md`, `SOURCE_PROJECT_SELECTION.md`.
- `DC_5_Report/` still carries the pre-correction numbers. It is a past progress report,
  not a live artifact.

## Remaining before submission

1. **Compile `main.tex` and proofread the rendered diff.** No `pdflatex`/`bibtex` in the
   agent environment. Structural checks pass (braces, math delimiters, environments all
   balance; every `\ref` resolves; the previously broken `ef{tab:stratified}` is fixed),
   but the PDF has not been rendered. `main.pdf` is now two correction passes behind.
2. **Re-version the Zenodo replication package.** Rebuild
   `zenodo_upload/cpl_code_artifact.zip` and `cpl_data_artifact.zip`, then update the
   DOI at the end of `main.tex` (Data Availability). The author does the upload.
3. **Sync the public `main` branch** from `master`.
4. Read the three "new items surfaced" above.
