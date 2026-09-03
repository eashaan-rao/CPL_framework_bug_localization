# TODO — MRR/MAP denominator correction

Status as of 2026-09-03. Branch `agents/data-claims-validation`.

## What the bug was

The per-model `evaluate()` code that produced the submitted results averaged
**MRR and MAP only over the test bug reports that reached the reranking stage
with a resolvable ground-truth file**, while **Top-K was averaged over all test
reports**. For TRANP-CNN that "reachable" subset is the bugs whose ground truth
was retrieved into the FAISS top-300; for BLAZE it is the bugs whose ground-truth
blob resolves in the commit snapshot. Result: MRR/MAP were inflated relative to
Top-K, and several TRANP-CNN MRR values in Table 7 exceeded the maximum their own
Top-1/5/10 permit (e.g. WP-small MRR 0.326 vs a ceiling of 0.262).

- Fixed in code post-submission: `src/tranp_cnn/pipeline.py` (commit `b06497a`),
  `src/blaze/pipeline.py` (commit `bb933dc`). COOBA's `evaluate_cooba`
  (`src/cooba/pipeline.py`) divided by `len(predictions)` and was never affected.
- The submitted result CSVs (`results/obj1_experimental_results.csv`,
  `results/paper_results_complete.csv`) were last generated before those fixes.

## Recomputation (done, no retraining)

Per-bug ground-truth ranks for all 756 runs are archived in
`results/{tranp_cnn,blaze,cooba}_ph1_diagnostics/*.csv`. From those, MRR/MAP were
recomputed with **all** test reports in the denominator (an unretrieved ground
truth scores reciprocal rank / AP = 0). The true per-run test-set size was
recovered by inverting the already-correct Top-K fractions; it matches
`ceil(0.20 * project_bugs)` within +/-3 and is identical across BLAZE and
TRANP-CNN for each target project.

Validation: for all 504 TRANP-CNN + BLAZE runs, the reachable-only recomputation
reproduces the published MRR to a maximum error of 0.00005 — i.e. the published
values are exactly `sum(1/rank) / n_reachable`. COOBA differs by ~0.09 (not
broken).

Corrected per-run data: **`results/paper_results_complete_corrected.csv`**
(Top-K byte-identical to the original; MRR/MAP corrected; extra columns
`n_reachable`, `n_test`, `denominator_corrected`). Recompute scripts:
scratchpad `recompute3.py` / `finalize.py` (stdlib only).

## Corrected Table 7 (`tab:scenario_means`) — APPLIED to main.tex

| Model | Scenario | MRR old -> new | MAP old -> new |
|-------|----------|----------------|----------------|
| BLAZE | WP-small | 0.256 -> 0.254 | 0.210 -> 0.209 |
| BLAZE | WP-large | 0.471 -> 0.466 | 0.391 -> 0.387 |
| BLAZE | CP-cold-start | 0.262 -> 0.259 | 0.214 -> 0.211 |
| BLAZE | CP-transfer | 0.392 -> 0.388 | 0.322 -> 0.319 |
| COOBA | all | unchanged | unchanged |
| TRANP-CNN | WP-small | 0.326 -> 0.185 | 0.288 -> 0.159 |
| TRANP-CNN | WP-large | 0.389 -> 0.219 | 0.349 -> 0.192 |
| TRANP-CNN | CP-cold-start | 0.075 -> 0.038 | 0.065 -> 0.033 |
| TRANP-CNN | CP-transfer | 0.380 -> 0.214 | 0.337 -> 0.185 |

Top-1/5/10 unchanged everywhere. Every corrected MRR now sits under its Top-K
ceiling. RQ1 conclusion survives: TRANP-CNN CP-transfer still beats WP-small
(+0.029, 68.3% win rate, still p<0.001) and still approximately equals its own
WP-large (0.214 vs 0.219).

## Already edited in main.tex

- Table 7 — all BLAZE + TRANP-CNN MRR/MAP cells.
- Table 7 caption — note that MRR/MAP use the all-reports denominator.
- Section "Evaluation Metrics" (`sec:metrics`) — footnote documenting the correction.
- Section 4.1.1 prose — BLAZE `0.392/0.256/+0.137 -> 0.388/0.254/+0.134`;
  TRANP-CNN `0.380/0.326/+0.054/17% -> 0.214/0.185/+0.029/16%`, WP-large `0.389 -> 0.219`.
  Fragile TOST/Wilcoxon p-values moved into an inline `% CORRECTION-PENDING` note.
- Table 4 (`tab:dataset`) caption — Recall@300 bound scoped to TRANP-CNN/COOBA;
  notes BLAZE reranks the full snapshot.
- A `% CORRECTION-PENDING` block after the Scenario Performance Overview heading.

## Remaining — needs pandas/scipy on the corrected CSV

Point `RESULTS_CSV` in the analysis scripts at
`results/paper_results_complete_corrected.csv` and rerun.

### 1. Table 8 (`tab:wilcoxon`) — `main_results_table.py` + `effect_size_analysis.py`
- TRANP-CNN MRR row: d 0.59 -> 0.65, delta-mean +0.054 -> +0.029,
  delta-median +0.041 -> +0.021, win% 68.3 and `***` unchanged.
  **95% CI(d) and r_rb must be regenerated** (that CI method is not in any repo script).
- TRANP-CNN MAP row: d 0.56 -> 0.66, delta-mean +0.049 -> +0.026,
  delta-median +0.029 -> +0.018, win% 71.4 and `***` unchanged; CI + r_rb regenerate.
- BLAZE MRR/MAP rows: delta-mean +0.137 -> +0.134 and +0.112 -> +0.110; d ~unchanged (1.19 / 1.18).
- COOBA rows: unchanged.

### 2. TOST / two-sided Wilcoxon p-values
- TRANP-CNN CP-transfer vs WP-large (Section 4.1.1 inline note): gap is now only
  -0.005, so +/-0.05 equivalence holds and +/-0.02 likely now holds too — rerun for exact p.
- BLAZE cold-start TOST `p = 0.007` (abstract, contributions, cold-start section,
  summary list): BLAZE cold-start gap is now +0.005; rerun.

### 3. Prose with stale `d = 0.59` or MRR/MAP values
- Abstract (~L69): `d = 0.59` -> `0.65`.
- Contributions section (~L164-166): `d = 0.59` -> `0.65`.
- Section 4.1.2 (~L1077, L1085): d range `0.32-0.59` -> `0.32-0.66`;
  `BLAZE (d = 1.19) and TRANP-CNN (d = 0.59)` -> `(d = 0.65)`;
  ordering `1.19 / 0.59 / 0.27` -> `1.19 / 0.65 / 0.27`.
- Cold-start viability section (~L1184-1196): BLAZE `MRR = 0.262` -> `0.259`;
  **TRANP-CNN `MRR = 0.075 ... vs 0.326` -> `0.038 ... vs 0.185`, and
  "2.6x the FAISS baseline" is now ~1.3x — reword**; COOBA `0.027` unchanged.
- Summary list (~L1208-1213), Section 6/7 discussion (~L1791, L1808-1810, L1887),
  Conclusion (~L1980-1983): repeated `d = 0.59` -> `0.65`;
  `MRR 0.262 vs 0.392` -> `0.259 vs 0.388`; `MRR 0.075 vs 0.326` -> `0.038 vs 0.185`.

### 4. TRANP-CNN negative-transfer rate (+/-0.01 MRR band on CP-transfer minus WP-small)
- `27.0% -> 15.9%` (win / neutral / loss counts `39 / 7 / 17` -> `36 / 17 / 10`).
- COOBA's 31.7% (abstract) is unchanged.
- Check `tab:crossdomain` and Section 4.2 for any TRANP-CNN band-based win rate.

### 5. Stratified-by-size table (`tab:stratified`, ~L1275) and `effect_size_stratified_table.csv`
- Regenerate; the TRANP-CNN "large-target mean 0.242" and similar figures shift.

### 6. Case studies
- Case 1 (~L1618-1636, prefect -> lightning): BLAZE row **unchanged**
  (n_reachable = n_test = 75). TRANP-CNN row `0.349/0.405/0.062/0.371 ->
  0.344/0.399/0.061/0.367` (trivial). COOBA unchanged. Likely no edit.
- Case 2 (~L1588-1610, ansible -> jupyterlab): BLAZE gain `+0.374 -> +0.355`;
  COOBA `-0.081` unchanged; TRANP-CNN `+0.146 -> +0.073` from baseline
  `0.654 -> 0.327` (drop "from a high within-project baseline"). Second
  paragraph is about jupyterlab -> scikit-learn ("surpasses 4x the local data",
  "collapse to 0.023", "WPL 0.138"): TRANP-CNN values there shrink ~72%
  (n_reachable = 13 vs n_test = 46) — re-examine whether "surpasses WPL" still holds.
- Case at ~L1688-1712 (`BLAZE 0.236/0.510/0.141/0.380` table): identify the pair;
  BLAZE per-pair barely moves, but recompute any TRANP-CNN / COOBA rows.

### 7. Figures — rerun `journal_figures.py`
- `fig_neg_transfer_mrr_map`, `fig_cohens_d_all_metrics`, `fig_win_rate_metrics`,
  `fig_coldstart_metrics`, `fig_coldstart_heatmap`.

### 8. Stale derived result CSVs (regenerate)
- `main_results_scenario_means.csv`, `main_results_wilcoxon_tests.csv`,
  `effect_size_summary.csv`, `effect_size_stratified_table.csv`,
  `cold_start_viability.csv`, `negative_transfer_analysis.csv`,
  `source_quality_transfer.csv`, `cross_model_agreement.csv`.

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
