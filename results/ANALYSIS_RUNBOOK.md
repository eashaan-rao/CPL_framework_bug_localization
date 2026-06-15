# Analysis Runbook — COOBA + BLAZE Results

**Created**: 2026-06-15  
**Purpose**: Step-by-step guide to run all analysis scripts and produce paper-ready results for COOBA + BLAZE.  
**Primary data**: `results/obj1_experimental_results.csv` (BLAZE: 62 pairs, COOBA: 121 pairs)

---

## Current Input Status

| File | Status | Note |
|---|---|---|
| `results/obj1_experimental_results.csv` | ✅ Ready | 252 COOBA rows + 249 BLAZE rows + partial TRANP-CNN |
| `data/processed/project_metadata.parquet` | ✅ Ready | All 13 paper projects |
| `results/all_project_domain_gaps.csv` | ✅ Ready | All 13 projects covered |
| `data/processed/bug_reports_clean.parquet` | ✅ Ready | Used by FAISS |
| `data/processed/embedding_dbs/*.pkl` | ✅ Ready | Used by FAISS |
| FAISS results (paper projects) | ❌ **Missing** | Run Step 0 first |

---

## Always activate the environment first

```bash
source /home/cs21d002_eashaan/PhD/Objective1/obj1/bin/activate
cd /home/cs21d002_eashaan/PhD/Objective1
```

---

## Step 0 — FAISS Baseline (run in background, CPU-only)

**Why**: Establishes the retrieval ceiling (are failures ranking failures or retrieval failures?). Required for E3/E5 in CPL desirability and for the "ranking failure" claim in the paper.

**Command**:
```bash
nohup python Scripts/run_faiss_experiment.py > logs/faiss_run.log 2>&1 &
echo "FAISS PID: $!"
```

> Create the logs directory first if it doesn't exist: `mkdir -p logs`

**Estimated time**: ~2–4 hrs (CPU-only, no GPU needed, runs independently of TRANP-CNN)

**Expected outputs** (one file per project in `results/faiss_recall_results/`):
```
semantic_search_jupyterlab_jupyterlab_results.csv
semantic_search_lightning-ai_lightning_results.csv
... (13 files total)
```

Each CSV has columns: `Recall@30, Recall@50, Recall@75, ..., Recall@700`

**Skip check**: Already-complete projects are skipped automatically.

---

## Step 1 — Main Results Table (no dependencies, run now)

**Why**: Generates the primary Table 3 of the paper — mean metrics per scenario × model, Wilcoxon significance tests (RQ1), and CPL gain magnitude. No FAISS needed.

**Command**:
```bash
python Scripts/analysis/main_results_table.py
```

**Inputs**:
- `results/obj1_experimental_results.csv`

**Outputs**:
- `results/main_results_scenario_means.csv` — mean MRR/MAP/Top-K per model × scenario
- `results/main_results_wilcoxon_tests.csv` — Wilcoxon p-values for all comparisons × metrics
- `results/main_results_cpl_gain_summary.csv` — win rate, median win/loss, asymmetry ratio

**Key numbers to extract for paper**:
- From `scenario_means`: the 4-row × 5-metric table per model → **Table 3** body
- From `wilcoxon_tests` where `comparison = 'CPT vs WPS'` and `metric = 'MRR'`: the `win_rate`, `mean_delta`, `p_greater`, `significance` → **RQ1 headline**
- From `cpl_gain_summary`: `asymmetry_ratio` → the "wins are larger than losses" claim

**LaTeX output**: The script prints LaTeX-formatted tables directly to stdout. Copy them as-is.

---

## Step 2 — Negative Transfer + Commutativity + Cross-domain (no FAISS needed)

**Why**: Characterizes when CPL hurts vs helps (A1 + A6 from CPL_DESIRABILITY). Adds the cross-domain vs within-domain breakdown (E6 test).

**Command**:
```bash
python Scripts/analysis/negative_transfer_analysis.py
```

**Inputs**:
- `results/obj1_experimental_results.csv`
- `data/processed/project_metadata.parquet`
- `results/all_project_domain_gaps.csv`

**Outputs**:
- `results/negative_transfer_analysis.csv` — per-pair deltas, features, transfer label
- `results/negative_transfer_analysis_commutativity.csv` — symmetric pair analysis
- `results/cross_domain_breakdown.csv` — within-domain vs cross-domain CPL gain
- `results/images/nt_feature_correlations.png`
- `results/images/nt_delta_by_target_loc.png`
- `results/images/nt_commutativity.png`
- `results/images/nt_cross_domain_breakdown.png`

**Key numbers to extract for paper**:
- From stdout: COOBA negative transfer count (expected ~12 pairs), BLAZE negative transfer count (expected ~2 pairs) → **RQ1 nuance paragraph**
- From `cross_domain_breakdown.csv`: win rates for within-domain vs cross-domain → **E6 / RQ2 domain gap claim**
- From `commutativity.csv`: "smaller-target benefits more" consistency rate → **commutativity finding**
- From `nt_feature_correlations.png`: which features have significant ρ → **Figure for RQ2**

---

## Step 3 — Effect Size Analysis + Stratified Table (no FAISS needed)

**Why**: Quantifies whether CPL gains are large and losses small (A4). The stratified table shows how performance varies by target codebase size — essential for the "small targets benefit more" claim.

**Command**:
```bash
python Scripts/analysis/effect_size_analysis.py
```

**Inputs**:
- `results/obj1_experimental_results.csv`
- `data/processed/project_metadata.parquet`

**Outputs**:
- `results/effect_size_summary.csv` — Cohen's d + Wilcoxon p per model × comparison × metric
- `results/effect_size_stratified_table.csv` — mean metrics by model × scenario × Small/Medium/Large
- `results/images/es_delta_distributions.png`
- `results/images/es_delta_by_target_size.png`
- `results/images/es_cohens_d_heatmap.png`

**Key numbers to extract for paper**:
- From `effect_size_summary` where `comparison = 'CPT vs WPS'`: `cohens_d` per model → **effect size claim**
- From `effect_size_stratified_table`: MRR for CP-transfer vs WP-small in Small group → **E4 "CPL most valuable for small targets"**
- From `es_delta_distributions.png`: the gain distribution histogram → **Figure for RQ1**
- From `es_cohens_d_heatmap.png`: Cohen's d across all comparisons → **Figure for RQ1 appendix**

---

## Step 4 — Cold Start Viability (no FAISS needed)

**Why**: Identifies when zero-shot CPL (no target labels) is sufficient in practice (A3). The BLAZE cold-start MRR is reportedly ~0.302 — very different from COOBA ~0.029. This is a practical deployment finding.

**Command**:
```bash
python Scripts/analysis/cold_start_viability_analysis.py
```

**Inputs**:
- `results/obj1_experimental_results.csv`
- `data/processed/project_metadata.parquet`

**Outputs**:
- `results/cold_start_viability.csv` — per-pair CP-cold-start stats + viability flag (MRR > 0.20)
- `results/images/cs_viability_heatmap_BLAZE.png`
- `results/images/cs_viability_heatmap_COOBA.png`
- `results/images/cs_loc_threshold.png`

**Key numbers to extract for paper**:
- Mean CP-cold-start MRR for BLAZE vs COOBA → **architecture contrast paragraph**
- % of BLAZE cold-start pairs where MRR > 0.20 → **"deployable without labels" claim**
- Target LoC threshold below which cold-start is viable → **RQ2 practical recommendation**

---

## Step 5 — Cross-Model Consistency (no FAISS needed)

**Why**: Tests whether BLAZE and COOBA agree on which pairs benefit from CPL (A2). The "architecture-agnostic CPL evidence" is the headline journal contribution.

**Command**:
```bash
python Scripts/analysis/cross_model_consistency_analysis.py
```

**Inputs**:
- `results/obj1_experimental_results.csv`
- `data/processed/project_metadata.parquet`

**Outputs**:
- `results/cross_model_agreement.csv` — per-pair deltas, direction labels, agreement flag
- `results/images/cm_agreement_scatter.png`
- `results/images/cm_delta_comparison.png`

**Key numbers to extract for paper**:
- Agreement rate on CPL benefit direction → **cross-model consistency headline**
- Pairs where both models agree CPL helps → **strongest evidence cases**
- Pairs where models disagree → **architecture-dependent cases**

---

## Step 6 — Source Quality as Transfer Predictor (no FAISS needed)

**Why**: Tests whether a source where the model works well within-project also transfers well (A5). Provides a principled source selection criterion beyond "most bugs."

**Command**:
```bash
python Scripts/analysis/source_quality_transfer_analysis.py
```

**Inputs**:
- `results/obj1_experimental_results.csv`
- `data/processed/project_metadata.parquet`

**Outputs**:
- `results/source_quality_transfer.csv` — per-pair source quality + CPL metrics
- `results/images/sq_scatter_BLAZE.png`
- `results/images/sq_scatter_COOBA.png`

**Key numbers to extract for paper**:
- Spearman ρ between source WP-large MRR and CP-transfer MRR → **RQ3 source quality predictor**
- Whether source quality predicts transfer sign (positive vs negative) → **negative transfer predictor**

---

## Step 7 — Source Selection Analysis (no FAISS needed for core)

**Why**: Validates the source selection heuristics (RQ3). Uses the rewritten script that reads from the new 13-project dataset.

**Command**:
```bash
python Scripts/analysis/source_selection_analysis.py
```

**Inputs**:
- `results/obj1_experimental_results.csv`
- `data/processed/project_metadata.parquet`
- `results/all_project_domain_gaps.csv`

**Outputs** (per model: BLAZE + COOBA):
- `results/source_selection_validation_BLAZE.csv`
- `results/source_selection_validation_COOBA.csv`
- `results/images/source_selection_feature_correlations_BLAZE.png`
- `results/images/source_selection_feature_correlations_COOBA.png`
- `results/images/source_selection_target_viability_BLAZE.png`
- `results/images/source_selection_target_viability_COOBA.png`
- `results/images/source_selection_ranking_quality_BLAZE.png`
- `results/images/source_selection_ranking_quality_COOBA.png`

**Key numbers to extract for paper**:
- `hit1_composite` and `hit1_bugs` means per model → **Hit@1 source selection accuracy**
- `tau_composite` and `tau_bugs` medians → **Kendall τ source ranking quality**
- Feature correlations: which source/target features have significant ρ → **RQ3 feature analysis**
- E3/E5 will say "PENDING" until FAISS is done — fill in after Step 0 completes

---

## Step 8 — After FAISS Completes: Re-run Source Selection

Once Step 0 finishes, re-run source_selection_analysis.py to fill the E3/E5 evidence:

```bash
# Check FAISS is done (should see all 13 files)
ls results/faiss_recall_results/semantic_search_*jupyterlab*.csv
ls results/faiss_recall_results/semantic_search_*qiskit*.csv

# Re-run source selection to fill E3/E5
python Scripts/analysis/source_selection_analysis.py
```

Also update `main_results_table.py` if you want to add a FAISS baseline row to the scenario means table.

---

## Recommended Run Order

Steps 1–7 can all run now (no FAISS dependency). Step 0 runs in the background.

```bash
# Terminal A: FAISS in background
source /home/cs21d002_eashaan/PhD/Objective1/obj1/bin/activate
cd /home/cs21d002_eashaan/PhD/Objective1
mkdir -p logs
nohup python Scripts/run_faiss_experiment.py > logs/faiss_run.log 2>&1 &

# Terminal B: Analysis scripts (run in sequence, ~2 min each)
source /home/cs21d002_eashaan/PhD/Objective1/obj1/bin/activate
cd /home/cs21d002_eashaan/PhD/Objective1
python Scripts/analysis/main_results_table.py               2>&1 | tee logs/main_results.log
python Scripts/analysis/negative_transfer_analysis.py       2>&1 | tee logs/negative_transfer.log
python Scripts/analysis/effect_size_analysis.py             2>&1 | tee logs/effect_size.log
python Scripts/analysis/cold_start_viability_analysis.py    2>&1 | tee logs/cold_start.log
python Scripts/analysis/cross_model_consistency_analysis.py 2>&1 | tee logs/cross_model.log
python Scripts/analysis/source_quality_transfer_analysis.py 2>&1 | tee logs/source_quality.log
python Scripts/analysis/source_selection_analysis.py        2>&1 | tee logs/source_selection.log
```

---

## Output → Paper Section Mapping

| Script output | Paper section | RQ |
|---|---|---|
| `main_results_scenario_means.csv` | §Results Table 3 | RQ1 |
| `main_results_wilcoxon_tests.csv` | §Results statistical significance | RQ1 |
| `main_results_cpl_gain_summary.csv` | §Results CPL gain magnitude | RQ1 |
| `effect_size_stratified_table.csv` | §Results Table 4 (stratified) | RQ2 |
| `es_delta_distributions.png` | §Results Figure (gain distribution) | RQ1 |
| `cross_model_agreement.csv` | §Results Table 5 (consistency) | RQ4 |
| `negative_transfer_analysis.csv` | §Discussion negative transfer | RQ1/RQ2 |
| `cross_domain_breakdown.csv` | §Discussion domain gap (E6) | RQ2 |
| `negative_transfer_analysis_commutativity.csv` | §Discussion commutativity | RQ2 |
| `cold_start_viability.csv` | §Discussion deployment (A3) | RQ1/RQ2 |
| `source_selection_validation_BLAZE.csv` | §Results Table 6 (RQ3) | RQ3 |
| `source_selection_validation_COOBA.csv` | §Results Table 6 (RQ3) | RQ3 |
| `source_quality_transfer.csv` | §Results source quality predictor | RQ3 |

---

## Expected Completion Time

| Step | Time | Blocking? |
|---|---|---|
| Step 0 — FAISS | 2–4 hrs (background) | No (run in background) |
| Steps 1–7 — Analysis | ~15 min total | No FAISS needed |
| Step 8 — Re-run source selection | 2 min | After Step 0 |

**Total wall-clock time before writing starts**: ~15 min (Steps 1–7 complete immediately).

---

*Script root: `Scripts/analysis/`*  
*Results root: `results/`*  
*Log directory: `logs/` (create with `mkdir -p logs`)*
