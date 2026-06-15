# Journal Paper Run Plan — CPL Multi-Model Study

**Created**: 2026-05-24  
**Status**: Active — COOBA stopped, BLAZE + TRANP-CNN queued  
**Target venue**: Journal of Systems and Software (JSS) or Empirical Software Engineering (EMSE)

---

## Why We Stopped at This Point

COOBA ran for 3+ weeks and reached 121/156 pairs. Running the remaining 35 pairs plus running BLAZE and TRANP-CNN on all 156 pairs would take 2+ more months. Instead, we define a **63-pair paper set** that:
- Covers the full LoC range (35K–572K) of all 13 target projects
- Ensures every target has ≥3 source candidates (required for RQ3 source selection analysis)
- Maximises reuse of already-computed COOBA + BLAZE results (42 pairs need zero new compute)
- Provides adequate statistical power for all 4 RQs

---

## Compute Status at Decision Point (2026-05-24)

| Model | Pairs Complete | Pairs Needed | Action |
|---|---|---|---|
| COOBA | 121 / 156 | 63 (all in paper set) | **Done — stop now** |
| BLAZE | 42 / 156 | 63 (81 new runs needed) | Run Phase 2a |
| TRANP-CNN | 0 / 156 (new project set) | 63 (252 new runs) | Run Phase 2b |

---

## The Paper Pair Set — 63 Directional Pairs

### Group A — DS×DS (30 pairs) | All models: COOBA ✓ BLAZE ✓ TRANP-CNN needed

All bidirectional pairs among the 6 Data Science projects:

| Projects (both directions) |
|---|
| jupyterlab ↔ lightning-ai |
| jupyterlab ↔ prefecthq |
| jupyterlab ↔ pydata/xarray |
| jupyterlab ↔ numpy |
| jupyterlab ↔ scikit-learn |
| lightning-ai ↔ prefecthq |
| lightning-ai ↔ pydata/xarray |
| lightning-ai ↔ numpy |
| lightning-ai ↔ scikit-learn |
| prefecthq ↔ pydata/xarray |
| prefecthq ↔ numpy |
| prefecthq ↔ scikit-learn |
| pydata/xarray ↔ numpy |
| pydata/xarray ↔ scikit-learn |
| numpy ↔ scikit-learn |

**Role**: Core statistical backbone. Replicates the initial DS-wave results with all 3 models. Provides the within-domain baseline and the commutativity evidence (all 15 symmetric pairs).

---

### Group B — jupyterlab as Cross-Domain Hub (14 pairs) | COOBA ✓ BLAZE ✓ (5 gap runs) TRANP-CNN needed

Both directions between jupyterlab and each non-DS project:

| Source | Target | LoC contrast | Domain crossing |
|---|---|---|---|
| jupyterlab (39K) | ipython (78K) | ×2 | DS → DevTools |
| ipython (78K) | jupyterlab (39K) | 0.5× | DevTools → DS |
| jupyterlab (39K) | mesonbuild (121K) | ×3 | DS → DevTools |
| mesonbuild (121K) | jupyterlab (39K) | 0.3× | DevTools → DS |
| jupyterlab (39K) | ansible (348K) | ×9 | DS → DevOps |
| ansible (348K) | jupyterlab (39K) | 0.1× | DevOps → DS |
| jupyterlab (39K) | docker (35K) | ×1 | DS → Systems |
| docker (35K) | jupyterlab (39K) | ×1 | Systems → DS |
| jupyterlab (39K) | localstack (572K) | ×15 | DS → Systems |
| localstack (572K) | jupyterlab (39K) | 0.07× | Systems → DS |
| jupyterlab (39K) | wagtail (286K) | ×7 | DS → Apps |
| wagtail (286K) | jupyterlab (39K) | 0.14× | Apps → DS |
| jupyterlab (39K) | qiskit (435K) | ×11 | DS → Apps |
| qiskit (435K) | jupyterlab (39K) | 0.09× | Apps → DS |

**Role**: Cross-domain signal with a fixed small-LoC target (jupyterlab). Essential for RQ2 (tgt_LoC dominance) and for showing CPL works across domains (RQ1 + E6 from CPL_DESIRABILITY).

---

### Group C — Strategic Cross-Domain Diversity (19 pairs) | COOBA ✓ BLAZE needed TRANP-CNN needed

| Source | Target | Analytical purpose |
|---|---|---|
| lightning (51K) | ansible (348K) | Medium DS → large DevOps; cross-domain |
| ansible (348K) | lightning (51K) | Reverse — commutativity with LoC swap |
| lightning (51K) | localstack (572K) | Medium DS → XLarge; hardest target |
| localstack (572K) | lightning (51K) | Reverse |
| numpy (277K) | qiskit (435K) | Large DS → XLarge Apps |
| qiskit (435K) | numpy (277K) | Reverse |
| numpy (277K) | localstack (572K) | Large DS → XLarge; both large |
| localstack (572K) | numpy (277K) | Reverse |
| prefect (107K) | ansible (348K) | Medium DS → large DevOps |
| ansible (348K) | prefect (107K) | Reverse |
| ipython (78K) | docker (35K) | DevOps → Systems; cross-domain |
| docker (35K) | ipython (78K) | Reverse |
| ipython (78K) | qiskit (435K) | DevOps → XLarge Apps |
| wagtail (286K) | docker (35K) | Large Apps → small Systems |
| lightning (51K) | ipython (78K) | Ensures ipython has 3rd source |
| numpy (277K) | mesonbuild (121K) | Ensures meson has 2nd source |
| prefect (107K) | mesonbuild (121K) | Ensures meson has 3rd source |
| lightning (51K) | wagtail (286K) | Ensures wagtail has 2nd source |
| prefect (107K) | wagtail (286K) | Ensures wagtail has 3rd source |

**Role**: Fills the analytical gaps that Groups A+B leave — large-to-large cross-domain, DevOps/Systems/Apps interactions, and minimum source coverage for every target.

---

## Source Coverage Per Target (RQ3 Requirement)

All 13 targets have ≥3 sources, satisfying the source selection analysis requirement.

| Target | LoC | Sources in paper set | Count |
|---|---|---|---|
| jupyterlab | 39K | all 12 other projects | 12 |
| lightning-ai | 51K | 5 DS + ansible + localstack | 7 |
| prefecthq | 107K | 5 DS + ansible + meson + wagtail | 8 |
| pydata/xarray | 142K | 5 DS | 5 |
| numpy | 277K | 5 DS + qiskit + localstack | 7 |
| scikit-learn | 376K | 5 DS | 5 |
| ipython | 78K | jupyterlab + docker + lightning | 3 |
| mesonbuild | 121K | jupyterlab + numpy + prefect | 3 |
| ansible | 348K | jupyterlab + lightning + prefect | 3 |
| docker | 35K | jupyterlab + ipython + wagtail | 3 |
| localstack | 572K | jupyterlab + lightning + numpy | 3 |
| wagtail | 286K | jupyterlab + lightning + prefect | 3 |
| qiskit | 435K | jupyterlab + numpy + ipython | 3 |

---

## Statistical Power Check

| Analysis | Requirement | Paper set |
|---|---|---|
| RQ1 Wilcoxon (CP-transfer vs WP-small) | ≥50 paired obs. | 63 ✓ |
| RQ2 Spearman ρ = 0.85 (tgt_LoC) | 22 pairs for 80% power | 7× overpowered ✓ |
| RQ2 Spearman ρ = 0.35 (moderate) | 46 pairs | 1.4× overpowered ✓ |
| RQ3 source selection (Kendall τ, Hit@1) | ≥3 sources × all targets | 13/13 targets ✓ |
| Cross-model consistency (A2) | Same pairs, 3 models | 63 pairs × 3 models ✓ |
| RQ4 FAISS ceiling | All test pairs | ✓ |

---

## Execution Phases

### Phase 2a — BLAZE (81 new runs)

**Config in `run_expts_ph1.py`**: `MODELS_TO_RUN = {'BLAZE': run_blaze_experiment}`, `NUM_SHARDS = 2`

- 42 pairs already done → skipped automatically by the existing-result check
- 19 Group C pairs × 4 scenarios = 76 new runs
- 5 qiskit↔jupyterlab gap runs = 5 new runs
- **Total: 81 new runs**
- Estimated: ~61 hrs @ 45 min/run → **~1.5 days with 2 shards**

Run shard 0: `SHARD=0 python Scripts/run_expts_ph1.py`  
Run shard 1: `SHARD=1 python Scripts/run_expts_ph1.py`

### Phase 2b — TRANP-CNN (252 new runs)

**Config in `run_expts_ph1.py`**: swap to `MODELS_TO_RUN = {'TRANP-CNN': run_tranp_cnn_experiment}`, `NUM_SHARDS = 3`

- 63 pairs × 4 scenarios = 252 new runs (all new — first time on this project set)
- **Total: 252 new runs**
- Estimated: ~189 hrs @ 45 min/run → **~3 days with 3 shards**

Run shards 0, 1, 2 in parallel across sessions.

### Phase 3 — Analysis (1 week)

Run all analysis scripts from `Scripts/analysis/` on the completed 3-model, 63-pair dataset:
- `negative_transfer_analysis.py` — A1 + A6 (negative transfer + commutativity)
- `cross_model_consistency_analysis.py` — A2 (BLAZE vs COOBA vs TRANP-CNN agreement)
- `cold_start_viability_analysis.py` — A3 (BLAZE cold-start viability heatmap)
- `effect_size_analysis.py` — A4 (delta distributions, Cohen's d)
- `source_quality_transfer_analysis.py` — A5 (source WP-large → CP-transfer correlation)

### Phase 4 — Paper Writing (4–5 weeks)

Structure follows 4 RQs in `DC_4_5_Report/main1.tex`, extended with:
- Multi-model comparison as primary contribution (vs TRANP-CNN-only DC report)
- Cross-model consistency as a headline finding (architecture-agnostic CPL evidence)
- BLAZE cold-start viability as a practical deployment finding

---

## Total Timeline

| Phase | Duration | Notes |
|---|---|---|
| Stop COOBA + deploy script changes | Day 1 | Done |
| BLAZE Phase 2a | Days 2–3 | 2 shards |
| TRANP-CNN Phase 2b | Days 3–6 | 3 shards, sequential after BLAZE |
| Analysis Phase 3 | Week 2 | Light compute |
| Paper writing Phase 4 | Weeks 3–7 | |
| **Submission** | **~Week 8** | **~7 weeks from today** |

---

## COOBA Data Being Retained

121 complete COOBA pairs are used as-is. The 35 incomplete pairs are excluded from the paper analysis:
- All 35 are cross-pairs between {mesonbuild, ansible, docker, localstack, wagtail, qiskit} without DS involvement
- 13 of those 35 DO appear in Group C but only one direction, which is COOBA-complete
- No data is deleted — the 121 complete results remain in `obj1_experimental_results.csv`

---

*Script config: `Scripts/run_expts_ph1.py` — see `PAPER_PAIRS` set and `MODELS_TO_RUN` comments*  
*Results file: `results/obj1_experimental_results.csv`*
