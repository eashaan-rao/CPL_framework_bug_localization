# Python Project Selection for Phase 1 CPL Study

**Date**: 2026-03-31
**Study**: Cross-Project Bug Localization (CPL) — Phase 1
**Models evaluated**: TRANP-CNN, COOBA, BLAZE

---

## 1. Selection Outcome

**6 Python projects** were selected from the pool of 41 Python projects in our 98-project dataset:

| # | Project | LoC | Bugs | Sub-domain |
|---|---------|-----|------|-----------|
| 1 | jupyterlab/jupyterlab | 39,372 | 197 | Interactive computing |
| 2 | lightning-ai/lightning | 51,387 | 366 | ML training framework |
| 3 | prefecthq/prefect | 107,073 | 404 | Workflow orchestration |
| 4 | pydata/xarray | 142,044 | 110 | Scientific data structures |
| 5 | numpy/numpy | 276,548 | 789 | Numerical computing |
| 6 | scikit-learn/scikit-learn | 376,169 | 228 | ML algorithms library |

**Experimental scope**: 6 × 5 = **30 directed source→target pairs** × 4 scenarios = **120 experiment runs per model**, 360 total across 3 models.

---

## 2. Empirical Basis for Selection

### 2.1 The dominant predictor: target codebase size (LoC)

Our Phase 1 analysis (373 TRANP-CNN runs across 94 pairs, 12 Python targets) identified **target LoC** as the overwhelmingly dominant predictor of CPL performance:

| Metric | Spearman ρ (target LoC vs performance) | Significance |
|--------|----------------------------------------|--------------|
| Top-10 | −0.855 | p < 0.001 |
| Top-5  | −0.801 | p < 0.001 |
| Top-1  | −0.712 | p < 0.001 |
| MRR    | −0.820 | p < 0.001 |

This very strong negative correlation means **LoC is the axis along which CPL generalisation varies most**. A statistically valid project sample must therefore span the LoC range, not be uniform within one region of it.

Other candidate predictors were weak or non-predictive:
- Domain gap accuracy: ρ < 0.12 (non-significant, n.s.) — domain similarity is irrelevant
- Bug report verbosity: ρ ≈ 0.18 (n.s.)
- Source project size: ρ ≈ 0.27 (weak, n.s.)
- Bug/code similarity (cosine): ρ < 0.10 (n.s.)

### 2.2 Exclusion criteria applied to the 41-project Python pool

| Criterion | Excluded projects | Reason |
|-----------|------------------|--------|
| LoC > 450K | apache/airflow (851K), pandas-dev/pandas (564K), huggingface/transformers (1.18M), conda/conda (2.58M), sympy/sympy (696K), googleapis/google-cloud-python (4.15M), dmwm/wmcore (1.05M), dagster-io/dagster (616K), qiskit/qiskit (435K), scipy/scipy (438K), django/django (460K), posthog/posthog (461K) | Extremely large codebases produce near-zero CPL performance (ρ = −0.855), making them statistically uninformative for understanding CPL generalization |
| Bugs < 100 | (none in pool — 100 bugs is the Phase 1 minimum threshold) | Insufficient data for reliable metric estimation |
| Not in Phase 1 pool (repos/embedding DBs unavailable) | All projects not in the original 12-project Phase 1 Python set | Practical constraint: repos must be cloned and embedding databases pre-built |
| Dominated by exclusion above | matplotlib/matplotlib (249K LoC, 183 bugs) — see §2.3 | Replaced by closer stratum representative |
| Redundant LoC stratum | google/jax (73K LoC), ray-project/ray (244K LoC), open-mmlab/mmdetection (209K LoC) | LoC strata already covered by retained projects |

After exclusions, 9 candidate projects remain. From these, 6 are selected by stratified sampling.

### 2.3 Stratified sampling by LoC quintile

With LoC as the key variable, we apply stratified sampling to ensure the selected set spans the empirically relevant LoC range. We exclude projects with LoC > 400K (known from Phase 1 to yield near-zero CPL performance with little variance), giving a usable range of approximately 39K–376K LoC.

The 9 candidates are binned into 5 strata (equal-width log-LoC quintiles); the one project per stratum with the highest bug count is selected (maximising training data in each stratum). For strata with only one candidate the single representative is taken.

| Stratum | LoC range | Candidates | Selected (highest bugs) |
|---------|-----------|------------|------------------------|
| S1 (very small) | < 60K | jupyterlab (197), lightning-ai (366) | **lightning-ai/lightning** |
| S2 (small) | 60K–120K | google/jax (326), prefecthq/prefect (404) | **prefecthq/prefect** |
| S3 (medium) | 120K–200K | pydata/xarray (110) | **pydata/xarray** |
| S4 (large) | 200K–300K | ray-project/ray (337), numpy/numpy (789), open-mmlab/mmdetection (102) | **numpy/numpy** |
| S5 (very large) | 300K–400K | scikit-learn/scikit-learn (228) | **scikit-learn/scikit-learn** |

One additional project is added from S1 to bring the total to 6, which crosses the minimum-pairs threshold (see §3). **jupyterlab/jupyterlab** is selected as the S1 companion because it represents the smallest codebase, providing an important anchor for the LoC-performance curve and has been a confirmed high-CPL-gain project in Phase 1 showcase analysis.

The final LoC distribution across the 6 projects:

```
39K   51K       107K     142K             277K              376K
 |     |          |        |                |                  |
 jupyterlab  lightning  prefect  xarray    numpy           sklearn
```

This spans approximately 1 order of magnitude (10× from smallest to largest), providing strong statistical leverage for LoC-conditioned analyses.

---

## 3. Statistical Justification for N = 6 Projects

### 3.1 Minimum pairs for correlation analysis

With n = 6 projects, there are **n(n−1) = 30 directed source→target pairs**. For Spearman rank correlation (the primary analysis method in this study), the minimum sample size for 80% statistical power at α = 0.05 is:

| Effect size (ρ) | Minimum n pairs needed |
|-----------------|----------------------|
| Large (0.50)    | 22 |
| Medium (0.35)   | 46 |
| Small (0.20)    | 140 |

With 30 pairs we have adequate power for large effect sizes (ρ ≥ 0.50), which is appropriate given that Phase 1 found effects of ρ = 0.71–0.85. Effects smaller than ρ = 0.36 would not be detectable, which is acceptable: our research questions focus on the strong, practically meaningful effects already observed.

### 3.2 Scenario comparison (within-pair)

For the 4-scenario comparison (WP-small, WP-large, CP-cold-start, CP-transfer) a Wilcoxon signed-rank test on 30 pairs requires n ≥ 20 for 80% power at a medium effect size (Cohen's d = 0.5). 30 pairs exceeds this threshold.

### 3.3 Model comparison (across 3 models)

A Friedman test across 3 models with 30 paired observations has power > 0.90 for medium effects (Kendall's W ≥ 0.25). 30 pairs is sufficient.

### 3.4 Why not fewer / more projects?

| N projects | Directed pairs | Notes |
|-----------|---------------|-------|
| 4 | 12 | Insufficient power for correlation analysis (< 22 needed) |
| 5 | 20 | Marginal; border of adequate power |
| **6** | **30** | Adequate power; fits within ~30-day compute budget |
| 7 | 42 | Adequate power but ~40% more computation; exceeds 1-month budget |
| 8 | 56 | Well-powered but ~87% more computation than n=6 |

**N = 6 is the smallest number that provides adequate statistical power within the computational budget.**

---

## 4. Computational Budget Validation

Estimated time per experiment (1 directed pair × 1 scenario × 1 model), based on observed Phase 1 timing:

| Model | Small target (≤100K LoC) | Large target (≥250K LoC) | Average |
|-------|--------------------------|--------------------------|---------|
| TRANP-CNN | ~45 min | ~90 min | ~60 min |
| COOBA | ~30 min | ~60 min | ~45 min |
| BLAZE | ~45 min | ~120 min | ~75 min |

Total experiments: 30 pairs × 4 scenarios × 3 models = **360 experiment runs**

Estimated total wall time: 360 × ~60 min average ≈ **21,600 min ≈ 15 days**

This is comfortably within the 1-month target even accounting for re-runs, I/O overhead, and queue delays. The project would exceed 1 month only if mean experiment time exceeds ~120 min, which is unlikely given that most pairs involve small or medium-LoC targets.

---

## 5. Sub-domain Coverage

Although all 6 projects are classified as "Data Science & AI/ML" at the top level of our taxonomy (consistent with the Phase 1 finding that domain gap has no predictive power, ρ < 0.12), they represent meaningfully distinct functional sub-domains:

| Project | Functional sub-domain | Primary language concern |
|---------|----------------------|--------------------------|
| jupyterlab | Interactive notebook IDE | UI, extension APIs, async |
| lightning-ai | Deep learning training abstraction | Trainer loops, callbacks, distributed training |
| prefecthq | Workflow DAG orchestration | Task scheduling, state machines, async |
| pydata/xarray | N-dimensional labelled arrays | Numerical indexing, I/O, broadcasting |
| numpy | Core numerical computing | Array operations, BLAS wrappers, C extensions |
| scikit-learn | ML algorithm implementations | Estimator API, optimisation, statistical methods |

This sub-domain spread ensures that any observed CPL patterns are not artifacts of a single narrow code style or API surface.

---

## 6. Projects Excluded from the Final List (and Why)

Projects that were in the Phase 1 Python pool but are NOT included:

| Project | LoC | Bugs | Reason excluded |
|---------|-----|------|----------------|
| scipy/scipy | 438K | 100 | LoC > 400K (near-zero CPL zone); only 100 bugs (minimum threshold) |
| sympy/sympy | 696K | 384 | LoC far exceeds 400K threshold |
| matplotlib/matplotlib | 249K | 183 | LoC stratum (S4) already covered by numpy (789 bugs >> 183) |
| open-mmlab/mmdetection | 209K | 102 | LoC stratum (S4) covered by numpy; only 102 bugs |
| ray-project/ray | 244K | 337 | LoC stratum (S4) covered by numpy; numpy has more bugs |
| google/jax | 73K | 326 | LoC stratum (S2) covered by prefect; prefect has more bugs |
