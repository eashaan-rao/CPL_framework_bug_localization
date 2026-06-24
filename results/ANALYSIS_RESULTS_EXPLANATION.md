# Analysis Results Explanation — COOBA + BLAZE + TRANP-CNN (complete)

**Updated**: 2026-06-24  
**Data**: `results/paper_results_complete.csv`  
**Pair counts**: BLAZE = 63 pairs, COOBA = 63 pairs, TRANP-CNN = 63 pairs (all complete — same 13-project set)  
**Purpose**: Plain-language explanation of each analysis test: what it measured, what the numbers say, what we can infer.

> **Models**: BLAZE = embedding-based reranker; COOBA = GNN on AST embeddings; TRANP-CNN = CNN reranker on FAISS candidates.  
> **Scenarios**: WP-small = within-project 20% target data; WP-large = within-project 80% target data; CP-cold-start (CPC) = source-only, no target data; CP-transfer (CPT) = source + 20% target data.  
> **Negative transfer threshold**: ±0.01 MRR (positive if Δ > +0.01, negative if Δ < −0.01, neutral otherwise).

---

## Step 1 — Main Results Table

### 1a. Scenario Performance Means

**What it measures**: Mean MRR/MAP/Top-K across all pairs per model per scenario. The reference ordering should be WP-small < CP-transfer < WP-large (CPL exceeds limited training, but full training is still best).

| Model | n | WP-small MRR | WP-large MRR | CPC MRR | **CPT MRR** |
|---|---|---|---|---|---|
| **BLAZE** | 63 | 0.256 | 0.471 | 0.262 | **0.392** |
| **COOBA** | 63 | 0.138 | 0.192 | 0.027 | **0.156** |
| **TRANP-CNN** | 63 | 0.326 | 0.389 | 0.075 | **0.380** |

Full metric table:

**BLAZE** (63 pairs):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.256 | 0.210 | 0.156 | 0.348 | 0.446 |
| WP-large | 0.471 | 0.391 | 0.343 | 0.608 | 0.705 |
| CP-cold-start | 0.262 | 0.214 | 0.154 | 0.359 | 0.471 |
| **CP-transfer** | **0.392** | **0.322** | **0.267** | **0.527** | **0.627** |

**COOBA** (63 pairs):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.138 | 0.120 | 0.057 | 0.216 | 0.313 |
| WP-large | 0.192 | 0.165 | 0.112 | 0.280 | 0.370 |
| CP-cold-start | 0.027 | 0.023 | 0.008 | 0.035 | 0.059 |
| **CP-transfer** | **0.156** | **0.134** | **0.075** | **0.237** | **0.339** |

**TRANP-CNN** (63 pairs):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.326 | 0.288 | 0.107 | 0.274 | 0.351 |
| WP-large | 0.389 | 0.349 | 0.132 | 0.314 | 0.387 |
| CP-cold-start | 0.075 | 0.065 | 0.010 | 0.055 | 0.095 |
| **CP-transfer** | **0.380** | **0.337** | **0.134** | **0.306** | **0.370** |

**Key observations:**

- **BLAZE**: CP-cold-start (MRR=0.262) exceeds WP-small (0.256) — training on a source project alone gives slightly better performance than 20% of the actual target. CP-transfer (0.392) reaches 83.2% of WP-large.
- **COOBA**: CP-cold-start collapses to near-random (0.027, −80% vs WP-small). CP-transfer (0.156) modestly beats WP-small (+13%). The ordering WPS < CPT < WPL holds, but absolute gains are small.
- **TRANP-CNN**: CPT (0.380) reaches **97.8%** of WP-large (0.389) — the CPT/WPL ratio is far higher than BLAZE (83.2%) or COOBA (81.1%). Cold-start (0.075) collapses like COOBA. TRANP-CNN's CPT is statistically indistinguishable from WP-large (see Step 1b), meaning 20% target fine-tuning on a cross-project initialisation effectively matches 80% within-project training.

**Inference**: BLAZE's embedding representation generalises without fine-tuning; COOBA's AST features do not generalise at all; TRANP-CNN's CNN reranker generalises strongly after target-side adaptation — to the point where CPT ≈ WPL. If the goal is maximum CPL quality with 20% target labels, TRANP-CNN is the best model. If the goal is zero-annotation deployment, BLAZE is the only viable option.

---

### 1b. Wilcoxon Significance Tests — Is the CPL advantage real?

**What it measures**: Paired Wilcoxon tests comparing CPT against baselines. Win rate = fraction of pairs where CPT > baseline.

**CP-transfer vs WP-small (core CPL claim):**

| Model | n | MRR win rate | MAP win rate | Top-5 win rate | MRR mean Δ | MRR sig. |
|---|---|---|---|---|---|---|
| **BLAZE** | 63 | **90.5%** | **93.7%** | **90.5%** | **+0.137** | *** |
| **COOBA** | 63 | 60.3% | 60.3% | 55.6% | +0.018 | * (p=0.034) |
| **TRANP-CNN** | 63 | **68.3%** | **71.4%** | **60.3%** | **+0.054** | *** |

**CP-cold-start vs WP-small:**

| Model | n | MRR win rate | MRR mean Δ | Significance |
|---|---|---|---|---|
| **BLAZE** | 63 | 42.9% | +0.006 | **ns — statistically tied** |
| **COOBA** | 63 | 6.3% | −0.111 | negative (p→1 for CPC>WPS) |
| **TRANP-CNN** | 63 | 0.0% | −0.251 | negative (p→1 for CPC>WPS) |

**CP-transfer vs WP-large (can CPL match full training?):**

| Model | n | MRR win rate | MRR mean Δ | Significance |
|---|---|---|---|---|
| **BLAZE** | 63 | 4.8% | −0.079 | WPL > CPT (significant) |
| **COOBA** | 63 | 20.6% | −0.036 | WPL > CPT (significant) |
| **TRANP-CNN** | 63 | 49.2% | −0.009 | **ns — CPT statistically ties WPL (p=0.19)** |

**Inference**: CPT significantly outperforms WPS for BLAZE (large effect, ***) and TRANP-CNN (medium effect, ***). With 63 pairs, COOBA's advantage is now statistically significant at the 5% level (* p=0.034), a change from the borderline ns result at 62 pairs. However, the effect remains small (see Step 3a) and the win rate of 60.3% means roughly 40% of pairs do not benefit. The most striking finding is TRANP-CNN vs WPL: with 49.2% win rate and mean delta of only −0.009 MRR, CPT is statistically indistinguishable from full within-project training. BLAZE is the only model where cold-start ties WPS; for COOBA and TRANP-CNN, cold-start is significantly worse than even limited within-project training.

---

### 1c. CPL Gain Summary — Are wins large and losses small?

**What it measures**: The asymmetry ratio (median win / |median loss|) tells whether gains justify adoption risk. Ratio >> 1 = good risk profile.

| Model | MRR win% | Med win | Med loss | Asym. ratio | Risk profile |
|---|---|---|---|---|---|
| **BLAZE** | 90.5% | +0.134 | −0.005 | **27.3×** | Very low risk |
| **COOBA** | 60.3% | +0.032 | −0.022 | 1.46× | Symmetric risk |
| **TRANP-CNN** | 68.3% | +0.068 | −0.025 | 2.77× | Moderate risk |

**Inference**: BLAZE has an excellent risk profile — gains are 27× larger than losses, and the rare losses are negligible (−0.005 MRR). COOBA's 1.46× ratio means adopting CPL for COOBA delivers gains and losses of similar magnitude; the 39.7% of pairs that don't benefit experience real harm. TRANP-CNN's 2.77× ratio is moderate — gains are meaningfully larger than losses, but not dramatically so. Its median loss is −0.025 MRR across 31.7% of pairs, concentrated in cross-domain and large-specificity-target settings.

---

## Step 2 — Negative Transfer, Commutativity, Cross-Domain

### 2a. Negative Transfer Characterization

**What it measures**: Per-pair CPT−WPS delta classified as positive (Δ > +0.01), neutral (|Δ| ≤ 0.01), or negative (Δ < −0.01).

| Model | n | Positive | Neutral | Negative |
|---|---|---|---|---|
| **BLAZE** | 63 | 54 (85.7%) | 7 (11.1%) | **2 (3.2%)** |
| **COOBA** | 63 | 28 (44.4%) | 15 (23.8%) | **20 (31.7%)** |
| **TRANP-CNN** | 63 | 39 (61.9%) | 7 (11.1%) | **17 (27.0%)** |

COOBA's worst negative transfer cases:
- mesonbuild/meson → jupyterlab/jupyterlab: −0.152 MRR
- qiskit/qiskit → jupyterlab/jupyterlab: −0.084 MRR
- ansible/ansible → jupyterlab/jupyterlab: −0.080 MRR
- wagtail/wagtail → docker/compose: −0.074 MRR
- numpy/numpy → lightning-ai/lightning: −0.074 MRR

TRANP-CNN's worst negative transfer cases:
- ansible/ansible → prefecthq/prefect: −0.094 MRR
- jupyterlab/jupyterlab → mesonbuild/meson: −0.087 MRR
- lightning-ai/lightning → ansible/ansible: −0.078 MRR
- prefecthq/prefect → ansible/ansible: −0.077 MRR

BLAZE's two negative cases:
- prefecthq/prefect → ansible/ansible: −0.023 MRR
- jupyterlab/jupyterlab → localstack/localstack: −0.016 MRR

**Inference**: BLAZE negative transfer is negligible (2 cases, near-zero magnitude). COOBA suffers genuine negative transfer in nearly a third of pairs (31.7%) — a serious deployment risk. TRANP-CNN shows negative transfer in 27.0% of pairs concentrated in cross-domain pairs involving ansible, prefect, and meson as targets (high-specificity codebases where source-project patterns appear to conflict). COOBA's negative transfer pattern persists: cases are concentrated where AST overfitting from large sources is most severe.

---

### 2b. Cross-Domain Breakdown — Does domain gap limit CPL?

**What it measures**: CPT win rate and mean delta split into within-domain (DS×DS pairs) vs cross-domain pairs.

| Model | Within-domain n | Mean Δ | Win% | Cross-domain n | Mean Δ | Win% |
|---|---|---|---|---|---|---|
| **BLAZE** | 30 | +0.180 | 93.3% | 33 | +0.097 | 78.8% |
| **COOBA** | 30 | +0.027 | 43.3% | 33 | +0.010 | 45.5% |
| **TRANP-CNN** | 30 | +0.078 | 76.7% | 33 | +0.033 | 48.5% |

**Inference**: Domain gap matters most for TRANP-CNN: its within-domain win rate (76.7%) drops substantially to 48.5% for cross-domain pairs. BLAZE shows degradation but remains strong — cross-domain pairs still achieve 78.8% win rate, confirming embedding-based representations are more robust to domain shift than reranking-based approaches. COOBA is unusual: its within-domain and cross-domain win rates are comparable (43.3% vs 45.5%), and both are mediocre — suggesting domain is not the primary driver of COOBA's failure; AST overfitting occurs regardless. TRANP-CNN's 28-point cross-domain drop confirms that its superior CPL performance is concentrated in within-domain settings.

---

### 2c. Commutativity — Does the smaller target benefit more?

**What it measures**: For symmetric pairs (A→B and B→A both present), does the smaller-LoC target consistently gain more from CPL?

| Model | Symmetric pairs | Smaller target benefits more | Rate |
|---|---|---|---|
| **BLAZE** | 28 | 19 | 67.9% |
| **COOBA** | 28 | 17 | 60.7% |
| **TRANP-CNN** | 28 | 14 | 50.0% |

**Inference**: For BLAZE, the smaller target benefits more in 67.9% of symmetric pairs, supporting the hypothesis that small codebases gain most from cross-project knowledge. COOBA shows a moderate asymmetry (60.7%), suggesting LoC is a weak but non-trivial predictor for COOBA. TRANP-CNN shows essentially random commutativity (50.0%), meaning LoC asymmetry does not govern TRANP-CNN's transfer direction — other factors (domain alignment, bug distribution) dominate.

---

## Step 3 — Effect Size Analysis and LoC-Stratified Table

### 3a. Effect Size (Cohen's d) for CPT vs WPS

**What it measures**: Standardised effect size for the CPT−WPS MRR delta. d ≈ 0.2 = small, 0.5 = medium, 0.8 = large.

| Model | n | MRR d | MAP d | Top-1 d | Top-5 d | Top-10 d | Classification |
|---|---|---|---|---|---|---|---|
| **BLAZE** | 63 | **1.191** | **1.179** | **1.128** | **1.140** | **1.042** | Large (all metrics) |
| **COOBA** | 63 | 0.272 | 0.243 | 0.310 | 0.204 | 0.298 | Small (all metrics) |
| **TRANP-CNN** | 63 | 0.591 | 0.564 | 0.453 | 0.525 | 0.315 | Medium (MRR/MAP/Top-5); Small (Top-1/Top-10) |

**Inference**: BLAZE's CPL advantage is a large effect (d > 1.0) consistently across all five metrics — detectable with as few as 10 pairs. COOBA's small effect (d ≈ 0.25–0.31) explains why its Wilcoxon test is only marginally significant even at 63 pairs. TRANP-CNN's medium d=0.591 on MRR confirms meaningful CPL gain at full scale. The lower Top-1 effect for TRANP-CNN (d=0.453) reflects its difficulty surfacing the ground-truth file as rank-1 even after cross-project transfer.

---

### 3b. Stratified by Target Codebase Size

**What it measures**: Mean CPT MRR by target LoC tier (Small <100K, Medium 100K–300K, Large >300K). Reveals whether CPL benefit depends on how large the target codebase is.

**BLAZE** (n=25 Small, 24 Medium, 14 Large):

| Size | WP-small MRR | WP-large MRR | CPC MRR | CPT MRR | CPT−WPS |
|---|---|---|---|---|---|
| Small (<100K) | 0.219 | 0.498 | 0.275 | 0.398 | +0.179 |
| Medium (100K–300K) | 0.278 | 0.459 | 0.254 | 0.388 | +0.110 |
| Large (>300K) | 0.283 | 0.444 | 0.252 | 0.389 | +0.106 |

**COOBA** (n=25 Small, 24 Medium, 14 Large):

| Size | WP-small MRR | WP-large MRR | CPC MRR | CPT MRR | CPT−WPS |
|---|---|---|---|---|---|
| Small (<100K) | 0.219 | 0.294 | 0.041 | 0.243 | +0.025 |
| Medium (100K–300K) | 0.099 | 0.152 | 0.016 | 0.116 | +0.018 |
| Large (>300K) | 0.062 | 0.080 | 0.021 | 0.068 | +0.006 |

**TRANP-CNN** (n=25 Small, 24 Medium, 14 Large):

| Size | WP-small MRR | WP-large MRR | CPC MRR | CPT MRR | CPT−WPS |
|---|---|---|---|---|---|
| Small (<100K) | 0.454 | 0.518 | 0.069 | **0.532** | +0.079 |
| Medium (100K–300K) | 0.278 | 0.340 | 0.108 | 0.302 | +0.024 |
| Large (>300K) | 0.181 | 0.242 | 0.030 | **0.243** | +0.062 |

**Inference**: BLAZE's CPL gain degrades gently with target size (+0.179 → +0.106) but remains substantial even for large targets. COOBA shows severe LoC degradation — large-target CPL gain is near zero (+0.006) and absolute performance collapses with size. TRANP-CNN reveals two striking results: (1) for **small targets**, CPT (0.532) exceeds WP-large (0.518) — cross-project transfer with 20% target fine-tuning actually outperforms 80% within-project training; (2) for **large targets**, CPT (0.243) ties WP-large (0.242) despite cold-start failing completely (0.030). TRANP-CNN's CPT performance is robust to codebase size in absolute terms (0.532 → 0.302 → 0.243), unlike COOBA (0.243 → 0.116 → 0.068) which degrades catastrophically.

---

## Step 4 — Cold Start Viability

**What it measures**: Whether CP-cold-start (zero target labels) achieves MRR > 0.20 — the "deployable without annotation" threshold.

| Model | n | Mean CPC MRR | WPS MRR (reference) | Viable (>0.20) | Rate |
|---|---|---|---|---|---|
| **BLAZE** | 63 | **0.262** | 0.256 | 47/63 | **74.6%** |
| **COOBA** | 63 | 0.027 | 0.138 | 0/63 | **0.0%** |
| **TRANP-CNN** | 63 | 0.075 | 0.326 | 6/63 | **9.5%** |

BLAZE cold-start MRR (0.262) exceeds WPS (0.256). TRANP-CNN cold-start (0.075) is 77% below its own WPS (0.326) — catastrophic cold-start failure.

**Inference**: BLAZE is the only model where zero-annotation CPL is deployment-viable. Its cold-start performance statistically ties WPS across all metrics, meaning annotating 20% of target bugs adds no value over what the source-trained model already knows. For COOBA and TRANP-CNN, cold-start produces near-random rankings. The 6 TRANP-CNN pairs that clear the 0.20 threshold are within-domain pairs where source and target share domain vocabulary — the reranker happens to learn partially transferable score calibrations. The cold-start dichotomy between BLAZE (embedding similarity generalises directly) vs TRANP-CNN/COOBA (reranking scores do not) is a core architectural finding.

---

## Step 5 — Cross-Model Consistency

**What it measures**: For the 63 pairs where all three models have results, whether models agree on CPL benefit direction (positive/neutral/negative).

**BLAZE vs COOBA** (n=63):

| Agreement type | Count | Rate |
|---|---|---|
| Strict (same direction label) | 26 / 63 | **41%** |
| Spearman ρ (delta magnitudes) | ρ=0.076, p=0.554 | No correlation |

**BLAZE vs TRANP-CNN** (n=63):

| Agreement type | Count | Rate |
|---|---|---|
| Strict (same direction label) | 42 / 63 | **67%** |
| Spearman ρ (delta magnitudes) | ρ=0.559, p<0.001 | Moderate positive correlation |

**COOBA vs TRANP-CNN** (n=63):

| Agreement type | Count | Rate |
|---|---|---|
| Strict (same direction label) | 31 / 63 | **49%** |
| Spearman ρ (delta magnitudes) | ρ=0.133, p=0.297 | No correlation |

Direction distribution across all 63 pairs:

| Direction | BLAZE | COOBA | TRANP-CNN |
|---|---|---|---|
| Positive | 54 (86%) | 28 (44%) | 39 (62%) |
| Neutral | 7 (11%) | 15 (24%) | 7 (11%) |
| Negative | 2 (3%) | 20 (32%) | 17 (27%) |

Notable disagreements (BLAZE gains, COOBA loses significantly):
- ansible/ansible → jupyterlab/jupyterlab: BLAZE +0.374 vs COOBA −0.080
- mesonbuild/meson → jupyterlab/jupyterlab: BLAZE +0.354 vs COOBA −0.152
- numpy/numpy → jupyterlab/jupyterlab: BLAZE +0.351 vs COOBA −0.026
- qiskit/qiskit → jupyterlab/jupyterlab: BLAZE +0.330 vs COOBA −0.084

Notable disagreements (BLAZE gains, TRANP-CNN loses):
- jupyterlab/jupyterlab → prefecthq/prefect: BLAZE +0.153 vs TRANP-CNN −0.049
- lightning-ai/lightning → prefecthq/prefect: BLAZE +0.151 vs TRANP-CNN −0.061
- lightning-ai/lightning → ansible/ansible: BLAZE +0.088 vs TRANP-CNN −0.078

**Inference**: BLAZE and COOBA agree on CPL direction in only 41% of shared pairs, with uncorrelated delta magnitudes (ρ=0.076, p=0.55). BLAZE and TRANP-CNN agree in 67% of pairs with a moderate correlation (ρ=0.559, p<0.001) — these two reranker-family models share more CPL sensitivity than either shares with COOBA. COOBA and TRANP-CNN agree in 49% of pairs (near chance) with no significant correlation (ρ=0.133, p=0.30), confirming they exploit fundamentally different aspects of the source project. Architecture-agnostic CPL claims are not supported — CPL benefit is model-family-specific, with embedding-based and CNN-reranker models being more aligned with each other than with GNN-based approaches.

---

## Step 6 — Source Quality as Transfer Predictor

**What it measures**: Spearman ρ between source WP-large MRR (how well the model works on source within-project) and CP-transfer MRR on the target. Tests "pick the source where the model already performs best."

| Model | n | ρ (src WPL vs CPT MRR) | p-value | ρ (src WPL vs CPT delta) | Interpretation |
|---|---|---|---|---|---|
| **BLAZE** | 63 | −0.000 | 0.999 | +0.023 | No relationship |
| **COOBA** | 63 | **−0.296** | **0.019** | +0.035 | Neg. relation to absolute MRR |
| **TRANP-CNN** | 63 | −0.208 | 0.102 | +0.014 | Marginal negative (not significant) |

**Inference**: For BLAZE, source quality has no predictive power — any source is equally valid for transfer. For COOBA, there is a statistically significant negative correlation (ρ=−0.296): sources where COOBA achieves high WP-large MRR transfer *worse*, not better. This suggests COOBA overfits to source-specific AST patterns when it performs well within-project. For TRANP-CNN, the negative trend (ρ=−0.208) is in the same direction as COOBA but does not reach significance at 63 pairs (p=0.10). The practical takeaway: "pick the source where the model works best" is a flawed heuristic for COOBA and potentially for TRANP-CNN. For BLAZE, source quality is irrelevant to transfer quality — source selection should focus on target-side properties (LoC, bug count).

---

## Step 7 — Source Selection Analysis

**What it measures**: Whether simple heuristics identify the oracle-best source for each target. Hit@1 = heuristic picks the oracle source. Kendall τ = how well heuristic ranks all candidate sources.

**BLAZE source selection** (13 targets):

| Metric | Value |
|---|---|
| Hit@1 (most-bugs heuristic) | 15.4% (2/13 targets) |
| Hit@1 (composite score) | 7.7% |
| Mean Kendall τ (most-bugs) | −0.055 |
| Mean Kendall τ (composite) | −0.033 |
| Mean % of oracle MRR achieved | **88.6%** |

**COOBA source selection** (13 targets):

| Metric | Value |
|---|---|
| Hit@1 (most-bugs heuristic) | 38.5% (5/13 targets) |
| Hit@1 (composite score) | 38.5% |
| Mean Kendall τ (most-bugs) | −0.194 |
| Mean Kendall τ (composite) | −0.177 |
| Mean % of oracle MRR achieved | **78.8%** |

**TRANP-CNN source selection** (13 targets):

| Metric | Value |
|---|---|
| Hit@1 (most-bugs heuristic) | 30.8% (4/13 targets) |
| Hit@1 (composite score) | 38.5% |
| Mean Kendall τ (most-bugs) | +0.110 |
| Mean Kendall τ (composite) | −0.079 |
| Mean % of oracle MRR achieved | **86.7%** |

**Inference**: For BLAZE, the most-bugs heuristic achieves Hit@1 in 15% of targets but the practical cost of a suboptimal choice is low — any chosen source achieves 89% of oracle. Kendall τ is slightly negative (−0.055), meaning larger sources (by bug count) weakly but not reliably rank best. For COOBA, the most-bugs heuristic achieves Hit@1 in 38.5% of targets but has negative Kendall τ (−0.194) — it wins the top slot in the right target sometimes but does not rank all sources well. Because COOBA's oracle MRR achieved is only 79%, a suboptimal source choice carries greater harm than for BLAZE. For TRANP-CNN, the most-bugs heuristic has positive mean Kendall τ (+0.110), the only model where larger-bug-count sources tend to rank slightly better for the purposes of transfer — the opposite of BLAZE and COOBA. TRANP-CNN achieves 87% of oracle with the heuristic-selected source.

**Note on τ sign sensitivity**: With only 13 targets, Kendall τ estimates are sensitive to individual target outcomes. One pair change can flip the mean sign, so the τ values should be interpreted directionally but not treated as precise estimates.

---

## Full Three-Model Comparison Summary

| Property | BLAZE | COOBA | TRANP-CNN |
|---|---|---|---|
| CPT MRR | 0.392 | 0.156 | **0.380** |
| CPT/WPL ratio | 83.2% | 81.1% | **97.8%** |
| Cold-start MRR | **0.262** | 0.027 | 0.075 |
| CPT win rate (vs WPS) | **90.5% ***| 60.3% * | 68.3% *** |
| CPT vs WPL | significant underperformance | significant underperformance | **ns — ties** |
| Neg. transfer rate (Δ < −0.01) | **3.2% (2/63)** | 31.7% (20/63) | 27.0% (17/63) |
| Asym. ratio (MRR) | **27.3×** | 1.46× | 2.77× |
| Cold-start viable (MRR > 0.20) | **74.6% of pairs** | 0% of pairs | 9.5% of pairs |
| LoC sensitivity (CPT) | Low (±0.009) | Very high (4× across tiers) | Moderate |
| Source quality predicts transfer | No (ρ=−0.00) | Negatively (ρ=−0.30*) | No (ρ=−0.21, ns) |
| Heuristic achieves % of oracle | 89% | 79% | 87% |
| Cohen's d (CPT vs WPS, MRR) | 1.191 (large) | 0.272 (small) | 0.591 (medium) |

**Summary**: BLAZE is the confirmed **safe** CPL architecture — large effect, near-zero risk, cold-start capable. COOBA is the confirmed **weak** CPL architecture — small but now marginally significant effect (*), nearly one-third negative transfer, zero cold-start, severe LoC sensitivity. TRANP-CNN is the confirmed **high-ceiling** CPL architecture — its CPT matches WP-large statistically (the only model with this property), and exceeds WP-large for small targets, but it has no cold-start capability and substantial negative transfer in cross-domain settings (27% of pairs). The paper's headline finding: **CPL benefit is architecture-dependent, mediated by whether the representation space generalises across project boundaries** — embedding-based BLAZE generalises without fine-tuning; structure-based COOBA barely generalises even with fine-tuning; CNN-reranker TRANP-CNN generalises powerfully with fine-tuning but not without it.

---

*Scripts: `Scripts/analysis/` (main_results_table.py, negative_transfer_analysis.py, effect_size_analysis.py, cold_start_viability_analysis.py, cross_model_consistency_analysis.py, source_quality_transfer_analysis.py, source_selection_analysis.py)*  
*Data: `results/paper_results_complete.csv`*
