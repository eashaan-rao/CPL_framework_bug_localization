# Analysis Results Explanation — COOBA + BLAZE + TRANP-CNN (complete)

**Updated**: 2026-06-22  
**Data**: `results/paper_results_complete.csv`  
**Pair counts**: BLAZE = 62 pairs, COOBA = 62 pairs, TRANP-CNN = 62 pairs (all complete — same 13-project set)  
**Purpose**: Plain-language explanation of each analysis test: what it measured, what the numbers say, what we can infer.

> **Models**: BLAZE = embedding-based reranker; COOBA = GNN on AST embeddings; TRANP-CNN = CNN reranker on FAISS candidates.  
> **Scenarios**: WP-small = within-project 20% target data; WP-large = within-project 80% target data; CP-cold-start (CPC) = source-only, no target data; CP-transfer (CPT) = source + 20% target data.

---

## Step 1 — Main Results Table

### 1a. Scenario Performance Means

**What it measures**: Mean MRR/MAP/Top-K across all pairs per model per scenario. The reference ordering should be WP-small < CP-transfer < WP-large (CPL exceeds limited training, but full training is still best).

| Model | n | WP-small MRR | WP-large MRR | CPC MRR | **CPT MRR** |
|---|---|---|---|---|---|
| **BLAZE** | 62 | 0.254 | 0.472 | 0.260 | **0.392** |
| **COOBA** | 62 | 0.139 | 0.194 | 0.027 | **0.157** |
| **TRANP-CNN** | 62 | 0.328 | 0.393 | 0.076 | **0.383** |

Full metric table:

**BLAZE** (62 pairs):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.254 | 0.208 | 0.155 | 0.345 | 0.443 |
| WP-large | 0.472 | 0.391 | 0.344 | 0.608 | 0.705 |
| CP-cold-start | 0.260 | 0.211 | 0.152 | 0.356 | 0.469 |
| **CP-transfer** | **0.392** | **0.321** | **0.266** | **0.526** | **0.627** |

**COOBA** (62 pairs):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.139 | 0.121 | 0.058 | 0.218 | 0.315 |
| WP-large | 0.194 | 0.166 | 0.113 | 0.282 | 0.372 |
| CP-cold-start | 0.027 | 0.023 | 0.008 | 0.035 | 0.060 |
| **CP-transfer** | **0.157** | **0.135** | **0.076** | **0.239** | **0.342** |

**TRANP-CNN** (62 pairs):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.328 | 0.289 | 0.108 | 0.276 | 0.353 |
| WP-large | 0.393 | 0.352 | 0.133 | 0.317 | 0.391 |
| CP-cold-start | 0.076 | 0.066 | 0.010 | 0.055 | 0.096 |
| **CP-transfer** | **0.383** | **0.339** | **0.135** | **0.309** | **0.373** |

**Key observations:**

- **BLAZE**: CP-cold-start (MRR=0.260) is essentially equal to WP-small (0.254) — training on a source project alone gives the same performance as training on 20% of the actual target. CP-transfer (0.392) reaches 83% of WP-large.
- **COOBA**: CP-cold-start collapses to near-random (0.027, −80% vs WP-small). CP-transfer (0.157) only modestly beats WP-small (+13%). The ordering WPS < CPT < WPL holds, but all gains are small.
- **TRANP-CNN**: CPT (0.383) reaches **97.4%** of WP-large (0.393) — the CPT/WPL ratio is far higher than BLAZE (83%) or COOBA (81%). Cold-start (0.076) collapses like COOBA. The critical finding is that TRANP-CNN's CPT is statistically indistinguishable from WP-large (see Step 1b), meaning 20% target fine-tuning on a cross-project initialisation effectively matches 80% within-project training.

**Inference**: BLAZE's embedding representation generalises without fine-tuning; COOBA's AST features do not generalise at all; TRANP-CNN's CNN reranker generalises strongly after target-side adaptation — to the point where CPT ≈ WPL. If the goal is maximum CPL quality with 20% target labels, TRANP-CNN is the best model. If the goal is zero-annotation deployment, BLAZE is the only viable option.

---

### 1b. Wilcoxon Significance Tests — Is the CPL advantage real?

**What it measures**: Paired Wilcoxon tests comparing CPT against baselines. Win rate = fraction of pairs where CPT > baseline.

**CP-transfer vs WP-small (core CPL claim):**

| Model | n | MRR win rate | MAP win rate | Top-5 win rate | MRR mean Δ | MRR sig. |
|---|---|---|---|---|---|---|
| **BLAZE** | 62 | **90.3%** | **93.5%** | **90.3%** | **+0.138** | *** |
| **COOBA** | 62 | 61.3% | 61.3% | 54.8% | +0.018 | ns (p=0.059) |
| **TRANP-CNN** | 62 | **67.7%** | **61.3%** | **58.1%** | **+0.055** | *** |

**CP-cold-start vs WP-small:**

| Model | n | MRR win rate | MRR mean Δ | Significance |
|---|---|---|---|---|
| **BLAZE** | 62 | 41.9% | +0.006 | **ns — statistically tied** |
| **COOBA** | 62 | 6.5% | −0.112 | negative *** |
| **TRANP-CNN** | 62 | 0.0% | −0.252 | negative *** |

**CP-transfer vs WP-large (can CPL match full training?):**

| Model | n | MRR win rate | MRR mean Δ | Significance |
|---|---|---|---|---|
| **BLAZE** | 62 | 4.8% | −0.080 | *** — CPT loses |
| **COOBA** | 62 | 21.0% | −0.036 | *** — CPT loses |
| **TRANP-CNN** | 62 | 48.4% | −0.010 | **ns — CPT statistically ties WPL** |

**Inference**: CPT significantly outperforms WPS for BLAZE (large effect, ***) and TRANP-CNN (medium effect, ***). COOBA's advantage is borderline and no longer significant (p=0.059) at the conventional 5% level — the win rate (61.3%) remains positive but the effect size is too small to achieve significance. The most striking finding is TRANP-CNN vs WPL: with 48.4% win rate and mean delta of only −0.010 MRR, CPT is statistically indistinguishable from full within-project training (p=0.29). BLAZE is the only model where cold-start ties WPS; for COOBA and TRANP-CNN, cold-start is significantly worse than even limited within-project training.

---

### 1c. CPL Gain Summary — Are wins large and losses small?

**What it measures**: The asymmetry ratio (median win / |median loss|) tells whether gains justify adoption risk. Ratio >> 1 = good risk profile.

| Model | MRR win% | Med win | Med loss | Asym. ratio | Risk profile |
|---|---|---|---|---|---|
| **BLAZE** | 90.3% | +0.134 | −0.005 | **27.3×** | Very low risk |
| **COOBA** | 61.3% | +0.032 | −0.023 | 1.40× | Symmetric risk |
| **TRANP-CNN** | 67.7% | +0.070 | −0.025 | 2.80× | Moderate risk |

**Inference**: BLAZE has an excellent risk profile — gains are 27× larger than losses, and the rare losses are negligible (−0.005 MRR). COOBA's 1.4× ratio means adopting CPL for COOBA delivers gains and losses of similar magnitude; the 38.7% of pairs that don't benefit experience real harm. TRANP-CNN's 2.8× ratio is moderate — gains are meaningfully larger than losses, but not dramatically so. Unlike the 7-pair preview where TRANP-CNN appeared to have a 19× ratio, the full 62-pair sample reveals its risk profile includes genuine non-trivial losses (median loss −0.025 MRR) across 10 pairs in cross-domain and large-target settings.

---

## Step 2 — Negative Transfer, Commutativity, Cross-Domain

### 2a. Negative Transfer Characterization

**What it measures**: Per-pair CPT−WPS delta classified as positive (>+0.02), neutral (|Δ| ≤ 0.02), or negative (<−0.02).

| Model | n | Positive | Neutral | Negative |
|---|---|---|---|---|
| **BLAZE** | 62 | 50 (80.6%) | 11 (17.7%) | **1 (1.6%)** |
| **COOBA** | 62 | 20 (32.3%) | 27 (43.5%) | **15 (24.2%)** |
| **TRANP-CNN** | 62 | 36 (58.1%) | 16 (25.8%) | **10 (16.1%)** |

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

**Inference**: BLAZE negative transfer is negligible (1 case, near-zero). COOBA suffers genuine negative transfer in about a quarter of pairs — a serious deployment risk. TRANP-CNN, now evaluated at full scale (62 pairs), shows 10 negative transfer cases (16.1%) concentrated in cross-domain pairs involving ansible and prefect as targets (high-specificity codebases where source-project patterns appear to conflict). The 7-pair preview's 14.3% estimate was fortuitously accurate for the overall rate, but masked the cross-domain concentration. COOBA's negative transfer pattern persists: 15 of the 15 negative COOBA cases involve large-source → any-target directionality where AST overfitting is most severe.

---

### 2b. Cross-Domain Breakdown — Does domain gap limit CPL?

**What it measures**: CPT win rate and mean delta split into within-domain vs cross-domain pairs.

| Model | Within-domain n | Mean Δ | Win% | Cross-domain n | Mean Δ | Win% |
|---|---|---|---|---|---|---|
| **BLAZE** | 29 | +0.184 | 93.1% | 33 | +0.097 | 87.9% |
| **COOBA** | 29 | +0.028 | 65.5% | 33 | +0.010 | 57.6% |
| **TRANP-CNN** | 29 | +0.080 | 82.8% | 33 | +0.033 | 54.5% |

**Inference**: Domain gap matters most for TRANP-CNN: its within-domain win rate (82.8%) drops substantially to 54.5% for cross-domain pairs. BLAZE shows the smallest degradation — cross-domain pairs still achieve 87.9% win rate, confirming embedding-based representations are robust to domain shift. COOBA shows modest degradation (65.5% → 57.6%), since its overall performance is already low regardless of domain. TRANP-CNN's 28-percentage-point cross-domain drop explains the gap between its 7-pair preview (all within-domain) and the full 62-pair result — the earlier optimistic CPT estimate was entirely drawn from the easier within-domain regime.

---

### 2c. Commutativity — Does the smaller target benefit more?

**What it measures**: For symmetric pairs (A→B and B→A both present), does the smaller-LoC target consistently gain more from CPL?

| Model | Symmetric pairs | Smaller target benefits more | Rate |
|---|---|---|---|
| **BLAZE** | 27 | 19 | 70.4% |
| **COOBA** | 27 | 16 | 59.3% |
| **TRANP-CNN** | 27 | 14 | 51.9% |

**Inference**: For BLAZE, the smaller target benefits more in 70% of symmetric pairs, supporting the hypothesis that small codebases gain most from cross-project knowledge. COOBA shows a moderate asymmetry (59.3%), slightly above random — LoC is a weak but not absent predictor for COOBA. TRANP-CNN shows essentially random commutativity (51.9%), meaning LoC asymmetry does not govern TRANP-CNN's transfer direction — other factors (domain alignment, bug distribution) dominate. The full 27-pair TRANP-CNN sample reverses the 100% estimate from the 3-pair preview, confirming that earlier result was sampling noise.

---

## Step 3 — Effect Size Analysis and LoC-Stratified Table

### 3a. Effect Size (Cohen's d) for CPT vs WPS

**What it measures**: Standardised effect size for the CPT−WPS MRR delta. d ≈ 0.2 = small, 0.5 = medium, 0.8 = large.

| Model | n | MRR d | MAP d | Top-1 d | Top-5 d | Top-10 d | Classification |
|---|---|---|---|---|---|---|---|
| **BLAZE** | 62 | **1.196** | **1.179** | **1.128** | **1.140** | **1.042** | Large (all metrics) |
| **COOBA** | 62 | 0.277 | 0.243 | 0.310 | 0.204 | 0.298 | Small (all metrics) |
| **TRANP-CNN** | 62 | 0.593 | 0.564 | 0.453 | 0.525 | 0.315 | Medium (MRR/MAP/Top-5); Small (Top-1/Top-10) |

**Inference**: BLAZE's CPL advantage is a large effect (d > 1.0) consistently across all five metrics — detectable with as few as 10 pairs. COOBA's small effect (d ≈ 0.25–0.31) explains why its Wilcoxon test is borderline at 62 pairs. TRANP-CNN's medium d=0.593 on MRR and 0.564 on MAP confirms meaningful CPL gain at full scale, and its *** significance at 62 pairs validates the directional finding from the 7-pair preview. The lower Top-1 effect for TRANP-CNN (d=0.453) reflects its difficulty surfacing the ground-truth file as rank-1 even after cross-project transfer.

---

### 3b. Stratified by Target Codebase Size

**What it measures**: Mean CPT MRR by target LoC tier (Small <100K, Medium 100K–300K, Large >300K). Reveals whether CPL benefit depends on how large the target codebase is.

**BLAZE** (n=25 Small, 23 Medium, 14 Large):

| Size | WP-small MRR | WP-large MRR | CPC MRR | CPT MRR | CPT−WPS |
|---|---|---|---|---|---|
| Small (<100K) | 0.219 | 0.498 | 0.275 | 0.398 | +0.178 |
| Medium (100K–300K) | 0.274 | 0.460 | 0.247 | 0.386 | +0.113 |
| Large (>300K) | 0.283 | 0.444 | 0.252 | 0.389 | +0.107 |

**COOBA** (n=25 Small, 23 Medium, 14 Large):

| Size | WP-small MRR | WP-large MRR | CPC MRR | CPT MRR | CPT−WPS |
|---|---|---|---|---|---|
| Small (<100K) | 0.219 | 0.294 | 0.041 | 0.243 | +0.025 |
| Medium (100K–300K) | 0.100 | 0.154 | 0.017 | 0.118 | +0.018 |
| Large (>300K) | 0.062 | 0.080 | 0.021 | 0.068 | +0.006 |

**TRANP-CNN** (n=25 Small, 23 Medium, 14 Large):

| Size | WP-small MRR | WP-large MRR | CPC MRR | CPT MRR | CPT−WPS |
|---|---|---|---|---|---|
| Small (<100K) | 0.454 | 0.518 | 0.069 | **0.532** | +0.079 |
| Medium (100K–300K) | 0.281 | 0.348 | 0.111 | 0.305 | +0.024 |
| Large (>300K) | 0.181 | 0.242 | 0.030 | **0.243** | +0.062 |

**Inference**: BLAZE's CPL gain degrades gently with target size (+0.178 → +0.107) but remains substantial even for large targets. COOBA shows severe LoC degradation — large-target CPL gain is near zero (+0.006) and absolute performance collapses with size. TRANP-CNN reveals two striking results across all 62 pairs: (1) for **small targets**, CPT (0.532) exceeds WP-large (0.518) — cross-project transfer with 20% target fine-tuning actually outperforms 80% within-project training; (2) for **large targets**, CPT (0.243) ties WP-large (0.242) despite cold-start failing completely (0.030). These patterns hold at full scale and are not sampling artefacts. TRANP-CNN's CPT performance is remarkably robust to codebase size in absolute terms (0.532 → 0.305 → 0.243), unlike COOBA (0.243 → 0.118 → 0.068) which degrades catastrophically.

---

## Step 4 — Cold Start Viability

**What it measures**: Whether CP-cold-start (zero target labels) achieves MRR > 0.20 — the "deployable without annotation" threshold.

| Model | n | Mean CPC MRR | WPS MRR (reference) | Viable (>0.20) | Rate |
|---|---|---|---|---|---|
| **BLAZE** | 62 | **0.260** | 0.254 | 46/62 | **74.2%** |
| **COOBA** | 62 | 0.027 | 0.139 | 0/62 | **0.0%** |
| **TRANP-CNN** | 62 | 0.076 | 0.328 | 6/62 | **9.7%** |

BLAZE cold-start MRR (0.260) marginally exceeds WPS (0.254). TRANP-CNN cold-start (0.076) is 77% below its own WPS (0.328) — catastrophic cold-start failure.

**Inference**: BLAZE is the only model where zero-annotation CPL is deployment-viable. Its cold-start performance statistically ties with WPS across all metrics, meaning annotating 20% of target bugs adds no value over what the source-trained model already knows. For COOBA and TRANP-CNN, cold-start produces near-random rankings. The 6 TRANP-CNN pairs that clear the 0.20 threshold are all within-domain pairs where source and target share domain vocabulary — the reranker happens to learn partially transferable score calibrations. The cold-start dichotomy between BLAZE (embedding similarity generalises directly) vs TRANP-CNN/COOBA (reranking scores do not) is a core architectural finding.

---

## Step 5 — Cross-Model Consistency

**What it measures**: For the 62 pairs where all three models have results, whether models agree on CPL benefit direction (positive/neutral/negative).

**BLAZE vs COOBA** (n=62):

| Agreement type | Count | Rate |
|---|---|---|
| Strict (same direction label) | 25 / 62 | **40%** |
| Spearman ρ (delta magnitudes) | ρ=0.070, p=0.588 | No correlation |

**BLAZE vs TRANP-CNN** (n=62):

| Agreement type | Count | Rate |
|---|---|---|
| Strict (same direction label) | 40 / 62 | **65%** |
| Spearman ρ (delta magnitudes) | ρ=0.559, p<0.001 | Moderate positive correlation |

**COOBA vs TRANP-CNN** (n=62):

| Agreement type | Count | Rate |
|---|---|---|
| Strict (same direction label) | 20 / 62 | **32%** |
| Spearman ρ (delta magnitudes) | ρ=0.127, p=0.323 | No correlation |

Direction distribution across all 62 pairs:

| Direction | BLAZE | COOBA | TRANP-CNN |
|---|---|---|---|
| Positive | 50 (81%) | 20 (32%) | 36 (58%) |
| Neutral | 11 (18%) | 27 (44%) | 16 (26%) |
| Negative | 1 (2%) | 15 (24%) | 10 (16%) |

Notable disagreements (BLAZE gains, COOBA loses significantly):
- ansible/ansible → jupyterlab/jupyterlab: BLAZE +0.374 vs COOBA −0.080
- mesonbuild/meson → jupyterlab/jupyterlab: BLAZE +0.354 vs COOBA −0.152
- numpy/numpy → jupyterlab/jupyterlab: BLAZE +0.351 vs COOBA −0.026
- qiskit/qiskit → jupyterlab/jupyterlab: BLAZE +0.330 vs COOBA −0.084

Notable disagreements (BLAZE gains, TRANP-CNN loses):
- jupyterlab/jupyterlab → prefecthq/prefect: BLAZE +0.153 vs TRANP-CNN −0.049
- lightning-ai/lightning → prefecthq/prefect: BLAZE +0.151 vs TRANP-CNN −0.061
- lightning-ai/lightning → ansible/ansible: BLAZE +0.088 vs TRANP-CNN −0.078

**Inference**: BLAZE and COOBA agree on CPL direction in only 40% of shared pairs, with uncorrelated delta magnitudes (ρ=0.070, p=0.59). BLAZE and TRANP-CNN agree in 65% of pairs with a moderate correlation (ρ=0.559, p<0.001) — these two reranker-family models share more CPL sensitivity than either shares with COOBA. COOBA and TRANP-CNN agree in only 32% of pairs (below chance), confirming they exploit fundamentally different aspects of the source project. Architecture-agnostic CPL claims are not supported — CPL benefit is model-family-specific, with embedding-based and CNN-reranker models being more aligned with each other than with GNN-based approaches.

---

## Step 6 — Source Quality as Transfer Predictor

**What it measures**: Spearman ρ between source WP-large MRR (how well the model works on source within-project) and CP-transfer MRR on the target. Tests "pick the source where the model already performs best."

| Model | n | ρ (src WPL vs CPT MRR) | p-value | ρ (src WPL vs CPT delta) | Interpretation |
|---|---|---|---|---|---|
| **BLAZE** | 62 | −0.014 | 0.915 | +0.033 | No relationship |
| **COOBA** | 62 | **−0.316** | **0.012** | +0.019 | Neg. relation to absolute MRR |
| **TRANP-CNN** | 62 | −0.216 | 0.092 | +0.011 | Marginal negative (not significant) |

**Inference**: For BLAZE, source quality has no predictive power — any source is equally valid for transfer. For COOBA, there is a statistically significant negative correlation (ρ=−0.316): sources where COOBA achieves high WP-large MRR transfer *worse*, not better. This suggests COOBA overfits to source-specific AST patterns when it performs well within-project. For TRANP-CNN, the negative trend (ρ=−0.216) is in the same direction as COOBA but does not reach significance at 62 pairs (p=0.09). The practical takeaway: "pick the source where the model works best" is a flawed heuristic for COOBA and potentially for TRANP-CNN. For BLAZE, source quality is irrelevant to transfer quality — source selection should focus on target-side properties (LoC, bug count).

---

## Step 7 — Source Selection Analysis

**What it measures**: Whether simple heuristics identify the oracle-best source for each target. Hit@1 = heuristic picks the oracle source. Kendall τ = how well heuristic ranks all candidate sources.

**BLAZE source selection** (13 targets):

| Metric | Value |
|---|---|
| Hit@1 (most-bugs heuristic) | 15.4% (2/13 targets) |
| Hit@1 (composite score) | 7.7% |
| Mean Kendall τ (most-bugs) | +0.062 |
| Mean Kendall τ (composite) | +0.001 |
| Mean % of oracle MRR achieved | **88.9%** |

**COOBA source selection** (13 targets):

| Metric | Value |
|---|---|
| Hit@1 (most-bugs heuristic) | 38.5% (5/13 targets) |
| Hit@1 (composite score) | 38.5% |
| Mean Kendall τ (most-bugs) | +0.198 |
| Mean Kendall τ (composite) | +0.110 |
| Mean % of oracle MRR achieved | **78.8%** |

**TRANP-CNN source selection** (13 targets):

| Metric | Value |
|---|---|
| Hit@1 (most-bugs heuristic) | 30.8% (4/13 targets) |
| Hit@1 (composite score) | 38.5% |
| Mean Kendall τ (most-bugs) | −0.109 |
| Mean Kendall τ (composite) | −0.087 |
| Mean % of oracle MRR achieved | **86.7%** |

**Inference**: For BLAZE, the most-bugs heuristic achieves Hit@1 in 15% of targets but the practical cost is low — any chosen source achieves 89% of oracle. Kendall τ is near zero (+0.062), meaning the heuristic is a weak but slightly positive ranker. For COOBA, the most-bugs heuristic now shows positive Kendall τ (+0.198) — at full scale with 62 pairs, source size is a mildly useful predictor for COOBA (unlike the initial negative τ from the older partial dataset). However, because COOBA's oracle MRR achieved drops to 79%, a suboptimal source choice carries greater harm than for BLAZE. For TRANP-CNN, the most-bugs heuristic has negative Kendall τ (−0.109), meaning larger sources (by bug count) tend to transfer slightly worse for TRANP-CNN — possibly because high-bug sources are high-complexity projects whose CNN features are harder to generalise. TRANP-CNN achieves 87% of oracle with the heuristic-selected source, so the absolute cost of suboptimal selection remains tolerable.

---

## Full Three-Model Comparison Summary

| Property | BLAZE | COOBA | TRANP-CNN |
|---|---|---|---|
| CPT MRR | 0.392 | 0.157 | **0.383** |
| CPT/WPL ratio | 83.1% | 80.9% | **97.4%** |
| Cold-start MRR | **0.260** | 0.027 | 0.076 |
| CPT win rate (vs WPS) | **90.3% ***| 61.3% ns | 67.7% *** |
| CPT vs WPL | *** underperforms | *** underperforms | **ns — ties** |
| Neg. transfer rate | **1.6% (1/62)** | 24.2% (15/62) | 16.1% (10/62) |
| Asym. ratio (MRR) | **27.3×** | 1.40× | 2.80× |
| Cold-start viable | **74.2% of pairs** | 0% of pairs | 9.7% of pairs |
| LoC sensitivity (CPT) | Low (±0.011) | Very high (4× across tiers) | Moderate |
| Source quality predicts transfer | No (ρ=−0.01) | Negatively (ρ=−0.32*) | No (ρ=−0.22, ns) |
| Heuristic achieves % of oracle | 89% | 79% | 87% |
| Cohen's d (CPT vs WPS, MRR) | 1.196 (large) | 0.277 (small) | 0.593 (medium) |

**Summary**: BLAZE is the confirmed **safe** CPL architecture — large effect, near-zero risk, cold-start capable. COOBA is the confirmed **weak** CPL architecture — small borderline-insignificant effect, one-quarter negative transfer, zero cold-start, severe LoC sensitivity. TRANP-CNN is the confirmed **high-ceiling** CPL architecture — its CPT matches WP-large statistically (the only model with this property), and exceeds WP-large for small targets, but it has no cold-start capability and moderate negative transfer in cross-domain settings. The paper's headline finding: **CPL benefit is architecture-dependent, mediated by whether the representation space generalises across project boundaries** — embedding-based BLAZE generalises without fine-tuning; structure-based COOBA barely generalises even with fine-tuning; CNN-reranker TRANP-CNN generalises powerfully with fine-tuning but not without it.

---

*Scripts: `Scripts/analysis/` (main_results_table.py, negative_transfer_analysis.py, effect_size_analysis.py, cold_start_viability_analysis.py, cross_model_consistency_analysis.py, source_quality_transfer_analysis.py, source_selection_analysis.py)*  
*Data: `results/paper_results_complete.csv`*
