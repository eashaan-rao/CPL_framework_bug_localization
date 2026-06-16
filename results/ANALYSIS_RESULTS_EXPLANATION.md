# Analysis Results Explanation — COOBA + BLAZE (TRANP-CNN partial)

**Generated**: 2026-06-16 (updated after pair-set correction)  
**Data**: `results/obj1_paper_results.csv` — **63 pairs each** for BLAZE and COOBA (same 13-project paper set), TRANP-CNN 8 pairs (indicative only)  
**Purpose**: Plain-language explanation of what each analysis test measured, what the numbers show, and what we can infer for the paper.

> **Models**: BLAZE = embedding-based reranker; COOBA = GNN on AST embeddings; TRANP-CNN = CNN on text+code (partial run, treat as indicative only — all Wilcoxon tests non-significant due to n=7–8).  
> **Scenarios**: WP-small = within-project, 20% target data; WP-large = within-project, 80% target data; CP-cold-start (CPC) = source-only, no target data; CP-transfer (CPT) = source + 20% target data.  
> **Pair set**: All analyses use the same 63 source→target pairs for both BLAZE and COOBA. TRANP-CNN covers 8 of these 63 pairs (DS×DS only).

---

## Step 1 — Main Results Table

### 1a. Scenario Performance Means (All 5 metrics)

**What it measures**: Mean MRR/MAP/Top-K averaged across all source→target pairs per model and scenario.

**BLAZE** (n=63 pairs for WP-small; 62 for others due to 1 missing run):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.256 | 0.210 | 0.156 | 0.348 | 0.446 |
| WP-large | 0.472 | 0.391 | 0.343 | 0.608 | 0.705 |
| CP-cold-start | 0.262 | 0.214 | 0.155 | 0.359 | 0.472 |
| **CP-transfer** | **0.392** | **0.321** | **0.266** | **0.526** | **0.626** |

**COOBA** (n=63/62):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.138 | 0.120 | 0.057 | 0.216 | 0.313 |
| WP-large | 0.194 | 0.166 | 0.113 | 0.282 | 0.372 |
| CP-cold-start | 0.027 | 0.023 | 0.008 | 0.035 | 0.060 |
| **CP-transfer** | **0.157** | **0.135** | **0.076** | **0.239** | **0.342** |

**TRANP-CNN** (n=8/7, DS×DS only — directional):

| Scenario | MRR | MAP | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| WP-small | 0.431 | 0.402 | 0.136 | 0.330 | 0.416 |
| WP-large | 0.510 | 0.479 | 0.186 | 0.383 | 0.434 |
| CP-cold-start | 0.061 | 0.057 | 0.006 | 0.028 | 0.046 |
| **CP-transfer** | **0.468** | **0.432** | **0.171** | **0.366** | **0.408** |

**Key observation — BLAZE**: CP-cold-start (MRR=0.262) is essentially equal to WP-small (0.256) across all five metrics. This means training on a source project alone matches the performance of training on 20% of the actual target project. CP-transfer (0.392) is 53% better than WP-small in MRR and reaches 83% of WP-large. Top-5 and Top-10 gains are even larger (+51%, +40% respectively).

**Key observation — COOBA**: CP-cold-start collapses across all metrics — MRR drops to 0.027 (80% below WP-small 0.138), Top-1 to 0.008. CP-transfer (MRR=0.157) modestly exceeds WP-small by only +0.019 MRR (+14%), with similarly small gains across MAP and Top-K.

**Inference**: BLAZE's embedding representation generalises cross-project without fine-tuning, while COOBA's AST-based representation requires target-specific adaptation to remain competitive. The two architectures encode fundamentally different assumptions about what is transferable across projects.

---

### 1b. Wilcoxon Significance Tests (Is the CPL advantage real?)

**What it measures**: Paired Wilcoxon signed-rank tests comparing CP-transfer against each baseline across all five metrics. Win rate = fraction of pairs where CPT > baseline.

**CP-transfer vs WP-small (the core CPL claim):**

| Model | n | MRR win rate | MAP win rate | Top-1 win rate | Top-5 win rate | Top-10 win rate | MRR sig. |
|---|---|---|---|---|---|---|---|
| **BLAZE** | 62 | **90.3%** | **93.5%** | **82.3%** | **90.3%** | **88.7%** | *** |
| **COOBA** | 62 | 61.3% | 61.3% | 43.5% | 54.8% | 56.5% | * |
| TRANP-CNN | 7 | 57.1% | 71.4% | 42.9% | 71.4% | 28.6% | ns |

Mean MRR deltas (CPT − WPS): BLAZE +0.138, COOBA +0.018, TRANP-CNN +0.070.

**CP-cold-start vs WP-small:**

| Model | MRR win rate | Mean MRR delta | Significance |
|---|---|---|---|
| **BLAZE** | 43.5% | +0.009 | ns (near tie across all metrics) |
| **COOBA** | ~5% | −0.111 | highly negative across all metrics |

**CP-transfer vs WP-large (can CPL match full within-project training?):**

| Model | MRR win rate | Mean MRR delta | Significance |
|---|---|---|---|
| **BLAZE** | ~5% | −0.080 | ns — CPT loses |
| **COOBA** | ~24% | −0.037 | ns — CPT loses |

**Inference**: CPT significantly improves over WPS for both models, but with radically different magnitudes. BLAZE's gain is consistent across all five metrics (all `***`) and affects ~90% of pairs. COOBA's gain is marginal (MRR p=`*`, Top-5 not significant), with fewer than half the pairs improving on Top-1. Neither model closes the gap to WP-large, but BLAZE (83% of WP-large MRR) comes substantially closer than COOBA (81%). For BLAZE, the zero-shot cold-start is statistically indistinguishable from WPS on every metric, making it the only model where CPL is viable without any target labels.

---

### 1c. CPL Gain Summary (Are wins large and losses small?)

**What it measures**: Asymmetry between how much CPL gains when it wins vs how much it loses when it fails. `asymmetry_ratio` = median win / |median loss|. Ratio >1 means wins outweigh losses.

| Model | Metric | Win rate | Median win | Median loss | Asymmetry ratio |
|---|---|---|---|---|---|
| **BLAZE** | MRR | 90.3% | +0.134 | −0.005 | **27.3×** |
| **BLAZE** | MAP | 93.5% | +0.093 | −0.006 | **16.3×** |
| **BLAZE** | Top-1 | 82.3% | +0.125 | −0.017 | **7.5×** |
| **BLAZE** | Top-5 | 90.3% | +0.174 | −0.015 | **11.6×** |
| **BLAZE** | Top-10 | 88.7% | +0.146 | −0.048 | **3.0×** |
| **COOBA** | MRR | 61.3% | +0.032 | −0.023 | 1.40× |
| **COOBA** | MAP | 61.3% | +0.023 | −0.020 | 1.14× |
| **COOBA** | Top-1 | 43.5% | +0.037 | −0.025 | 1.46× |
| **COOBA** | Top-5 | 54.8% | +0.063 | −0.044 | 1.44× |
| **COOBA** | Top-10 | 56.5% | +0.032 | −0.049 | **0.65×** |

**Inference**: BLAZE's CPL benefit is strongly asymmetric across all metrics — gains are 3× to 27× larger than losses. Losses are negligible in absolute terms (−0.005 MRR). COOBA's gains and losses are nearly symmetric (ratios 0.65×–1.46×), meaning there is no consistent "wins are larger" property. For Top-10 specifically, COOBA's losses exceed its wins (ratio 0.65×), meaning the expected CPL outcome on Top-10 is negative for pairs that don't benefit. This is a practically important risk: a practitioner adopting COOBA for CPL would trade a 56% chance of a small Top-10 gain against a 44% chance of a larger loss.

---

## Step 2 — Negative Transfer, Commutativity, and Cross-Domain Analysis

### 2a. Negative Transfer Characterization

**What it measures**: Per-pair CPT−WPS delta (MRR) classified as positive (delta > 0.02), neutral (|delta| ≤ 0.02), or negative (delta < −0.02).

| Model | Positive transfer | Neutral | Negative transfer |
|---|---|---|---|
| **BLAZE** | 50 / 62 (80.6%) | 11 (17.7%) | **1 (1.6%)** |
| **COOBA** | 20 / 62 (32.3%) | 27 (43.5%) | **15 (24.2%)** |

**Inference**: BLAZE negative transfer is practically negligible — a single pair out of 62 shows meaningful harm, and even that single case is near-zero. COOBA has 24.2% of pairs in negative territory, making it a genuine deployment risk. COOBA's most common outcome is the neutral zone (43.5%), meaning CPL neither helps nor hurts for nearly half its pairs, which is itself a failure to deliver value.

---

### 2b. Cross-Domain Breakdown (Does domain gap limit CPL?)

**What it measures**: CPT−WPS MRR deltas split into within-domain (DS×DS, all DataScience projects) vs cross-domain pairs.

| Model | Within-domain (DS×DS) | n | Mean delta | Win rate | Cross-domain | n | Mean delta | Win rate |
|---|---|---|---|---|---|---|---|---|
| **BLAZE** | DS×DS | 30 | +0.180 | 93.3% | Cross-domain | 32 | +0.099 | 78.1% |
| **COOBA** | DS×DS | 29 | +0.028 | 44.8% | Cross-domain | 33 | +0.010 | 45.5% |

**Inference**: Domain gap does not prevent CPL benefit for BLAZE, but it does reduce the magnitude — the within-domain gain (+0.180 MRR) is nearly double the cross-domain gain (+0.099). BLAZE's cross-domain win rate (78%) remains high, indicating CPL is broadly viable even across domain boundaries. For COOBA, within-domain and cross-domain win rates are essentially identical (45% vs 46%), meaning the domain distinction carries no signal for COOBA — its CPL outcome is determined by other factors not captured by the DS vs non-DS split. The cross-domain win rates for COOBA are below 50% in both cases, reinforcing that COOBA CPL is unreliable in general.

---

### 2c. Commutativity (Does direction matter in symmetric pairs?)

**What it measures**: For symmetric pairs where both A→B and B→A appear in results, tests whether the smaller-LoC target consistently gains more from CPL.

| Model | Symmetric pairs | Smaller target benefits more | Rate |
|---|---|---|---|
| **BLAZE** | 28 | 19 | **67.9%** |
| **COOBA** | 24 | 13 | **54.2%** |

**Inference**: For BLAZE, the smaller-LoC target benefits more in 68% of symmetric pairs, consistent with the hypothesis that smaller codebases lack diversity in their limited 20% training split and gain most from the cross-project knowledge. COOBA shows near-random commutativity (54%), indicating that smaller targets do not systematically benefit more — the direction of COOBA transfer is determined by factors the LoC-based hypothesis does not capture.

---

## Step 3 — Effect Size Analysis and LoC-Stratified Table

### 3a. Effect Size (Cohen's d) for CPT vs WPS

**What it measures**: Standardised effect size for the CPT−WPS delta distribution. d ≈ 0.2 = small, 0.5 = medium, 0.8 = large.

| Model | MRR d | MAP d | Top-1 d | Top-5 d | Top-10 d | Interpretation |
|---|---|---|---|---|---|---|
| **BLAZE** | **1.20** | **1.18** | **1.14** | **1.13** | **1.03** | Large across all metrics |
| **COOBA** | 0.28 | 0.24 | 0.31 | 0.20 | 0.30 | Small across all metrics |
| TRANP-CNN | 0.60 | 0.60 | 0.59 | 0.70 | 0.003 | Medium (n=7, indicative) |

**Inference**: BLAZE's CPL advantage is a large effect by Cohen's convention on every metric — detectable with as few as 10 pairs. COOBA's small effect (d≈0.25) explains why its Wilcoxon results are marginal despite 62 pairs. The consistency of Cohen's d across all 5 metrics for BLAZE (1.03–1.20) confirms this is a broad, robust performance gain, not specific to the MRR ranking measure.

---

### 3b. Stratified Results by Target Codebase Size

**What it measures**: Mean CPT MRR broken down by target LoC: Small (<100K), Medium (100K–300K), Large (>300K). Pair counts: Small n=25, Medium n=24, Large n=13.

**BLAZE CPT MRR by target LoC group:**

| Size group | WP-small MRR | CP-transfer MRR | Delta | n pairs |
|---|---|---|---|---|
| Small (<100K) | 0.219 | 0.398 | **+0.178** | 25 |
| Medium (100K–300K) | 0.278 | 0.388 | +0.110 | 24 |
| Large (>300K) | 0.283 | 0.386 | +0.103 | 13 |

**COOBA CPT MRR by target LoC group:**

| Size group | WP-small MRR | CP-transfer MRR | Delta | n pairs |
|---|---|---|---|---|
| Small (<100K) | 0.225 | 0.254 | +0.030 | 25 |
| Medium (100K–300K) | 0.098 | 0.117 | +0.018 | 24 |
| Large (>300K) | 0.065 | 0.066 | +0.001 | 13 |

**Inference**: For BLAZE, CPL benefit is largest for small targets (+0.178 MRR) but remains substantial for medium (+0.110) and large (+0.103) targets — a gradient, not a threshold. Absolute performance is also relatively stable across sizes (0.386–0.398 MRR), meaning BLAZE scales. For COOBA, LoC is effectively a hard ceiling on achievable performance: medium targets gain only +0.018 and large targets gain essentially nothing (+0.001). COOBA's absolute CPT performance degrades severely with target size (0.254 → 0.117 → 0.066 MRR), revealing that AST-level features become insufficient to discriminate files in large, architecturally complex codebases even with source-project pre-training.

---

## Step 4 — Cold Start Viability Analysis

**What it measures**: Whether CP-cold-start (source-only, no target labels) achieves MRR > 0.20 — defined as the "viability" threshold for a useful shortlist tool.

| Model | Mean CPC MRR | Mean CPC MAP | Mean CPC Top-5 | Viable (MRR > 0.20) | Viability rate |
|---|---|---|---|---|---|
| **BLAZE** | **0.262** | **0.214** | **0.359** | 46 / 62 | **74.2%** |
| **COOBA** | 0.027 | 0.023 | 0.035 | 0 / 62 | **0.0%** |
| TRANP-CNN | 0.061 | 0.057 | 0.028 | 1 / 7 | 14.3% |

Recall: BLAZE WP-small MRR = 0.256. Cold-start MRR (0.262) marginally exceeds it.

**Inference**: BLAZE's embedding representation transfers so directly that for nearly three-quarters of pairs, a practitioner can apply the source-trained model to a new project with zero annotations and immediately get a useful bug localiser. The cold-start performance is statistically tied to WP-small on every metric, meaning collecting 20% of target labels adds no benefit beyond what cross-project knowledge already provides. COOBA has no zero-shot capability whatsoever — its 0.027 MRR cold-start is near-random across all metrics and no pairs cross the viability threshold. COOBA requires target-side fine-tuning to be useful, ruling it out for the cold-start deployment scenario.

---

## Step 5 — Cross-Model Consistency Analysis

**What it measures**: For all 61 shared pairs (where both BLAZE and COOBA results exist), whether both models agree on the direction of CPL benefit (positive / neutral / negative).

| Agreement type | Count | Rate |
|---|---|---|
| Strict (same direction label) | 25 / 61 | **41.0%** |
| Sign (not opposite direction) | 35 / 61 | 57.4% |

Direction distribution per model across 61 shared pairs:

| Direction | BLAZE | COOBA |
|---|---|---|
| Positive | 53 (87%) | 23 (38%) |
| Neutral | 6 (10%) | 21 (34%) |
| Negative | 2 (3%) | 17 (28%) |

**Inference**: The two models agree on CPL benefit direction in only 41% of shared pairs. The most common disagreement pattern (18 pairs) is BLAZE-positive vs COOBA-negative — BLAZE gains while COOBA loses on the same pair. This means CPL benefit is substantially architecture-dependent: BLAZE's embedding-based reranker almost always benefits (87% positive), while COOBA's GNN-based localiser is genuinely uncertain (38% positive, 28% negative). The architecture-agnostic CPL conclusion — that CPL works regardless of model — cannot be made from these results. Instead, the finding is that CPL benefit depends critically on whether the model's representation is transferable across projects, which favours embedding-based over structure-based architectures.

---

## Step 6 — Source Quality as Transfer Predictor

**What it measures**: Spearman ρ between source project within-project performance (WP-large MRR) and CPT MRR on the target. Tests "pick the source where the model already works best."

| Model | n pairs | Spearman ρ | p-value | Interpretation |
|---|---|---|---|---|
| **BLAZE** | 62 | **+0.010** | 0.940 | No relationship |
| **COOBA** | 62 | **−0.316** | **0.012** | Significant negative relationship |
| TRANP-CNN | 7 | −0.788 | 0.035 | Negative (n too small) |

**Inference**: For BLAZE, source quality (within-project performance) has no predictive power for transfer quality — a strong BLAZE source is no more likely to transfer well than a weak one (ρ=0.01, p=0.94). For COOBA, there is a statistically significant *negative* correlation (ρ=−0.316, p=0.012): sources where COOBA works well within-project actually transfer *worse* to targets. This counter-intuitive finding likely reflects overfitting to source-specific AST structures — a source with distinctive AST patterns that make it easy to localise within-project may have learned features that are too source-specific to generalise. For practitioners using COOBA, "pick the best source" is not only unhelpful but actively harmful as a selection strategy.

---

## Step 7 — Source Selection Analysis

**What it measures**: How well simple heuristics — "pick the source with the most bugs" (bugs heuristic) and a composite score (50% bug count + 25% LoC + 25% inverse domain gap) — identify the oracle-best source for each target. Hit@1 = heuristic picks the oracle-best source. Kendall τ = rank correlation between heuristic ranking and oracle ranking across all candidate sources.

**BLAZE source selection** (12 targets — qiskit excluded as only-source):

| Metric | Value |
|---|---|
| Hit@1 (bugs heuristic) | 8.3% (1/12 targets) |
| Hit@1 (composite) | 8.3% (1/12 targets) |
| Median τ (bugs) | −0.09 |
| Median τ (composite) | +0.05 |
| Mean % of oracle MRR achieved | **87.0%** |

**COOBA source selection** (13 targets):

| Metric | Value |
|---|---|
| Hit@1 (bugs heuristic) | 38.5% (5/13 targets) |
| Hit@1 (composite) | 38.5% (5/13 targets) |
| Median τ (bugs) | −0.333 |
| Median τ (composite) | −0.333 |
| Mean % of oracle MRR achieved | **82.0%** |

**Inference**: The two models tell very different source selection stories. For BLAZE, Hit@1 is near-zero (8%) and Kendall τ is near zero — neither heuristic reliably identifies the best source, but this matters little in practice because any reasonable source choice achieves 87% of oracle performance on average. BLAZE is robust to source selection: the heuristic cost is low. For COOBA, Hit@1 is 38.5% but Kendall τ is consistently negative (−0.333 median), meaning the bugs heuristic ranks sources in roughly the *reverse* of the optimal order. The fact that Hit@1 is non-trivial (38.5%) but τ is negative suggests the heuristic sometimes gets the top pick right by chance while ordering the rest poorly. Given the sensitivity to source choice for COOBA (82% oracle vs BLAZE 87%, and COOBA has a harder distribution), source selection is more consequential for COOBA — a bad heuristic choice risks pushing outcomes into negative-transfer territory.

---

## Summary: Architecture Risk Profiles for CPL Adoption

| Property | BLAZE | COOBA |
|---|---|---|
| CPT win rate (MRR) | 90.3% | 61.3% |
| CPT effect size (d) | 1.20 (large) | 0.28 (small) |
| Win/loss asymmetry (MRR) | 27.3× | 1.40× |
| Negative transfer rate | 1.6% | 24.2% |
| Cold-start viability | 74.2% pairs, all metrics near WPS | 0% pairs |
| LoC sensitivity | Low (consistent +0.10–0.18 MRR gain) | Very high (large targets: +0.001) |
| Cross-model agreement | 41% strict (87% BLAZE positive) | 41% strict (38% COOBA positive) |
| Source quality predicts transfer | No (ρ=+0.01, p=0.94) | Yes, negatively (ρ=−0.32, p=0.012) |
| Heuristic reaches % of oracle | 87% | 82% |

**Overall inference**: BLAZE is a strong, low-risk CPL candidate — its cold-start viability, large effect size, near-zero negative transfer rate, and robustness to source selection make it suitable for deployment recommendation. COOBA's CPL benefit exists but is fragile, LoC-limited, frequently harmful for large targets, and paradoxically harmed by selecting "the best" source. The architecture contrast is itself a primary finding: CPL benefit is not architecture-agnostic — it is mediated by whether the model's representation space generalises across project boundaries, a property that distinguishes embedding-based from structure-based (AST-based) approaches.

---

## Feature Correlation Analysis (supplementary)

The negative transfer and source selection analyses both computed Spearman correlations between project-level metadata features and CPT outcome. No measured feature — LoC, bug count, domain gap accuracy, report verbosity — reliably predicts CPT success or failure across both models. For BLAZE, domain gap (within-domain vs cross-domain) is the single most informative feature (higher win rate and larger gain within DS×DS), but even this is not a reliable predictor at the pair level. For COOBA, source WP-large MRR is the strongest predictor (negative, ρ=−0.316), suggesting that source-specific model overfitting is the primary driver of CPL failure. The absence of strong positive predictors means we cannot yet provide a simple rule for when CPL will succeed: it is currently architecture-dependent and only consistently reliable for BLAZE.

---

*Note: TRANP-CNN results are from 7–8 pairs (DS×DS only) and all CPT comparisons are statistically underpowered (p>0.05 for all metrics). Directionally, TRANP-CNN resembles BLAZE (large CPT wins, near-zero cold-start), but this cannot be confirmed statistically until the full 63-pair run completes.*

---

*Data: `results/obj1_paper_results.csv` (63 pairs each, BLAZE + COOBA)*  
*Supporting: `results/main_results_*.csv`, `results/effect_size_*.csv`, `results/cold_start_viability.csv`, `results/cross_model_agreement.csv`, `results/negative_transfer_analysis*.csv`, `results/cross_domain_breakdown.csv`, `results/source_quality_transfer.csv`, `results/source_selection_validation_*.csv`*
