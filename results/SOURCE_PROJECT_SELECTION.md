# Source Project Selection for Cross-Project Bug Localization

**Study**: TRANP-CNN Phase 1 — Cross-Project Bug Localization on 20 Python Projects
**Scope**: 91 CP-transfer rows, 12 targets with ≥3 source candidates, 3 ranking strategies validated
**Generated**: 2026-03-14

---

## The Central Question

> *Given a target project that needs bug localisation, which source project should you train on?*

This is the practical bottleneck for deploying CPL. If source selection requires an oracle (knowing the answer in advance), CPL is not a usable framework. If a principled strategy exists, it becomes a deployable, reproducible methodology.

**Conclusion from this data**: No single feature reliably ranks source projects. What we can offer is a **decision framework** — a set of target-first filters and source-side heuristics — that performs comparably to the composite score and better than random. This is a set of strategies, not an algorithm.

---

## What We Tested

### 12 Targets with ≥ 3 Source Candidates

| Target | Sources | Oracle MRR | Composite MRR | Most-Bugs MRR | Random MRR |
|---|---|---|---|---|---|
| jax | 6 | 0.221 | 0.206 | 0.206 | 0.195 |
| jupyterlab | 5 | 0.668 | 0.666 | 0.666 | 0.625 |
| lightning | 4 | 0.208 | 0.204 | 0.204 | 0.192 |
| matplotlib | 11 | 0.292 | 0.209 | 0.209 | 0.205 |
| numpy | 11 | 0.185 | 0.129 | 0.185 | 0.155 |
| mmdetection | 7 | 0.138 | 0.138 | 0.138 | 0.092 |
| prefect | 4 | 0.180 | 0.175 | 0.175 | 0.157 |
| xarray | 4 | 0.209 | 0.179 | 0.179 | 0.192 |
| ray | 6 | 0.175 | 0.175 | 0.175 | 0.130 |
| scikit-learn | 5 | 0.259 | 0.200 | 0.200 | 0.193 |
| scipy | 11 | 0.024 | 0.024 | 0.024 | 0.017 |
| sympy | 11 | 0.158 | 0.120 | 0.120 | 0.109 |

### Ranking Strategies Evaluated

1. **Oracle**: always picks the best source (upper bound, not deployable)
2. **Composite score**: weighted combination of source features (src_bugs×0.35 + bug_sim×0.25 + domain_gap_inv×0.20 + src_LoC×0.10 + code_sim×0.10)
3. **Most-bugs heuristic**: always picks the source with the most bug reports
4. **Random**: uniform random selection (lower bound)

### Ranking Quality (Kendall τ)

Kendall τ measures whether a strategy correctly orders all source candidates for a given target. τ = 1 means perfect ordering; τ = −1 means perfectly reversed; τ = 0 means no signal.

| Strategy | Mean Kendall τ | Targets with τ > 0 |
|---|---|---|
| Composite score | −0.104 | 3/12 (25%) |
| Most-bugs heuristic | −0.183 | 3/12 (25%) |

### Hit@1 Accuracy (Did the strategy pick the best source?)

| Strategy | Hit@1 |
|---|---|
| Composite score | **25.0%** (3/12) |
| Most-bugs heuristic | **33.3%** (4/12) |
| Random (expected) | ~8–17% (1/n_sources) |

---

## Key Finding: Source-Side Features Are Weak Predictors

### Spearman Correlations with CP-transfer MRR

| Feature | ρ | p-value | Signal |
|---|---|---|---|
| tgt_bug_report_verbosity | +0.581 | <0.001 | **Strong — target property** |
| tgt_LoC | −0.516 | <0.001 | **Strong — target property** |
| tgt_n_bugs | +0.303 | 0.003 | Moderate — target property |
| domain_gap | −0.265 | 0.011 | Weak — pair property |
| src_n_bugs | +0.230 | 0.028 | Weak — source property |
| bug_report_similarity | −0.164 | 0.122 | Non-significant |
| src_LoC | +0.145 | 0.168 | Non-significant |
| code_similarity | −0.052 | 0.626 | Non-significant |

**The pattern is clear**: target-side properties explain most of the variance in achievable MRR. Source-side properties (number of bugs, codebase size, similarity) have weak or non-significant correlation.

### Correlations with CPL Gain (CP-transfer − WP-small)

| Feature | ρ | p-value |
|---|---|---|
| tgt_n_bugs | −0.375 | <0.001 |
| tgt_LoC | −0.239 | 0.022 |
| domain_gap | −0.184 | 0.080 |
| src_n_bugs | +0.019 | 0.862 |
| bug_report_similarity | −0.133 | 0.208 |

CPL gain is largest when the target is small and data-scarce — confirming E4 from CPL_DESIRABILITY.md.

---

## Is This an Algorithm or a Set of Strategies?

### The Honest Answer: A Set of Strategies

An algorithm implies a deterministic, validated procedure that reliably produces the best outcome. Our data does not support that claim:

- Hit@1 = 25–33% (no better than choosing the largest source)
- Mean Kendall τ = −0.10 (near-zero, slightly negative)
- No individual feature has strong predictive power for source ranking
- The composite score does not consistently outperform the simple most-bugs heuristic

**What we *can* claim** is a **decision framework** with two phases:

1. **Target viability check** (high confidence, data-supported): determine whether the target is a good CPL candidate at all
2. **Source candidate filtering** (moderate confidence, heuristic): among viable sources, apply coarse filters before picking

---

## The Source Selection Decision Framework

### Phase 1 — Target Viability (Apply Before Selecting Source)

These filters use target-side properties, which are strong predictors (ρ > 0.5):

| Check | Threshold | Action |
|---|---|---|
| **Bug report verbosity** | < 50 words mean | LOW CONFIDENCE — CPL may not localise well regardless of source |
| **Target codebase size** | > 200K LoC | LOW CONFIDENCE — correct file is buried in a very large search space |
| **Target bug count** | ≥ 50 bugs | RECONSIDER — WP-large may be more reliable than CP-transfer |

If the target fails these checks, CPL is still deployable, but expectations should be calibrated downward.

### Phase 2 — Source Candidate Filtering

Among all available source projects, apply these heuristics in order. None is individually reliable; together they reduce the candidate pool:

#### Strategy S1 — Prefer Larger Source Bug Corpora (Hit@1 = 33%)
The most-bugs heuristic is the single best deployable strategy. A larger source corpus provides more diversity in bug pattern coverage.

> **Rule**: prefer sources with ≥ 2× the number of bugs in the target.

#### Strategy S2 — Avoid Extreme Domain Gaps (weak signal, ρ = −0.265)
Domain gap (logistic regression accuracy between blob embeddings) has a weak negative correlation with CPL gain. Sources with domain gap > 0.95 are not reliably better or worse, but a domain gap > 0.97 corresponds to the far tail of poor pairs.

> **Rule**: exclude sources where domain gap > 0.97, unless no alternative exists.

#### Strategy S3 — Do Not Use Bug Report Similarity as a Ranking Signal
Bug report similarity has a non-significant correlation with CPL performance (ρ = −0.164, p = 0.122) and **negative mean Kendall τ = −0.094** — meaning it is if anything a misleading signal. Using bug similarity to select sources may actively harm source ranking.

> **Rule**: do not use bug report similarity as a selection criterion.

#### Strategy S4 — If Unsure, Prefer the Source with the Most Bugs
In the absence of better information, the most-bugs heuristic matches or exceeds the composite score in most targets. It is simple, interpretable, and reproducible.

> **Rule**: default to the source with the largest bug corpus.

---

## Why This Matters as a Contribution

Even a negative result on source selection is a contribution:

1. **It rules out intuitive but wrong approaches.** Bug report similarity sounds like a natural proxy for "related bug patterns" — but it fails empirically. Without this experiment, practitioners would waste time computing bug similarities or using them as a selection criterion.

2. **It isolates where the field should focus.** The dominant factors are target-side. This means: future work on CPL should focus on target viability prediction and adaptive architectures for large codebases — not on source selection algorithms.

3. **It establishes an honest baseline.** Hit@1 = 33% with the most-bugs heuristic is the current bar. Future methods (e.g., meta-learning based source selection, transfer learning with domain adaptation) can be evaluated against this.

4. **It shapes the research agenda.** The two questions this data cannot answer — which will motivate future work — are:
   - Can a learned source selection model (trained on pair outcomes) generalise to unseen targets?
   - Does ensembling multiple source projects outperform any single source?

---

## Detailed Observations by Target

### jupyterlab (best CPL target)
- Oracle MRR: 0.668; composite: 0.666 — **composite nearly matches oracle**
- Hit@1: Yes (composite correctly picked the best source)
- Why it works: 39K LoC target — small codebase, easier ranking
- τ (composite): −0.600 — ordering is poor but top-1 happens to be correct

### scipy (worst CPL target)
- Oracle MRR: 0.024 — **even the best possible source produces near-zero MRR**
- No strategy can fix this: 438K LoC, correct file buried in massive search space
- Source selection is irrelevant here — target viability check would have flagged this

### matplotlib (11 sources, hardest selection)
- Oracle MRR: 0.292; composite: 0.209 — **large gap between oracle and any strategy**
- τ (composite): −0.055 — no ordering signal across 11 sources
- Most-bugs heuristic also 0.209 — sources are nearly interchangeable in performance

### numpy (11 sources)
- Oracle: 0.185; most-bugs: 0.185; composite: 0.129 — **most-bugs outperforms composite**
- composite score hurt by bug_sim weight (negative signal)
- Demonstrates that composite weights are not universally beneficial

---

## Recommendations for Future Work

| Priority | Direction | Motivation |
|---|---|---|
| **High** | Learn source selection from pair outcomes (meta-model) | Current heuristics fail; learned selection from historical pairs may generalise |
| **High** | Ensemble multiple sources instead of picking one | All sources provide signal; combining may be more robust than selection |
| **Medium** | Investigate architecture for large-codebase targets | scipy/numpy→scipy: scale is the bottleneck, not source choice |
| **Medium** | Collect multi-language pairs | Current results are Python-only; Java/C++ behaviour unknown |
| **Low** | Explore domain adaptation instead of domain gap filtering | Bridges high domain-gap pairs rather than filtering them out |

---

## Summary Table

| Claim | Evidence | Confidence |
|---|---|---|
| CPL works better with data-rich sources | src_n_bugs ρ=+0.230 | Low (weak correlation) |
| CPL works better with similar bug reports | bug_sim ρ=−0.164 | **Negative — avoid this heuristic** |
| Low domain gap helps | domain_gap ρ=−0.265 | Low (weak) |
| Target size governs achievable MRR | tgt_LoC ρ=−0.516 | **High** |
| Bug verbosity governs achievable MRR | tgt_verbosity ρ=+0.581 | **High** |
| Most-bugs heuristic is best simple strategy | Hit@1 = 33% | Moderate (beats random, beats composite) |
| Source selection is a solved problem | Hit@1 = 25–33% | **No — unsolved** |

---

*Data sources: `results/tranp_cnn_ph1_summary.csv`, `results/source_selection_validation.csv`*
*Analysis scripts: `Scripts/analysis/source_selection_analysis.py`*
*Plots: `results/images/source_selection_*.png`*
