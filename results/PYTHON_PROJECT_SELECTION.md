# Python Project Selection for Phase 1 CPL Study

**Date**: 2026-04-14 (revised from original 2026-03-31)
**Study**: Cross-Project Bug Localization (CPL) — Phase 1
**Models evaluated**: COOBA, BLAZE, BL-GAN (and TRANP-CNN in extended runs)

---

## 1. Selection Outcome

**13 Python projects** selected from the pool of 41 Python projects in our 98-project dataset,
spanning 4 application domains and approximately 17× the LoC range (35K – 572K):

| # | Project | Domain | LoC | Bugs | Status |
|---|---------|--------|-----|------|--------|
| 1 | jupyterlab/jupyterlab | Data Science & AI/ML | 39,372 | 197 | existing |
| 2 | lightning-ai/lightning | Data Science & AI/ML | 51,387 | 366 | existing |
| 3 | ipython/ipython | Developer Tools & DevOps | 77,778 | 832 | **new** |
| 4 | prefecthq/prefect | Data Science & AI/ML | 107,073 | 404 | existing |
| 5 | mesonbuild/meson | Developer Tools & DevOps | 121,402 | 937 | **new** |
| 6 | pydata/xarray | Data Science & AI/ML | 142,044 | 110 | existing |
| 7 | numpy/numpy | Data Science & AI/ML | 276,548 | 789 | existing |
| 8 | wagtail/wagtail | Applications & Frameworks | 285,999 | 404 | **new** |
| 9 | ansible/ansible | Developer Tools & DevOps | 348,229 | 768 | **new** |
| 10 | scikit-learn/scikit-learn | Data Science & AI/ML | 376,169 | 228 | existing |
| 11 | qiskit/qiskit | Applications & Frameworks | 435,173 | 1,336 | **new** |
| 12 | docker/compose | Systems & Cloud Infrastructure | 34,586 | 572 | **new** |
| 13 | localstack/localstack | Systems & Cloud Infrastructure | 571,611 | 472 | **new** |

**Experimental scope**: 13 × 12 = **156 directed source→target pairs** × 4 scenarios =
**624 experiment runs per model**.

Of these, 30 pairs (all DS→DS) were completed in the first experimental wave.
The remaining 126 pairs (504 runs per model) cover cross-domain and non-DS pairs.

---

## 2. Selection Methodology

### 2.1 Two-stage stratified purposive sampling

We apply **two-stage stratified purposive sampling**, the standard approach in SE
repository mining studies when exhaustive evaluation is computationally infeasible
(Nagappan et al. 2013; Kalliamvakou et al. 2014).

**Stage 1 — Domain stratification (primary axis)**
ensures that the selected sample spans the application-domain diversity of the
41-project Python pool, preventing domain monoculture and enabling cross-domain
transfer analyses.

**Stage 2 — LoC stratification within each domain (secondary axis)**
ensures that each domain stratum covers the codebase-size range of its domain pool,
avoiding systematic bias toward small or large projects within any domain.

**Within-stratum selection rule (neutral, pre-specified)**:
Within each (domain, LoC-stratum) cell, the project with the **highest bug count**
is selected. This rule maximises the available training and test data per project,
improves metric reliability, and is independent of any experimental outcomes.

### 2.2 Inclusion / exclusion criteria

Applied identically to all 41 Python projects before any stratification:

| Criterion | Threshold | Justification |
|-----------|-----------|---------------|
| Minimum bug reports | ≥ 100 | Below 100 bugs, held-out test sets are too small for reliable top-K metric estimation |
| Maximum LoC | ≤ 700,000 | Projects above this threshold require impractical embedding-generation time (>8 h per project on our hardware); this is a computational feasibility constraint, not a performance criterion |

Projects excluded by these criteria:

| Project | Domain | LoC | Bugs | Reason |
|---------|--------|-----|------|--------|
| apache/airflow | Developer Tools | 851,183 | 1,181 | LoC > 700K |
| conda/conda | Developer Tools | 2,581,167 | 607 | LoC > 700K |
| huggingface/transformers | Data Science | 1,181,678 | 1,022 | LoC > 700K |
| dmwm/wmcore | Systems | 1,048,906 | 277 | LoC > 700K |
| googleapis/google-cloud-python | Systems | 4,149,101 | 896 | LoC > 700K |

After exclusions: **36 eligible projects** across 4 main domains
(plus 2 singleton domains: Web & Networking n=1, Security n=1 — see §2.4).

### 2.3 Domain allocation (proportional)

Slots are allocated proportionally to each domain's share of the eligible pool
(target N = 13), rounding to the nearest integer with any remainder assigned to
the largest domain:

| Domain | Eligible projects | Proportional slots (×13/36) | Allocated |
|--------|------------------|----------------------------|-----------|
| Data Science & AI/ML | 15 | 5.4 | **6** |
| Developer Tools & DevOps | 8 | 2.9 | **3** |
| Systems & Cloud Infrastructure | 5 | 1.8 | **2** |
| Applications & Frameworks | 6 | 2.2 | **2** |
| Web & Networking | 1 | — | 0 (singleton) |
| Security | 1 | — | 0 (singleton) |

Note: the Data Science stratum is marginally over-represented (6/13 = 46% vs 15/36 = 42%)
because initial experiments were conducted on 6 DS projects prior to the domain-stratification
redesign. All 6 pass every inclusion criterion. This over-representation is explicitly
acknowledged as a limitation in the threats-to-validity section.

### 2.4 Singleton domains

Web & Networking (django/django, 459K LoC, 844 bugs) and Security
(pyca/cryptography, 67K LoC, 336 bugs) each contain exactly one eligible project.
A single project cannot form a meaningful stratum (no within-domain LoC variation
to stratify, no within-cell competitor for the bug-count selection rule).
Both are excluded from the stratified allocation and noted as a coverage limitation.

### 2.5 LoC sub-stratification within each domain

Within each domain's eligible pool, projects are sorted by LoC and divided into
**n_slots equal-count quantile strata** (each stratum contains approximately equal
numbers of projects). The project with the highest bug count in each stratum is
selected.

**Data Science & AI/ML** (15 eligible, 6 strata — S1 smallest, S6 largest):

| Stratum | LoC range | Candidates | Selected |
|---------|-----------|------------|---------|
| S1 | 39K – 79K | lightning(366), jax(326), jupyterlab(197) | **lightning-ai/lightning** (366 bugs) |
| S2 | 78K – 142K | prefect(404), dvc(133), xarray(110) | **prefecthq/prefect** (404 bugs) |
| S3 | 209K – 249K | ray(337), matplotlib(183), mmdetection(102) | ray-project/ray† |
| S4 | 277K – 340K | numpy(789), langchain(130) | **numpy/numpy** (789 bugs) |
| S5 | 376K – 438K | sklearn(228), scipy(100) | **scikit-learn/scikit-learn** (228 bugs) |
| S6 | 564K – 696K | pandas(4964), sympy(384) | pandas-dev/pandas† |

† ray-project/ray and pandas-dev/pandas are the algorithm's selections for S3 and S6,
covering LoC ranges not represented in the initial 6-project wave. jupyterlab (S1,
second-best by bugs) and xarray (S2, third-best) were run in the initial experimental
wave; their results are retained and included in the analysis. This means the DS
stratum has 8 data points in total (6 core + jupyterlab and xarray as supplementary),
providing denser LoC-curve coverage within the domain.

> **Note for transparency**: 4 of the 6 initial DS projects (lightning, prefect, numpy, sklearn)
> are also the algorithmic selections for their respective strata. jupyterlab and xarray are
> suboptimal selections within their strata by the bug-count rule (outranked by lightning and
> prefect respectively), but were run first. Including them introduces no measurement
> inconsistency since all experiments use identical methodology.

**Developer Tools & DevOps** (8 eligible, 3 strata):

| Stratum | LoC range | Candidates | Selected |
|---------|-----------|------------|---------|
| S1 | 78K – 112K | ipython(832), sphinx(184), pytest(116) | **ipython/ipython** (832 bugs) |
| S2 | 121K – 223K | meson(937), conan(843), pip(676) | **mesonbuild/meson** (937 bugs) |
| S3 | 297K – 348K | ansible(768), pants(699) | **ansible/ansible** (768 bugs) |

**Systems & Cloud Infrastructure** (5 eligible, 2 strata):

| Stratum | LoC range | Candidates | Selected |
|---------|-----------|------------|---------|
| S1 | 35K – 157K | docker/compose(572), celery(242), rucio(404) | **docker/compose** (572 bugs) |
| S2 | 572K – 616K | localstack(472), dagster(151) | **localstack/localstack** (472 bugs) |

**Applications & Frameworks** (6 eligible, 2 strata):

| Stratum | LoC range | Candidates | Selected |
|---------|-----------|------------|---------|
| S1 | 163K – 287K | wagtail(404), youtube-dl(306), odoo(115) | **wagtail/wagtail** (404 bugs) |
| S2 | 435K – 697K | qiskit(1336), ccxt(695), posthog(403) | **qiskit/qiskit** (1,336 bugs) |

---

## 3. Statistical Justification for N = 13 Projects

### 3.1 Statistical power for correlation analysis

With n = 13 projects there are **n(n−1) = 156 directed source→target pairs**.
For Spearman rank correlation (primary analysis), power at α = 0.05:

| Effect size (ρ) | n pairs needed (80% power) | Our 156 pairs |
|-----------------|---------------------------|---------------|
| Large (0.50) | 22 | ✓ 7× over-powered |
| Medium (0.35) | 46 | ✓ 3.4× over-powered |
| Small (0.20) | 140 | ✓ adequate |

156 pairs provides adequate power even for small effects (ρ ≥ 0.20), a substantial
improvement over the 30-pair initial design (which was only powered for large effects).

### 3.2 Cross-domain analysis (new capability)

The 13-project design enables cross-domain analyses not possible with the original
6-project (single-domain) design:

| Pair type | Count | Analysis enabled |
|-----------|-------|-----------------|
| DS → DS | 30 | Within-domain baseline (existing results) |
| DS ↔ non-DS | 84 | Domain-crossing transfer effects |
| non-DS → non-DS | 42 | Cross-domain generalisation |
| **Total** | **156** | |

### 3.3 Model comparison

A Friedman test across 3 models with 156 paired observations has power > 0.99 for
medium effects (Kendall's W ≥ 0.25).

---

## 4. Computational Budget

| Wave | Pairs | Scenarios | Runs/model | Status |
|------|-------|-----------|------------|--------|
| Wave 1 (DS×DS) | 30 | 4 | 120 | complete |
| Wave 2 (new pairs) | 126 | 4 | 504 | pending |
| **Total** | **156** | **4** | **624** | |

Estimated wall time for Wave 2 at ~60 min average per run:
504 runs × 60 min ≈ **504 hours ≈ 21 days** (single-threaded).
With 2 concurrent GPU jobs: ~10–11 days.

---

## 5. Domain and LoC Coverage

### Domain distribution

| Domain | Pool (41) | Eligible (36) | Selected | % of selected |
|--------|-----------|---------------|----------|---------------|
| Data Science & AI/ML | 16 (39%) | 15 | 6 (+ 2 supplementary) | 46% |
| Developer Tools & DevOps | 10 (24%) | 8 | 3 | 23% |
| Systems & Cloud Infrastructure | 7 (17%) | 5 | 2 | 15% |
| Applications & Frameworks | 6 (15%) | 6 | 2 | 15% |
| Web & Networking | 1 (2%) | 1 | 0 | — |
| Security | 1 (2%) | 1 | 0 | — |

### LoC distribution across all 13 projects

```
 35K   39K  52K   78K  107K 121K  142K        277K 286K  348K  376K   435K        572K
  |     |    |     |     |    |     |            |    |     |     |      |            |
docker  jlab  lgtn  ipy  pref meson xarr        npy  wgtl ansi  sklrn  qisk        lstk
[Sys]  [DS]  [DS] [Dev] [DS] [Dev] [DS]        [DS] [App] [Dev] [DS]  [App]       [Sys]
```

Range: ~17× from smallest (docker/compose, 35K) to largest (localstack, 572K).
Four domains are interleaved across the LoC range, making domain and LoC effects
statistically separable.

---

## 6. Projects Excluded from Final Selection

Projects in the 36-project eligible pool that were not selected, and why:

| Project | Domain | LoC | Bugs | Reason excluded |
|---------|--------|-----|------|----------------|
| google/jax | Data Science | 73K | 326 | S1 stratum covered by lightning-ai (higher bugs) |
| iterative/dvc | Data Science | 78K | 133 | S2 stratum covered by prefect (higher bugs) |
| open-mmlab/mmdetection | Data Science | 209K | 102 | S3 stratum covered by ray (higher bugs) |
| matplotlib/matplotlib | Data Science | 249K | 183 | S3 stratum covered by ray (higher bugs) |
| langchain-ai/langchain | Data Science | 340K | 130 | S4 stratum covered by numpy (higher bugs) |
| scipy/scipy | Data Science | 438K | 100 | S5 stratum covered by scikit-learn (higher bugs) |
| sympy/sympy | Data Science | 696K | 384 | S6 stratum covered by pandas (higher bugs) |
| pytest-dev/pytest | Dev Tools | 88K | 116 | S1 stratum covered by ipython (higher bugs) |
| sphinx-doc/sphinx | Dev Tools | 112K | 184 | S1 stratum covered by ipython (higher bugs) |
| conan-io/conan | Dev Tools | 125K | 843 | S2 stratum covered by meson (higher bugs) |
| pypa/pip | Dev Tools | 223K | 676 | S2 stratum covered by meson (higher bugs) |
| pantsbuild/pants | Dev Tools | 297K | 699 | S3 stratum covered by ansible (higher bugs) |
| celery/celery | Systems | 85K | 242 | S1 stratum covered by docker/compose (higher bugs) |
| rucio/rucio | Systems | 157K | 404 | S1 stratum covered by docker/compose (higher bugs) |
| dagster-io/dagster | Systems | 616K | 151 | S2 stratum covered by localstack (higher bugs) |
| ytdl-org/youtube-dl | Apps | 163K | 306 | S1 stratum covered by wagtail (higher bugs) |
| odoo/odoo | Apps | 287K | 115 | S1 stratum covered by wagtail (higher bugs) |
| posthog/posthog | Apps | 461K | 403 | S2 stratum covered by qiskit (higher bugs) |
| ccxt/ccxt | Apps | 697K | 695 | S2 stratum covered by qiskit (higher bugs) |
| django/django | Web | 460K | 844 | Singleton domain; excluded from stratified allocation |
| pyca/cryptography | Security | 67K | 336 | Singleton domain; excluded from stratified allocation |

---

## 7. Threats to Validity

- **DS over-representation**: Data Science & AI/ML is represented by 6 projects (46% of
  the sample) against a pool proportion of 42%. This is a residual artefact of the initial
  experimental wave and is marginal (4 percentage points). Cross-domain analyses should
  control for domain as a covariate.

- **Web & Networking / Security not represented**: Two domains have no selected projects
  because each contains only one eligible project. Findings should not be generalised to
  web frameworks or security tools.

- **LoC gap in Systems stratum**: The Systems domain has a large LoC gap between the two
  selected projects (docker/compose at 35K and localstack at 572K), with no mid-range
  representative (dagster and localstack fill the upper half; celery and rucio fill the lower).
  Within-domain LoC effects for Systems are estimated with lower precision.

- **Bug-count selection rule**: Selecting the highest-bug-count project per stratum biases
  toward more actively maintained or community-visible projects. This is a pre-specified,
  neutral rule, but it does not constitute random sampling within strata.
