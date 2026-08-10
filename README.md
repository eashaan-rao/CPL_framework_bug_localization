# Cross-Project Bug Localisation (CPL) — Replication Package (Code)

Code artifact accompanying the paper *"What Works, What Doesn't, and Why: An Empirical
Investigation of Cross-Project Bug Localisation"* (A Eashaan Rao, Sridhar Chimalakonda,
Indian Institute of Technology Tirupati).

This is a research framework that benchmarks three bug localisation models — **TRANP-CNN**,
**COOBA**, and **BLAZE** — across 98 open-source projects, evaluating how well each transfers
from a source project to a target project under four scenarios (within-project small/large,
cross-project cold-start, cross-project transfer).

The companion **data artifact** (parquet files with project/bug-report metadata and a sample
embedding database) is archived separately on Zenodo; see the Data Availability statement in the
paper for both DOIs.

## Repository layout

```
src/                          # Model implementations
├── cooba/                    # COOBA — GNN-based bug localisation (AST embeddings)
├── tranp_cnn/                # TRANP-CNN — CNN-based reranker (text + code)
└── blaze/                    # BLAZE — embedding-based reranker

Scripts/                      # Entry-point pipeline scripts
├── build_embed_database_pipeline.py   # Stage 1: build embedding databases
├── run_expts_ph1.py                   # Stage 3: main experiment runner (tranp_cnn / cooba)
├── run_faiss_experiment.py            # Stage 4: FAISS semantic-search recall baseline
├── domain_gap.py                      # Stage 5: pairwise domain-gap analysis
├── analysis/                          # Post-hoc analysis & reporting scripts
└── utils/                             # One-off analysis / utility scripts

benchmark_dataset_analysis/   # Notebooks documenting selection/curation of the 98-project dataset
docs/                          # Model guides (COOBA, TRANP-CNN)
results/                       # Experiment outputs referenced in the paper (CSVs, markdown reports, figures)
```

## Environment setup

Requires **Python 3.12**. A GPU (CUDA) is used automatically if available (`torch.cuda.is_available()`);
all scripts fall back to CPU otherwise.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the pipeline

All scripts use **absolute, hardcoded paths** at the top of each file as module-level constants —
edit these to point at your local checkout and data locations before running.

```bash
# Stage 1: build embedding databases (multi-process, uses BAAI/bge-code-v1)
python Scripts/build_embed_database_pipeline.py

# Stage 2: run TRANP-CNN / COOBA / BLAZE experiments (4 scenarios per project pair)
python Scripts/run_expts_ph1.py

# Stage 3: FAISS semantic-search recall baseline (standalone, no model)
python Scripts/run_faiss_experiment.py

# Stage 4: pairwise domain-gap analysis (requires embedding DBs from Stage 1)
python Scripts/domain_gap.py
```

Analysis/reporting, run after experiments complete (all read from
`results/phase1_experimental_results.csv`):

```bash
python Scripts/analysis/aggregate_tranp_cnn_results.py    # -> tranp_cnn_ph1_summary.csv
python Scripts/analysis/source_selection_analysis.py      # Kendall tau / Hit@1 source selection
python Scripts/analysis/generate_showcase_pairs.py        # per-pair figures -> results/showcase_pairs/
python Scripts/analysis/generate_presentation.py          # PPTX summary + speaker notes
```

The raw source-code embedding caches, tokenised training caches, and cloned project repositories
used to produce these results are not included in either Zenodo artifact (too large — see the
companion data artifact's `DATA_README.md` for exact regeneration commands and file sizes).

## Key results files (in `results/`)

| File | Description |
|---|---|
| `phase1_experimental_results.csv` | Authoritative results — 373 rows, columns: `model_name, source_project, target_project, scenario, top-1, top-5, top-10, MAP, MRR` |
| `tranp_cnn_ph1_summary.csv` | Per-pair aggregated summary including FAISS baseline MRR and project metadata |
| `all_project_domain_gaps.csv` | Columns: `project_A, project_B, domain_gap_accuracy` |
| `SUMMARY.md` | Findings summary and statistical methods |
| `DETAILED_ANALYSIS.md` | Deep-dive per-finding analysis |
| `CPL_DESIRABILITY.md` | Evidence for/against cross-project-localisation deployment |
| `SOURCE_PROJECT_SELECTION.md` | Source-selection framework and Kendall tau analysis |

## Dataset

98 open-source projects (41 Python, 34 Java, 9 Kotlin, 6 JS, 5 C++, 3 Go) drawn from 5 existing
benchmarks, selected with thresholds of ≥100 bugs (Python/Java) or ≥20 bugs (other languages).
Phase 1 results in this artifact cover 14 Python projects (87 source→target pairs, 345 runs). See
`benchmark_dataset_analysis/` for the selection/curation notebooks.

## Citation

If you use this code, please cite the paper (see `journal_draft/` for the manuscript, or the
Zenodo record's citation metadata).
