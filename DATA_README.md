# Cross-Project Bug Localisation (CPL) — Replication Package (Data)

Data artifact accompanying the paper *"What Works, What Doesn't, and Why: An Empirical
Investigation of Cross-Project Bug Localisation"* (A Eashaan Rao, Sridhar Chimalakonda,
Indian Institute of Technology Tirupati). The companion **code artifact** is archived separately
on Zenodo — see the Data Availability statement in the paper for both DOIs.

## What's included

| File | Rows/entries | Description |
|---|---|---|
| `bug_reports.parquet` | 71,452 | Raw bug reports across the 98-project dataset |
| `bug_reports_clean.parquet` | 57,214 | Filtered/deduplicated bug reports used in experiments |
| `project_metadata.parquet` | 98 | One row per benchmarked project |
| `embedding_dbs_sample/` | 6 projects, 12 files | Example embedding-database output (see below) |

**`bug_reports.parquet` / `bug_reports_clean.parquet` columns:**
`repo_name, bug_id, bug_report_text, ground_truth_files, creation_date, fix_date,
pre_fix_commit_sha, fix_commit_sha, language, source_dataset, bug_report_url, fix_url`

**`project_metadata.parquet` columns:**
`Dataset, repo_name, repo_link, language, total_unique_bug_reports, LoC, age_years,
median_bug_year, num_authors, num_commits, num_dependencies, polyglot_index, bug_density,
code_complexity, bug_report_verbosity, domain, project_size, project_age`

**`embedding_dbs_sample/`** — output of `build_embed_database_pipeline.py` (see the code
artifact) for 6 representative projects (`apache/commons-math`, `apache/hive`,
`clickhouse/clickhouse`, `fastify/fastify`, `spring-projects/spring-data-mongodb`,
`square/anvil`), one language/size pair per project. Two files per project:
- `<project>_blob_embeddings.pkl` — `dict[blob_sha (str) -> numpy.ndarray[float32, shape=(1536,)]]`,
  one entry per source-code chunk (chunked at 512 tokens, 50-token overlap, embedded with
  `BAAI/bge-code-v1`).
- `<project>_bug_metadata.pkl` — `dict[bug_id (str) -> dict]`, each value has keys `embedding`
  (`numpy.ndarray[float16, shape=(1536,)]`), `commit_sha`, and `ground_truth_files` (array of
  fixed file paths for that bug).

## What's excluded (and how to regenerate it)

The full dataset behind these experiments is **506 GB**, dominated by regeneratable intermediate
artifacts that exceed Zenodo's per-record limits and add no information beyond what the code
+ these files already capture:

| Excluded | Approx. size | Regenerate with |
|---|---|---|
| `data/repos/` — cloned source repositories for all 98 projects | 54 GB | Re-clone from each project's `repo_link` in `project_metadata.parquet` |
| `data/processed/embedding_dbs/` — full embedding DBs for all 98 projects | 16 GB | `python Scripts/build_embed_database_pipeline.py` (code artifact) against `data/repos/` |
| `data/processed/tranp_cnn_cache/` — tokenised/mmap'd TRANP-CNN training cache | 141 GB | `src/tranp_cnn/data_prep.py`, invoked via `Scripts/run_expts_ph1.py` |
| `data/processed/cooba_cache/` — per-pair AST embedding cache for COOBA | 206 GB | `src/cooba/pipeline.py`, invoked via `Scripts/run_expts_ph1.py` |
| `data/glove/` — third-party GloVe vectors | 2.3 GB | Download from the original GloVe release; not produced by this project |

All regeneration commands assume the code artifact's `requirements.txt` environment and expect
`data/repos/<language>/<owner>_<repo>/` to already contain the cloned project (paths are hardcoded
per project in each script — see the code artifact's README).

## Loading the data

```python
import pandas as pd, pickle

bugs = pd.read_parquet("bug_reports_clean.parquet")
projects = pd.read_parquet("project_metadata.parquet")

with open("embedding_dbs_sample/apache_commons-math_blob_embeddings.pkl", "rb") as f:
    blob_embeddings = pickle.load(f)   # {blob_sha: float32 vector}

with open("embedding_dbs_sample/apache_commons-math_bug_metadata.pkl", "rb") as f:
    bug_metadata = pickle.load(f)      # {bug_id: {"embedding": ..., "commit_sha": ..., "ground_truth_files": [...]}}
```
