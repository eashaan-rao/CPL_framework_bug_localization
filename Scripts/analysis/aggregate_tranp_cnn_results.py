"""
aggregate_tranp_cnn_results.py
──────────────────────────────
Step 0 of the Phase 1 analysis pipeline.

Scans results/tranp_cnn_ph1_diagnostics/, loads every per-bug CSV,
computes standard IR metrics for both TRANP-CNN and the FAISS baseline,
adds FAISS-ceiling statistics, then joins project metadata + domain gap.

Output
──────
results/tranp_cnn_ph1_summary.csv
    One row per (source_project, target_project, scenario).

Run
───
    python Scripts/analysis/aggregate_tranp_cnn_results.py
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DIAG_DIR      = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_diagnostics"
OUTPUT_CSV    = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_summary.csv"
META_PATH     = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
DOMAIN_GAP    = "/home/cs21d002_eashaan/PhD/Objective1/results/all_project_domain_gaps.csv"

# Must match TOP_K_CANDIDATES in src/tranp_cnn/pipeline.py
TOP_K_CANDIDATES = 300

SCENARIOS = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
K_VALS    = [1, 5, 10]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_safe_name_map(meta_df):
    """Returns {safe_name: repo_name} where safe_name = repo_name.replace('/', '_')."""
    return {row['repo_name'].replace('/', '_'): row['repo_name']
            for _, row in meta_df.iterrows()}


def parse_filename(filename, safe_to_repo):
    """
    Parse '{source_safe}_{target_safe}_{scenario}_diagnostics.csv'.

    Project names contain underscores (e.g. google_jax, open-mmlab_mmdetection),
    so we try every split point and validate against the known project list.
    Returns (source_repo, target_repo, scenario) or (None, None, None).
    """
    name = filename.replace('_diagnostics.csv', '')
    for scenario in SCENARIOS:
        suffix = '_' + scenario
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            parts  = prefix.split('_')
            for i in range(1, len(parts)):
                src = '_'.join(parts[:i])
                tgt = '_'.join(parts[i:])
                if src in safe_to_repo and tgt in safe_to_repo:
                    return safe_to_repo[src], safe_to_repo[tgt], scenario
    return None, None, None


def compute_ir_metrics(ranks: np.ndarray, prefix: str) -> dict:
    """
    Compute Top-K recall, MRR for a 1-indexed rank array.
    prefix: 'model' or 'faiss'
    """
    n = len(ranks)
    metrics = {f'{prefix}_n_bugs': n}
    for k in K_VALS:
        metrics[f'{prefix}_top{k}'] = float((ranks <= k).sum() / n)
    metrics[f'{prefix}_mrr']         = float((1.0 / ranks).mean())
    metrics[f'{prefix}_mean_rank']   = float(ranks.mean())
    metrics[f'{prefix}_median_rank'] = float(np.median(ranks))
    return metrics


def compute_ceiling_metrics(df: pd.DataFrame) -> dict:
    """
    Three-way classification per bug using TOP_K_CANDIDATES:

        unreachable  : faiss_rank > TOP_K  → GT not in FAISS candidates,
                       TRANP-CNN structurally cannot find it.
        ceiling_miss : faiss_rank <= TOP_K AND model_rank > 10
                       → GT was a candidate but model ranked it outside top-10.
        model_hit    : faiss_rank <= TOP_K AND model_rank <= 10
                       → GT was a candidate AND model ranked it top-10.

    Also reports the faiss_only_hit rate (faiss_rank <= 10 regardless of model).
    """
    n = len(df)
    unreachable  = df['faiss_rank'] > TOP_K_CANDIDATES
    reachable    = ~unreachable
    model_top10  = df['model_rank'] <= 10
    faiss_top10  = df['faiss_rank'] <= 10

    return {
        'pct_unreachable'        : float(unreachable.sum() / n),
        'pct_reachable_model_hit': float((reachable & model_top10).sum() / n),
        'pct_reachable_model_miss': float((reachable & ~model_top10).sum() / n),
        'pct_faiss_top10'        : float(faiss_top10.sum() / n),
        'pct_model_improved'     : float((df['rank_displacement'] > 0).sum() / n),
        'pct_model_hurt'         : float((df['rank_displacement'] < 0).sum() / n),
        'mean_displacement'      : float(df['rank_displacement'].mean()),
        'median_displacement'    : float(df['rank_displacement'].median()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    meta      = pd.read_parquet(META_PATH)
    safe_map  = build_safe_name_map(meta)
    meta_idx  = meta.set_index('repo_name')

    # Load domain gap — symmetrise: store both (A,B) and (B,A)
    dg = pd.read_csv(DOMAIN_GAP)
    dg_dict = {}
    for _, row in dg.iterrows():
        dg_dict[(row['project_A'], row['project_B'])] = row['domain_gap_accuracy']
        dg_dict[(row['project_B'], row['project_A'])] = row['domain_gap_accuracy']

    files   = [f for f in os.listdir(DIAG_DIR) if f.endswith('_diagnostics.csv')]
    records = []
    skipped = []

    for fname in tqdm(sorted(files), desc='Aggregating'):
        src_repo, tgt_repo, scenario = parse_filename(fname, safe_map)
        if src_repo is None:
            skipped.append(fname)
            continue

        df = pd.read_csv(os.path.join(DIAG_DIR, fname))

        # Drop rows with missing ranks (shouldn't happen but be safe)
        df = df.dropna(subset=['model_rank', 'faiss_rank'])
        if df.empty:
            skipped.append(fname)
            continue

        model_ranks = df['model_rank'].astype(float).values
        faiss_ranks = df['faiss_rank'].astype(float).values

        row = {
            'source_project': src_repo,
            'target_project': tgt_repo,
            'scenario'      : scenario,
        }
        row.update(compute_ir_metrics(model_ranks, 'model'))
        row.update(compute_ir_metrics(faiss_ranks, 'faiss'))
        row.update(compute_ceiling_metrics(df))

        # Domain gap
        row['domain_gap'] = dg_dict.get((src_repo, tgt_repo), np.nan)

        # Source project metadata (prefixed src_)
        if src_repo in meta_idx.index:
            for col in ['language', 'LoC', 'total_unique_bug_reports', 'code_complexity',
                        'bug_report_verbosity', 'domain', 'project_size', 'polyglot_index',
                        'bug_density', 'age_years']:
                row[f'src_{col}'] = meta_idx.loc[src_repo, col]

        # Target project metadata (prefixed tgt_)
        if tgt_repo in meta_idx.index:
            for col in ['language', 'LoC', 'total_unique_bug_reports', 'code_complexity',
                        'bug_report_verbosity', 'domain', 'project_size', 'polyglot_index',
                        'bug_density', 'age_years']:
                row[f'tgt_{col}'] = meta_idx.loc[tgt_repo, col]

        # Same domain flag
        row['same_domain'] = (
            row.get('src_domain') == row.get('tgt_domain')
            and row.get('src_domain') is not None
        )

        records.append(row)

    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved {len(summary)} rows → {OUTPUT_CSV}")

    if skipped:
        print(f"⚠ Skipped {len(skipped)} files (unparseable): {skipped[:5]}")

    # Quick sanity print
    print("\nScenario counts:")
    print(summary['scenario'].value_counts().to_string())
    print("\nMean model MRR per scenario:")
    print(summary.groupby('scenario')['model_mrr'].mean().sort_values(ascending=False).round(4).to_string())


if __name__ == '__main__':
    main()
