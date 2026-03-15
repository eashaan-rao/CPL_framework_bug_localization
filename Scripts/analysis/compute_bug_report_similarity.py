"""
compute_bug_report_similarity.py
─────────────────────────────────
For every pair of projects that appear together in the diagnostics results,
computes the mean cosine similarity between their bug report embeddings.

Uses the precomputed *_bug_metadata32.pkl files (embedding field) so no
re-encoding is needed.

Output
──────
results/bug_report_similarity.csv
    Columns: project_A, project_B, n_bugs_A, n_bugs_B,
             mean_sim, std_sim, median_sim

Run
───
    python Scripts/analysis/compute_bug_report_similarity.py
"""

import os
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
DIAG_DIR    = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_diagnostics"
EMBED_DIR   = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
META_PATH   = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
OUTPUT_CSV  = "/home/cs21d002_eashaan/PhD/Objective1/results/bug_report_similarity.csv"

SCENARIOS = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_safe_name_map(meta_df):
    return {row['repo_name'].replace('/', '_'): row['repo_name']
            for _, row in meta_df.iterrows()}


def parse_filename(filename, safe_to_repo):
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
                    return safe_to_repo[src], safe_to_repo[tgt]
    return None, None


def load_bug_embeddings(repo_name: str) -> np.ndarray | None:
    """
    Loads *_bug_metadata32.pkl and returns an (N, D) float32 embedding matrix.
    Returns None if the file doesn't exist.
    """
    safe   = repo_name.replace('/', '_')
    fpath  = os.path.join(EMBED_DIR, f'{safe}_bug_metadata32.pkl')
    if not os.path.exists(fpath):
        print(f"  ⚠ Missing: {fpath}")
        return None
    with open(fpath, 'rb') as f:
        db = pickle.load(f)  # {bug_id: {'embedding': array, ...}}
    embeddings = np.array([v['embedding'] for v in db.values()], dtype=np.float32)
    return embeddings


def mean_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray,
                            sample_limit: int = 500) -> tuple[float, float, float]:
    """
    Computes mean / std / median cosine similarity between two embedding matrices.

    To avoid an O(N²) matrix when projects have thousands of bugs, we randomly
    sample up to `sample_limit` bugs from each side.
    """
    rng = np.random.default_rng(42)
    if len(emb_a) > sample_limit:
        emb_a = emb_a[rng.choice(len(emb_a), sample_limit, replace=False)]
    if len(emb_b) > sample_limit:
        emb_b = emb_b[rng.choice(len(emb_b), sample_limit, replace=False)]

    # Normalise rows
    def norm(m):
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return m / norms

    sim_matrix = norm(emb_a) @ norm(emb_b).T   # (Na, Nb)
    flat       = sim_matrix.flatten()
    return float(flat.mean()), float(flat.std()), float(np.median(flat))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    meta      = pd.read_parquet(META_PATH)
    safe_map  = build_safe_name_map(meta)

    # Collect all unique project pairs from diagnostics filenames
    diag_files = [f for f in os.listdir(DIAG_DIR) if f.endswith('_diagnostics.csv')]
    pairs = set()
    for fname in diag_files:
        src, tgt = parse_filename(fname, safe_map)
        if src and tgt:
            pairs.add(tuple(sorted([src, tgt])))  # canonical (alphabetical) order

    print(f"Found {len(pairs)} unique project pairs to process.")

    # Cache embeddings — load each project only once
    embed_cache: dict[str, np.ndarray | None] = {}
    all_projects = {p for pair in pairs for p in pair}
    print(f"Loading embeddings for {len(all_projects)} projects...")
    for proj in tqdm(sorted(all_projects), desc='Loading embeddings'):
        embed_cache[proj] = load_bug_embeddings(proj)

    records = []
    for proj_a, proj_b in tqdm(sorted(pairs), desc='Computing similarity'):
        emb_a = embed_cache.get(proj_a)
        emb_b = embed_cache.get(proj_b)
        if emb_a is None or emb_b is None:
            continue

        mean_s, std_s, med_s = mean_cosine_similarity(emb_a, emb_b)
        records.append({
            'project_A'  : proj_a,
            'project_B'  : proj_b,
            'n_bugs_A'   : len(emb_a),
            'n_bugs_B'   : len(emb_b),
            'mean_sim'   : round(mean_s, 6),
            'std_sim'    : round(std_s,  6),
            'median_sim' : round(med_s,  6),
        })

    out = pd.DataFrame(records)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved {len(out)} pairs → {OUTPUT_CSV}")
    print("\nTop-5 most similar pairs:")
    print(out.nlargest(5, 'mean_sim')[['project_A', 'project_B', 'mean_sim']].to_string(index=False))
    print("\nTop-5 least similar pairs:")
    print(out.nsmallest(5, 'mean_sim')[['project_A', 'project_B', 'mean_sim']].to_string(index=False))


if __name__ == '__main__':
    main()
