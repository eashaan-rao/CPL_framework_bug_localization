"""
compute_code_similarity.py
──────────────────────────
Computes mean cosine similarity between project codebases using the
precomputed *_blob_embeddings32.pkl files (source code embeddings).

This complements domain_gap_accuracy (a classifier-based measure) with
a direct embedding-space distance between code corpora.

Output
──────
results/code_similarity.csv
    Columns: project_A, project_B, n_files_A, n_files_B,
             mean_code_sim, std_code_sim, median_code_sim

Run
───
    python Scripts/analysis/compute_code_similarity.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DIAG_DIR   = "/home/cs21d002_eashaan/PhD/Objective1/results/tranp_cnn_ph1_diagnostics"
EMBED_DIR  = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
META_PATH  = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
OUTPUT_CSV = "/home/cs21d002_eashaan/PhD/Objective1/results/code_similarity.csv"

SCENARIOS = ['WP-small', 'WP-large', 'CP-cold-start', 'CP-transfer']
# Sample at most this many files per project to keep computation tractable
FILE_SAMPLE_LIMIT = 1000


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


def load_blob_embeddings(repo_name: str) -> np.ndarray | None:
    """
    Loads *_blob_embeddings32.pkl and returns an (N, D) float32 matrix.
    blob_embedding_db format: {file_path: embedding_array}
    """
    safe  = repo_name.replace('/', '_')
    fpath = os.path.join(EMBED_DIR, f'{safe}_blob_embeddings32.pkl')
    if not os.path.exists(fpath):
        print(f"  ⚠ Missing: {fpath}")
        return None
    with open(fpath, 'rb') as f:
        db = pickle.load(f)   # {file_path: np.ndarray}
    embs = np.array(list(db.values()), dtype=np.float32)
    return embs


def mean_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray,
                            sample_limit: int = FILE_SAMPLE_LIMIT
                            ) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    if len(emb_a) > sample_limit:
        emb_a = emb_a[rng.choice(len(emb_a), sample_limit, replace=False)]
    if len(emb_b) > sample_limit:
        emb_b = emb_b[rng.choice(len(emb_b), sample_limit, replace=False)]

    def norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    sim   = norm(emb_a) @ norm(emb_b).T
    flat  = sim.flatten()
    return float(flat.mean()), float(flat.std()), float(np.median(flat))


def main():
    meta     = pd.read_parquet(META_PATH)
    safe_map = build_safe_name_map(meta)

    diag_files = [f for f in os.listdir(DIAG_DIR) if f.endswith('_diagnostics.csv')]
    pairs = set()
    for fname in diag_files:
        src, tgt = parse_filename(fname, safe_map)
        if src and tgt:
            pairs.add(tuple(sorted([src, tgt])))

    print(f"Found {len(pairs)} unique project pairs.")

    embed_cache: dict[str, np.ndarray | None] = {}
    all_projects = {p for pair in pairs for p in pair}
    print(f"Loading blob embeddings for {len(all_projects)} projects...")
    for proj in tqdm(sorted(all_projects), desc='Loading'):
        embed_cache[proj] = load_blob_embeddings(proj)

    records = []
    for proj_a, proj_b in tqdm(sorted(pairs), desc='Computing similarity'):
        emb_a = embed_cache.get(proj_a)
        emb_b = embed_cache.get(proj_b)
        if emb_a is None or emb_b is None:
            continue

        mean_s, std_s, med_s = mean_cosine_similarity(emb_a, emb_b)
        records.append({
            'project_A'       : proj_a,
            'project_B'       : proj_b,
            'n_files_A'       : len(emb_a),
            'n_files_B'       : len(emb_b),
            'mean_code_sim'   : round(mean_s, 6),
            'std_code_sim'    : round(std_s,  6),
            'median_code_sim' : round(med_s,  6),
        })

    out = pd.DataFrame(records)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved {len(out)} pairs → {OUTPUT_CSV}")
    print("\nTop-5 most similar codebases:")
    print(out.nlargest(5, 'mean_code_sim')[['project_A', 'project_B', 'mean_code_sim']].to_string(index=False))


if __name__ == '__main__':
    main()
