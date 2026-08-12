"""
compare_faiss_bm25.py
──────────────────────
Compares the FAISS/BGE dense-retrieval baseline against the BM25 lexical
baseline (both computed on the same 13 projects, same bug snapshots, same
K_VALUES) to justify the paper's choice of dense retrieval as the shared
candidate-generation backbone (journal_draft/main.tex Section 3.5 / 5.4).

Inputs
──────
results/faiss_recall_results/semantic_search_SAMPLE_recall_results.csv
results/faiss_recall_results/semantic_search_{repo}_results.csv (per project)
results/bm25_recall_results/bm25_SAMPLE_recall_results.csv
results/bm25_recall_results/bm25_{repo}_results.csv (per project)

Outputs
───────
results/faiss_vs_bm25_comparison.csv       aggregate Recall@K, both methods + delta
results/faiss_vs_bm25_per_project.csv      per-project Recall@300, both methods + delta
results/images/faiss_vs_bm25_recall_vs_k.png

Run
───
    python Scripts/analysis/compare_faiss_bm25.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
FAISS_DIR = os.path.join(RESULT_PATH, "faiss_recall_results")
BM25_DIR = os.path.join(RESULT_PATH, "bm25_recall_results")

REPOS = [
    'jupyterlab/jupyterlab', 'lightning-ai/lightning', 'prefecthq/prefect',
    'pydata/xarray', 'numpy/numpy', 'scikit-learn/scikit-learn',
    'ipython/ipython', 'mesonbuild/meson', 'ansible/ansible',
    'docker/compose', 'localstack/localstack', 'wagtail/wagtail', 'qiskit/qiskit',
]

# Representative subset for the paper's comparison table.
TABLE_K_VALUES = [30, 100, 300, 700]


def load_aggregate(path, method_label):
    df = pd.read_csv(path)
    df = df.rename(columns={'Recall': method_label})
    return df[['K', method_label]]


def compare_aggregate():
    faiss_path = os.path.join(FAISS_DIR, 'semantic_search_SAMPLE_recall_results.csv')
    bm25_path = os.path.join(BM25_DIR, 'bm25_SAMPLE_recall_results.csv')
    if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
        raise FileNotFoundError(
            f"Missing aggregate recall CSV(s). Expected both:\n  {faiss_path}\n  {bm25_path}"
        )

    df_faiss = load_aggregate(faiss_path, 'FAISS_Recall')
    df_bm25 = load_aggregate(bm25_path, 'BM25_Recall')

    df = pd.merge(df_faiss, df_bm25, on='K', how='outer').sort_values('K')
    df['Delta'] = df['FAISS_Recall'] - df['BM25_Recall']

    out_path = os.path.join(RESULT_PATH, 'faiss_vs_bm25_comparison.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved aggregate comparison to {out_path}")
    print(df.to_string(index=False))

    print("\nCompact table for the paper (K in", TABLE_K_VALUES, "):")
    print(df[df['K'].isin(TABLE_K_VALUES)].to_string(index=False))

    return df


def compare_per_project():
    rows = []
    for repo in REPOS:
        repo_slug = repo.replace('/', '_')
        faiss_path = os.path.join(FAISS_DIR, f'semantic_search_{repo_slug}_results.csv')
        bm25_path = os.path.join(BM25_DIR, f'bm25_{repo_slug}_results.csv')
        if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
            print(f"  Skipping {repo}: missing per-project result file(s).")
            continue
        df_faiss = pd.read_csv(faiss_path)
        df_bm25 = pd.read_csv(bm25_path)
        faiss_r300 = df_faiss['Recall@300'].iloc[0]
        bm25_r300 = df_bm25['Recall@300'].iloc[0]
        rows.append({
            'repo_name': repo,
            'FAISS_Recall@300': faiss_r300,
            'BM25_Recall@300': bm25_r300,
            'Delta': faiss_r300 - bm25_r300,
        })

    if not rows:
        print("No per-project result pairs found; skipping per-project comparison.")
        return None

    df = pd.DataFrame(rows).sort_values('repo_name')
    out_path = os.path.join(RESULT_PATH, 'faiss_vs_bm25_per_project.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-project comparison to {out_path}")
    print(df.to_string(index=False))
    return df


def plot_comparison(df):
    os.makedirs(os.path.join(RESULT_PATH, 'images'), exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['K'], df['FAISS_Recall'], marker='o', linestyle='-', label='FAISS / BGE (dense)')
    ax.plot(df['K'], df['BM25_Recall'], marker='s', linestyle='--', label='BM25 (lexical)')
    ax.set_title('Retrieval Recall vs. Candidate Pool Size (K): FAISS/BGE vs. BM25', fontsize=15)
    ax.set_xlabel('Number of Candidates (K)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(RESULT_PATH, 'images', 'faiss_vs_bm25_recall_vs_k.png')
    fig.savefig(plot_path, dpi=300)
    print(f"\nSaved comparison plot to {plot_path}")


if __name__ == '__main__':
    df_agg = compare_aggregate()
    compare_per_project()
    plot_comparison(df_agg)
