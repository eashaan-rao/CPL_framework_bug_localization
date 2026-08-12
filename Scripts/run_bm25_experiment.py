"""
BM25 (lexical IR) retrieval baseline, mirroring run_faiss_experiment.py exactly
(same projects, same K_VALUES, same per-bug commit-snapshot walk, same
ground-truth matching and Recall@K bookkeeping) so its output is directly
comparable to the FAISS/BGE dense-retrieval results. Built to answer the
"why was FAISS/dense retrieval chosen over classical lexical IR" question
for the journal paper (Section 3.5 / Section 5.4).
"""
import os
import re
import pickle

import pandas as pd
import git
from tqdm import tqdm
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi

# Configuration
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"

LANGUAGE_EXTENSIONS = {
    'c++': ['.ac', '.alpine', '.am', '.arm64', '.armv7', '.arrow', '.asar', '.avro', '.bash', '.bat', '.bazel', '.bin', '.bzl', '.c', '.cc', '.cfg', '.clj', '.cmake', '.conf', '.config', '.cpp', '.cs', '.csproj', '.css', '.dat', '.data', '.def', '.dict', '.expect', '.expected', '.fragment', '.gd', '.gemspec', '.glsl', '.gn', '.gni', '.go', '.grdp', '.gyp', '.gypi', '.h', '.hpp', '.html', '.ico', '.in', '.inc', '.include', '.init', '.install', '.j2', '.java', '.js', '.json', '.json5', '.kt', '.lib', '.limits', '.lock', '.m', '.m4', '.make', '.manifest', '.md', '.mjs', '.mk', '.mm', '.mp4', '.npy', '.out', '.patch', '.pb', '.pbxproj', '.php', '.plist', '.png', '.postinst', '.postinstall', '.preinst', '.prerm', '.props', '.proto', '.ps1', '.py', '.python', '.queries', '.rb', '.reference', '.scm', '.scss', '.sh', '.sigs', '.so', '.sql', '.supp', '.svg', '.targets', '.templates', '.ts', '.txt', '.ubuntu', '.ui', '.vcxproj', '.xib', '.xml', '.yaml', '.yml', '.zst'],
    'go': ['.bash', '.bats', '.conf', '.css', '.cue', '.dockerfile', '.ex', '.exs', '.fragment', '.go', '.graphqls', '.gtpl', '.hbs', '.ignore', '.js', '.json', '.key', '.lock', '.md', '.mdx', '.mjs', '.mod', '.mts', '.nightly', '.pem', '.png', '.py', '.rs', '.scss', '.sh', '.sum', '.tmpl', '.toml', '.ts', '.txt', '.win64', '.xml', '.yaml', '.yml'],
    'java': ['.asciidoc', '.bat', '.bazel', '.bz2', '.bzl', '.cer', '.clusterfilter', '.cmd', '.commandstep', '.conf', '.config', '.cpp', '.crt', '.crx', '.cs', '.css', '.csv', '.csv-spec', '.db', '.desktop', '.enc', '.executor', '.factories', '.filter', '.gemspec', '.gradle', '.groovy', '.gz', '.h', '.html', '.importorder', '.ini', '.install4j', '.iss', '.jar', '.java', '.jj', '.jpg', '.js', '.json', '.key', '.kt', '.less', '.liquibasedatatype', '.lock', '.md', '.meshenvlistenerfactory', '.mustache', '.nuspec', '.ods', '.plist', '.png', '.policy', '.properties', '.protocol', '.ps1', '.py', '.rb', '.rs', '.rst', '.scss', '.sh', '.sha1', '.snapshotgenerator', '.sql', '.sql-spec', '.sqlgenerator', '.st', '.svg', '.targets', '.toml', '.tpl', '.ts', '.tsv', '.tsx', '.txt', '.typebuilder', '.validation', '.vm', '.vt', '.vue', '.xls', '.xml', '.xsd', '.yaml', '.yml', '.zip'],
    'javascript': ['.avif', '.cjs', '.coffee', '.css', '.cts', '.dockerfile', '.example', '.graphql', '.graphqls', '.html', '.ico', '.jpg', '.js', '.jsm', '.json', '.jsx', '.link', '.lock', '.map', '.md', '.mdx', '.mjs', '.mts', '.opts', '.pdf', '.png', '.properties', '.rb', '.rs', '.scss', '.sh', '.snap', '.sqlite', '.stderr', '.svelte', '.svg', '.toml', '.ts', '.tsx', '.ttf', '.txt', '.wasm', '.webp', '.xml', '.yaml', '.yml'],
    'kotlin': ['.java', '.kt', '.py'],
    'python': ['.0', '.1', '.acl', '.ambr', '.base', '.bat', '.build', '.c', '.cfg', '.ci', '.cmd', '.cnf', '.conf', '.cs', '.csproj', '.css', '.css_t', '.csv', '.db', '.dockerfile', '.example', '.expected', '.g4', '.gif', '.h', '.html', '.in', '.ini', '.interp', '.inv', '.inventory', '.ipynb', '.j2', '.jar', '.java', '.jinja2', '.js', '.json', '.json5', '.jsx', '.kubernetes-helm-yaml', '.less', '.lock', '.manifest', '.md', '.mdx', '.mkv', '.mp4', '.nodejs14x', '.php', '.pip', '.png', '.pot', '.ps1', '.psm1', '.pxd', '.pxi', '.py', '.pyi', '.pyx', '.r', '.rdb', '.rst', '.run', '.scss', '.sh', '.sha256', '.sln', '.stderr', '.stdout', '.svg', '.template', '.tf', '.tgz', '.toml', '.tpl', '.ts', '.tsx', '.txt', '.xml', '.yaml', '.yml', '.zip'],
}

K_VALUES = [30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700]

# Max bytes read per blob; skip anything larger (binary/huge generated files
# that would dominate tokenization cost with no retrieval benefit).
MAX_BLOB_BYTES = 2_000_000

# Paper set: same 13 Python projects across 4 domains used by run_faiss_experiment.py,
# so BM25 and FAISS recall are directly comparable.
experiments = [
    {'repo_name': 'jupyterlab/jupyterlab'},
    {'repo_name': 'lightning-ai/lightning'},
    {'repo_name': 'prefecthq/prefect'},
    {'repo_name': 'pydata/xarray'},
    {'repo_name': 'numpy/numpy'},
    {'repo_name': 'scikit-learn/scikit-learn'},
    {'repo_name': 'ipython/ipython'},
    {'repo_name': 'mesonbuild/meson'},
    {'repo_name': 'ansible/ansible'},
    {'repo_name': 'docker/compose'},
    {'repo_name': 'localstack/localstack'},
    {'repo_name': 'wagtail/wagtail'},
    {'repo_name': 'qiskit/qiskit'},
]

_TOKEN_SPLIT_RE = re.compile(r'[^a-zA-Z0-9]+')
_CAMEL_SPLIT_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')


def tokenize(text: str) -> list:
    """
    Code-aware tokenizer: splits on non-alphanumeric boundaries (handles
    snake_case, punctuation, paths) and on camelCase boundaries, then
    lowercases. This gives BM25 a fair shot at code-identifier vocabulary
    rather than treating it as English prose.
    """
    if not text:
        return []
    tokens = []
    for raw in _TOKEN_SPLIT_RE.split(text):
        if not raw:
            continue
        for part in _CAMEL_SPLIT_RE.split(raw):
            if part:
                tokens.append(part.lower())
    return tokens


def get_project_path(repo_name, language):
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))


def run_bm25_experiment():
    print("---Starting BM25 Experiment----")
    os.makedirs(os.path.join(RESULT_PATH, 'bm25_recall_results'), exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, 'images'), exist_ok=True)

    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    all_extensions = set(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts)
    all_project_results = []

    for experiment in tqdm(experiments, desc="Experiments"):
        repo_name = experiment['repo_name']
        project_row = df_meta[df_meta['repo_name'] == repo_name].iloc[0]
        language = project_row['language']

        print(f"\n--- Processing project: {repo_name} ---")

        recall_path = os.path.join(RESULT_PATH, 'bm25_recall_results',
                                    f'bm25_{repo_name.replace("/", "_")}_results.csv')
        if os.path.exists(recall_path):
            print("  Already computed. Skipping.")
            continue

        repo_path = get_project_path(repo_name, language)
        if not os.path.exists(repo_path):
            print(f"ERROR: Repo path not found for {repo_name}. Skipping.")
            continue

        repo = git.Repo(repo_path)
        project_bugs = df_bugs[df_bugs['repo_name'] == repo_name].copy()
        if project_bugs.empty:
            print(f"ERROR: No bug reports found for {repo_name}. Skipping.")
            continue

        total_bugs_processed = 0
        commit_failures = 0
        snapshots_without_indexable_files = 0
        hits_per_k = {k: 0 for k in K_VALUES}

        for _, bug in tqdm(project_bugs.iterrows(), total=len(project_bugs),
                            desc=f"Bugs in {repo_name}", leave=False):
            total_bugs_processed += 1
            snapshot_sha = bug['pre_fix_commit_sha']
            if not snapshot_sha:
                commit_failures += 1
                continue

            try:
                commit = repo.commit(snapshot_sha)
            except Exception:
                commit_failures += 1
                continue

            # Walk the snapshot, tokenize every indexable source blob.
            corpus_tokens = []
            corpus_paths = []
            for blob in commit.tree.traverse():
                if blob.type != 'blob' or not any(blob.name.endswith(ext) for ext in all_extensions):
                    continue
                if blob.size > MAX_BLOB_BYTES:
                    continue
                try:
                    raw = blob.data_stream.read().decode('utf-8', errors='ignore')
                except Exception:
                    continue
                tokens = tokenize(raw)
                if not tokens:
                    continue
                corpus_tokens.append(tokens)
                corpus_paths.append(blob.path)

            if not corpus_tokens:
                snapshots_without_indexable_files += 1
                continue

            bm25 = BM25Okapi(corpus_tokens)
            query_tokens = tokenize(bug['bug_report_text'])
            scores = bm25.get_scores(query_tokens)

            # Rank paths by descending BM25 score, keep top max(K_VALUES).
            top_n = min(max(K_VALUES), len(corpus_paths))
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
            candidates_relative = [corpus_paths[i] for i in ranked_indices]

            ground_truth = set(bug['ground_truth_files'])

            found_rank = -1
            for i, candidate in enumerate(candidates_relative):
                if any(candidate.endswith(gt_file) for gt_file in ground_truth):
                    found_rank = i + 1
                    break

            if found_rank != -1:
                for k in K_VALUES:
                    if found_rank <= k:
                        hits_per_k[k] += 1

        print(f"\n--- BM25 Search for {repo_name} Complete ----")
        print(f"Total bug reports processed: {total_bugs_processed}")
        print(f"Commit failures (invalid SHA): {commit_failures}")
        print(f"Snapshots without indexable files: {snapshots_without_indexable_files}")

        valid_bugs = total_bugs_processed - commit_failures - snapshots_without_indexable_files
        if valid_bugs > 0:
            all_project_results.append({
                'repo_name': repo_name,
                'valid_bugs': valid_bugs,
                'hits_per_k': hits_per_k,
            })

        final_recall = {}
        print("\n-- Recall @ K Results --")
        for k, hits in hits_per_k.items():
            recall = hits / valid_bugs if valid_bugs > 0 else 0
            final_recall[f"Recall@{k}"] = recall
            print(f"Recall@{k}: {recall:.4f} ({hits}/{valid_bugs})")

        df_final_recall = pd.DataFrame([final_recall])
        df_final_recall.to_csv(recall_path, index=False)
        print(f"\nFinal recall results saved to {recall_path}")

    print("\nOverall summary across all sample projects")
    if not all_project_results:
        print("No results to aggregate.")
        return

    total_valid_bugs = sum(res['valid_bugs'] for res in all_project_results)
    total_hits_per_k = {k: 0 for k in K_VALUES}
    for res in all_project_results:
        for k in K_VALUES:
            total_hits_per_k[k] += res['hits_per_k'][k]

    print(f"Total valid bug reports across all projects: {total_valid_bugs}")
    print("\nOverall Recall @ K Results")

    overall_recall = {}
    for k, total_hits in total_hits_per_k.items():
        recall = total_hits / total_valid_bugs if total_valid_bugs > 0 else 0
        overall_recall[k] = recall
        print(f"Recall@{k}: {recall:.4f} ({total_hits}/{total_valid_bugs})")

    df_overall_recall = pd.DataFrame({
        'K': list(overall_recall.keys()),
        'Recall': list(overall_recall.values()),
    })
    recall_path = os.path.join(RESULT_PATH, 'bm25_recall_results', 'bm25_SAMPLE_recall_results.csv')
    df_overall_recall.to_csv(recall_path, index=False)
    print(f"\nOverall recall results saved to {recall_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_overall_recall['K'], df_overall_recall['Recall'], marker='o', linestyle='-')
    ax.set_title('BM25 Overall Average Recall vs. Candidate Pool Size (K)', fontsize=16)
    ax.set_xlabel('Number of Candidates (K)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.grid(True)
    for i, row in df_overall_recall.iterrows():
        ax.text(row['K'], row['Recall'], f" {row['Recall']:.1%}", verticalalignment='bottom')
    plt.tight_layout()
    plot_path = os.path.join(RESULT_PATH, 'images', 'bm25_recall_vs_k.png')
    fig.savefig(plot_path, dpi=300)
    print(f"Saved final plot to: {plot_path}")


if __name__ == '__main__':
    run_bm25_experiment()
