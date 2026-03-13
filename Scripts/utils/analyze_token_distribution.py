import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import git
from git import Blob
from git.util import hex_to_bin
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"

# BGE tokenizer for accurate token count
BGE_MODEL_NAME = "BAAI/bge-code-v1"

def analyze_token_distribution(projects_to_analyze=None):
    '''
    Analyze token distribution across code files in repositories.
    ''' 
    # Load project metadata
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)

    if projects_to_analyze is None:
        projects_to_analyze = df_meta['repo_name'].unique()[:10] # Analyze first 10 projects

    # Initialize tokenizer
    print("Loading BGE tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)

    # Storage for statistics
    all_token_counts = []
    token_counts_by_language = defaultdict(list)
    token_counts_by_project = defaultdict(list)

    for project_name in projects_to_analyze:
        print(f"\n Analyzing project: {project_name}")

        # Get project language
        language = df_meta[df_meta['repo_name'] == project_name]['language'].iloc[0]

        # Load blob database to get all file SHAs
        blob_db_path = os.path.join(BLOB_DB_DIR, project_name.replace('/', '_') + '_blob_embeddings32.pkl')
        if not os.path.exists(blob_db_path):
            print(f" Skipping {project_name} - blob database not found")
            continue

        with open(blob_db_path, 'rb') as f:
            blob_db = pickle.load(f)

        # Get repository
        repo_path = os.path.join(REPO_BASE_PATH, language.lower(), project_name.replace('/', '_'))
        if not os.path.exists(repo_path):
            print(f" Skipping {project_name} - repository not found")
            continue

        repo = git.Repo(repo_path)

        # Sample files analysis (analyzing all would be too slow)
        blob_shas = list(blob_db.keys())
        sample_size = len(blob_shas) # Sample up to 1000 files per project
        sampled_shas = np.random.choice(blob_shas, size=sample_size, replace=False)

        print(f" Analyzing {sample_size} files from {project_name} ({language})")

        for blob_sha in tqdm(sampled_shas, desc=f" Processing {project_name}"):
            try:
                # Get file content
                blob = Blob(repo, hex_to_bin(blob_sha))
                content = blob.data_stream.read().decode('utf-8', 'ignore')

                # Skip empty files
                if not content.strip():
                    continue

                # Tokenize to count tokens
                tokens = tokenizer.tokenize(content)
                token_count = len(tokens)

                # Store statistics
                all_token_counts.append(token_count)
                token_counts_by_language[language].append(token_count)
                token_counts_by_project[project_name].append(token_count)
            
            except Exception as e:
                continue
    return all_token_counts, token_counts_by_language, token_counts_by_project

def print_statistics(token_counts, label="Overall"):
    '''
    Print percentile statistics for token counts.
    '''
    if not token_counts:
        print(f"No data for {label}")
        return

    counts = np.array(token_counts)

    print(f"\n{'='*50}")
    print(f"Token Distribution Statistics - {label}")
    print(f"{'='*50}")
    print(f"Total files analyzed: {len(counts):,}")
    print(f"Mean tokens: {np.mean(counts):.1f}")
    print(f"Median tokens: {np.median(counts):.1f}")
    print(f"Std deviation: {np.std(counts):.1f}")
    print(f"Min tokens: {np.min(counts)}")
    print(f"Max tokens: {np.max(counts):,}")
    
    print(f"\nPercentiles:")
    percentiles = [50, 75, 80, 85, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(counts, p)
        print(f" {p}th percentile: {value:,.0f} tokens")
    
    # Recommend MAX_CODE_TOKENS based on 90th percentile
    recommended = int(np.percentile(counts, 90))
    print(f"\n Recommended MAX_CODE_TOKENS: {recommended}")

    # Show impact of different thresholds
    print(f"\nCoverage with different MAX_CODE_TOKENS:")
    thresholds = [1024, 2048, 4096, 8192, 16384]
    for threshold in thresholds:
        coverage = (counts <= threshold).mean() * 100
        print(f"  {threshold:,} tokens: {coverage:.1f}% files fully covered")
    
    return counts

def plot_distribution(all_counts, language_counts, output_dir="/home/cs21d002_eashaan/PhD/Objective1/results"):
    """
    Create visualization of token distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.percentile(all_counts, 90), color='r', linestyle='--', label='90th percentile')
    plt.axvline(np.percentile(all_counts, 95), color='g', linestyle='--', label='95th percentile')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title('Overall Token Distribution')
    plt.legend()
    plt.xlim(0, min(20000, np.max(all_counts)))
    
    # By language
    plt.subplot(1, 2, 2)
    languages = list(language_counts.keys())
    for lang in languages:
        if language_counts[lang]:
            counts = np.array(language_counts[lang])
            # Plot CDF
            sorted_counts = np.sort(counts)
            p = np.arange(len(sorted_counts)) / len(sorted_counts)
            plt.plot(sorted_counts, p, label=f'{lang} (n={len(counts)})', alpha=0.7)
    
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF by Language')
    plt.legend()
    plt.xlim(0, 20000)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_distribution.png'), dpi=150)
    plt.show()
    
    print(f"\nPlot saved to {output_dir}/token_distribution.png")

def analyze_bug_report_tokens(projects_to_analyze=None):
    '''
    Analyze token distribution in bug reports (overall, per-project, per-language).
    '''
    print("\n Analyzing bug report token distribution....")

    # Load bug reports
    BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    # Filter to specific projects if provided
    if projects_to_analyze:
        df_bugs = df_bugs[df_bugs['repo_name'].isin(projects_to_analyze)]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)

    ## Storage
    bug_token_counts = []                     # overall
    token_counts_by_project = defaultdict(list)
    token_counts_by_language = defaultdict(list)

    # Iterate over bug reports
    for _, row in tqdm(df_bugs.iterrows(), total=len(df_bugs), desc="Tokenizing bug reports"):
        text = row['bug_report_text']
        project = row['repo_name']
        language = row['language']

        if pd.notna(text):
            tokens = tokenizer.tokenize(text)
            count = len(tokens)

            # Store statistics
            bug_token_counts.append(count)
            token_counts_by_project[project].append(count)
            token_counts_by_language[language].append(count)

    # --- Print overall statistics ---
    if bug_token_counts:
        print_statistics(bug_token_counts, "Bug Reports (Overall)")

    # --- Print per-project statistics ---
    for project, counts in token_counts_by_project.items():
        if counts:
            print_statistics(counts, f"Bug Reports - Project: {project}")

    # --- Print per-language statistics ---
    for lang, counts in token_counts_by_language.items():
        if counts:
            print_statistics(counts, f"Bug Reports - Language: {lang}")

    return bug_token_counts, token_counts_by_project, token_counts_by_language

def main():
    '''
    Main analysis function.
    '''
    print("COOBA Token Length Analysis")

    # Specify projects to analyze (or None for default)
    projects = [
        "scipy/scipy",
        "sympy/sympy",
        "matplotlib/matplotlib",
        "open-mmlab/mmdetection",
        "ray-project/ray",
        "scikit-learn/scikit-learn",
        "google/jax",
        "jupyterlab/jupyterlab",
        "lightning-ai/lightning",
        "prefecthq/prefect",
        "pydata/xarray"
    ]

    # Analyze code files
    print("\n Analyzing code file token distributions....")
    all_counts, language_counts, project_counts = analyze_token_distribution(projects)

    # Per-project statistics
    for project, counts in project_counts.items():
        if counts:
            print_statistics(counts, f"Project: {project}")

    if all_counts:
        # Print overall statistics
        overall_stats = print_statistics(all_counts, "Overall")

        # Print per-language statistics
        for lang, counts in language_counts.items():
            if counts:
                print_statistics(counts, f"Language: {lang}")

        # Create visualizations
        plot_distribution(all_counts, language_counts)

        # Final recommendations
        print("Final recommendations for Cooba")

        p90_overall = int(np.percentile(all_counts, 90))
        p95_overall = int(np.percentile(all_counts, 95))

        print(f"MAX_CODE_TOKENS (90th percentile): {p90_overall:,}")
        print(f"MAX_CODE_TOKENS (95th percentile): {p95_overall:,}")

        # Language-specific recommendations
        for lang, counts in language_counts.items():
            if counts:
                p90 = int(np.percentile(counts, 90))
                print(f' {lang}: {p90:,} tokens (90th percentile)')

    # Analyze bug reports
    # Analyze bug reports for the same projects
    bug_counts, bug_project_counts, bug_language_counts = analyze_bug_report_tokens(projects)

    print("Analysis Complete!")

if __name__ == "__main__":
    main()