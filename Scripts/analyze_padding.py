import os
import pandas as pd
import git
from git import Repo, Blob
from git.util import hex_to_bin
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# CONFIGURATION

# Add the repository names for the projects that need to be analyze
JAVA_PROJECTS = [
    "elastic/elasticsearch",
    "apache/dolphinscheduler",
    "apache/dubbo",
    "liquibase/liquibase",
    "openrefine/openrefine",
    "seleniumhq/selenium"
]

PYTHON_PROJECTS = [
    "apache/airflow",
    "localstack/localstack",
    "posthog/posthog",
    "ansible/ansible",
    "langchain-ai/langchain",
    "odoo/odoo"
]

# Paths
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
TOKENIZER = "BAAI/bge-code-v1"
TOP_K_CANDIDATES = 300 

# HELPER FUNCTIONS
def get_project_path(repo_name, language):
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

def load_project_databases(project_name, meta_df):
    """
    Loads the bug and blob databases for a given project, with robust error handling.
    """
    # Filter for the project
    project_info = meta_df[meta_df['repo_name'] == project_name]

    # --- THIS IS THE FIX ---
    # Check if the project was found in the metadata
    if project_info.empty:
        # If not, raise a clear, informative error
        raise ValueError(
            f"Error: Project '{project_name}' was not found in the metadata file located at:\n"
            f"{PROJECTS_METADATA_PATH}\n"
            f"Please ensure the name in your JAVA_PROJECTS or PYTHON_PROJECTS list is spelled correctly."
        )
    language = project_info['language'].iloc[0]
    blob_db_path = os.path.join(BLOB_DB_DIR, project_name.replace('/', '_') + '_blob_embeddings32.pkl')
    bug_db_path = os.path.join(BUG_METADATA_DIR, project_name.replace('/', '_') + '_bug_metadata32.pkl')
    with open(blob_db_path, 'rb') as f:
        blob_db = pickle.load(f)
    with open(bug_db_path, 'rb') as f:
        bug_db = pickle.load(f)
    return bug_db, blob_db, language

def get_path_to_sha_map(repo, commit_sha):
    try:
        commit = repo.commit(commit_sha)
        return {blob.path: blob.hexsha for blob in commit.tree.traverse() if blob.type == 'blob'}
    except Exception:
        return {}
    
def create_labeled_samples(project_name, language, bug_metadata_db, blob_embedding_db, top_k):
    repo_path = os.path.join(REPO_BASE_PATH, language.lower(), project_name.replace('/', '_'))
    repo = git.Repo(repo_path)
    bugs_by_commit = {}
    for bug_id, meta in bug_metadata_db.items():
        sha = meta['commit_sha']
        if sha not in bugs_by_commit:
            bugs_by_commit[sha] = []
        bugs_by_commit[sha].append(bug_id)
    
    labeled_samples = []
    # Using dummy faiss for sample generation, as we only need the blob_shas
    for commit_sha, bug_ids in tqdm(bugs_by_commit.items(), desc=f"Generating samples for {project_name}"):
        path_to_sha_map = get_path_to_sha_map(repo, commit_sha)
        for bug_id in bug_ids:
            # For analysis, we can just consider all files in the snapshot as potential candidates
            for blob_sha in path_to_sha_map.values():
                labeled_samples.append((bug_id, blob_sha, 0, 'dummy'))
    return labeled_samples

# ANALYSIS AND REPORTING FUNCTIONS
def get_lengths_for_samples(raw_samples, repo, tokenizer):
    '''
    Analyzes file and line lengths for a given set of samples. Returns two lists: line_lengths and tokens_per_line
    '''
    line_lengths = []
    tokens_per_line = []

    unique_blobs = {sha for _, sha, _, _ in raw_samples}
    
    for blob_sha in tqdm(unique_blobs, desc="Analysing file lengths"):
        try: 
            source_content = Blob(repo, hex_to_bin(blob_sha)).data_stream.read().decode('utf-8', 'ignore')
            lines = source_content.splitlines()
            if not lines:
                continue

            line_lengths.append(len(lines))

            tokenized_lines = tokenizer(lines, truncation=True, max_length=1024)['input_ids']
            for line in tokenized_lines:
                tokens_per_line.append(len(line))
        except Exception:
            continue
    return line_lengths, tokens_per_line

def print_percentile_report(lenghts, title):
    '''
    Prints a formatted table of percentiles for a given list of lengths.
    '''
    if not lenghts:
        print(f"\n--{title}--")
        print(" No data to analyze.")
        return
    
    print(f"\n--{title}--")
    percentiles_to_check = range(50, 96) # 50 to 96 inclusive

    # Header
    print(f"{"Percentile":<12} | {'Value':<10}")
    print("-"*25)

    for p in percentiles_to_check:
        value = int(np.percentile(lenghts, p))
        print(f"{p:<12} | {value:<10}")
    
    print("-" * 25)
    print(f"{'Max Found': <12} | {np.max(lenghts):<10}")
    print(f"{'Average': <12} | {int(np.mean(lenghts)):<10}")
    print(f"{'Median': <12} | {int(np.median(lenghts)):<10}")

def save_report_to_csv(all_lengths, output_filename="padding_analysis_report.csv"):
    '''Calculates detailed percentiles and saves them to a single CSV file.'''
    print(f"\n Saving detailed percentile report to {output_filename}")
    results = []
    # User requested percentiles from 50 to 96 inclusive
    percentiles_to_check = range(50,97)

    # Loop through each category (java, python, global) and data types (lines, tokens)
    for category, length_data in all_lengths.items():
        for data_type, lengths in length_data.items():
            if not lengths:
                continue

            # Calculate all percentiles at once for efficiency
            percentile_values = np.percentile(lengths, percentiles_to_check)

            # Add a row for each percentile
            for i, p in enumerate(percentiles_to_check):
                results.append({
                    'category': category,
                    'type': data_type,
                    'percentile': p,
                    'value': int(percentile_values[i])
                })
    
    # Convert the list of results into a pandas DataFrame and save to CSV
    df_report = pd.DataFrame(results)
    output_path = os.path.join(RESULT_PATH, output_filename)
    df_report.to_csv(output_path, index=False)
    print(f"Report saved sucessfully to {output_path}")

def main():
    '''
    Main orchestrator to run the global analysis.
    '''
    print("Initializing tokenizer..")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    print("Loading project metadata...")
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)

    all_projects = [
        (name, 'java') for name in JAVA_PROJECTS
    ] + [
        (name, 'python') for name in PYTHON_PROJECTS
    ]

    # Dictionaries to hold all collected lengths
    all_lengths = {
        'java': {'lines':[], 'tokens':[]},
        'python': {'lines':[], 'tokens':[]},
        'global': {'lines':[], 'tokens':[]}
    }

    for project_name, lang_key in all_projects:
        print(f"\n{'='*20} Processing Project: {project_name} ({lang_key.upper()}) {'='*20}")

        # 1. Load data for the project
        bug_db, blob_db, language = load_project_databases(project_name, df_meta)

        # 2. Generate candidate samples
        raw_samples = create_labeled_samples(project_name, language, bug_db, blob_db, TOP_K_CANDIDATES)
        if not raw_samples:
            print(f"No sample generated for {project_name}. SKipping..")
            continue

        # 3. Analyze the samples
        repo = git.Repo(get_project_path(project_name, language))
        line_lengths, tokens_per_line = get_lengths_for_samples(raw_samples, repo, tokenizer)

        # 4. Store the results
        all_lengths[lang_key]['lines'].extend(line_lengths)
        all_lengths[lang_key]['tokens'].extend(tokens_per_line)
        all_lengths['global']['lines'].extend(line_lengths)
        all_lengths['global']['lines'].extend(tokens_per_line)
        print(f"Finished processing {project_name}. Found {len(line_lengths)} unique files.")
    
    # 5. Print Final Reports
    print(f"\n\n{'='*30} FINAL ANALYSIS RESULTS {'='*30}")
    
    save_report_to_csv(all_lengths)

if __name__ == '__main__':
    main()