import os
import time
import pandas as pd
import git
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROJECTS_METADATA_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/data/processed/project_metadata.parquet'
BUG_REPORTS_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/data/processed/bug_reports.parquet'
REPO_BASE_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/data/repos' # Base folder where all repos are cloned
RESULT_PATH = '/home/user/CS21D002_A_Eashaan_Rao/Research/PhD/Objective1/results' # base folder where all results will be stored.
NUM_SAMPLE_PROJECTS = 6

LANGUAGE_EXTENSIONS = {
    'c++': ['.c', '.cc', '.cmake', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.in', '.json', '.make', '.py', '.sh', '.xml'],
    'go': ['.go', '.json', '.proto', '.sh', '.yaml', '.yml'],
    'java': ['.gradle', '.groovy', '.java', '.json', '.properties', '.xml', '.yml', '.yaml'],
    'javascript': ['.css', '.html', '.js', '.json', '.jsx', '.mjs', '.scss', '.sh', '.ts', '.tsx', '.yaml', '.yml'],
    'kotlin': ['.gradle', '.json', '.kt', '.kts', '.properties', '.xml', '.yaml', '.yml'],
    'python': ['.bash', '.cfg', '.in', '.ini', '.json', '.py', '.sh', '.toml', '.yaml', '.yml']
}

BUCKET_STRATEGIES = {
    'annual': 'Y',
    'semi-annual': '2Q',
    'quartely': 'Q'
}
K_VALUES = [30, 50, 75, 100, 125, 150, 200]

#####################################################
# PART 1: ElasticSearch and Git Management
#####################################################

class ElasticSearchManager:
    '''
    Manages Elasticseach indexing, querying, and stats.
    '''
    def __init__(self, host='localhost', port=9200):
        self.client = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}], timeout=30)
        if not self.client.ping():
            raise ConnectionError("Could not connect to Elasticsearch.")
        
    def create_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        mapping = {
            "properties": {
                "file_path": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"}
            }
        }
        self.client.indices.create(index=index_name, mappings=mapping)
    
    def index_source_files(self, index_name: str, project_dir: str):
        # Flatten all extensions from LANGUAGE_EXTENSIONS
        all_extensions = set(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts)
        actions = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                # A simple filter for common source files, can be expanded via LANGUAGE_EXTENSIONS
                if any(file.endswith(ext) for ext in all_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        actions.append({
                            "_index": index_name,
                            "_source": {"file_path": file_path, "content": content}
                        })
                    except Exception:
                        continue
        if actions:
            bulk(self.client, actions)
        
    def get_candidate_files(self, index_name: str, bug_report_text: str, top_k: int):
        query = {"match": {"content": bug_report_text}}
        try:
            response = self.client.search(index=index_name, query=query, size=top_k)
            return [hit['_source']['file'] for hit in response['hits']['hits']]
        except Exception:
            return []
        
    def get_index_stats(self, index_name: str):
        stats = self.client.indices.stats(index=index_name, metric='store')
        size_gb = stats['_all']['primaries']['store']['size_in_bytes'] / (1024**3)
        return {"size_gb": size_gb}
    
    def delete_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

def get_project_path(repo_name, language):
    '''
    Constructs the local path to a cloned repository.
    '''
    # Assumes repo_name might be 'owner/repo' and we want to convert it to 'owner_repo'
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

def checkout_commit(repo_path, sha):
    '''
    Checks out a specific commit in a Git repository.
    '''
    try:
        repo = git.Repo(repo_path)
        repo.git.checkout(sha, f=True)
        return True
    except git.exc.GitCommandError as e:
        print(f"Error checking out {sha} in {repo_path}: {e}")
        return False

#####################################################
# PART 2: Experimental logic
#####################################################

def select_sample_projects(df_meta):
    '''
    Selects a stratified sample of projects.
    '''
    print("Selecting a stratified sample of projects....")
    # Create strata from size and age
    df_meta['size_cat'] = pd.qcut(df_meta['project_size'], q=[0, 0.33, 0.66, 1], labels=['small', 'medium', 'large'])
    df_meta['age_cat'] = pd.qcut(df_meta['project_age'], q=[0, 0.5, 1], labels=['new', 'mid-era'])
    df_meta['strata'] = df_meta['size_cat'].astype(str) + '_' + df_meta['age_cat'].astype(str)

    # Ensure we get one project from each stratum
    sample_df = df_meta.groupby('strata', group_keys=False).apply(lambda x: x.samples(1))
    print("Selected Projects:")
    print(sample_df[['repo_name', 'strata', 'language']])
    return sample_df

def group_bugs_into_buckets(df_bugs, strategy_freq, repo_path):
    '''
    Group bug reports into temporal buckets using commit_dates derived from their SHAs. 
    '''
    print(f"Extracting commit dates for {len(df_bugs)} bugs... (This may take a moment)")
    repo = git.Repo(repo_path)
    commit_dates = {}
    for sha in tqdm(df_bugs['pre_fix_commit_sha'].unique(), desc="Getting commit dates"):
        try:
            commit_dates[sha] = repo.commit(sha).committed_datetime
        except Exception:
            commit_dates[sha] = pd.NaT # Handles cases SHA might be invalid SHAs
    
    df_bugs['commit_date'] = df_bugs['pre_fix_commit_sha'].map(commit_dates)
    df_bugs.dropna(subset=['commit_date'], inplace=True) # Drop bugs invalid 

    df_bugs['bucket'] = df_bugs['commit_date'].dt.to_period(strategy_freq)
    
    # for each bucket, find the bug with the latest creation date to get the snapshot SHA
    latest_shas = df_bugs.loc[df_bugs.groupby('bucket')['commit_date'].idxmax()]
    bucket_to_sha = dict(zip(latest_shas['bucket'], latest_shas['pre_fix_commit_sha']))

    # Group all bug reports by their bucket
    grouped = df_bugs.groupby('bucket')
    return {bucket: (group.to_dict('records'), bucket_to_sha.get(bucket)) for bucket, group in grouped}

def generate_and_save_plots(summary_df):
    '''
    Generates and saves plots to help analyze the results.
    '''
    # Plot 1: Recall vs K for each strategy
    # This plot helps to see how recall saturates for each strategy

    plt.style.use('searborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    recall_cols = [f'Recall@{k}' for k in K_VALUES]
    plot_data = summary_df[recall_cols].T
    plot_data.index = K_VALUES

    plot_data.plot(ax=ax1, marker='o')

    ax1.set_title('Recall vs. Candidate Pool Size (K) for Each Strategy', fontsize=16)
    ax1.set_xlabel('Number of Candidates (K)', fontsize=12)
    ax1.set_ylabel('Recall', fontsize=12)
    ax1.legend(title='Bucket Strategy')
    ax1.grid(True)

    plt.tight_layout()
    fig1_path = os.path.join(RESULT_PATH, 'images', 'recall_vs_k.png')
    fig1.savefig(fig1_path, dpi=300)
    print(f"Saved plot 1: {fig1_path}")

    # Plot 2: The 'knee' curve (cost vs benefit)
    # This is the most important plot for the final decision

    fig2, ax2= plt.subplots(figsize=(10, 6))

    # Indexing time as the cost and Recall@100 as the benefit
    cost = summary_df['indexing_time_min']
    benefit = summary_df['Recall@100']
    labels = summary_df.index

    sns.scatterplot(x=cost, y=benefit, s=150, ax=ax2)

    # Annotate points
    for i, label in enumerate(labels):
        ax2.text(cost[i], benefit[i], f'  {label}', fontsize=12)

    ax2.set_title('Cost (Indexing Time) vs. Benefit (Recall@100)', fontsize=16)
    ax2.set_xlabel('Total Indexing Time (minutes)', fontsize=12)
    ax2.set_ylabel('Recall @ k=100', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    fig2_path = os.path.join(RESULT_PATH, 'images', 'cost_vs_recall_100.png')
    fig2.savefig(fig2_path, dpi=300)
    print(f"Saved plot 2: {fig2_path}")

def run_experiment():
    '''
    Main function to execute the entire experimental protocol.
    '''
    es_manager = ElasticSearchManager()

    # Load data
    print("Loading metadata and bug reports...")
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)
    df_bugs['creation_date'] = pd.to_datetime(df_bugs['creation_date'])

    # Step 1: Select sample projects
    sample_projects_df = select_sample_projects(df_meta)
    sample_repo_names = sample_projects_df['repo_name'].tolist()

    results = []

    for _, project_row in tqdm(sample_projects_df.iterrows(), total=len(sample_projects_df), desc="Projects"):
        repo_name = project_row['repo_name']
        language = project_row['language']
        repo_path = get_project_path(repo_name, language)

        if not os.path.exists(repo_path):
            print(f"Warning: Repo path not found at {repo_path}. Skipping project.")
            continue

        project_bugs = df_bugs[df_bugs['repo_name'] == repo_name].copy()
        if project_bugs.empty:
            continue

        for strategy_name, freq in BUCKET_STRATEGIES.items():
            print(f"\n---- Processing: {repo_name} | Strategy: {strategy_name} ----")

            # Step 2: Group bugs and define indexing tasks
            bucketing_plan = group_bugs_into_buckets(project_bugs, freq, repo_path)

            for bucket, (bugs_in_bucket, snapshot_sha) in tqdm(bucketing_plan.items(), desc=f"Buckets ({strategy_name})"):
                if not snapshot_sha:
                    continue

                # Define a unique index name
                index_name = re.sub(r'[^a-z0-9]', '_', f"{repo_name}_{strategy_name}_{bucket}".lower())

                # Checkout, index, and record costs
                if not checkout_commit(repo_path, snapshot_sha):
                    continue

                start_time = time.time()
                es_manager.create_index(index_name)
                es_manager.index_source_files(index_name, repo_path)
                indexing_time = (time.time() - start_time) / 60  # in minutes
                index_stats = es_manager.get_index_stats(index_name)


                # Step 3: Run retrieval for all bugs in the bucket
                for bug in bugs_in_bucket:
                    candidates = es_manager.get_candidate_files(index_name, bug['bug_report_text', max(K_VALUES)])

                    result_row = {
                        'repo_name': repo_name,
                        'strategy': strategy_name,
                        'num_indexes': len(bucketing_plan),
                        'indexing_time_min': indexing_time,
                        'index_size_gb': index_stats['size_gb']
                    }

                    # Calculate hits for each K
                    for k in K_VALUES:
                        candidates_at_k = set(candidates[:k])
                        ground_truth = set(bug['ground_truth_files'])
                        hit = 1 if not candidates_at_k.isdisjoint(ground_truth) else 0
                        result_row[f'hit@{k}'] = hit

                    results.append(result_row)

                # Cleanup to save space
                es_manager.delete_index(index_name)

    # Step 4: Aggregrate and Analyze Results
    print("--- Experiment Complete. Aggregating results ---")
    df_results = pd.DataFrame(results)       

    summary_cols = {f'hit@{k}': 'mean' for k in K_VALUES}
    summary_cols.update({
        'indexing_time_min': 'sum',
        'index_size_gb': 'sum',
        'num_indexes': 'first' # All rows for a strategy have the same num_indexes
    })     

    # Calculate recall (mean of hits) and aggregate costs
    final_summary = df_results.groupby(['repo_name', 'strategy']).agg(summary_cols)

    # Rename columns for clarity
    final_summary.rename(columns={f'hit@{k}': f'Recall@{k}' for k in K_VALUES}, inplace=True)
    
    # Average across all sample projects
    overall_summary = final_summary.groupby('strategy').mean()

    print("--- Overall Summary Across All Sample Projects ---")
    print(overall_summary)

    # Save results for further analysis and plotting
    bucketing_results_path = os.path.join(RESULT_PATH, 'detailed_bucketing_results.csv')
    df_results.to_csv(bucketing_results_path, index=False)
    overall_summary_path = os.path.join(RESULT_PATH, 'summary_bucketing_results.csv')
    overall_summary.to_csv(overall_summary_path, index=False)
    print("\n Detailed and summary results saved to CSV files.")

    generate_and_save_plots(overall_summary)

if __name__ == '__main__':
    run_experiment()



    




