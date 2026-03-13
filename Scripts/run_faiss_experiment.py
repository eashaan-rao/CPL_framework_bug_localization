import os
import pandas as pd
import git
import re
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import torch
import matplotlib.pyplot as plt

# Configuration
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
DATABASE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
MODEL_NAME = "BAAI/bge-code-v1" # long-context model

LANGUAGE_EXTENSIONS = {
    'c++': ['.ac', '.alpine', '.am', '.arm64', '.armv7', '.arrow', '.asar', '.avro', '.bash', '.bat', '.bazel', '.bin', '.bzl', '.c', '.cc', '.cfg', '.clj', '.cmake', '.conf', '.config', '.cpp', '.cs', '.csproj', '.css', '.dat', '.data', '.def', '.dict', '.expect', '.expected', '.fragment', '.gd', '.gemspec', '.glsl', '.gn', '.gni', '.go', '.grdp', '.gyp', '.gypi', '.h', '.hpp', '.html', '.ico', '.in', '.inc', '.include', '.init', '.install', '.j2', '.java', '.js', '.json', '.json5', '.kt', '.lib', '.limits', '.lock', '.m', '.m4', '.make', '.manifest', '.md', '.mjs', '.mk', '.mm', '.mp4', '.npy', '.out', '.patch', '.pb', '.pbxproj', '.php', '.plist', '.png', '.postinst', '.postinstall', '.preinst', '.prerm', '.props', '.proto', '.ps1', '.py', '.python', '.queries', '.rb', '.reference', '.scm', '.scss', '.sh', '.sigs', '.so', '.sql', '.supp', '.svg', '.targets', '.templates', '.ts', '.txt', '.ubuntu', '.ui', '.vcxproj', '.xib', '.xml', '.yaml', '.yml', '.zst'],
    'go': ['.bash', '.bats', '.conf', '.css', '.cue', '.dockerfile', '.ex', '.exs', '.fragment', '.go', '.graphqls', '.gtpl', '.hbs', '.ignore', '.js', '.json', '.key', '.lock', '.md', '.mdx', '.mjs', '.mod', '.mts', '.nightly', '.pem', '.png', '.py', '.rs', '.scss', '.sh', '.sum', '.tmpl', '.toml', '.ts', '.txt', '.win64', '.xml', '.yaml', '.yml'],
    'java': ['.asciidoc', '.bat', '.bazel', '.bz2', '.bzl', '.cer', '.clusterfilter', '.cmd', '.commandstep', '.conf', '.config', '.cpp', '.crt', '.crx', '.cs', '.css', '.csv', '.csv-spec', '.db', '.desktop', '.enc', '.executor', '.factories', '.filter', '.gemspec', '.gradle', '.groovy', '.gz', '.h', '.html', '.importorder', '.ini', '.install4j', '.iss', '.jar', '.java', '.jj', '.jpg', '.js', '.json', '.key', '.kt', '.less', '.liquibasedatatype', '.lock', '.md', '.meshenvlistenerfactory', '.mustache', '.nuspec', '.ods', '.plist', '.png', '.policy', '.properties', '.protocol', '.ps1', '.py', '.rb', '.rs', '.rst', '.scss', '.sh', '.sha1', '.snapshotgenerator', '.sql', '.sql-spec', '.sqlgenerator', '.st', '.svg', '.targets', '.toml', '.tpl', '.ts', '.tsv', '.tsx', '.txt', '.typebuilder', '.validation', '.vm', '.vt', '.vue', '.xls', '.xml', '.xsd', '.yaml', '.yml', '.zip'],
    'javascript': ['.avif', '.cjs', '.coffee', '.css', '.cts', '.dockerfile', '.example', '.graphql', '.graphqls', '.html', '.ico', '.jpg', '.js', '.jsm', '.json', '.jsx', '.link', '.lock', '.map', '.md', '.mdx', '.mjs', '.mts', '.opts', '.pdf', '.png', '.properties', '.rb', '.rs', '.scss', '.sh', '.snap', '.sqlite', '.stderr', '.svelte', '.svg', '.toml', '.ts', '.tsx', '.ttf', '.txt', '.wasm', '.webp', '.xml', '.yaml', '.yml'],
    'kotlin': ['.java', '.kt', '.py'],
    'python': ['.0', '.1', '.acl', '.ambr', '.base', '.bat', '.build', '.c', '.cfg', '.ci', '.cmd', '.cnf', '.conf', '.cs', '.csproj', '.css', '.css_t', '.csv', '.db', '.dockerfile', '.example', '.expected', '.g4', '.gif', '.h', '.html', '.in', '.ini', '.interp', '.inv', '.inventory', '.ipynb', '.j2', '.jar', '.java', '.jinja2', '.js', '.json', '.json5', '.jsx', '.kubernetes-helm-yaml', '.less', '.lock', '.manifest', '.md', '.mdx', '.mkv', '.mp4', '.nodejs14x', '.php', '.pip', '.png', '.pot', '.ps1', '.psm1', '.pxd', '.pxi', '.py', '.pyi', '.pyx', '.r', '.rdb', '.rst', '.run', '.scss', '.sh', '.sha256', '.sln', '.stderr', '.stdout', '.svg', '.template', '.tf', '.tgz', '.toml', '.tpl', '.ts', '.tsx', '.txt', '.xml', '.yaml', '.yml', '.zip'],
}

# A safe chunk for the BGE model
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
K_VALUES = [30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700]

# # Semantic Search Manager
class SemanticSearchManager:
    '''
    Manages embedding for bug reports and Faiss similarity search.
    '''
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def build_faiss_index(self, doc_embeddings: np.ndarray):
        '''
        Builds a Faiss index from document embeddings.
        '''
        if doc_embeddings.dtype != 'float32':
            doc_embeddings = doc_embeddings.astype('float32')
        faiss.normalize_L2(doc_embeddings)
        index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        return index
    
    def search(self, query_embedding: np.ndarray, index: faiss.Index, top_k: int):
        '''
        Searches the Faiss index for the top_k most similar vectors.
        '''
        if query_embedding.dtype != 'float32':
            query_embedding = query_embedding.astype('float32')
        query_embedding_2d = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding_2d)
        _, indices = index.search(query_embedding_2d, top_k)
        return indices[0]

# Helper Functions
def get_project_path(repo_name, language):
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

def select_sample_projects(df_meta):
    '''
    Selects a stratified sample of projects using existing categorical columns.
    '''
    print("Selecting a stratified sample of projects....")
    df_meta['strata'] = df_meta['project_size'] + '_' + df_meta['project_age']
    sample_df = df_meta.groupby('strata').apply(
        lambda x: x.sample(1) if not x.empty else None,
        include_groups=False
    ).reset_index()
    print("Selected Projects:")
    print(sample_df[['repo_name', 'strata', 'language']])
    return sample_df

def pre_run_check(repo_name, project_bugs, repo, blob_embedding_db):
    '''
    Find out how many bug reports based on their ground truth files are valid, partially valid or invalid (no ground truth file is found in commit.)
    '''
    # Pre-computation: Build a map of {path: blob_sha} for each commit
    commit_file_map = {}
    for sha in tqdm(project_bugs['pre_fix_commit_sha'].dropna().unique(), desc=f"Mapping files in {repo_name}", leave=False):
        try:
            commit = repo.commit(sha)
            commit_file_map[sha] = {blob.path: blob.hexsha for blob in commit.tree.traverse() if blob.type == 'blob'}
        except Exception:
            commit_file_map[sha] = {}

    cleaned_ground_truth_map = {}
    fully_valid_bugs = set()
    partially_valid_bugs = set()
    invalid_bugs_commit = set() # Bugs invalid because No GT File was in the commit
    invalid_bugs_embedding = set() # Bugs invalid because no GT file had an embedding

    partially_missing_commit = 0 # Count of individual GT Files missing from commits
    partially_missing_embedding = 0 # Count of individual GT files missing from embeddings
    missing_extensions_counter = {}

    # Check each bug against the pre-computed map and embedding DB
    for _, bug in project_bugs.iterrows():
        snapshot_sha = bug['pre_fix_commit_sha']
        if not snapshot_sha or snapshot_sha not in commit_file_map:
            continue

        files_in_commit = commit_file_map[snapshot_sha]

        valid_gt_files_for_this_bug = [] # keep the bug if at least one gt file is valid
        gt_found_in_commit = 0
        gt_found_in_db = 0

        for gt_file in bug['ground_truth_files']:
            found_path = next((path for path in files_in_commit if path.endswith(gt_file)), None)

            if found_path:
                # File exists in the commit. Now check if its embedding exists.
                gt_found_in_commit += 1
                blob_sha = files_in_commit[found_path]
                if blob_sha in blob_embedding_db:
                    # this ground_truth file is valid! adding to our list
                    gt_found_in_db += 1
                    valid_gt_files_for_this_bug.append(gt_file)
                else:
                    # gt_file is not present in embedding, record the missing extension
                    ext = os.path.splitext(gt_file)[1]
                    if not ext: ext =  "[No Extenstion]"
                    missing_extensions_counter[ext] = missing_extensions_counter.get(ext, 0) + 1
            else:
                # Record the missing extension
                ext = os.path.splitext(gt_file)[1]
                if not ext: ext =  "[No Extenstion]"
                missing_extensions_counter[ext] = missing_extensions_counter.get(ext, 0) + 1
        
        # Classify the bug based on the counts
        total_gt_count = len(bug['ground_truth_files'])

        if gt_found_in_db == total_gt_count:
            # All GT files were found and embedded
            fully_valid_bugs.add(bug['bug_id'])
            cleaned_ground_truth_map[bug['bug_id']] = valid_gt_files_for_this_bug
        elif gt_found_in_db > 0:
            # Some, but not all, GT files were invalid
            partially_valid_bugs.add(bug['bug_id'])
            cleaned_ground_truth_map[bug['bug_id']] = valid_gt_files_for_this_bug
            # Log the reasons for the missing ones
            partially_missing_commit += (total_gt_count - gt_found_in_commit)
            partially_missing_embedding += (gt_found_in_commit - gt_found_in_db)
        else:
            # No valid GT files were found for this bug. It is invalid.
            if gt_found_in_commit > 0:
                invalid_bugs_embedding.add(bug['bug_id'])  
            else:
                invalid_bugs_commit.add(bug['bug_id'])  
        
        
    # Detailed reporting for this project
    print(f"\n--- Pre-check Reporty for : {repo_name} ---")
    print(f"Total Bug reports in project: {len(project_bugs)}")
    print(f"    - Fully Valid (all GT Files found): {len(fully_valid_bugs)}")
    print(f"    - Partially Valid (some GT files found): {len(partially_valid_bugs)}")

    total_invalid = len(invalid_bugs_commit) + len(invalid_bugs_embedding)
    print(f" - Invalid (No GT Files Found): {total_invalid}")
    if total_invalid > 0:
        print(f"    - Reason: Not in commit: {len(invalid_bugs_commit)}")
        print(f"    - Reason: Not in embeddinf DB: {len(invalid_bugs_embedding)}")

    if len(partially_valid_bugs) > 0:
        print(f"\n Details for Partially Valid Bugs:")
        print(f"  - Total individual GT Files missing from commits: {partially_missing_commit} ")
        print(f"  - Total individual GT files missing from embedding DB: {partially_missing_embedding}")

    if missing_extensions_counter:
        print("\n Missing file extensions breakdown:")
        sorted_ext = sorted(missing_extensions_counter.items(), key=lambda item: item[1], reverse=True)
        for ext, count in sorted_ext:
            print(f" -- Extension '{ext}' : {count} missing instances")

    # Add the valid bugs (both full and partial) from this project to the overall list
    valid_bug_ids_for_project = fully_valid_bugs.union(partially_valid_bugs)
    print(f"Totally Valid Bugs (both full and parital): {len(valid_bug_ids_for_project)}")
    print(f"Total invalid bugs: {total_invalid}")


# Main experiment function
def run_faiss_experiment():
    print("---Starting FAISS Experiment----")
    os.makedirs(os.path.join(RESULT_PATH, 'faiss_recall_results'), exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, 'images'), exist_ok=True)
    semantic_manager = SemanticSearchManager(MODEL_NAME)

    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)
    # sample_projects_df = select_sample_projects(df_meta)

    # Define the list of target repositories
    # target_repos = [
    #     "pandas-dev/pandas",
    #     "huggingface/transformers",
    #     "numpy/numpy",
    #     "apache/dubbo",
    #     "apache/commons-lang",
    #     "square/kotlinpoet"
    # ]
    
    # We can add more repositories or precision types here later.
    experiments = [
        {'repo_name': 'spring-projects/spring-roo', 'precision': 'fp32', 'blob_suffix': '32.pkl', 'bug_suffix': '32.pkl'},
        # {'repo_name': 'spring-projects/spring-roo', 'precision': 'fp16', 'blob_suffix': '16.pkl', 'bug_suffix': '16.pkl'},
        {'repo_name': 'pypa/pip', 'precision': 'fp32', 'blob_suffix': '32.pkl', 'bug_suffix': '32.pkl'},
        # {'repo_name': 'pypa/pip', 'precision': 'fp16', 'blob_suffix': '16.pkl', 'bug_suffix': '16.pkl'},
        {'repo_name': 'protocolbuffers/protobuf', 'precision': 'fp32', 'blob_suffix': '32.pkl', 'bug_suffix': '32.pkl'},
        # {'repo_name': 'protocolbuffers/protobuf', 'precision': 'fp16', 'blob_suffix': '16.pkl', 'bug_suffix': '16.pkl'},
        {'repo_name': 'nats-io/nats-server', 'precision': 'fp32', 'blob_suffix': '32.pkl', 'bug_suffix': '32.pkl'},
        # {'repo_name': 'nats-io/nats-server', 'precision': 'fp16', 'blob_suffix': '16.pkl', 'bug_suffix': '16.pkl'},
        {'repo_name': 'square/anvil', 'precision': 'fp32', 'blob_suffix': '32.pkl', 'bug_suffix': '32.pkl'},
        # {'repo_name': 'square/anvil', 'precision': 'fp16', 'blob_suffix': '16.pkl', 'bug_suffix': '16.pkl'}
        
    ]

    # Filter df_meta to include only these repositories
    # sample_projects_df = df_meta[df_meta['repo_name'].isin(target_repos)].reset_index(drop=True)
        
    all_extensions = set(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts)
    # 1. initialize a list to store results from all projects
    all_project_results = []
    detailed_results = []
    
    # for _, project_row in tqdm(sample_projects_df.iterrows(), total=len(sample_projects_df), desc="Projects"):
    for experiment in tqdm(experiments, desc="Experiments"):
        # repo_name = project_row['repo_name']
        repo_name = experiment['repo_name']
        # language = project_row['language']
        project_row = df_meta[df_meta['repo_name'] == repo_name].iloc[0]
        language = project_row['language']

        print(f"\n--- Processing project: {repo_name} (Precision: {experiment['precision']}) ---")
        # 3. Dynamically build the database paths based on the experiment config.
        blob_filename = repo_name.replace('/', '_') + '_blob_embeddings' + experiment['blob_suffix']
        bug_filename = repo_name.replace('/', '_') + '_bug_metadata' + experiment['bug_suffix'] # Assuming bug files are also separated
        
        project_blob_db_path = os.path.join(DATABASE_DIR, blob_filename)
        project_bug_db_path = os.path.join(DATABASE_DIR, bug_filename)

        print(f"Loading blob DB: {project_blob_db_path}")
        print(f"Loading bug DB: {project_bug_db_path}")

        if not os.path.exists(project_blob_db_path) or not os.path.exists(project_bug_db_path):
            print(f"WARNING: Embedding databases for {repo_name} not found. Skipping.")
            continue

        print("Loading project-specific embedding databases...")
        with open(project_blob_db_path, 'rb') as f:
            blob_embedding_db = pickle.load(f)
        with open(project_bug_db_path, 'rb') as f:
            bug_metadata_db = pickle.load(f)
        print(f"Loaded {len(blob_embedding_db)} blob embeddings and {len(bug_metadata_db)} bug embeddings.")

        repo_path = get_project_path(repo_name, language)
        if not os.path.exists(repo_path):
            print(f"ERROR: Repo path not found for {repo_name}. Skipping.")
            continue
    
        repo = git.Repo(repo_path)
        project_bugs = df_bugs[df_bugs['repo_name'] == repo_name].copy()
        if project_bugs.empty: 
            print(f"ERROR: No bug reports found for {repo_name}. Exiting.")
            continue

        # If you want to run pre check test to know about invalid bugs, uncomment these line of code:
        # pre_run_check(repo_name, project_bugs, repo, blob_embedding_db)
        # continue
        
        # Counters
        total_bugs_processed = 0
        commit_failures = 0
        snapshots_without_embeddings = 0
        bugs_without_embeddings = 0
        hits_per_k = {k: 0 for k in K_VALUES}

        for _, bug in tqdm(project_bugs.iterrows(), total=len(project_bugs), desc=f"Bugs in {repo_name}", leave=False):
            total_bugs_processed += 1
            bug_id = bug['bug_id']
            snapshot_sha = bug['pre_fix_commit_sha']
            if not snapshot_sha:
                commit_failures += 1
                continue

            # Retrieve pre-computed bug embedding instead of generating it
            if bug_id not in bug_metadata_db:
                bugs_without_embeddings += 1
                continue
            bug_report_embedding = bug_metadata_db[bug_id]['embedding']

            try:
                commit = repo.commit(snapshot_sha)
            except Exception:
                commit_failures += 1
                continue

            # Assemble the snapshot from the project-specific database
            snapshot_embeddings = []
            embeddings_to_path_map = []
            for blob in commit.tree.traverse():
                if blob.type == 'blob' and any(blob.name.endswith(ext) for ext in all_extensions):
                    # Check against the project-specific blob_embedding_db
                    if blob.hexsha in blob_embedding_db:
                        snapshot_embeddings.append(blob_embedding_db[blob.hexsha])
                        embeddings_to_path_map.append(blob.path)

            if not snapshot_embeddings:
                snapshots_without_embeddings += 1
                continue

            snapshot_embeddings_np = np.array(snapshot_embeddings)

            # Build temporary Faiss index and search
            faiss_index = semantic_manager.build_faiss_index(snapshot_embeddings_np)
            result_indices = semantic_manager.search(bug_report_embedding, faiss_index, min(max(K_VALUES), len(snapshot_embeddings_np)))

            candidates_relative = [embeddings_to_path_map[i] for i in result_indices]

            # Calculate hits
            # original_ground_truth_list = df_bugs.loc[df_bugs['bug_id'] == bug_id, 'ground_truth_files'].iloc[0]
            ground_truth = set(bug['ground_truth_files'])

            # Create a dictionary to quickly map a candidate path to its rank
            candidate_rank_map = {path: i + 1 for i, path in enumerate(candidates_relative)}

            if repo_name == 'pypa/pip':
                print(f"\n--- DEBUGGING PATHS for bug {bug['bug_id']} in pypa/pip ---")
                
                print("\nSample of Ground Truth Paths:")
                for i, gt in enumerate(list(ground_truth)[:5]): # Print up to 5
                    print(f"  - '{gt}'")

                print("\nif ground truth file is present in Candidate Paths:")
                for gt_path in ground_truth:
                    for candidate_path, rank in candidate_rank_map.items():
                        if candidate_path.endswith(gt_path):
                            print(candidate_path)

                # for i, can in enumerate(candidates_relative[:5]): # Print up to 5
                #     print(f"  - '{can}'")

                print("\n--- END OF DEBUG. Stopping script. ---")
                exit() # Stop the script after printing for one bug

            # Find the rank for EVERY for ground truth file
            for gt_path in ground_truth:
                found_rank = float('inf') # Use infinity to represent "not found"

                # find the ground truth file in the candidate list
                for candidate_path, rank in candidate_rank_map.items():
                    if candidate_path.endswith(gt_path):
                        found_rank = rank
                        break # Found the best match for this gt_path, move to the next one
                
                # Record the result for this specific ground truth file instance
                detailed_results.append({
                    'bug_id': bug['bug_id'],
                    'repo': repo_name,
                    'gt_path': gt_path,
                    'rank': found_rank,
                    'extension': os.path.splitext(gt_path)[1] or '[No Ext]'
                })
                
            # First, find the rank of the first correct file we find
            found_rank = -1
            for i, candidate in enumerate(candidates_relative):
                # Use the simple, robust endswith() check
                is_match = any(candidate.endswith(gt_file) for gt_file in ground_truth)
                if is_match:
                    found_rank = i + 1 # Rank is 1-based (position 1, 2, 3...)
                    break # Stop after finding the first match

            # Now, if we found a match, add a "hit" to all K values
            # that are greater than or equal to its rank.
            if found_rank != -1:
                for k in K_VALUES:
                    if found_rank <= k:
                        hits_per_k[k] += 1
    
        # Aggregate and analyze final results
        print(f"\n --- Semantic Search for {repo_name} Complete ----")
        print(f"Total bug reports processed: {total_bugs_processed}")
        print(f"COmmit Failures (invalid SHA): {commit_failures}")
        print(f"Snapshots Missing Embeddings: {snapshots_without_embeddings}")

        print("\n -- Recal @ K Results --")
        final_recall = {}
        # Calculate recall only on bugs that could be processed
        valid_bugs = total_bugs_processed - commit_failures - snapshots_without_embeddings
        if valid_bugs > 0:
            all_project_results.append({
                'repo_name': repo_name,
                'valid_bugs': valid_bugs,
                'hits_per_k': hits_per_k
            })
        for k, hits in hits_per_k.items():
            recall = hits / valid_bugs if valid_bugs > 0 else 0
            final_recall[f"Recall@{k}"] = recall
            print(f"Recall@{k}: {recall:.4f} ({hits}/{valid_bugs})")

        # Save final results
        df_final_recall = pd.DataFrame([final_recall])
        recall_path = os.path.join(RESULT_PATH, 'faiss_recall_results', f'semantic_search_{repo_name.replace("/", "_")}_results.csv')
        df_final_recall.to_csv(recall_path, index=False)
        print(f"\n Final recall results save to {recall_path}")

    print("\n Overall summary across all sample projects")
    if not all_project_results:
        print("No results to aggregate.")
        return
    
    # Calculate overall micro-average recall
    total_valid_bugs = sum(res['valid_bugs'] for res in all_project_results)
    total_hits_per_k = {k: 0 for k in K_VALUES}
    for res in all_project_results:
        for k in K_VALUES:
            total_hits_per_k[k] += res['hits_per_k'][k]

    print(f"Total valid bug reports across all projects: {total_valid_bugs}")
    print("\n Overall Recall @ K Results")

    overall_recall = {}
    for k, total_hits in total_hits_per_k.items():
        recall = total_hits / total_valid_bugs if total_valid_bugs > 0 else 0
        overall_recall[k] = recall
        print(f"Recall@{k}: {recall:.4f} ({total_hits}/{total_valid_bugs})")

    # Extension analysis by rank section
    print("\n--- Detailed Analysis by File Extension ---")
    df_detailed = pd.DataFrame(detailed_results)
    # Calculate Recall@300 for each file extension
    recall_at_300 = df_detailed.groupby('extension')['rank'].apply(
        lambda x: (x <= 300).sum() / len(x)
    ).sort_values(ascending=False)
    print("\n Recall@300 by extension:")
    print(recall_at_300)

    # Calculate Mean Reciprocal Rank (MRR) for each file extension
    # This tells us the average quality of the ranking for each file type
    df_detailed['reciprocal_rank'] = 1 / df_detailed['rank']
    mrr_by_ext = df_detailed.groupby('extension')['reciprocal_rank'].mean().sort_values(ascending=False)
    print("\n Mean Reciprocal Rank (MRR) by Extension: ")
    print(mrr_by_ext)

    # Calculate MRR by extension per unique repo
    mrr_by_repo_ext = df_detailed.groupby(['repo', 'extension'])['reciprocal_rank'].mean().unstack().fillna(0)
    print("\n Mean Receiprocal Rank (MRR) by Extension per Repository: ")
    print(mrr_by_repo_ext)
    mrr_by_repo_ext_path = os.path.join(RESULT_PATH, 'mrr_per_ext_repo.csv')
    mrr_by_repo_ext.to_csv(mrr_by_repo_ext_path, index=False)
    print(f"Saved detailed rank results to {mrr_by_repo_ext_path}")

    # Save and plot the final aggregated results
    # Save final summary
    df_overall_recall = pd.DataFrame({
        'K': list(overall_recall.keys()),
        'Recall': list(overall_recall.values())
    })
    recall_path = os.path.join(RESULT_PATH, 'faiss_recall_results', 'semantic_search_SAMPLE_recall_results.csv')
    df_overall_recall.to_csv(recall_path, index=False)
    print(f"\nOverall recall results saved to {recall_path}")

    # Create and save the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_overall_recall['K'], df_overall_recall['Recall'], marker='o', linestyle='-')
    ax.set_title('Overall Average Recall vs. Candidate Pool Size (K)', fontsize=16)
    ax.set_xlabel('Number of Candidates (K)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.grid(True)
    
    # Add percentage labels to points
    for i, row in df_overall_recall.iterrows():
        ax.text(row['K'], row['Recall'], f" {row['Recall']:.1%}", verticalalignment='bottom')
    
    plt.tight_layout()
    plot_path = os.path.join(RESULT_PATH, 'images', 'sample_recall_vs_k.png')
    fig.savefig(plot_path, dpi=300)
    print(f"Saved final plot to: {plot_path}")  

if __name__ == '__main__':
    run_faiss_experiment()
