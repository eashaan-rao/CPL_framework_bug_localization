import os
import pandas as pd
import git
from git import Repo, Blob
from git.util import hex_to_bin
import numpy as np
import pickle
import faiss
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


# Configuration
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
BLOB_CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/cache"
PARAMS = {
    'nl_embedding_dim': 512, # Based on the BGE model
    'code_embedding_dim' : 512,
    'vocab_size': 0,  # Will be set in main()
    'nl_kernels': 128,
    'nl_kernel_sizes': [3, 4, 5],  # For sequence processing
    'max_bug_len': 512, # Added: max length for bug report text
    'stmt_kernels': 100,
    'stmt_kernel_sizes': [3, 4, 5],
    'max_lines': 768, # Max statements per file
    'max_line_len': 21, # Max tokens per statement
    'file_kernels': 100,
    'file_kernel_sizes': [3, 5, 7],
    'hidden_dim': 256,
    'num_classes': 2, # Buggy vs Non Buggy
    'dropout':0.5
}

# Step 1: Data Preparation & Candidate Generation using Faiss
def get_path_to_sha_map(repo, commit_sha):
    '''
    For a given commit, creates a dictionary mapping file paths to blob SHAs.
    '''
    try:
        commit = repo.commit(commit_sha)
        return {blob.path: blob.hexsha for blob in commit.tree.traverse() if blob.type == 'blob'}
    except Exception:
        return {}

def create_labeled_samples(project_name, target_project, language, bug_metadata_db, blob_embedding_db, top_k, is_training_data=True, bug_ids_to_process=None):
    '''
    The main data preparation function. It generates candidates using Faiss, creates labeled pairs, and ensures
    ground truth is included for training. It now processes only the specified bug_ids
    '''
    repo_path = os.path.join(REPO_BASE_PATH, language.lower(), project_name.replace('/', '_'))
    repo = git.Repo(repo_path)

    # if no specific bug IDs are given, process all bugs in the database
    if bug_ids_to_process is None:
        bug_ids_to_process = list(bug_metadata_db.keys())

    # Create a filtered dictionary containing only the bugs we need to process
    filtered_bug_db = {bug_id: bug_metadata_db[bug_id] for bug_id in bug_ids_to_process if bug_id in bug_metadata_db}

    # Groupt bugs by commit for efficient FAISS indexing
    bugs_by_commit = {}
    for bug_id, meta in filtered_bug_db.items():
        sha = meta['commit_sha']
        if sha not in bugs_by_commit:
            bugs_by_commit[sha] = []
        bugs_by_commit[sha].append(bug_id)
    
    labeled_samples = []

    for commit_sha, bug_ids in tqdm(bugs_by_commit.items(), desc=f"Processing commits for {project_name}"):
        path_to_sha_map = get_path_to_sha_map(repo, commit_sha)

        # Assemble snapshot embeddings for Faiss
        snapshot_blobs = list(path_to_sha_map.values())
        snapshot_embeddings = np.array([blob_embedding_db[sha] for sha in snapshot_blobs if sha in blob_embedding_db]).astype('float32')
        if snapshot_embeddings.shape[0] == 0:
            continue
        
        # Build Faiss index
        faiss.normalize_L2(snapshot_embeddings)
        index = faiss.IndexFlatIP(snapshot_embeddings.shape[1])
        index.add(snapshot_embeddings)

        for bug_id in bug_ids:
            bug_info = bug_metadata_db[bug_id]
            bug_embedding = bug_info['embedding'].astype('float32')

            # Search for candidates
            query_vec = np.expand_dims(bug_embedding, axis=0)
            faiss.normalize_L2(query_vec)
            k = min(top_k, len(snapshot_blobs))

            # Capture Faiss scores (D)
            scores, result_indices = index.search(query_vec, k)
            # Create a map of blob_sha -> faiss score
            sha_to_score_map = {snapshot_blobs[idx]: scores[0][i] for i, idx in enumerate(result_indices[0])}
            candidate_shas = set(sha_to_score_map.keys())

            # Get ground truth blob SHAs
            ground_truth_shas = {
                sha for candidate, sha in path_to_sha_map.items()
                if any(candidate.endswith(gt_file) for gt_file in bug_info['ground_truth_files'])
            }

            # For training data, force-include the ground truth
            if is_training_data:
                candidate_shas.update(ground_truth_shas)

            # Create Labeled pairs
            for blob_sha in candidate_shas:
                label = 1 if blob_sha in ground_truth_shas else 0
                # Use projec_name to determine project_type, not the global TARGET_PROJECT
                project_type = 'target' if project_name == target_project else 'source'

                # Get the score from the map. Default to -1.0 if not found
                # e.g., a force-included ground truth file
                faiss_score = sha_to_score_map.get(blob_sha, -1.0)
                labeled_samples.append((bug_id, blob_sha, label, project_type, faiss_score))

    return labeled_samples

# Pre-processing function
def preprocess_and_cache_samples(raw_samples, bug_reports_df, repo, tokenizer, cache_meta_path, cache_code_ids_path, dtype=np.int32):
    '''
    Performs the slow pre-processing (reading files, tokenizing) and saves results into two files:
    a) A Parquet file for metadata (labels, bug_ids, etc.)
    b) A Numpy .npy file for the large code_ids tensors. 
    Stores unique blobs only once, metadata references them by index
    '''
    # Check if dtype is sufficient
    vocab_size = len(tokenizer)
    max_token_id = vocab_size - 1
    
    if dtype == np.int16 and max_token_id > 32767:
        print(f"WARNING: vocab_size={vocab_size} exceeds int16 range. Using int32.")
        dtype = np.int32
    elif dtype == np.int8 and max_token_id > 127:
        print(f"WARNING: vocab_size={vocab_size} exceeds int8 range. Using int16.")
        dtype = np.int16
    
    print(f"Using dtype: {dtype.__name__} (vocab_size: {vocab_size:,})")
    metadata = []

    # Create a dictionary for fast bug text lookup
    bug_texts = pd.Series(
        bug_reports_df['bug_report_text'].values,
        index= bug_reports_df['bug_id']
    ).to_dict()

    # Get unique blobs
    unique_blobs = list(set(blob_sha for _, blob_sha, _, _, _ in raw_samples))
    unique_bugs = list(set(bug_id for bug_id, _, _, _, _ in raw_samples))
    print(f"Total samples: {len(raw_samples)}")
    print(f"Unique blobs: {len(unique_blobs)}")
    print(f"Deduplication ratio: {len(raw_samples) / len(unique_blobs):.2f}x")

    # === OPTIMIZATION 1: Tokenize bugs once ===
    print("\nTokenizing unique bugs...")
    bug_tokens = {}
    for bug_id in tqdm(unique_bugs, desc="Tokenizing bugs"):
        bug_text = bug_texts.get(bug_id, "")
        tokenized_bug = tokenizer(
            bug_text, padding='max_length', truncation=True,
            max_length=PARAMS['max_bug_len'], return_tensors='pt'
        )['input_ids'].squeeze(0).numpy()
        bug_tokens[bug_id] = tokenized_bug.tolist()

    # Create blob_sha -> index mapping
    blob_to_idx = {sha: idx for idx, sha in enumerate(unique_blobs)}

    # Process unique blobs only
    num_blobs = len(unique_blobs)

    # Set up memory-mapped file before the loop
    # num_samples = len(raw_samples)
    code_ids_shape = (num_blobs, PARAMS['max_lines'] * PARAMS['max_line_len'])

    # This creates the large file on disk but doesn't load it into RAM
    print(f"Creating memory-mapped file at {cache_code_ids_path} with shape {code_ids_shape} for {num_blobs} blobs...")
    # Create an empty array and save it first to create proper .npy format
    # This writes the header and allocates space
    empty_array = np.empty(code_ids_shape, dtype=np.int32)
    np.save(cache_code_ids_path, empty_array)
    del empty_array  # Free memory immediately
    # Now open it as a memory-mapped array in read-write mode
    # This reads the .npy header and maps to the data portion
    code_ids_mmap = np.load(cache_code_ids_path, mmap_mode='r+')

    # Process each unique blob once
    for blob_sha in tqdm(unique_blobs, desc=f"Preprocessing unqiue blobs"):
        blob_idx = blob_to_idx[blob_sha] # Get index from mapping
        # process source code
        try:
            source_content = Blob(repo, hex_to_bin(blob_sha)).data_stream.read().decode('utf-8', 'ignore')
            lines = source_content.splitlines()[:PARAMS['max_lines']]
        except Exception:
            lines = [] # Handle cases where blob might be missing

        if not lines:
            # For empty files, create a tensor of zeros
            padded_tensor = torch.zeros((PARAMS['max_lines'], PARAMS['max_line_len']), dtype=torch.long)
        else:
            # For non-empty files, tokenize and pad as before
            tokenized_lines = tokenizer(
                lines, padding='max_length', truncation=True, max_length=PARAMS['max_line_len'], return_tensors='pt'
            )['input_ids']

            padded_tensor = torch.zeros((PARAMS['max_lines'], PARAMS['max_line_len']), dtype=torch.long)
            num_lines_to_copy = min(len(tokenized_lines), PARAMS['max_lines'])
            padded_tensor[:num_lines_to_copy] = tokenized_lines[:num_lines_to_copy]
        
        # Write data directly to disk instead of appending to a list
        code_ids_mmap[blob_idx] = padded_tensor.flatten().numpy()

    code_ids_mmap.flush()
    del code_ids_mmap

     # === OPTIMIZATION 2: Create metadata WITHOUT re-tokenizing ===
    print("\nCreating metadata (fast - no tokenization)...")
    metadata = []
    for bug_id, blob_sha, label, project_type, faiss_score in tqdm(raw_samples, desc="Building metadata"):
        metadata.append({
            'bug_ids': bug_tokens[bug_id],  # Lookup pre-tokenized bug
            'blob_idx': blob_to_idx[blob_sha],
            'label': label,
            'project_type': project_type,
            'bug_id': bug_id,
            'blob_sha': blob_sha,
            'faiss_score': faiss_score
        })
    
    meta_df = pd.DataFrame(metadata)
    meta_df.to_parquet(cache_meta_path, index=False)
    
    print(f"\n✓ Successfully cached:")
    print(f"  - Metadata: {cache_meta_path}")
    print(f"  - Code IDs: {cache_code_ids_path}")
    print(f"  - Unique blobs: {num_blobs:,}")
    print(f"  - Total samples: {len(raw_samples):,}")
    print(f"  - File size: {num_blobs * PARAMS['max_lines'] * PARAMS['max_line_len'] * 4 / 1e9:.2f} GB")

def get_project_path(repo_name, language):
    """
    Constructs the local path to a project repository based on its name and language.

    Args:
        repo_name (str): The repository name, e.g., 'apache/commons-lang'.
        language (str): The programming language, e.g., 'java', 'c++'.

    Returns:
        str: The full path to the repository directory.
    """
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

# Helper function for loading data
def load_project_databases(project_name, meta_df):
    '''
    Loads the bug and blob databases for a given project.
    '''
    language = meta_df[meta_df['repo_name'] == project_name]['language'].iloc[0]
    blob_db_path = os.path.join(BLOB_DB_DIR, project_name.replace('/', '_') + '_blob_embeddings32.pkl')
    bug_db_path = os.path.join(BUG_METADATA_DIR, project_name.replace('/', '_') + '_bug_metadata32.pkl')

    with open(blob_db_path, 'rb') as f:
        blob_db = pickle.load(f)
    with open(bug_db_path, 'rb') as f:
        bug_db = pickle.load(f)

    return bug_db, blob_db, language