import os
import pandas as pd
import git
from git import Repo, Blob
from git.util import hex_to_bin
import pickle
import numpy as np
import faiss 
import torch
import torch.nn as nn
import torch.optim as optim
# from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import itertools
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch.nn.functional as F

# Import the model definition from TRANPCNN model
from tranp_cnn_model import TRANPCNN

# Configuration
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
BLOB_CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/cache"

# Experiment Settings
SOURCE_PROJECT = "huggingface/transformers"  # Example source project
TARGET_PROJECT = "pandas-dev/pandas" # Example target project
TOP_K_CANDIDATES = 300
TARGET_TRAIN_SPLIT = 0.10 # Use 10% of target data for training
TOKENIZER_NAME = "BAAI/bge-code-v1"

# Model Hyperparameters (assumed, as original paper doesn't specify them)
PARAMS = {
    'nl_embedding_dim': 512, # Based on the BGE model
    'code_embedding_dim' : 512,
    'vocab_size': 0,  # Will be set in main()
    'nl_kernels': 128,
    'nl_kernel_sizes': [3, 4, 5],  # For sequence processing
    'max_bug_len': 512, # Added: max length for bug report text
    'stmt_kernels': 100,
    'stmt_kernel_sizes': [3, 4, 5],
    'max_lines': 500, # Max statements per file
    'max_line_len': 100, # Max tokens per statement
    'file_kernels': 100,
    'file_kernel_sizes': [3, 5, 7],
    'hidden_dim': 256,
    'num_classes': 2, # Buggy vs Non Buggy
    'dropout':0.5
}
# Training Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5 # For regularization

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

def create_labeled_samples(project_name, language, bug_metadata_db, blob_embedding_db, top_k, is_training_data=True, bug_ids_to_process=None):
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
            _, result_indices = index.search(query_vec, k)
            candidate_shas = {snapshot_blobs[i] for i in result_indices[0]}

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
                project_type = 'target' if project_name == TARGET_PROJECT else 'source'
                labeled_samples.append((bug_id, blob_sha, label, project_type))

    return labeled_samples

# Pre-processing function
def preprocess_and_cache_samples(raw_samples, bug_reports_df, repo, tokenizer, cache_path):
    '''
    Performs the slow pre-processing (reading files, tokenizing) and saves the results to a parquet file for fast
    loading in the future.
    '''
    processed_data = []

    # Create a dictionary for fast bug text lookup
    bug_texts = pd.Series(
        bug_reports_df['bug_report_text'].values,
        index= bug_reports_df['bug_id']
    ).to_dict()

    for bug_id, blob_sha, label, project_type in tqdm(raw_samples, desc=f"Preprocessing and cachng"):
        # process bug report
        bug_text = bug_texts.get(bug_id, "")
        tokenized_bug = tokenizer(
            bug_text,
            padding='max_length',
            truncation=True,
            max_length=PARAMS['max_bug_len'],
            return_tensors='pt'
        )['input_ids'].squeeze(0).tolist() # Convert to list for DataFrame storage

        # process source code
        try:
            source_content = Blob(repo, hex_to_bin(blob_sha)).data_stream.read().decode('utf-8', 'ignore')
            lines = source_content.splitlines()[:PARAMS['max_lines']]
        except Exception:
            lines = [] # Handle cases where blob might be missing

        if not lines:
            padded_tokenized_lines = torch.zeros((PARAMS['max_lines'], PARAMS['max_line_len']), dtype=torch.long).tolist()
        else:
            tokenized_lines = tokenizer(
                lines, padding='max_length', truncation=True, max_length=PARAMS['max_line_len'], return_tensors='pt'
            )['input_ids']

            padded_tensor = torch.zeros((PARAMS['max_lines'], PARAMS['max_line_len']), dtype=torch.long)
            num_lines_to_copy = min(len(tokenized_lines), PARAMS['max_lines'])
            padded_tensor[:num_lines_to_copy] = tokenized_lines[:num_lines_to_copy]
            padded_tokenized_lines = padded_tensor.flatten().tolist() # Convert to list

            processed_data.append({
                'bug_ids': tokenized_bug, 
                'code_ids': padded_tokenized_lines,
                'label': label,
                'project_type': project_type,
                'bug_id': bug_id,
                'blob_sha': blob_sha
            })

    # Save to Parquet file
    df = pd.DataFrame(processed_data)
    df.to_parquet(cache_path, index=False)
    print(f"Successfully cache {len(df)} samples to {cache_path}")
    return df

# Step 2: Data Preparation & Pytorch Dataset
class BugLocalizationDataset(Dataset):
    '''Custom PyTorch Dataset for the hybrid model.'''
    def __init__(self, cache_path):
        # Load the entire pre-processed dataframe. This is fast
        self.data = pd.read_parquet(cache_path)
        # self.samples = samples
        # self.bug_db = bug_metadata_db
        # self.repo = repo
        # self.tokenizer = tokenizer
        # self.max_lines = 500 # Max statements per file
        # self.max_line_len = 100 # Max tokens per statement
        # self.max_bug_len = PARAMS['max_bug_len']

        # Create a dictionary for fast bug text lookup
        # bug_reports_df has 'bug_id' and a column with the clean text
        # self.bug_texts = pd.Series(
        #     bug_reports_df['bug_report_text'].values,
        #     index=bug_reports_df['bug_id']
        # ).to_dict()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the row .iloc is fast for integer-location based indexing.
        sample = self.data.iloc[idx]

        # Reshape the 1D list back into a 2D tensor
        code_ids_tensor = torch.tensor(sample['code_ids'], dtype=torch.long).reshape(
            PARAMS['max_lines'], PARAMS['max_line_len']
        )

        # 
        # bug_id, blob_sha, label, project_type = self.samples[idx]

        # # 1. Get bug report text
        # bug_text = self.bug_texts.get(bug_id, "") # Get text, default to empty strinf if not found

        # # 2. Tokenize and pad bug report text
        # tokenized_bug = self.tokenizer(
        #     bug_text,
        #     padding = 'max_length',
        #     truncation=True,
        #     max_length=self.max_bug_len,
        #     return_tensors='pt'
        # )['input_ids'].squeeze(0) # Squeeze to make it a 1D tensor

        # # Instantiate a Blob object directly using the repo and the binary-formatted SHA
        # blob_object = Blob(self.repo, hex_to_bin(blob_sha))
        # source_content = blob_object.data_stream.read().decode('utf-8', 'ignore')
        # lines = source_content.splitlines()[:self.max_lines]

        # # Handle cases where the source file is empty
        # if not lines:
        #     # If the file is empty, the code is just a tensor of padding tokens (zeros)
        #     padded_tokenized_lines = torch.zeros((self.max_lines, self.max_line_len), dtype=torch.long)
        # else: 
        #     tokenized_lines = self.tokenizer(
        #         lines, padding='max_length', truncation=True, max_length=self.max_line_len, return_tensors='pt'
        #     )['input_ids']

            # # Pad the number of lines
            # padded_tokenized_lines = torch.zeros((self.max_lines, self.max_line_len), dtype=torch.long)
            # num_lines_to_copy = min(len(tokenized_lines), self.max_lines)
            # padded_tokenized_lines[:num_lines_to_copy] = tokenized_lines[:num_lines_to_copy]

        # Convert pre-tokenized data (Stored as lists/value) back to tensors
        return {
            'bug_ids': torch.tensor(sample['bug_ids'], dtype=torch.long),
            'code_ids': code_ids_tensor, # Used the reshaped tensor
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'project_type': sample['project_type'],
            'bug_id': sample['bug_id'],
            'blob_sha': sample['blob_sha']
        }

# Step 3: Training and Evaluation Functions
def train(model, source_loader, target_loader, optimizer, criterion, epoch, device):
    model.train()
    progress_bar = tqdm(source_loader, desc=f'Epoch {epoch + 1}/ {EPOCHS}')
    iter_target = iter(itertools.cycle(target_loader))

    for source_batch in progress_bar:
        target_batch = next(iter_target)
        optimizer.zero_grad()

        # Process source batch
        outputs_s = model(source_batch['bug_ids'].to(device), source_batch['code_ids'].to(device), 'source')
        loss_s = criterion(outputs_s, source_batch['label'].to(device))

        # Process target batch
        outputs_t = model(target_batch['bug_ids'].to(device), target_batch['code_ids'].to(device), 'target')
        loss_t = criterion(outputs_t, target_batch['label'].to(device))

        combined_loss = loss_s + loss_t
        combined_loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'loss': f'{combined_loss.item(): .4f}'})

def evaluate(model, language, test_bug_ids, bug_db, blob_db, repo, tokenizer, device):
    '''
    Evaluates the model on the test set.
    '''
    model.eval()

    # Generate candidate pairs for the test set using our create_labeled_samples()
    # We set is_training_data = False to ensure the ground truth is NOT force included
    print("Generating candidates for the test set...")
    test_samples = create_labeled_samples(
        TARGET_PROJECT, language,
        bug_db, blob_db, TOP_K_CANDIDATES, is_training_data=False, bug_ids_to_process=test_bug_ids
    )

    if not test_samples:
        print("No valid samples generated for the test set. Cannot evaluate.")
        return {"Top-1 Acc": 0, "Top-5 Acc": 0, "Top-10 Acc": 0, "MAP": 0, "MRR": 0}

    # Group samples by bug_id to evaluate each bug's ranked list
    bugs_to_evaluate = {}
    for bug_id, blob_sha, _, _ in test_samples:
        if bug_id not in bugs_to_evaluate:
            bugs_to_evaluate[bug_id] = []
        bugs_to_evaluate[bug_id].append(blob_sha)

    # Create a single dataset for efficient scoring
    eval_dataset = BugLocalizationDataset(test_samples, bug_db, repo, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get model predictions for all candidates
    predictions = {}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Scoring Test Candidates"):
            outputs = model(batch['bug_ids'].to(device), batch['code_ids'].to(device), 'target')
            scores = F.softmax(outputs, dim=1)[:, 1] # Get probability of being 'buggy'

            for i in range(len(batch['bug_id'])):
                bug_id = batch['bug_id'][i]
                if bug_id not in predictions:
                    predictions[bug_id] = []
                predictions[bug_id].append((batch['blob_sha'][i], scores[i].item()))

    # 4. Calculate metrics
    top_k_hits = {k: 0 for k in [1, 5, 10]}
    average_precisions = [] 
    reciprocal_ranks = [] 

    for bug_id, candidates in bugs_to_evaluate.items():
        ranked_preds = sorted(predictions.get(bug_id, []), key=lambda x: x[1], reverse=True)
        ranked_shas = [sha for sha, score in ranked_preds]

        ground_truth_shas = {
            sha 
            for path, sha in get_path_to_sha_map(repo, bug_db[bug_id]['commit_sha']).items() 
            if any(path.endswith(gt_file) for gt_file in bug_db[bug_id]['ground_truth_files'])
            }

        # Calculate Top-K
        for k in top_k_hits:
            if not set(ranked_shas[:k]).isdisjoint(ground_truth_shas):
                top_k_hits[k] += 1

        # MRR
        found_rank = -1
        for i, sha in enumerate(ranked_shas):
            if sha in ground_truth_shas:
                found_rank = i + 1
                break
        if found_rank != -1:
            reciprocal_ranks.append(1 / found_rank)
        else:
            reciprocal_ranks.append(0)

        # MAP
        hits = 0
        precision_at_k = []
        for i, sha in enumerate(ranked_shas):
            if sha in ground_truth_shas:
                hits += 1
                precision_at_k.append(hits / (i + 1))
        if precision_at_k:
            average_precisions.append(np.mean(precision_at_k))
        else:
            average_precisions.append(0)


    num_bugs = len(bugs_to_evaluate)
    final_metrics = {
        "Top-1": top_k_hits[1] / num_bugs,
        "Top-5": top_k_hits[5] / num_bugs,
        "Top-10": top_k_hits[10] / num_bugs,
        "MAP": np.mean(average_precisions),
        "MRR": np.mean(reciprocal_ranks)
    }
    
    return final_metrics

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
 
# Step 4: Main Orchestrator
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize tokenizer once and reuse it
    # Initialize the tokenizer object
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Get the vocab size from THIS tokenizer object and update PARAMS
    # PARAMS['vocab_size'] = tokenizer.vocab_size
    PARAMS['vocab_size'] = len(tokenizer)
    
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    # 1. Load Databases
    print("Loading databases")
    source_bug_db, source_blob_db, source_language = load_project_databases(SOURCE_PROJECT, df_meta)
    target_bug_db, target_blob_db, target_language = load_project_databases(TARGET_PROJECT, df_meta)

    # Define cache paths
    os.makedirs(BLOB_CACHE_DIR, exist_ok=True)
    source_cache_path = os.path.join(BLOB_CACHE_DIR, f'{SOURCE_PROJECT.replace("/", "_")}_source_cache.parquet')
    target_cache_path = os.path.join(BLOB_CACHE_DIR, f'{TARGET_PROJECT.replace("/", "_")}_target_train_cache.parquet')

    source_repo = git.Repo(get_project_path(SOURCE_PROJECT, source_language))
    target_repo = git.Repo(get_project_path(TARGET_PROJECT, target_language))
    
    # 2. Preprocess source data (with caching)
    if not os.path.exists(source_cache_path):
        print("Source cache not found. Generating samples and preprocessing..")
        source_samples = source_samples = create_labeled_samples(SOURCE_PROJECT, source_language, source_bug_db, source_blob_db,
                                            TOP_K_CANDIDATES, is_training_data=True)
        preprocess_and_cache_samples(source_samples, df_bugs, source_repo, tokenizer, source_cache_path)
    else:
        print(f"Loading preprocessed source data from {source_cache_path}")

    # 3. Pre-process target data (with caching)
    if not os.path.exists(target_cache_path):
        print("Target train cache not found. Generating samples and preprocessing..")
        target_bug_ids = list(target_bug_db.keys())
        # 3a. Split the BUG IDs into training and testing sets
        train_bug_ids, test_bug_ids = train_test_split(
            target_bug_ids,
            test_size=(1 - TARGET_TRAIN_SPLIT),
            random_state=42 # for reproducibility
        )
        # 3b. Create training samples using ONLY the bug IDs from the training split
        target_train_samples = create_labeled_samples(
            TARGET_PROJECT, target_language, target_bug_db, target_blob_db, 
            TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=train_bug_ids #Pass the specific IDs to process
        )
        preprocess_and_cache_samples(target_train_samples, df_bugs, target_repo, tokenizer, target_cache_path)
    else:
        print(f"Loading preprocessed target train data from {target_cache_path}")
        # Need to regenerate test_bug_ids if loading from cache
        target_bug_ids = list(target_bug_db.keys())
        _, test_bug_ids = train_test_split(
            target_bug_ids, test_size=(1 - TARGET_TRAIN_SPLIT), random_state=42
        )
    
    # 4. Create FAST Datasets and DataLoaders
    source_dataset = BugLocalizationDataset(source_cache_path)
    target_train_dataset = BugLocalizationDataset(target_cache_path)

    source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Data loading complete.")
    print(f"  - Source samples: {len(source_dataset)}")
    print(f"  - Target train samples: {len(target_train_dataset)}")
    # print(f"  - Target test bugs: {len(test_bug_ids)}")

    # 5. Initialize Model, Criterion, Optimizer
    model = TRANPCNN(PARAMS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. Joint Training Loop
    print("\n--- Starting Joint Training ---")
    for epoch in range(EPOCHS):
        # Call the standalone train function
        train(model, source_loader, target_train_loader, optimizer, criterion, epoch, device)

    # 7. Final Evaluation
    print("\n-- Starting Final Evaluation ---")
    final_metrics = evaluate(model, target_language, test_bug_ids, target_bug_db, target_blob_db, target_repo, tokenizer, device)
    print(f"\n Final Metrics on Target Project: \n {final_metrics}")

if __name__ == '__main__':
    main()