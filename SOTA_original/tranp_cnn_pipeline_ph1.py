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
from torch.amp import autocast, GradScaler
import time 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
SOURCE_PROJECT = "apache/dolphinscheduler" # Example source project
TARGET_PROJECT = "apache/dubbo" # Example target project
TOP_K_CANDIDATES = 300
TARGET_TRAIN_SIZE = 0.10 # Use 10% of target data for training
TEST_SET_SIZE = 0.2
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
    'max_lines': 512, # Max statements per file
    'max_line_len': 21, # Max tokens per statement
    'file_kernels': 100,
    'file_kernel_sizes': [3, 5, 7],
    'hidden_dim': 256,
    'num_classes': 2, # Buggy vs Non Buggy
    'dropout':0.5
}
# Training Hyperparameters
EPOCHS = 10
BATCH_SIZE = 192
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
    unique_blobs = list(set(blob_sha for _, blob_sha, _, _ in raw_samples))
    unique_bugs = list(set(bug_id for bug_id, _, _, _ in raw_samples))
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
    for bug_id, blob_sha, label, project_type in tqdm(raw_samples, desc="Building metadata"):
        metadata.append({
            'bug_ids': bug_tokens[bug_id],  # Lookup pre-tokenized bug
            'blob_idx': blob_to_idx[blob_sha],
            'label': label,
            'project_type': project_type,
            'bug_id': bug_id,
            'blob_sha': blob_sha
        })
    
    meta_df = pd.DataFrame(metadata)
    meta_df.to_parquet(cache_meta_path, index=False)
    
    print(f"\n✓ Successfully cached:")
    print(f"  - Metadata: {cache_meta_path}")
    print(f"  - Code IDs: {cache_code_ids_path}")
    print(f"  - Unique blobs: {num_blobs:,}")
    print(f"  - Total samples: {len(raw_samples):,}")
    print(f"  - File size: {num_blobs * PARAMS['max_lines'] * PARAMS['max_line_len'] * 4 / 1e9:.2f} GB")


def analyze_and_set_padding(raw_samples, repo, tokenizer):
    """
    Analyzes the distribution of file and line lengths in the candidate samples
    and dynamically sets the padding limits in the PARAMS dictionary.
    """
    print("\n--- Starting analysis of candidate files to determine optimal padding... ---")
    line_lengths = []
    tokens_per_line = []
    
    # Get unique blob SHAs to avoid analyzing the same file multiple times
    unique_blobs = {sha for _, sha, _, _ in raw_samples}
    
    for blob_sha in tqdm(unique_blobs, desc="Analyzing file lengths"):
        try:
            source_content = Blob(repo, hex_to_bin(blob_sha)).data_stream.read().decode('utf-8', 'ignore')
            lines = source_content.splitlines()
            if not lines:
                continue
            
            line_lengths.append(len(lines))
            
            # Tokenize without padding to get true lengths
            tokenized_lines = tokenizer(lines, truncation=True, max_length=512)['input_ids'] # Use a high max_length for analysis
            for line in tokenized_lines:
                tokens_per_line.append(len(line))
        except Exception:
            continue # Skip files that can't be read

    if not line_lengths or not tokens_per_line:
        print("Could not analyze any files. Using default padding values.")
        return

    # --- Calculate and report statistics ---
    max_lines_95th = int(np.percentile(line_lengths, 95))
    max_line_len_95th = int(np.percentile(tokens_per_line, 95))
    
    print("\n--- Analysis Complete ---")
    print(f"File Lines Distribution:")
    # print(f"  - 90th Percentile: {int(np.percentile(line_lengths, 90))} lines")
    # print(f"  - 95th Percentile: {max_lines_95th} lines")
    # print(f"  - 99th Percentile: {int(np.percentile(line_lengths, 99))} lines")
    # print(f"  - Max Found:       {np.max(line_lengths)} lines")
    
    # print(f"\nTokens per Line Distribution:")
    # print(f"  - 90th Percentile: {int(np.percentile(tokens_per_line, 90))} tokens")
    # print(f"  - 95th Percentile: {max_line_len_95th} tokens")
    # print(f"  - 99th Percentile: {int(np.percentile(tokens_per_line, 99))} tokens")
    # print(f"  - Max Found:       {np.max(tokens_per_line)} tokens")
    # Calculate more percentiles
    max_lines_75th = int(np.percentile(line_lengths, 75))
    max_lines_85th = int(np.percentile(line_lengths, 85))
    max_lines_90th = int(np.percentile(line_lengths, 90))

    max_line_len_75th = int(np.percentile(tokens_per_line, 75))
    max_line_len_85th = int(np.percentile(tokens_per_line, 85))
    max_line_len_90th = int(np.percentile(tokens_per_line, 90))

    print(f"\n--- Percentile Options ---")
    print(f"75th: max_lines={max_lines_75th}, max_line_len={max_line_len_75th}")
    print(f"85th: max_lines={max_lines_85th}, max_line_len={max_line_len_85th}")
    print(f"90th: max_lines={max_lines_90th}, max_line_len={max_line_len_90th}")

    # --- Dynamically update the global PARAMS ---
    print("\nUpdating PARAMS with 90th percentile values to optimize performance and disk space.")
    PARAMS['max_lines'] = max_lines_75th
    PARAMS['max_line_len'] = max_line_len_75th
    print(f"New 'max_lines': {PARAMS['max_lines']}")
    print(f"New 'max_line_len': {PARAMS['max_line_len']}\n")
   
# Step 2: Data Preparation & Pytorch Dataset
class BugLocalizationDataset(Dataset):
    '''Custom PyTorch Dataset for the hybrid model.
    Loads pre-processed data from a metadata file and a memory-mapped NumPY file.
    '''
    def __init__(self, cache_meta_path, cache_code_ids_path):
        # Load the metadata. This is small and fits in RAM.
        self.metadata = pd.read_parquet(cache_meta_path)

        # Load the massive code_ids array using memory mapping. This does not load the file into RAM. Its fast 
        # and memory-efficient
        print(f"Loading code_ids data from {os.path.basename(cache_code_ids_path)}...")
        try:
            # First, try the fast, memory-mapped approach
            # self.code_ids_data = np.load(cache_code_ids_path, mmap_mode='r')
            print("Attempting to load full array into RAM...")
            self.code_ids_data = np.load(cache_code_ids_path)  # No mmap_mode
            print(f"Successfully loaded {self.code_ids_data.nbytes / 1e9:.2f} GB into RAM")
        except MemoryError:
            print("Not enough RAM. Using memory-mapped mode (slower on HDD)...")
            self.code_ids_data = np.load(cache_code_ids_path, mmap_mode='r')
        
        print("Data loading complete.")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):        
        # Get the row .iloc is fast for integer-location based indexing.
        meta_sample = self.metadata.iloc[idx]

        # get the code_ids array from the memory-mapped filed. Very fast.
        # Get blob index (works for both old and new cache format)
        if 'blob_idx' in meta_sample:
            blob_idx = meta_sample['blob_idx']  # New deduplicated format
        else:
            blob_idx = idx  # Old format (backward compatible)
        code_ids_flat = self.code_ids_data[blob_idx]

        # Reshape the flat array back into a 2D tensor the model expects
        code_ids_tensor = torch.from_numpy(code_ids_flat.copy()).reshape(
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
            'bug_ids': torch.tensor(meta_sample['bug_ids'], dtype=torch.long),
            'code_ids': code_ids_tensor, # Used the reshaped tensor
            'label': torch.tensor(meta_sample['label'], dtype=torch.long),
            'project_type': meta_sample['project_type'],
            'bug_id': meta_sample['bug_id'],
            'blob_sha': meta_sample['blob_sha']
        }

# Step 3: Training and Evaluation Functions
def train(model, source_loader, target_loader, optimizer, criterion, epoch, device, scaler):
    model.train()

    # Scenario 1: Joint Training (CP-transfer)
    if source_loader and target_loader:
        print(f"\n--- Epoch {epoch + 1}: Joint Training on Source and Target Data ---")
        progress_bar = tqdm(source_loader, desc=f'Epoch {epoch + 1}/ {EPOCHS}')
        iter_target = iter(itertools.cycle(target_loader))
        batch_count = 0
        total_loss = 0.0

        for source_batch in progress_bar:
            target_batch = next(iter_target)

            bug_ids_s = source_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_s = source_batch['code_ids'].to(device, non_blocking=True)
            labels_s = source_batch['label'].to(device, non_blocking=True)
            
            bug_ids_t = target_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_t = target_batch['code_ids'].to(device, non_blocking=True)
            labels_t = target_batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward + loss
            with autocast(device_type='cuda'):
                outputs_s = model(bug_ids_s, code_ids_s, 'source')
                loss_s = criterion(outputs_s, labels_s)
                outputs_t = model(bug_ids_t, code_ids_t, 'target')
                loss_t = criterion(outputs_t, labels_t)
                combined_loss = loss_s + loss_t
            
        # Backward + optimizer step
            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track loss
            total_loss += combined_loss.item()
            batch_count += 1

            # Update progress bar every 10 batches
            if batch_count % 10 == 0:
                avg_loss = total_loss / batch_count
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
        # Epoch summary
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}")

    # Scenario 2: Source-Only Training (CP-cold-start)
    elif source_loader:
        print(f"\n--- Epoch {epoch + 1}: Training on Source Data Only ---")
        progress_bar = tqdm(source_loader, desc=f'Epoch {epoch + 1}/ {EPOCHS}')
        batch_count = 0
        total_loss = 0.0

        for source_batch in progress_bar:
            bug_ids_s = source_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_s = source_batch['code_ids'].to(device, non_blocking=True)
            labels_s = source_batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward + loss
            with autocast(device_type='cuda'):
                outputs_s = model(bug_ids_s, code_ids_s, 'source')
                loss_s = criterion(outputs_s, labels_s)
            
            scaler.scale(loss_s).backward()
            scaler.step(optimizer)
            scaler.update()
        
            # Track loss
            total_loss += loss_s.item()
            batch_count += 1        

            #  Update progress bar every 10 batches
            if batch_count % 10 == 0:
                avg_loss = total_loss / batch_count
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
        # Epoch summary
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}")

    # Scenario 3: Target-Only Training (WP-small, WP-large)
    elif target_loader:
        print(f"\n--- Epoch {epoch + 1}: Training on Target Data Only ---")
        progress_bar = tqdm(target_loader, desc=f'Epoch {epoch + 1}/ {EPOCHS}')
        batch_count = 0
        total_loss = 0.0

        for target_batch in progress_bar:
            bug_ids_t = target_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_t = target_batch['code_ids'].to(device, non_blocking=True)
            labels_t = target_batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward + loss
            with autocast(device_type='cuda'):
                outputs_t = model(bug_ids_t, code_ids_t, 'target')
                loss_t = criterion(outputs_t, labels_t)
            
            scaler.scale(loss_t).backward()
            scaler.step(optimizer)
            scaler.update()
        
            # Track loss
            total_loss += loss_t.item()
            batch_count += 1        

            #  Update progress bar every 10 batches
            if batch_count % 10 == 0:
                avg_loss = total_loss / batch_count
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
        # Epoch summary
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}")

    else:
        raise ValueError("At least one of source_loader or target_loader must be provided for training.")
        return # No data to train on at all

def evaluate(model, test_meta_path, test_code_ids_path, device):
    '''
    Evaluates the model on the pre-processed test set.
    '''
    model.eval()

    print("Loading pre-processed test data for evaluation...")
    # Use the same fast Dataset, but on the test cache files
    eval_dataset = BugLocalizationDataset(test_meta_path, test_code_ids_path)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Group samples by bug_id to evaluate each bug's ranked list
    # We can get this info from the loaded metadata
    bugs_to_evaluate = eval_dataset.metadata.groupby('bug_id')['blob_sha'].apply(list).to_dict()
    ground_truth_shas_df = eval_dataset.metadata[eval_dataset.metadata['label'] == 1]
    ground_truth_map = ground_truth_shas_df.groupby('bug_id')['blob_sha'].apply(set).to_dict()

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

    for bug_id in bugs_to_evaluate:
        ranked_preds = sorted(predictions.get(bug_id, []), key=lambda x: x[1], reverse=True)
        ranked_shas = [sha for sha, score in ranked_preds]
        ground_truth = ground_truth_map.get(bug_id, set())

        # Calculate Top-K
        for k in top_k_hits:
            if not set(ranked_shas[:k]).isdisjoint(ground_truth):
                top_k_hits[k] += 1

        # MRR
        found_rank = -1
        for i, sha in enumerate(ranked_shas):
            if sha in ground_truth:
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
            if sha in ground_truth:
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
def run_tranp_cnn_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario):
    '''
    This is the new entry point for the TRANP-CNN pipeline, refactored from the old main() function.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize tokenizer once and reuse it
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # Get the vocab size from THIS tokenizer object and update PARAMS
    PARAMS['vocab_size'] = len(tokenizer)
    
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    # 1. Load Databases
    print("Loading databases")
    source_bug_db, source_blob_db, source_language = load_project_databases(SOURCE_PROJECT, df_meta)
    target_bug_db, target_blob_db, target_language = load_project_databases(TARGET_PROJECT, df_meta)

    # Define cache paths
    os.makedirs(BLOB_CACHE_DIR, exist_ok=True)
    source_meta_path = os.path.join(BLOB_CACHE_DIR, f'{SOURCE_PROJECT.replace("/", "_")}_{scenario}_source_train_meta.parquet')
    source_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{SOURCE_PROJECT.replace("/", "_")}_{scenario}_source_code_ids.npy')
    
    target_train_meta_path = os.path.join(BLOB_CACHE_DIR, f'{TARGET_PROJECT.replace("/", "_")}_{scenario}_target_train_meta.parquet')
    target_train_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{TARGET_PROJECT.replace("/", "_")}_{scenario}_target_train_code_ids.npy')

    target_test_meta_path = os.path.join(BLOB_CACHE_DIR, f'{TARGET_PROJECT.replace("/", "_")}_{scenario}_target_test_meta.parquet')
    target_test_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{TARGET_PROJECT.replace("/", "_")}_{scenario}_target_test_code_ids.npy')

    # --- Generate RAW samples first ---
    print("Generating raw source samples...")
    source_samples = create_labeled_samples(SOURCE_PROJECT, source_language, source_bug_db, source_blob_db, 
                                            TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=source_train_ids)
    
    # ... (logic for target train/test split) ...
    print("Generating raw target train samples...")
    # 3b. Create training samples using ONLY the bug IDs from the training split
    target_train_samples = create_labeled_samples(
        TARGET_PROJECT, target_language, target_bug_db, target_blob_db, 
        TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=target_train_ids #Pass the specific IDs to process
    )
    # 3b. Create training samples using ONLY the bug IDs from the training split
    target_test_samples = create_labeled_samples(
        TARGET_PROJECT, target_language, target_bug_db, target_blob_db, 
        TOP_K_CANDIDATES, is_training_data=False, bug_ids_to_process=target_test_ids #Pass the specific IDs to process
    )
    
    source_repo = git.Repo(get_project_path(SOURCE_PROJECT, source_language))
    # --- NEW: Run analysis and dynamically set padding before caching ---
    # We analyze on the source project's candidates as it's typically the largest dataset
    # analyze_and_set_padding(source_samples, source_repo, tokenizer)
    target_repo = git.Repo(get_project_path(TARGET_PROJECT, target_language))
    
    
    # 4. Preprocess source data (with caching) and Create FAST Datasets and DataLoaders
    if source_train_ids:
        if not os.path.exists(source_meta_path) or not os.path.exists(source_code_ids_path):
            print("Source cache not found. Generating samples and preprocessing..")
            preprocess_and_cache_samples(source_samples, df_bugs, source_repo, tokenizer, source_meta_path, source_code_ids_path)
        else:
            print(f"Loading preprocessed source data from cache")
        source_dataset = BugLocalizationDataset(source_meta_path, source_code_ids_path)
        print(f"  - Source samples: {len(source_dataset)}")
        source_loader = DataLoader(
            source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    else:
        print("  - No source training samples.")
        source_loader = [] # Use an empty list as a flag



    # 4a. Pre-process target training data (with caching)
    if target_train_ids:
        if not os.path.exists(target_train_meta_path) or not os.path.exists(target_train_code_ids_path):
            print("Target train cache not found. Generating samples and preprocessing..")
            preprocess_and_cache_samples(target_train_samples, df_bugs, target_repo, tokenizer, target_train_meta_path, 
                                     target_train_code_ids_path)
        else:
            print(f"Loading preprocessed target train data from cache")

        target_train_dataset = BugLocalizationDataset(target_train_meta_path, target_train_code_ids_path)
        print(f"  - Target train samples: {len(target_train_dataset)}")
        target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, 
                                         pin_memory=True)
    else:
        print("  - No target training samples.")
        target_train_loader = [] # Use an empty list as a flag
    
    print("Data loading checks complete.") # <-- New print statement
    
    # 5. Initialize Model, Criterion, Optimizer
    model = TRANPCNN(PARAMS).to(device)

    # Calculate class weights to fix imbalance
    # We read the label columns from the caches we just created.
    print("Calculating class weights for loss function...")
    try:
        labels_s_df = pd.read_parquet(source_meta_path, columns=['label'])
    except:
        labels_s_df = pd.DataFrame(columns=['label']) # Handle empty source

    try:
        labels_t_df = pd.read_parquet(target_train_meta_path, columns=['label'])
    except:
        labels_t_df = pd.DataFrame(columns=['label']) # Handle empty target

    all_train_labels = pd.concat([labels_s_df['label'], labels_t_df['label']])
    
    # Calculate counts
    counts = all_train_labels.value_counts()
    count_0 = counts.get(0, 1) # Get count for label 0, default to 1
    count_1 = counts.get(1, 1) # Get count for label 1
    total_samples = count_0 + count_1
    
    # Calculate balanced weights: weight = total_samples / (num_classes * class_count)
    weight_0 = total_samples / (2.0 * count_0)
    weight_1 = total_samples / (2.0 * count_1)

    # Create the weight tensor and send to GPU
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
    
    print(f"  - Total Samples: {total_samples}")
    print(f"  - Labels 0 (neg): {count_0} (Weight: {weight_0:.2f})")
    print(f"  - Labels 1 (pos): {count_1} (Weight: {weight_1:.2f})")

    # Enable cuDNN autotuner for potential speedups on fixed-size inputs
    torch.backends.cudnn.benchmark = True

    # Pass the calculated weights to the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Use AdamW with fused=True for faster CUDA kernels
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
    # Intialize mixed precision scaler
    scaler = GradScaler("cuda")

    # 6. Joint Training Loop
    print("\n--- Starting Joint Training ---")
    for epoch in range(EPOCHS):
        # Call the standalone train function
        train(model, source_loader, target_train_loader, optimizer, criterion, epoch, device, scaler)

    # 7. Final Evaluation
    # Note: The test set should ALWAYS have data
    if not target_test_ids:
        print("WARNING: No test bugs found. Skipping evaluation.")
        final_metrics = {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MAP": 0, "MRR": 0}
    else:
        # 7a. Pre-process target testing data (with caching)
        if not os.path.exists(target_test_meta_path) or not os.path.exists(target_test_code_ids_path):
            print("Target test cache not found. Generating samples and preprocessing..")
            preprocess_and_cache_samples(target_test_samples, df_bugs, target_repo, tokenizer, target_test_meta_path, 
                                        target_test_code_ids_path)
        else:
            print(f"Loading preprocessed target test data from cache")
        final_metrics = evaluate(model, target_test_meta_path, target_test_code_ids_path, device)
    
    print(f"\n Final Metrics on Target Project: \n {final_metrics}\n")
    
    # 8. RETURN the metrics dict for the experiment runner
    return final_metrics
