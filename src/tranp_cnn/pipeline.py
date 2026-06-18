import os
import pandas as pd
import git
from git import Repo, Blob
from git.util import hex_to_bin
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
from torch.amp import autocast
import time 
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import the model definition from TRANPCNN model
from .model import TRANPCNN
from .data_prep import (
    create_labeled_samples, preprocess_and_cache_samples, load_project_databases, get_project_path,
    REPO_BASE_PATH, BUG_METADATA_DIR, BLOB_DB_DIR, BUG_REPORTS_PATH, BLOB_CACHE_DIR, PROJECTS_METADATA_PATH
)

# Configuration
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"


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
    'max_lines': 512, # Statements fed to model (cache stores 768; sliced in __getitem__)
    'max_line_len': 21, # Max tokens per statement
    'file_kernels': 100,
    'file_kernel_sizes': [3, 5, 7],
    'hidden_dim': 256,
    'num_classes': 2, # Buggy vs Non Buggy
    'dropout':0.5
}
# Lines stored per blob in existing .npy cache files.
# MUST match the max_lines used when the cache was built (data_prep.py PARAMS).
# Change PARAMS['max_lines'] above freely; change this only after a cache rebuild.
CACHE_MAX_LINES = 768

# Training Hyperparameters
EPOCHS = 20
BATCH_SIZE = 96
GRAD_ACCUM_STEPS = 2  # Effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 192 (same as original)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5 # For regularization


# Data Preparation & Pytorch Dataset
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

        # Reshape using cache dimensions, then truncate to model's max_lines.
        # Cache stores CACHE_MAX_LINES rows; model only needs PARAMS['max_lines'].
        # Slicing here avoids cache rebuild when tuning max_lines.
        code_ids_tensor = torch.from_numpy(code_ids_flat.copy()).reshape(
            CACHE_MAX_LINES, PARAMS['max_line_len']
        )[:PARAMS['max_lines']]

        # Convert pre-tokenized data (Stored as lists/value) back to tensors
        return {
            'bug_ids': torch.tensor(meta_sample['bug_ids'], dtype=torch.long),
            'code_ids': code_ids_tensor, # Used the reshaped tensor
            'label': torch.tensor(meta_sample['label'], dtype=torch.long),
            'project_type': meta_sample['project_type'],
            'bug_id': meta_sample['bug_id'],
            'blob_sha': meta_sample['blob_sha'],
            'faiss_score': torch.tensor(meta_sample['faiss_score'], dtype=torch.float32)
        }

# Step 3: Training and Evaluation Functions
def train(model, source_loader, target_loader, optimizer, criterion, epoch, device):
    model.train()

    # Scenario 1: Joint Training (CP-transfer)
    if source_loader and target_loader:
        print(f"\n--- Epoch {epoch + 1}: Joint Training on Source and Target Data ---")
        progress_bar = tqdm(source_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        iter_target = iter(itertools.cycle(target_loader))
        total_loss = 0.0
        num_batches = len(source_loader)

        optimizer.zero_grad(set_to_none=True)
        for step, source_batch in enumerate(progress_bar):
            target_batch = next(iter_target)

            bug_ids_s = source_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_s = source_batch['code_ids'].to(device, non_blocking=True)
            labels_s = source_batch['label'].to(device, non_blocking=True)
            bug_ids_t = target_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_t = target_batch['code_ids'].to(device, non_blocking=True)
            labels_t = target_batch['label'].to(device, non_blocking=True)

            # Sequential backward: source graph freed before target forward begins.
            # Divide by GRAD_ACCUM_STEPS so accumulated gradients equal a single
            # effective-batch step (effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS).
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_s = model(bug_ids_s, code_ids_s, 'source')
                loss_s = criterion(outputs_s, labels_s) / GRAD_ACCUM_STEPS
            loss_s.backward()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_t = model(bug_ids_t, code_ids_t, 'target')
                loss_t = criterion(outputs_t, labels_t) / GRAD_ACCUM_STEPS
            loss_t.backward()

            total_loss += (loss_s.item() + loss_t.item()) * GRAD_ACCUM_STEPS

            is_last = (step + 1 == num_batches)
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or is_last:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 10 == 0:
                progress_bar.set_postfix({'loss': f'{total_loss / (step + 1):.4f}'})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}")
        return avg_loss

    # Scenario 2: Source-Only Training (CP-cold-start)
    elif source_loader:
        print(f"\n--- Epoch {epoch + 1}: Training on Source Data Only ---")
        progress_bar = tqdm(source_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        total_loss = 0.0
        num_batches = len(source_loader)

        optimizer.zero_grad(set_to_none=True)
        for step, source_batch in enumerate(progress_bar):
            bug_ids_s = source_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_s = source_batch['code_ids'].to(device, non_blocking=True)
            labels_s = source_batch['label'].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_s = model(bug_ids_s, code_ids_s, 'source')
                loss_s = criterion(outputs_s, labels_s) / GRAD_ACCUM_STEPS
            loss_s.backward()

            total_loss += loss_s.item() * GRAD_ACCUM_STEPS

            is_last = (step + 1 == num_batches)
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or is_last:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 10 == 0:
                progress_bar.set_postfix({'loss': f'{total_loss / (step + 1):.4f}'})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}")
        return avg_loss

    # Scenario 3: Target-Only Training (WP-small, WP-large)
    elif target_loader:
        print(f"\n--- Epoch {epoch + 1}: Training on Target Data Only ---")
        progress_bar = tqdm(target_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        total_loss = 0.0
        num_batches = len(target_loader)

        optimizer.zero_grad(set_to_none=True)
        for step, target_batch in enumerate(progress_bar):
            bug_ids_t = target_batch['bug_ids'].to(device, non_blocking=True)
            code_ids_t = target_batch['code_ids'].to(device, non_blocking=True)
            labels_t = target_batch['label'].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_t = model(bug_ids_t, code_ids_t, 'target')
                loss_t = criterion(outputs_t, labels_t) / GRAD_ACCUM_STEPS
            loss_t.backward()

            total_loss += loss_t.item() * GRAD_ACCUM_STEPS

            is_last = (step + 1 == num_batches)
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or is_last:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 10 == 0:
                progress_bar.set_postfix({'loss': f'{total_loss / (step + 1):.4f}'})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}")
        return avg_loss

    else:
        raise ValueError("At least one of source_loader or target_loader must be provided for training.")

def evaluate(model, test_meta_path, test_code_ids_path, device, source_project, target_project, scenario):
    '''
    Evaluates the model on the pre-processed test set. Also performs diagnostic analysis comparing
    Faiss rank vs Model rank.
    '''
    model.eval()

    print("Loading pre-processed test data for evaluation...")
    # Use the same fast Dataset, but on the test cache files
    eval_dataset = BugLocalizationDataset(test_meta_path, test_code_ids_path)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Group samples by bug_id to evaluate each bug's ranked list
    # We can get this info from the loaded metadata
    bugs_to_evaluate = eval_dataset.metadata.groupby('bug_id')['blob_sha'].apply(list).to_dict()
    ground_truth_shas_df = eval_dataset.metadata[eval_dataset.metadata['label'] == 1]
    ground_truth_map = ground_truth_shas_df.groupby('bug_id')['blob_sha'].apply(set).to_dict()

    # Get model predictions for all candidates: {bug_id: [(blob_sha, model_score, faiss_score)]}
    predictions = {}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Scoring Test Candidates"):
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch['bug_ids'].to(device), batch['code_ids'].to(device), 'target')
            scores = F.softmax(outputs.float(), dim=1)[:, 1]

            faiss_scores_batch = batch['faiss_score'] # Get faiss scores from batch

            for i in range(len(batch['bug_id'])):
                bug_id = batch['bug_id'][i]
                if bug_id not in predictions:
                    predictions[bug_id] = []
                
                # Appenf all three pieces of info
                predictions[bug_id].append((batch['blob_sha'][i], scores[i].item(), faiss_scores_batch[i].item()))

    # 4. Calculate metrics
    top_k_hits = {k: 0 for k in [1, 5, 10]}
    average_precisions = [] 
    reciprocal_ranks = [] 

    diagnostic_results = [] # Store detailed ranking info

    for bug_id in bugs_to_evaluate:
        # Get all candidates for this bug
        preds = predictions.get(bug_id, [])
        if not preds:
            continue    # No predictions for this bug

        # Get the set of correct answers for this bug
        ground_truth = ground_truth_map.get(bug_id, set())
        if not ground_truth:
            continue    # No ground truth for this bug

        # RANKING ANALYSIS
        # 1. "Before" rank: Sort by Faiss score
        faiss_ranked_list = sorted(preds, key=lambda x: x[2], reverse=True) # Sort by faiss score

        # 2. "After" rank: Sort by Model score
        model_ranked_list = sorted(preds, key=lambda x: x[1], reverse=True) # Sort by model_score

        # --- STANDARD METRICS CALCULATION (uses model_ranked_list) ---
        ranked_shas = [sha for sha, _, _ in model_ranked_list]

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

        # --- DIAGNOSTIC CSV CALCULATION ---
        # Find the rank of *every* ground truth file in both lists
        for gt_sha in ground_truth:
            faiss_rank = -1
            model_rank = -1
            faiss_score = -1.0
            model_score = -1.0

            # Find in Faiss list
            for r, (sha, m_score, f_score) in enumerate(faiss_ranked_list):
                if sha == gt_sha:
                    faiss_rank = r + 1
                    faiss_score = f_score
                    model_score = m_score # Get model score for this sha
                    break
            
            # Find in Model list
            for r, (sha, m_score, f_score) in enumerate(model_ranked_list):
                if sha == gt_sha:
                    model_rank = r + 1
                    model_score = m_score # This should be the same
                    # If not found in faiss list (e.g. score was -1), get score from here
                    if faiss_rank == -1: 
                        faiss_score = f_score
                    break
            
            # Handle case where GT was not retrieved by Faiss at all
            if faiss_rank == -1:
                # Check if it was in the original 'preds' list at all
                # If it's not, it means it wasn't even in the top-K
                found_in_preds = False
                for (sha, m_score, f_score) in preds:
                     if sha == gt_sha:
                        faiss_score = f_score
                        model_score = m_score
                        found_in_preds = True
                        break
                # If still not found, it means it wasn't in the Top-K candidates
                # (This should not happen if is_training_data=False logic is correct,
                # but good to be safe)
                
            
            rank_displacement = -1
            if faiss_rank != -1 and model_rank != -1:
                rank_displacement = faiss_rank - model_rank # Positive = improved
            
            diagnostic_results.append({
                'bug_id': bug_id,
                'gt_blob_sha': gt_sha,
                'faiss_rank': faiss_rank,
                'model_rank': model_rank,
                'rank_displacement': rank_displacement,
                'faiss_score': faiss_score,
                'model_score': model_score
            })

    # --- SAVE DIAGNOSTIC FILE ---
    if diagnostic_results:
        diag_df = pd.DataFrame(diagnostic_results)
        # Create a unique, safe filename
        src_safe = source_project.replace('/', '_')
        tgt_safe = target_project.replace('/', '_')
        diag_filename = os.path.join(RESULT_PATH, 'tranp_cnn_ph1_diagnostics', f"{src_safe}_{tgt_safe}_{scenario}_diagnostics.csv")
        try:
            diag_df.to_csv(diag_filename, index=False)
            print(f"✓ Saved ranking diagnostics to {diag_filename}")
        except Exception as e:
            print(f"Error saving diagnostics: {e}")


    num_bugs = len(bugs_to_evaluate)
    if num_bugs == 0:
        print("Warning: num_bugs is 0. No metrics to calculate.")
        return {"Top-1": 0, "Top-5": 0, "Top-10": 0, "MAP": 0, "MRR": 0}

    final_metrics = {
        "Top-1": round(top_k_hits[1] / num_bugs, 4),
        "Top-5": round(top_k_hits[5] / num_bugs, 4),
        "Top-10": round(top_k_hits[10] / num_bugs, 4),
        "MAP": round(np.mean(average_precisions), 4),
        "MRR": round(np.mean(reciprocal_ranks), 4)
    }
    
    return final_metrics


def evaluate_validation(model, source_val_loader, target_val_loader, criterion, device):
    '''
    Calculates the validation loss for early stopping.
    '''
    model.eval() # Set model to evaluation mode
    total_val_loss = 0.0
    val_batch_count = 0

    with torch.no_grad():  # No gradients needed for validation
        # Scenario 1: Joint Validation (if both loaders exist)
        if source_val_loader and target_val_loader:
            # Determine the number of batches to run (use the longer loader)
            num_batches = max(len(source_val_loader), len(target_val_loader))
            iter_source = iter(itertools.cycle(source_val_loader))
            iter_target = iter(itertools.cycle(target_val_loader))

            print(f" -> Evaluating on joing validation set ({num_batches} steps)...")
            progress_bar = tqdm(range(num_batches), desc="Validation", leave=False)

            for _ in progress_bar:
                source_batch = next(iter_source)
                target_batch = next(iter_target)

                bug_ids_s = source_batch['bug_ids'].to(device, non_blocking=True)
                code_ids_s = source_batch['code_ids'].to(device, non_blocking=True)
                labels_s = source_batch['label'].to(device, non_blocking=True)

                bug_ids_t = target_batch['bug_ids'].to(device, non_blocking=True)
                code_ids_t = target_batch['code_ids'].to(device, non_blocking=True)
                labels_t = target_batch['label'].to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs_s = model(bug_ids_s, code_ids_s, 'source')
                    loss_s = criterion(outputs_s, labels_s)
                    outputs_t = model(bug_ids_t, code_ids_t, 'target')
                    loss_t = criterion(outputs_t, labels_t)
                combined_loss = loss_s + loss_t

                total_val_loss += combined_loss.item()
                val_batch_count += 1

        # Scenario 2: Source-Only Validation
        elif source_val_loader:
            print(f" -> Evaluating on source validation set ({len(source_val_loader)} steps)...")
            progress_bar = tqdm(source_val_loader, desc="Validation", leave=False)

            for source_batch in progress_bar:
                bug_ids_s = source_batch['bug_ids'].to(device, non_blocking=True)
                code_ids_s = source_batch['code_ids'].to(device, non_blocking=True)
                labels_s = source_batch['label'].to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs_s = model(bug_ids_s, code_ids_s, 'source')
                    loss_s = criterion(outputs_s, labels_s)
                total_val_loss += loss_s.item()
                val_batch_count += 1

        # Scenario 3: Target-Only Validation
        elif target_val_loader:
            print(f" -> Evaluating on target validation set ({len(target_val_loader)} steps)...")
            progress_bar = tqdm(target_val_loader, desc="Validation", leave=False)

            for target_batch in progress_bar:
                bug_ids_t = target_batch['bug_ids'].to(device, non_blocking=True)
                code_ids_t = target_batch['code_ids'].to(device, non_blocking=True)
                labels_t = target_batch['label'].to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs_t = model(bug_ids_t, code_ids_t, 'target')
                    loss_t = criterion(outputs_t, labels_t)
                total_val_loss += loss_t.item()
                val_batch_count += 1

        else:
            # No validation loaders provided
            print(" -> No validation data loaders found. SKippng validation.")
            return float('inf') # Return infinity if no validation possible

    if val_batch_count == 0:
        return float('inf')

    avg_val_loss = total_val_loss / val_batch_count
    print(f" -> Avg validation loss: {avg_val_loss:.4f}")
    return avg_val_loss


# Step 4: Main Orchestrator
def run_tranp_cnn_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids, scenario, validation_split_ratio=0.2):
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
    source_bug_db, source_blob_db, source_language = load_project_databases(source_project, df_meta)
    target_bug_db, target_blob_db, target_language = load_project_databases(target_project, df_meta)

    # Adding validation split
    print(f"\n Splitting training data using {validation_split_ratio*100:.0f}% for validation...")

    source_train_final_ids, source_val_ids = [], []
    if source_train_ids:
        if len(source_train_ids) * validation_split_ratio < 1:
            print("Warning: Not enough source train data for validation split. Using all for training.")
            source_train_final_ids = source_train_ids
        else:
            source_train_final_ids, source_val_ids = train_test_split(
                source_train_ids, test_size=validation_split_ratio, random_state=42
            )
        print(f" - Source: {len(source_train_final_ids)} train, {len(source_val_ids)} validation bugs")
    
    target_train_final_ids, target_val_ids = [], []
    if target_train_ids:
        if len(target_train_ids) * validation_split_ratio < 1:
            print("Warning: Not enough target train data for validation split. Using all for training.")
            target_train_final_ids = target_train_ids
        else:
            target_train_final_ids, target_val_ids = train_test_split(
                target_train_ids, test_size=validation_split_ratio, random_state=42
            )
        print(f" - Target: {len(target_train_final_ids)} train, {len(target_val_ids)} validation bugs")

    # Define all cache paths (train, val, test)
    os.makedirs(BLOB_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, 'tranp_cnn_ph1_diagnostics'), exist_ok=True)
    src_proj_safe = source_project.replace("/", "_")
    tgt_proj_safe = target_project.replace("/", "_")

    # Training Cache Paths
    source_train_meta_path = os.path.join(BLOB_CACHE_DIR, f'{src_proj_safe}_{scenario}_source_train_meta.parquet')
    source_train_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{src_proj_safe}_{scenario}_source_train_code_ids.npy')
    target_train_meta_path = os.path.join(BLOB_CACHE_DIR, f'{tgt_proj_safe}_{scenario}_target_train_meta.parquet')
    target_train_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{tgt_proj_safe}_{scenario}_target_train_code_ids.npy')

    # Validation Cache Paths
    source_val_meta_path = os.path.join(BLOB_CACHE_DIR, f'{src_proj_safe}_{scenario}_source_val_meta.parquet')
    source_val_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{src_proj_safe}_{scenario}_source_val_code_ids.npy')
    target_val_meta_path = os.path.join(BLOB_CACHE_DIR, f'{tgt_proj_safe}_{scenario}_target_val_meta.parquet')
    target_val_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{tgt_proj_safe}_{scenario}_target_val_code_ids.npy')

    # Test Cache Paths (remain the same)
    target_test_meta_path = os.path.join(BLOB_CACHE_DIR, f'{tgt_proj_safe}_{scenario}_target_test_meta.parquet')
    target_test_code_ids_path = os.path.join(BLOB_CACHE_DIR, f'{tgt_proj_safe}_{scenario}_target_test_code_ids.npy')
    
    # --- Generate RAW samples (train, val, test) ---
    print("\nGenerating raw samples...")
    # Source Train
    source_train_samples = create_labeled_samples(source_project, target_project, source_language, source_bug_db, source_blob_db, 
                                            TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=source_train_final_ids) # Use final train ids
    # Source Validation
    source_val_samples = create_labeled_samples(source_project, target_project, source_language, source_bug_db, source_blob_db, 
                                            TOP_K_CANDIDATES, is_training_data=False, bug_ids_to_process=source_val_ids) # Use val ids, is_training_data=False
    
    # Target Train
    target_train_samples = create_labeled_samples(
        target_project, target_project, target_language, target_bug_db, target_blob_db, 
        TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=target_train_final_ids # Use final train ids
    )
    # Target Validation
    target_val_samples = create_labeled_samples(
        target_project, target_project, target_language, target_bug_db, target_blob_db, 
        TOP_K_CANDIDATES, is_training_data=False, bug_ids_to_process=target_val_ids # Use val ids, is_training_data=False
    )
    # Target Test (remains the same)
    target_test_samples = create_labeled_samples(
        target_project, target_project, target_language, target_bug_db, target_blob_db, 
        TOP_K_CANDIDATES, is_training_data=False, bug_ids_to_process=target_test_ids 
    )
    
    source_repo = git.Repo(get_project_path(source_project, source_language))
    target_repo = git.Repo(get_project_path(target_project, target_language))
    
    
    # Process and Load Data (Train, Val)
    # 4. Source Train
    print("\nProcessing Source Train data...")
    if source_train_final_ids: # Use final ids
        if not os.path.exists(source_train_meta_path) or not os.path.exists(source_train_code_ids_path):
            print("Source train cache not found. Preprocessing...")
            preprocess_and_cache_samples(source_train_samples, df_bugs, source_repo, tokenizer, source_train_meta_path, source_train_code_ids_path)
        source_train_dataset = BugLocalizationDataset(source_train_meta_path, source_train_code_ids_path)
        print(f"  - Source train samples: {len(source_train_dataset)}")
        source_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        print("  - No source training samples.")
        source_loader = None # Use None instead of [] for clarity

    # 4a. Source Validation
    print("Processing Source Validation data...")
    if source_val_ids:
        if not os.path.exists(source_val_meta_path) or not os.path.exists(source_val_code_ids_path):
            print("Source validation cache not found. Preprocessing...")
            preprocess_and_cache_samples(source_val_samples, df_bugs, source_repo, tokenizer, source_val_meta_path, source_val_code_ids_path)
        source_val_dataset = BugLocalizationDataset(source_val_meta_path, source_val_code_ids_path)
        print(f"  - Source validation samples: {len(source_val_dataset)}")
        # No shuffle for validation
        source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        print("  - No source validation samples.")
        source_val_loader = None

    # 4b. Target Train
    print("Processing Target Train data...")
    if target_train_final_ids: # Use final ids
        if not os.path.exists(target_train_meta_path) or not os.path.exists(target_train_code_ids_path):
            print("Target train cache not found. Preprocessing...")
            preprocess_and_cache_samples(target_train_samples, df_bugs, target_repo, tokenizer, target_train_meta_path, target_train_code_ids_path)
        target_train_dataset = BugLocalizationDataset(target_train_meta_path, target_train_code_ids_path)
        print(f"  - Target train samples: {len(target_train_dataset)}")
        target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        print("  - No target training samples.")
        target_train_loader = None

    # 4c. Target Validation
    print("Processing Target Validation data...")
    if target_val_ids:
        if not os.path.exists(target_val_meta_path) or not os.path.exists(target_val_code_ids_path):
            print("Target validation cache not found. Preprocessing...")
            preprocess_and_cache_samples(target_val_samples, df_bugs, target_repo, tokenizer, target_val_meta_path, target_val_code_ids_path)
        target_val_dataset = BugLocalizationDataset(target_val_meta_path, target_val_code_ids_path)
        print(f"  - Target validation samples: {len(target_val_dataset)}")
        target_val_loader = DataLoader(target_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        print("  - No target validation samples.")
        target_val_loader = None
        
    print("\nData loading and processing complete.")
    
    print("Data loading checks complete.") # <-- New print statement
    
    # 5. Initialize Model, Criterion, Optimizer
    model = TRANPCNN(PARAMS).to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Calculate class weights (using ONLY final training data)
    print("\nCalculating class weights for loss function...")
    labels_s_df = pd.DataFrame(columns=['label']) 
    labels_t_df = pd.DataFrame(columns=['label']) 
    try:
        if os.path.exists(source_train_meta_path) and os.path.getsize(source_train_meta_path) > 0:
             labels_s_df = pd.read_parquet(source_train_meta_path, columns=['label'])
    except Exception as e:
        print(f"Warning: could not read source train parquet: {e}")
    try:
        if os.path.exists(target_train_meta_path) and os.path.getsize(target_train_meta_path) > 0:
            labels_t_df = pd.read_parquet(target_train_meta_path, columns=['label'])
    except Exception as e:
        print(f"Warning: could not read target train parquet: {e}")

    all_train_labels = pd.concat([labels_s_df['label'], labels_t_df['label']])
    # ... (rest of weight calculation remains the same) ...
    if all_train_labels.empty:
        print("Warning: No training labels found. Using default weights [1.0, 1.0]")
        count_0 = 1
        count_1 = 1
    else:
        counts = all_train_labels.value_counts()
        count_0 = counts.get(0, 1) # Get count for label 0, default to 1
        count_1 = counts.get(1, 1) # Get count for label 1
    
    total_samples = count_0 + count_1
    
    weight_0 = total_samples / (2.0 * count_0)
    weight_1 = total_samples / (2.0 * count_1)

    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
    
    print(f"  - Total Train Samples (for weights): {total_samples}")
    print(f"  - Labels 0 (neg): {count_0} (Weight: {weight_0:.2f})")
    print(f"  - Labels 1 (pos): {count_1} (Weight: {weight_1:.2f})")

    # Enable cuDNN autotuner for potential speedups on fixed-size inputs
    torch.backends.cudnn.benchmark = True
    # Pass the calculated weights to the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Use AdamW with fused=True for faster CUDA kernels
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)

    # --- Early Stopping Variables (use validation loss) ---
    best_val_loss = float('inf') # Now track validation loss
    patience_counter = 0
    patience_limit = 3 # Stop after 3 epochs with no improvement in validation loss
    best_model_state = None 
    best_epoch = -1

    # 6. Training Loop with Validation and Early Stopping
    print(f"\n--- Starting Training (Max {EPOCHS} Epochs) ---")
    for epoch in range(EPOCHS):
        
        # --- Run Training Epoch ---
        # Pass the FINAL training loaders
        _ = train(model, source_loader, target_train_loader, optimizer, criterion, epoch, device)
        
        # --- Run Validation Epoch ---
        # Pass the validation loaders
        current_val_loss = evaluate_validation(model, source_val_loader, target_val_loader, criterion, device)
        
        # --- EARLY STOPPING LOGIC (use validation loss) ---
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0 
            print(f"  -> Validation loss improved to {best_val_loss:.4f}.")
            best_model_state = copy.deepcopy(model.state_dict()) # Use deepcopy for safety
            best_epoch = epoch + 1
            # Optional: Save best model state to disk here
            # torch.save(model.state_dict(), f'best_model_{scenario}.pth')
        else:
            patience_counter += 1
            print(f"  -> Validation loss did not improve for {patience_counter} epoch(s).")

        if patience_counter >= patience_limit:
            print(f"\nEarly stopping triggered after epoch {epoch + 1}.")
            break

    # Load the best model state before final evaluation
    if best_model_state is not None:
        print(f"\nLoading model from best epoch: {best_epoch} with validation loss {best_val_loss:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("\nWarning: No improvement in validation loss observed. Using model from the last epoch.")
    
    # 7. Final Evaluation on TEST set
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
        final_metrics = evaluate(model, target_test_meta_path, target_test_code_ids_path, device,
                                 source_project, target_project, scenario)
    
    print(f"\n Final Metrics on Target Project: \n {final_metrics}\n")

    # Release PyTorch allocator cache so the next experiment starts with a clean
    # high-water mark. Critical when two shards share the same GPU.
    torch.cuda.empty_cache()

    # 8. RETURN the metrics dict for the experiment runner
    return final_metrics
