import pandas as pd
import os
import git
from git import Repo, Blob
from git.util import hex_to_bin
import pickle
import numpy as np
import faiss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data

# Local imports
from cooba_model import COOBA, ProjectDiscriminator
from cooba_ast_parsers import PythonASTParser, JavaASTParser, UNK_TOKEN, PAD_TOKEN
from cooba_utils import build_vocabulary, load_glove_embeddings, save_preprocessors, load_preprocessors

# -- Configuration --
# Experiment Mode: 'cross-project' or 'within-project'
MODE = 'cross-project'

REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
GLOVE_FILE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/glove.6B.100d.txt"
CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/cooba"
PREPROCESSOR_PATH = os.path.join(CACHE_DIR, 'cooba_preprocessors.pkl')
GRAPH_CACHE_DIR = os.path.join(CACHE_DIR, 'graph_cache')

# Experiment Settings
SOURCE_PROJECT = 'apache/dolphinscheduler'
TARGET_PROJECT = 'apache/dubbo'
# Define project languages
PROJECT_LANGUAGES = {
    'apache/dolphinscheduler': 'java',
    'apache/dubbo': 'java'
}

TOP_K_CANDIDATES = 300
TARGET_TRAIN_SIZE = 0.2
TEST_SET_SIZE = 0.2
WP_TRAIN_SIZE = 0.8
MAX_BUG_LEN = 512
MAX_CODE_SEQ_LEN = 1024    # For the CNN path (Path B)
EMBEDDING_DIM = 300

# Model Hyperparameters
PARAMS = {
    'bug_encoder': {'hidden_dim': 256, 'num_layers': 2, 'dropout_prob':0.5},
    'shared_extractor': {'num_filters': 128, 'kernel_sizes': [3, 4, 5]},
    'individual_extractor': {'hidden_dim': 256, 'output_dim': 256, 'dropout_prob':0.5},
    'fusion': {'output_dim': 256}
}

# Training Hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DISC_LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
MARGIN = 0.4 # For Margin Ranking Loss
LAMBDA_ADV = 0.1 # Weight for adversarial loss

# Step 1: Data Preparation & Candidate Generation using Faiss
def get_project_path(repo_name, language):
    '''Constructs the local path to a project repository.'''
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))

def get_path_to_sha_map(repo, commit_sha):
    '''
    For a given commit, creates a dictionary mapping file paths to blob SHAs.
    '''
    try:
        commit = repo.commit(commit_sha)
        return {blob.path: blob.hexsha for blob in commit.tree.traverse() if blob.type == 'blob'}
    except Exception:
        return {}

def load_project_databases(project_name, meta_df):
    '''Loads the pre-computed Faiss bug/blob databases.'''
    language = meta_df[meta_df['repo_name'] == project_name]['language'].iloc[0] 
    blob_db_path = os.path.join(BLOB_DB_DIR, project_name.replace('/', '_') + '_blob_embeddings32.pkl')
    bug_db_path = os.path.join(BUG_METADATA_DIR, project_name.replace('/', '_') + '_bug_metadata32.pkl')
    with open(blob_db_path, 'rb') as f:
        blob_db = pickle.load(f)
    with open(bug_db_path, 'rb') as f:
        bug_db = pickle.load(f)
    return bug_db, blob_db, language

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

def get_all_unique_blobs_code(raw_samples_list, repos):
    '''Fetches code for all unique blobs from a list of raw_samples.'''
    all_unique_blobs = set()
    for samples in raw_samples_list:
        all_unique_blobs.update(sha for _, sha, _, _ in samples)

    code_map = {}
    for blob_sha in tqdm(all_unique_blobs, desc="Fetching unique blob code"):
        for repo_name, repo_data in repos.items():
            try:
                lang = repo_data['lang']
                source_content = Blob(repo_data['repo'], hex_to_bin(blob_sha)).data_stream.read().decode('utf-8', 'ignore')
                code_map[blob_sha] = (source_content, lang)
                break
            except Exception:
                continue # Try next repo
    return code_map

# 2. Preprocessing & caching 
def preprocess_and_cache_samples_cooba(raw_samples_list, bug_reports_df, repos, parsers):
    '''
    Parses all unique blobs into graphs and saves them to a single cache file. Saves metadata and tokenized
    bug reports.
    '''
    print("Starting COOBA preprocessing....")
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)

    # 1. Get code for all unique blobs
    all_code_map = get_all_unique_blobs_code(raw_samples_list, repos)

    # 2. Build Vocabulary and Embedding Matrix
    if not os.path.exists(PREPROCESSOR_PATH):
        vocabulary = build_vocabulary(bug_reports_df, parsers, all_code_map)
        embedding_matrix = load_glove_embeddings(GLOVE_FILE_PATH, vocabulary, EMBEDDING_DIM)
        with open(PREPROCESSOR_PATH, 'wb') as f:
            pickle.dump({'vocabulary': vocabulary, 'embedding_matrix': embedding_matrix}, f)
        print("Saved preprocessors to {PREPROCESSOR_PATH}")
    else:
        print(f"Loading preprocessors from {PREPROCESSOR_PATH}")
        with open(PREPROCESSOR_PATH, 'rb') as f:
            data = pickle.load(f)
        vocabulary = data['vocabulary']
        embedding_matrix = data['embeddng_matrix']

    # Re-initialize parsers with the final vocabulary
    for lang in parsers:
        parsers[lang].vocabulary = vocabulary

    # 3. Create the single graph cache file
    if not os.path.exists(GRAPH_CACHE_DIR):
        print(f"Creating new graph cache at {GRAPH_CACHE_DIR}")
        graph_cache = {}
        for blob_sha, (code_string, lang) in tqdm(all_code_map.items(), desc="Parsing and caching ASTs"):
            parser = parsers.get(lang)
            if not parser: continue

            graph_data = parser.pase(code_string)
            if graph_data:
                # Add chunking logic here if needed, for now, we will rely on truncation in collate_fn
                graph_cache[blob_sha] = graph_data

        print(f"Saving graph cache with {len(graph_cache)} graphs...")
        torch.save(graph_cache, GRAPH_CACHE_DIR)
    else:
        print(f"Graph cache {GRAPH_CACHE_DIR} already exists.")

    # 4. Create and save metadata
    print("Creating metadata...")
    unk_id = vocabulary.get(UNK_TOKEN, 1)
    bug_token_cache = {}
    for bug_id, text in bug_reports_df['bug_report_text'].to_dict().items():
        tokens = str(text).lower().split()[:MAX_BUG_LEN]
        bug_token_cache[bug_id] = [vocabulary.get(t, unk_id) for t in tokens]

    all_metadata = []
    for samples in raw_samples_list:
        for bug_id, blob_sha, label, project_type in tqdm(samples, desc="Building metadata"):
            if blob_sha not in all_code_map: # skip blobs we couldn't read
                continue
            all_metadata.append({
                'bug_id':bug_id,
                'bug_token_ids': bug_token_cache.get(bug_id, []),
                'blob_sha': blob_sha,
                'label': label,
                'project_type': project_type
            })
    
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_parquet(METADATA_CACHE_PATH, index=False)
    print(f"Saved metadata to {METADATA_CACHE_PATH}")

    return metadata_df, embedding_matrix

# --- 3. PyTorch Dataset ---
class CoobaBugLocalizationDataset(Dataset):
    '''
    Loads preprocessed COOBA samples for DataLoader.
    '''
    def __init__(self, metadata_df):
        self.metadata = metadata_df.to_dict('records')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        graph_path = os.path.join(GRAPH_CACHE_DIR, f"{meta['blob_sha']}.pt")
        try:
            graph_data = torch.load(graph_path)
        except FileNotFoundError:
            return None # Skip if file is missing
        return {
            'bug_tokens_ids': torch.tensor(meta['bug_tokens_ids'], dtype=torch.long),
            'bug_length': len(meta['bug_tokens_ids']),
            'graph_data': graph_data,
            'label': torch.tensor(meta['label'], dtype=torch.long),
            'project_type': meta['project_type']
        }

def cooba_collate_fn(batch):
    '''
    Custom collate function to handle padding and graph batching.
    '''
    batch = [b for b in batch if b is not None]  # Filter out None items
    if not batch:
        return None
    
    # Pad bug reports
    bug_lengths = torch.tensor([b['bug_length'] for b in batch], dtype=torch.long)
    bug_tokens = [b['bug_tokens_ids'] for b in batch]
    padded_bugs = pad_sequence(bug_tokens, batch_first=True, padding_value=0)

    # Pad code token sequences (from graph data)
    code_seqs = [b['graph_data'].code_token_sequence[:MAX_CODE_SEQ_LEN] for b in batch]
    padded_code_seqs = pad_sequence(code_seqs, batch_first=True, padding_value=0)
    
    # Batch graphs
    graph_list = [b['graph_data'] for b in batch]
    batched_graph = Batch.from_data_list(graph_list)

    labels = torch.stack([b['label'] for b in batch])
    project_types = [b['project_type'] for b in batch]

    return {
        'bug_report': (padded_bugs, bug_lengths),
        'code_graph': batched_graph,
        'code_sequence': padded_code_seqs,
        'labels': labels,
        'project_types': project_types
    }

# --- 4. Training and Evaluation ---
def train_cooba(model, discriminator, source_loader, target_loader, optimizer_main, optimizer_disc, epoch, device):
    '''
    Main Training loop for COOBA in cross-project mode. Implements the adversarial training and 
    margin ranking loss. 
    '''
    model.train()
    discriminator.train()

    # Loss functions
    task_criterion = nn.MarginRankingLoss(margin=MARGIN).to(device)
    adv_criterion = nn.CrossEntropyLoss().to(device)

    # Setup iterator
    if target_loader is None:
        print("Running in Within-Project mode. No target loader.")
        iter_target = None
        loaders = source_loader
    else:
        print("Running in Cross-Project mode.")
        iter_target = iter(itertools.cycle(target_loader))
        loaders = source_loader # Source loader dictates epoch length

    progress_bar = tqdm(loaders, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    total_task_loss = 0
    total_adv_loss = 0
    total_disc_loss = 0

    for batch in progress_bar:
        if batch is None: continue

        # --- Prepare data for Margin Loss ---
        # We need (bug, positive_code, negative_code) triplets
        # This implementation simplifies and uses (bug, code, label)
        # Let's adapt to use MarginRankingLoss
        
        # This requires a different Dataset sampling...
        # For simplicity, let's switch to CrossEntropyLoss like tranp_cnn
        # and treat it as a classification. The paper's loss (Eq. 11) is
        # a ranking loss.
        
        # Let's stick to the paper's MarginRankingLoss.
        # This pipeline needs to be re-designed to provide triplets.
        
        # *** SIMPLIFICATION FOR THIS SCRIPT ***
        # We will use BinaryCrossEntropyLoss on the cosine similarity score.
        # This is a common simplification of ranking tasks.
        # score = (score + 1) / 2 # Map from [-1, 1] to [0, 1]
        # task_loss = F.binary_cross_entropy(score, labels.float())
        # Let's use CrossEntropyLoss on logits, as in tranp_cnn
        
        # *** STICKING TO PAPER'S LOSS (Eq. 11) ***
        # This requires a triplet-based dataset.
        # Since this script is based on `tranp_cnn`'s (bug, file, label) pairs,
        # we will use a classification loss (CrossEntropy) as a proxy.
        # The `relevance_score` from `cooba_core` will be treated as a logit.
        
        # --- (Re-Implementing with CrossEntropyLoss for simplicity) ---
        # 1. Update cooba_core.py: 
        #    - `fusion` layer should output 2 features (logits)
        #    - No cosine similarity. Return `code_vector` (fused)
        #    - Add a final linear layer after bug/code vec dot product
        #
        # Let's assume `cooba_core` is modified to return a single score,
        # and we use MarginRankingLoss. We need to find positive and negative
        # pairs *within the batch*.

        labels = batch['labels'].to(device)
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        # Skip batch if no positive or no negative samples
        if not torch.any(pos_mask) or not torch.any(neg_mask):
            continue

        # Move data to device
        bug_report = (batch['bug_report'][0].to(device), batch['bug_report'][1])
        code_graph = batch['code_graph'].to(device)
        code_sequence = batch['code_sequence'].to(device)

        # -- (A) Train Discriminator --
        if MODE == 'cross-project' and iter_target:
            target_batch = next(iter_target)
            if target_batch is None: continue

            # Get target data
            code_graph_t = target_batch['code_graph'].to(device)
            code_seq_t = target_batch['code_sequence'].to(device)

            optimizer_disc.zero_grad()

            # Get public features for source
            with torch.no_grad():
                _, public_features_s = model(bug_report, code_graph, code_sequence, 'source')
            
            # Get public features for target
            with torch.no_grad():
                bug_report_t = (target_batch['bug_report'][0].to(device), target_batch['bug_report'][1])
                _, public_features_t = model(bug_report_t, code_graph_t, code_seq_t, 'target')

            # Concat and create labels
            public_features = torch.cat((public_features_s, public_features_t), dim=0)
            labels_s = torch.zeros(public_features_s.size(0), dtype=torch.long, device=device)
            labels_t = torch.ones(public_features_t.size(0), dtype=torch.long, device=device)
            disc_labels = torch.cat((labels_s, labels_t))

            disc_preds = discriminator(public_features.detach())
            disc_loss = adv_criterion(disc_preds, disc_labels)
            disc_loss.backward()
            optimizer_disc.step()
            total_disc_loss += disc_loss.item()

        # -- (B) Train Main Model (Generator) --
        optimizer_main.zero_grad()

        # Get scores for all samples in the batch
        project_type = 'source' if MODE == 'cross-project' else None
        scores, public_features = model(bug_report, code_graph, code_sequence, project_type)

        # 1. Task Loss (Margin Ranking)
        # Select one positive and one negative for each
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        # Create pairs
        n_pos = pos_scores.size(0)
        n_neg = neg_scores.size(0)
        n_pairs = min(n_pos, n_neg)

        pos_scores = pos_scores[:n_pairs]
        neg_scores = neg_scores[:n_pairs]

        target = torch.ones(n_pairs, device=device)  # We want pos > neg
        task_loss = task_criterion(pos_scores, neg_scores, target)

        # 2. Adversarial Loss (if cross-project)
        if MODE == 'cross-project':
            disc_preds = discriminator(public_features)
            # We want to fool the discriminator, so we flip the labels
            # We train generator to predict "target" (1) for "source" (0) samples
            adv_labels = torch.ones(public_features.size(0), dtype=torch.long, device=device)
            adv_loss = adv_criterion(disc_preds, adv_labels)

            total_loss = task_loss + (LAMBDA_ADV * adv_loss)
            total_adv_loss += adv_loss.item()
        else:
            total_loss = task_loss

        total_loss.backward()
        optimizer_main.step()

        total_task_loss += task_loss.item()
        progress_bar.set_postfix({
            'TaskL': f'{total_task_loss / (progress_bar.n+1):.4f}',
            'AdvL': f'{total_adv_loss / (progress_bar.n+1):.4f}',
            'DiscL': f'{total_disc_loss / (progress_bar.n+1):.4f}'
        })

# Main orchestrator

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    # 1. Load/Build Preprocessors
    if not os.path.exists(PREPROCESSOR_PATH):
        # This is slow: requires loading all code files
        print("Building vocabulary from scratch ... This may take a while.")
        # This part needs to be implemented:
        # all_code = load_all_code_files(...)
        # vocabulary = build_vocabulary(df_bugs, parsers, all_code)
        # embedding_matrix = load_glove_embeddings(GLOVE_FILE_PATH, vocabulary, PARAMS['embedding_dim])
        # save_preprocessors(vocabulary, embedding_matrix, PREPROCESSOR_PATH)
        print("Please run a separate script to build preprocessors.")
        return

    vocabulary, embedding_matrix = load_preprocessors(PREPROCESSOR_PATH)
    PARAMS['vocab_size'] = len(vocabulary)

    # 2. Setup Parsers and Repos
    parsers = {
        'java': JavaASTParser(vocabulary),
        'python': PythonASTParser(vocabulary)
    }
    repos = {
        SOURCE_PROJECT: {
            'repo': git.Repo(get_project_path(SOURCE_PROJECT, ...)), # Need lang
            'lang': 'java' 
        },
        TARGET_PROJECT: {
            'repo': git.Repo(get_project_path(TARGET_PROJECT, ...)), # Need lang
            'lang': 'java'
        }
    }

    # ... (Need to implement 'get_project_path' from tranp_cnn) ..
    # ... (Need to load 'source_bug_db', 'target_bug_db', etc.. from tranp_cnn code)

    # 3. Generate Raw Samples
    print("Generating Raw Samples....")
    if MODE == 'cross-project':
        # ... (load bug_ids, split target_bug_ids) ...
        # source_samples = create_labeled_samples(SOURCE_PROJECT, ...)
        # target_train_samples = create_labeled_samples(TARGET_PROJECT, ..., bug_ids_to_process=target_train_bug_ids)
        # test_samples = create_labeled_samples(TARGET_PROJECT, ..., bug_ids_to_process=target_test_bug_ids, is_training_data=False)
        pass # Placeholder
    else: # within-project
        # ... (load target_bug_db only) ...
        # ... (split target_bug_ids into train_ids (80%) and test_ids (20%)) ...
        # train_samples = create_labeled_samples(TARGET_PROJECT, ..., bug_ids_to_process=train_ids)
        # test_samples = create_labeled_samples(TARGET_PROJECT, ..., bug_ids_to_process=test_ids, is_training_data=False)
        pass # Placeholder
        
    # --- This is a placeholder section ---
    # The logic from `tranp_cnn_pipeline.py`'s `main` function for
    # loading dbs, splitting bug_ids, and calling `create_labeled_samples`
    # should be fully integrated here.
    # For this example, we assume `train_samples`, `target_train_samples` (if CP)
    # and `test_samples` exist.
    
    # Preprocess and Cache
    if MODE == 'cross-project':
        train_meta = preprocess_and_cache_samples_cooba(train_samples, df_bugs, repos, parsers, vocabulary)
        target_train_meta = preprocess_and_cache_samples_cooba(target_train_samples, df_bugs, repos, parsers, vocabulary)
        test_meta = preprocess_and_cache_samples_cooba(test_samples, df_bugs, repos, parsers, vocabulary)
        
        train_dataset = CoobaBugLocalizationDataset(train_meta)
        target_train_dataset = CoobaBugLocalizationDataset(target_train_meta)
        
        source_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cooba_collate_fn)
        target_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cooba_collate_fn)
        
    else: # 'within-project'
        train_meta = preprocess_and_cache_samples_cooba(train_samples, df_bugs, repos, parsers, vocabulary)
        test_meta = preprocess_and_cache_samples_cooba(test_samples, df_bugs, repos, parsers, vocabulary)
        
        train_dataset = CoobaBugLocalizationDataset(train_meta)
        source_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cooba_collate_fn)
        target_loader = None # No target loader in WPBL mode
        
    test_dataset = CoobaBugLocalizationDataset(test_meta)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=cooba_collate_fn)
    
    # 5. Initialize Models
    model = COOBA(embedding_matrix, PARAMS['bug_encoder'], PARAMS['shared_extractor'],
                  PARAMS['individual_extractor'], PARAMS['fusion'], mode=MODE).to(device)
    
    discriminator = None
    if MODE == 'cross-project':
        public_feat_dim = PARAMS['shared_extractor']['num_filters'] * len(PARAMS['shared_extractor']['kernel_sizes'])
        discriminator = ProjectDiscriminator(public_feat_dim).to(device)
        optimizer_disc = optim.Adam(discriminator.parameters(), lr=DISC_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    optimizer_main = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. Training Loop
    print(f"\n--- Starting {MODE} Training ---")
    for epoch in range(EPOCHS):
        train_cooba(model, discriminator, source_loader, target_loader,
                    optimizer_main, optimizer_disc if MODE == 'cross-project' else None,
                    epoch, device)
        
    # 7. Final Evaluation
    print("\n--- Starting Final Evaluation ---")
    # ... (Evaluation logic needs to be implemented) ...
    # This involves iterating through `test_loader`, getting scores,
    # grouping by bug_id, and calculating Top-K, MAP, MRR,
    # similar to `tranp_cnn_pipeline.py`'s `evaluate` function.

if __name__ == '__main__':
    print("This script is a template. You must:")
    print("1. Fill in all CONFIG paths.")
    print("2. Implement the `utils.py` vocabulary building helper functions.")
    print("3. Fully integrate the data loading/splitting logic from `tranp_cnn_pipeline.py`'s main func.")
    print("4. Implement the `evaluate` function.")
    # main()