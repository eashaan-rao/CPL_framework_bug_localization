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
from transformers import AutoModel, AutoTokenizer
import time

# Local imports
from cooba_model import COOBA, ProjectDiscriminator
from cooba_ast_parsers import PythonASTParser, JavaASTParser
from cooba_utils import preprocess_code_to_ast_embeddings, prepare_bug_embeddings


REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/cooba_cache"

# BGE Model Configuration
BGE_MODEL_NAME = 'BAAI/bge-code-v1'
BGE_EMBEDDING_DIM = 512

# Experiment settings
TOP_K_CANDIDATES = 300
MAX_BUG_LEN = 512
MAX_CODE_LEN = 1024
MAX_AST_NODES = 500 # Maximum nodes in AST graph

# Model Hyperparameters
PARAMS = {
    'bug_encoder': {'hidden_dim': 256, 'num_layers': 2, 'dropout_prob':0.5},
    'shared_extractor': {'num_filters': 128, 'kernel_sizes': [3, 4, 5], 'dropout_prob':0.5},
    'individual_extractor': {'hidden_dim': 256, 'output_dim': 256, 'num_layers': 2, 'dropout_prob':0.5},
    'fusion': {'output_dim': 256, 'hidden_dim':384, 'dropout_prob':0.5}
}

# Training Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32 # Smaller batch size due to graph processing
LEARNING_RATE = 0.001
DISC_LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
MARGIN = 0.4 # For Margin Ranking Loss
LAMBDA_ADV = 0.1 # Weight for adversarial loss

# Helper functions
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
        blob_db = pickle.load(f)  # Dict: blob_sha -> embedding
    with open(bug_db_path, 'rb') as f:
        bug_db = pickle.load(f) # Dict: bug_id -> {'embedding': ..., 'ground_truth_files': ...}
    return bug_db, blob_db, language

def create_labeled_samples(project_name, language, bug_metadata_db, blob_embedding_db, top_k, 
                           is_training_data=True, bug_ids_to_process=None):
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
                project_type = 'target' if project_name == project_name else 'source'
                labeled_samples.append((bug_id, blob_sha, label, project_type))

    return labeled_samples

# --- 3. PyTorch Dataset ---
class CoobaBugLocalizationDataset(Dataset):
    '''
    PyTorch Dataset for COOBA with BGE embeddings and AST graphs.
    '''
    def __init__(self, samples, bug_reports_df, repo, language, bge_model, bge_tokenizer, ast_parser, cache_dir):
        """
        Args:
            samples: List of (bug_id, blob_sha, label) tuples
            bug_metadata_db: Pre-computed bug embeddings
            blob_embedding_db: Pre-computed code embeddings
            repo: Git repository object
            language: Programming language
            project_type: 'source' or 'target'
            cache_dir: Directory for caching AST graphs
        """
        self.samples = samples
        self.bug_reports_df = bug_reports_df
        self.repo = repo
        self.language = language
        self.bge_model = bge_model
        self.bge_tokenizer = bge_tokenizer
        self.ast_parser = ast_parser
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        # Initialize AST parser
        if language.lower() == 'python':
            self.ast_parser = PythonASTParser(max_nodes=MAX_AST_NODES)
        else:  # Default to Java
            self.ast_parser = JavaASTParser(max_nodes=MAX_AST_NODES)

        # Create bug text lookup
        self.bug_texts = pd.Series(
            bug_reports_df['bug_report_text'].values,
            index=bug_reports_df['bug_id']
        ).to_dict()

    def __len__(self):
        return len(self.samples)
    
    def _get_bge_embedding(self, text, max_length=512):
        '''Get BGE embedding for text.'''
        inputs = self.bge_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.bge_model(**inputs)
            # Use CLS token embedding or mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.squeeze(0)

    def _get_code_content(self, blob_sha):
        '''Fetch code content from blob SHA.'''
        try:
            blob = Blob(self.repo, hex_to_bin(blob_sha))
            return blob.data_stream.read().decode('utf-8', 'ignore')
        except:
            return ""
    
    def _cache_key(self, blob_sha):
        '''Generate cache key for blob.'''
        return os.path.join(self.cache_dir, f"{blob_sha}.pt")

    def __getitem__(self, idx):
        bug_id, blob_sha, label, project_type = self.samples[idx]
        
        # Check cache first
        cache_path = self._cache_key(blob_sha)

        if os.path.exists(cache_path):
            cached_data = torch.load(cache_path)
            code_embeddings = cached_data['code_embeddings']
            graph_data = cached_data['graph_data']
        else:
            # Get code content
            code_content = self._get_code_content(blob_sha)

            # Get BGE embeddings for code
            if code_content:
                # Split code into chunks for embedding
                lines = code_content.splitlines()[:MAX_CODE_LEN]
                code_text = '\n'.join(lines)
                code_embedding = self._get_bge_embedding(code_text, MAX_CODE_LEN)

                # Parse code to AST graph
                graph_data = self.ast_parser.parse(code_content)

                if graph_data is not None:
                    # Add BGE embeddings as node features
                    # For simplicity, use mean embedding for all nodes
                    num_nodes = graph_data.x.shape[0] if hasattr(graph_data, 'x') else 1
                    node_embeddings = code_embeddings.unsequeeze(0).expand(num_nodes, -1)
                    graph_data.x = node_embeddings
            else:
                # Empty file handling
                code_embeddings = torch.zeros(BGE_EMBEDDING_DIM)
                graph_data =  Data(
                    x=torch.zeros(1, BGE_EMBEDDING_DIM),
                    edge_index=torch.tensor([[], []], dtype=torch.long)
                )
            
            # Cache the processed data
            torch.save({
                'code_embeddings': code_embeddings,
                'graph_data': graph_data
            }, cache_path)

        # Get bug report embedding
        bug_text = self.bug_texts.get(bug_id, "")
        bug_embeddings = self._get_bge_embedding(bug_text, MAX_BUG_LEN)

        return {
            'bug_embeddings': bug_embeddings,
            'bug_length': min(len(bug_text.split()), MAX_BUG_LEN),
            'code_embeddings': code_embeddings.unsqueeze(0), # Add sequence dimensions
            'graph_data': graph_data,
            'label': torch.tensor(label, dtype=torch.long),
            'project_type': project_type,
            'bug_id': bug_id,
            'blob_sha': blob_sha
        }

def cooba_collate_fn(batch):
    '''
    Custom collate function for batching COOBA data.
    '''
    # Separate components
    bug_embeddings = torch.stack([item['bug_embeddings'] for item in batch])
    bug_lengths = torch.tensor([item['bug_length'] for item in batch])
    code_embeddings = torch.stack([item['code_embeddings'] for item in batch])
    labels = torch.stack([torch.tensor(item['label']) for item in batch])

    # Batch graphs using PyG's Batch
    graph_list = [item['graph_data'] for item in batch]
    batched_graph = Batch.from_data_list(graph_list)

    project_types = [item['project_type'] for item in batch]
    bug_ids =[item['bug_id'] for item in batch]
    blob_shas = [item['blob_sha'] for item in batch]
    
    return {
        'bug_embeddings': bug_embeddings.unsqueeze(1), # Add sequence dimension
        'bug_lengths': bug_lengths,
        'code_embeddings': code_embeddings,
        'graph_data': batched_graph,
        'labels': labels,
        'project_types': project_types,
        'bug_ids': bug_ids,
        'blob_shas': blob_shas
    }

# --- 4. Training and Evaluation ---
def train_cooba(model, discriminator, source_loader, target_loader, optimizer_main, 
                optimizer_disc, epoch, device, mode='cross-project'):
    '''
    Main Training loop for COOBA with adversarial learning. 
    '''
    model.train()
    if discriminator:
        discriminator.train()

    # Loss functions
    task_criterion = nn.MarginRankingLoss(margin=MARGIN).to(device)
    adv_criterion = nn.CrossEntropyLoss().to(device)

    # Setup iterators
    if mode == 'within-project':
        # Only use target loader
        data_loader = target_loader if target_loader else source_loader
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/ {EPOCHS}")
    else:
        # use both loader
        print("Running in Cross-Project mode.")
        progress_bar = tqdm(source_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        target_iter = iter(itertools.cycle(target_loader)) if target_loader else None
        
    total_task_loss = 0
    total_adv_loss = 0
    total_disc_loss = 0
    batch_count = 0

    for batch in progress_bar:
        # Move data to device
        bug_embeddings = batch['bug_embeddings'].to(device)
        bug_lengths = batch['bug_lengths'].to(device)
        code_embeddings = batch['code_embeddings'].to(device)
        graph_data = batch['graph_data'].to(device)
        labels = batch['labels'].to(device)

        # Determine project type
        project_type = 'source' if mode == 'cross-project' else None

        # Forward pass
        scores, public_features = model(
            bug_embeddings, bug_lengths, code_embeddings, graph_data, project_type
        )

        # Task loss (margin ranking)
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        if torch.any(pos_mask) or torch.any(neg_mask):
            pos_scores = scores[pos_mask]
            neg_scores = scores[neg_mask]

            n_pairs = min(len(pos_scores), len(neg_scores))
            if n_pairs > 0:
                pos_scores = pos_scores[:n_pairs]
                neg_scores = neg_scores[:n_pairs]

                target_rank = torch.ones(n_pairs, device=device)
                task_loss = task_criterion(pos_scores, neg_scores, target_rank)
            else:
                task_loss = torch.tensor(0.0, device=device)
        else:
            task_loss = torch.tensor(0.0, device=device)

        # Adversarial training (only in cross-project mode)
        if mode == 'cross-project' and discriminator and target_iter:
            # Train discriminator
            optimizer_disc.zero_grad()
            # Get target batch
            target_batch = next(target_iter)
            target_bug_embeddings = target_batch['bug_embeddings'].to(device)
            target_bug_lengths = target_batch['bug_lengths'].to(device)
            target_code_embeddings = target_batch['code_embeddings'].to(device)
            target_graph_data = target_batch['graph_data'].to(device)

            # Get features from both domains
            with torch.no_grad():
                _, source_public = model(bug_embeddings, bug_lengths, code_embeddings, graph_data, 'source')
                _, target_public = model(target_bug_embeddings, target_bug_lengths, target_code_embeddings, target_graph_data, 'target')

            # Discriminator predictions
            all_public = torch.cat([source_public, target_public], dim=0)
            disc_labels = torch.cat([
                torch.zeros(len(source_public), device=device, dtype=torch.long),
                torch.ones(len(target_public), device=device, dtype=torch.long)
            ])

            disc_preds = discriminator(all_public)
            disc_loss = adv_criterion(disc_preds, disc_labels)
            disc_loss.backward()
            optimizer_disc.step()

            # Train generator (model) to fool discriminator
            optimizer_main.zero_grad()

            # Re-compute features (with gradients this time)
            _, source_public = model(bug_embeddings, bug_lengths, code_embeddings, graph_data, 'source')

            # Reverse labels to fool discriminator
            fool_labels = torch.ones(len(source_public), device=device, dtype=torch.long)
            disc_preds = discriminator(source_public)
            adv_loss = adv_criterion(disc_preds, fool_labels)

            total_loss = task_loss + LAMBDA_ADV * adv_loss
            total_adv_loss += adv_loss.item()
            total_disc_loss += disc_loss.item()
        
        else:
            # Within-project or no adversarial training
            optimizer_main.zero_grad()
            total_loss = task_loss
        
        # Backward pass
        total_loss.backward()
        optimizer_main.step()

        # Update statistics
        total_task_loss += task_loss.item()
        batch_count += 1

        # Update progress bar
        progress_bar.set_postfix({
            'Task': f'{total_task_loss/batch_count: .4f}',
            'Adv': f'{total_adv_loss/batch_count:.4f}' if mode == 'cross-project' else 'N/A',
            'Disc': f'{total_disc_loss/batch_count:.4f}' if mode == 'cross-project' else 'N/A'
        })

def evaluate_cooba(model, test_loader, device):
    """Evaluate COOBA model."""
    model.eval()
    
    # Collect predictions
    predictions = {}
    ground_truths = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            bug_embeddings = batch['bug_embeddings'].to(device)
            bug_lengths = batch['bug_lengths'].to(device)
            code_embeddings = batch['code_embeddings'].to(device)
            graph_data = batch['graph_data'].to(device)
            labels = batch['labels']
            bug_ids = batch['bug_ids']
            blob_shas = batch['blob_shas']
            
            scores, _ = model(
                bug_embeddings, bug_lengths,
                code_embeddings, graph_data, 'target'
            )
            
            # Store predictions
            for i in range(len(bug_ids)):
                bug_id = bug_ids[i]
                if bug_id not in predictions:
                    predictions[bug_id] = []
                    ground_truths[bug_id] = []
                
                predictions[bug_id].append((blob_shas[i], scores[i].item()))
                if labels[i] == 1:
                    ground_truths[bug_id].append(blob_shas[i])
    
    # Calculate metrics
    top_k_hits = {1: 0, 5: 0, 10: 0}
    mrr_scores = []
    map_scores = []
    
    for bug_id in predictions:
        # Sort predictions by score
        ranked = sorted(predictions[bug_id], key=lambda x: x[1], reverse=True)
        ranked_shas = [sha for sha, _ in ranked]
        true_shas = set(ground_truths[bug_id])
        
        # Top-K accuracy
        for k in top_k_hits:
            if len(set(ranked_shas[:k]) & true_shas) > 0:
                top_k_hits[k] += 1
        
        # MRR
        for i, sha in enumerate(ranked_shas):
            if sha in true_shas:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0.0)
        
        # MAP
        precisions = []
        hits = 0
        for i, sha in enumerate(ranked_shas):
            if sha in true_shas:
                hits += 1
                precisions.append(hits / (i + 1))
        
        if precisions:
            map_scores.append(np.mean(precisions))
        else:
            map_scores.append(0.0)
    
    num_bugs = len(predictions)
    metrics = {
        'Top-1': top_k_hits[1] / num_bugs if num_bugs > 0 else 0,
        'Top-5': top_k_hits[5] / num_bugs if num_bugs > 0 else 0,
        'Top-10': top_k_hits[10] / num_bugs if num_bugs > 0 else 0,
        'MRR': np.mean(mrr_scores) if mrr_scores else 0,
        'MAP': np.mean(map_scores) if map_scores else 0
    }
    
    return metrics

def run_cooba_experiment(source_project, target_project, source_train_ids, target_train_ids, target_test_ids,
                          scenario):
    '''
    Main entry point for COOBA experiments 
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running COOBA experiment on {device}")
    print(f"Scenario: {scenario}")
    print(f"Source: {source_project}, Target: {target_project}")

    # load metadata    
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    df_bugs = pd.read_parquet(BUG_REPORTS_PATH)

    # Initialize BGE model
    print("Loading BGE model....")
    bge_tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
    bge_model = AutoModel.from_pretrained(BGE_MODEL_NAME).to(device)
    bge_model.eval()
    
    # load databases
    print("Loading project databases...")
    source_bug_db, source_blob_db, source_language = load_project_databases(source_project, df_meta)
    source_language = source_language.strip().lower()
    target_bug_db, target_blob_db, target_language = load_project_databases(target_project, df_meta)
    target_language = target_language.strip().lower()

    # Initialize AST parsers
    parsers = {
        'java': JavaASTParser(),
        'python': PythonASTParser()
    }

    # Get repos
    source_repo = git.Repo(get_project_path(source_project, source_language))
    target_repo = git.Repo(get_project_path(target_project, target_language))

    # Create cache directories
    source_cache = os.path.join(CACHE_DIR, f"{source_project.replace('/', '_')}_{scenario}")
    target_cache = os.path.join(CACHE_DIR, f"{target_project.replace('/', '_')}_{scenario}")

    # Generate samples
    print("Generating training samples...")
    source_samples = []
    target_train_samples = []

    if source_train_ids:
        source_samples = create_labeled_samples(
            source_project, source_language, source_bug_db, source_blob_db,
            TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=source_train_ids
        )
    if target_train_ids:
        target_train_samples = create_labeled_samples(
            target_project, target_language, target_bug_db, target_blob_db,
            TOP_K_CANDIDATES, is_training_data=False, bug_ids_to_process=target_train_ids
        )
    
    print("Generating test samples...")
    target_test_samples = create_labeled_samples(
        target_project, target_language, target_bug_db, target_blob_db,
        TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=target_test_ids
    )

    # Create datasets
    source_dataset = None
    target_train_dataset = None

    if source_samples:
        source_dataset = CoobaBugLocalizationDataset(
            source_samples, df_bugs, source_repo, source_language,
            bge_model, bge_tokenizer, parsers[source_language], source_cache
        )
    if target_train_samples:
        target_train_dataset = CoobaBugLocalizationDataset(
            target_train_samples, df_bugs, target_repo, target_language,
            bge_model, bge_tokenizer, parsers[target_language], target_cache
        )
 
    target_test_dataset = CoobaBugLocalizationDataset(
        target_test_samples, df_bugs, target_repo, target_language,
        bge_model, bge_tokenizer, parsers[target_language], target_cache
    )

    # Create data loaders
    source_loader = None
    target_train_loader = None 

    if source_dataset:
        source_loader = DataLoader(
            source_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cooba_collate_fn, num_workers=0
        )
    if target_train_dataset:
        target_train_loader = DataLoader(
            target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cooba_collate_fn, num_workers=0
        )    
    target_test_loader = DataLoader(
        target_test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=cooba_collate_fn, num_workers=0
    )

    # Determine mode based on scenario
    if 'WP' in scenario:
        mode = 'within-project'
    else:
        mode = 'cross-project'

    # Initialize model
    print("Initializing COOBA model...")
    model = COOBA(
        bug_embedding_dim=BGE_EMBEDDING_DIM,
        code_embedding_dim=BGE_EMBEDDING_DIM,
        bug_encoder_params=PARAMS['bug_encoder'],
        shared_extractor_params=PARAMS['shared_extractor'],
        individual_extractor_params=PARAMS['individual_extractor'],
        fusion_params=PARAMS['fusion'],
        mode=mode
    ).to(device)

    # Initialize discriminator for cross-project
    discriminator = None
    optimizer_disc = None
    if mode == 'cross-project':
        public_feat_dim = PARAMS['shared_extractor']['num_filters'] * len(PARAMS['shared_extractor']['kernel_sizes'])
        discriminator = ProjectDiscriminator(
            input_dim=public_feat_dim,
            hidden_dim=128,
            dropout_prob=0.5
            ).to(device)
        optimizer_disc = optim.Adam(discriminator.parameters(), lr=DISC_LEARNING_RATE)
    
    # Initialize optimizer
    optimizer_main = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training
    print("Starting Training")
    for epoch in range(EPOCHS):
        # Use appropriate loader based on scenario
        if mode == 'within-project':
            train_loader = target_train_loader if target_train_loader else source_loader
            train_cooba(model, None, train_loader, None,
                    optimizer_main, None, epoch, device, mode)
        else:
            train_cooba(
                model, discriminator, source_loader, target_train_loader, optimizer_main,
                optimizer_disc, epoch, device, mode)
        
    # Evaluation
    print("\nEvaluating model...")
    metrics = evaluate_cooba(model, target_test_loader, device)
    print(f"Final metrics: {metrics}")

    return metrics

