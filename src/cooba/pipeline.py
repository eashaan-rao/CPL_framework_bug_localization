import math
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
import time
import gc
import shutil

# Local imports
from .model import COOBA, ProjectDiscriminator, GradientReversalLayer
from .ast_parsers import PythonASTParser, JavaASTParser
from .utils import load_glove, glove_lookup, text_to_glove_sequence


REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/cooba_cache"

# GloVe configuration (used for COOBA training — retrieval stage still uses BGE)
GLOVE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/glove/glove.6B.300d.txt"
GLOVE_DIM = 300

# Experiment settings
TOP_K_CANDIDATES = 300
MAX_BUG_LEN = 512   # max words in bug report sequence
MAX_AST_NODES = 5000  # cap AST graph size; prevents OOM on auto-generated / minified files

# Model Hyperparameters - FIXED: Use 'hidden_size' instead of 'hidden_dim'
PARAMS = {
    'bug_encoder': {'hidden_size': 256, 'num_layers': 2, 'dropout_prob': 0.5},
    'shared_extractor': {'num_filters': 128, 'kernel_sizes': [3, 4, 5], 'dropout_prob': 0.5},
    'individual_extractor': {'hidden_dim': 256, 'output_dim': 256, 'num_layers': 2, 'dropout_prob': 0.5},
    'fusion': {'output_dim': 256, 'hidden_dim': 384, 'dropout_prob': 0.5}
}

# Training Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16  # Reduced to accommodate variable true-AST graph sizes
LEARNING_RATE = 0.001
DISC_LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
MARGIN = 1.0  # For Margin Ranking Loss; scaled for normalized L2² ∈ [0, 4]
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

            # Create Labeled pairs - FIXED: Don't hardcode project_type
            for blob_sha in candidate_shas:
                label = 1 if blob_sha in ground_truth_shas else 0
                labeled_samples.append((bug_id, blob_sha, label, 'unknown'))  # Will be set by dataset

    return labeled_samples

# --- 3. PyTorch Dataset ---
class CoobaBugLocalizationDataset(Dataset):
    '''
    PyTorch Dataset for COOBA with BGE embeddings and AST graphs.
    '''
    def __init__(self, samples, bug_reports_df, repo, language, glove_dict, ast_parser, cache_dir, project_type='target'):
        """
        Args:
            samples: List of (bug_id, blob_sha, label) tuples
            bug_reports_df: Bug reports dataframe
            repo: Git repository object
            language: Programming language
            glove_dict: GloVe {word: np.array(300)} — used for bug sequences and AST node features
            ast_parser: AST parser (must have glove_dict set)
            cache_dir: Directory for caching AST graphs
            project_type: 'source' or 'target'
        """
        self.samples = samples
        self.bug_reports_df = bug_reports_df
        self.repo = repo
        self.language = language
        self.glove_dict = glove_dict
        self.ast_parser = ast_parser
        self.cache_dir = cache_dir
        self.project_type = project_type

        os.makedirs(cache_dir, exist_ok=True)

        # Bug text lookup
        self.bug_texts = pd.Series(
            bug_reports_df['bug_report_text'].values,
            index=bug_reports_df['bug_id']
        ).to_dict()

    def __len__(self):
        return len(self.samples)
    
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
        bug_id, blob_sha, label, _ = self.samples[idx]

        # Bug report → GloVe word sequence (faithful to paper: variable-length word embeddings)
        bug_text = self.bug_texts.get(bug_id, "")
        bug_seq, bug_len = text_to_glove_sequence(bug_text, self.glove_dict, MAX_BUG_LEN)
        bug_embeddings = torch.tensor(bug_seq, dtype=torch.float32)  # (MAX_BUG_LEN, GLOVE_DIM)

        # AST graph with per-node GloVe features — load from cache or build
        cache_path = self._cache_key(blob_sha)
        if os.path.exists(cache_path):
            cached_data = torch.load(cache_path, weights_only=False)
            graph_data = cached_data['graph_data']
        else:
            code_content = self._get_code_content(blob_sha)
            if code_content:
                # ast_parser has glove_dict set — produces Data with per-node GloVe features
                graph_data = self.ast_parser.parse(code_content)
            else:
                graph_data = Data(
                    x=torch.zeros(1, GLOVE_DIM),
                    edge_index=torch.tensor([[], []], dtype=torch.long)
                )
            torch.save({'graph_data': graph_data}, cache_path)

        # Truncate oversized AST graphs to prevent OOM
        if graph_data.x.shape[0] > MAX_AST_NODES:
            graph_data = Data(
                x=graph_data.x[:MAX_AST_NODES],
                edge_index=graph_data.edge_index[
                    :, (graph_data.edge_index[0] < MAX_AST_NODES) &
                       (graph_data.edge_index[1] < MAX_AST_NODES)
                ]
            )

        return {
            'bug_embeddings': bug_embeddings,    # (MAX_BUG_LEN, GLOVE_DIM)
            'bug_length': bug_len,               # actual word count for BiLSTM packing
            'code_embeddings': graph_data.x,     # (N, GLOVE_DIM) variable — padded in collate_fn
            'graph_data': graph_data,
            'label': torch.tensor(label, dtype=torch.long),
            'project_type': self.project_type,
            'bug_id': bug_id,
            'blob_sha': blob_sha
        }

def cooba_collate_fn(batch):
    '''
    Custom collate function for batching COOBA data.
    bug_embeddings : already padded to MAX_BUG_LEN in __getitem__ → stack directly
    code_embeddings: variable-length node sequences → pad to max nodes in this batch
    '''
    # Bug: (MAX_BUG_LEN, GLOVE_DIM) per item — uniform shape, just stack
    bug_embeddings = torch.stack([item['bug_embeddings'] for item in batch])  # (B, MAX_BUG_LEN, GLOVE_DIM)
    bug_lengths = torch.tensor([item['bug_length'] for item in batch])

    # Code: (N_i, GLOVE_DIM) per item — pad to max nodes in this batch
    code_seqs = [item['code_embeddings'] for item in batch]
    max_nodes = max(s.shape[0] for s in code_seqs)
    glove_dim = code_seqs[0].shape[1]
    code_embeddings = torch.zeros(len(batch), max_nodes, glove_dim)
    for i, seq in enumerate(code_seqs):
        code_embeddings[i, :seq.shape[0]] = seq  # (B, max_nodes, GLOVE_DIM)

    labels = torch.stack([item['label'] for item in batch])

    # Batch graphs using PyG's Batch
    graph_list = [item['graph_data'] for item in batch]
    batched_graph = Batch.from_data_list(graph_list)

    project_types = [item['project_type'] for item in batch]
    bug_ids = [item['bug_id'] for item in batch]
    blob_shas = [item['blob_sha'] for item in batch]

    return {
        'bug_embeddings': bug_embeddings,
        'bug_lengths': bug_lengths,
        'code_embeddings': code_embeddings,
        'graph_data': batched_graph,
        'labels': labels,
        'project_types': project_types,
        'bug_ids': bug_ids,
        'blob_shas': blob_shas
    }

# --- 4. Training and Evaluation ---

def _compute_task_loss(scores, labels, criterion, device):
    """MarginRankingLoss over positive/negative pairs in a batch."""
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    if not (torch.any(pos_mask) and torch.any(neg_mask)):
        return torch.tensor(0.0, device=device)
    pos_s = scores[pos_mask]
    neg_s = scores[neg_mask]
    n = min(len(pos_s), len(neg_s))
    if n == 0:
        return torch.tensor(0.0, device=device)
    return criterion(pos_s[:n], neg_s[:n], torch.ones(n, device=device))


def train_cooba(model, discriminator, grl, source_loader, target_loader, optimizer,
                epoch, device, mode='cross-project'):
    """
    Training loop — paper-faithful implementation (Ganin et al. 2016 / COOBA Eq. 8, 11, 12):
      - GRL + single combined optimizer (paper: "adapt Adam to directly minimize L")
      - Task loss on BOTH source and target batches when labels are available (paper Eq. 12)
      - GRL lambda scheduled 0→1 over epochs (Ganin et al. schedule)
      - Gradient clipping for stability
    """
    model.train()
    if discriminator:
        discriminator.train()

    task_criterion = nn.MarginRankingLoss(margin=MARGIN).to(device)
    adv_criterion  = nn.CrossEntropyLoss().to(device)

    # GRL lambda: 0 at epoch 0, approaches 1 at final epoch (Ganin et al. schedule)
    p = epoch / max(EPOCHS - 1, 1)
    grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    if grl is not None:
        grl.alpha = grl_lambda

    all_params = list(model.parameters()) + (list(discriminator.parameters()) if discriminator else [])

    total_task_loss = 0.0
    total_disc_loss = 0.0
    batch_count = 0
    oom_skips = 0

    if mode == 'within-project':
        data_loader = target_loader if target_loader else source_loader
        num_batches = len(data_loader)

        for batch_idx, batch in enumerate(data_loader):
            try:
                scores, _ = model(
                    batch['bug_embeddings'].to(device), batch['bug_lengths'].to(device),
                    batch['code_embeddings'].to(device), batch['graph_data'].to(device), None
                )
                task_loss = _compute_task_loss(scores, batch['labels'].to(device), task_criterion, device)

                optimizer.zero_grad()
                if task_loss.grad_fn is not None:
                    task_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_task_loss += task_loss.item()
                batch_count += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_skips += 1
                continue

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                oom_str = f" OOM={oom_skips}" if oom_skips else ""
                print(f"  Epoch {epoch+1}/{EPOCHS} | {batch_idx+1}/{num_batches} | "
                      f"Task={total_task_loss/max(batch_count,1):.4f}{oom_str}", flush=True)

    else:  # cross-project
        has_target = target_loader is not None
        num_batches = len(source_loader)
        source_iter = iter(source_loader)
        target_iter = iter(itertools.cycle(target_loader)) if has_target else None

        for batch_idx in range(num_batches):
            try:
                task_loss = torch.tensor(0.0, device=device)
                disc_loss = torch.tensor(0.0, device=device)

                # Source batch: task loss (paper L^s) + domain features
                src = next(source_iter)
                src_scores, src_pub = model(
                    src['bug_embeddings'].to(device), src['bug_lengths'].to(device),
                    src['code_embeddings'].to(device), src['graph_data'].to(device), 'source'
                )
                task_loss = task_loss + _compute_task_loss(
                    src_scores, src['labels'].to(device), task_criterion, device
                )

                if has_target:
                    # Target batch: task loss (paper L^t, labeled in CP-transfer) + domain features
                    tgt = next(target_iter)
                    tgt_scores, tgt_pub = model(
                        tgt['bug_embeddings'].to(device), tgt['bug_lengths'].to(device),
                        tgt['code_embeddings'].to(device), tgt['graph_data'].to(device), 'target'
                    )
                    task_loss = task_loss + _compute_task_loss(
                        tgt_scores, tgt['labels'].to(device), task_criterion, device
                    )

                    # GRL adversarial: domain labels source=0, target=1.
                    # GRL reverses gradients into shared_extractor → domain-invariant public features.
                    if discriminator is not None and grl is not None:
                        all_pub = torch.cat([grl(src_pub), grl(tgt_pub)], dim=0)
                        domain_labels = torch.cat([
                            torch.zeros(src_pub.size(0), dtype=torch.long, device=device),
                            torch.ones(tgt_pub.size(0),  dtype=torch.long, device=device)
                        ])
                        disc_loss = adv_criterion(discriminator(all_pub), domain_labels)

                total_loss = task_loss + LAMBDA_ADV * disc_loss

                optimizer.zero_grad()
                if total_loss.grad_fn is not None:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()

                total_task_loss += task_loss.item()
                total_disc_loss += disc_loss.item()
                batch_count += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_skips += 1
                continue

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                oom_str = f" OOM={oom_skips}" if oom_skips else ""
                print(f"  Epoch {epoch+1}/{EPOCHS} | {batch_idx+1}/{num_batches} | "
                      f"Task={total_task_loss/max(batch_count,1):.4f} "
                      f"Disc={total_disc_loss/max(batch_count,1):.4f} "
                      f"λ={grl_lambda:.3f}{oom_str}", flush=True)

    print(f"  Epoch {epoch+1}/{EPOCHS} DONE | "
          f"Task={total_task_loss/max(batch_count,1):.4f} "
          f"Disc={total_disc_loss/max(batch_count,1):.4f} [{batch_count} batches]")

def evaluate_cooba(model, test_loader, device,
                   source_project=None, target_project=None, scenario=None,
                   target_bug_db=None, target_blob_db=None, target_repo=None):
    """Evaluate COOBA model. Writes per-bug diagnostic CSV when project/DB args are provided."""
    model.eval()

    # Collect predictions: bug_id -> [(blob_sha, model_score), ...]
    predictions = {}
    ground_truths = {}

    with torch.no_grad():
        for batch in test_loader:
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

            for i in range(len(bug_ids)):
                bug_id = bug_ids[i]
                if bug_id not in predictions:
                    predictions[bug_id] = []
                    ground_truths[bug_id] = []
                predictions[bug_id].append((blob_shas[i], scores[i].item()))
                if labels[i] == 1:
                    ground_truths[bug_id].append(blob_shas[i])

    # Standard metrics
    top_k_hits = {1: 0, 5: 0, 10: 0}
    mrr_scores = []
    map_scores = []

    for bug_id in predictions:
        ranked = sorted(predictions[bug_id], key=lambda x: x[1], reverse=True)
        ranked_shas = [sha for sha, _ in ranked]
        true_shas = set(ground_truths[bug_id])

        for k in top_k_hits:
            if len(set(ranked_shas[:k]) & true_shas) > 0:
                top_k_hits[k] += 1

        for i, sha in enumerate(ranked_shas):
            if sha in true_shas:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0.0)

        precisions = []
        hits = 0
        for i, sha in enumerate(ranked_shas):
            if sha in true_shas:
                hits += 1
                precisions.append(hits / (i + 1))
        map_scores.append(np.mean(precisions) if precisions else 0.0)

    # --- Diagnostics: FAISS rank vs COOBA model rank per ground-truth file ---
    run_diag = all(x is not None for x in [
        source_project, target_project, scenario,
        target_bug_db, target_blob_db, target_repo
    ])
    if run_diag:
        diagnostic_results = []

        # Group test bugs by commit so we build one FAISS index per snapshot
        bugs_by_commit = {}
        for bug_id in predictions:
            if bug_id not in target_bug_db:
                continue
            commit_sha = target_bug_db[bug_id]['commit_sha']
            bugs_by_commit.setdefault(commit_sha, []).append(bug_id)

        for commit_sha, bug_ids in bugs_by_commit.items():
            path_to_sha = get_path_to_sha_map(target_repo, commit_sha)
            snap_shas = [sha for sha in path_to_sha.values() if sha in target_blob_db]
            if not snap_shas:
                continue

            snap_embs = np.stack([target_blob_db[sha].astype('float32') for sha in snap_shas])
            faiss.normalize_L2(snap_embs)

            for bug_id in bug_ids:
                if bug_id not in predictions or bug_id not in target_bug_db:
                    continue

                bug_emb = target_bug_db[bug_id]['embedding'].astype('float32').reshape(1, -1)
                faiss.normalize_L2(bug_emb)
                faiss_sims = (snap_embs @ bug_emb.T).flatten()
                faiss_order = np.argsort(faiss_sims)[::-1]
                faiss_ranked_shas = [snap_shas[i] for i in faiss_order]

                ranked = sorted(predictions[bug_id], key=lambda x: x[1], reverse=True)
                model_ranked_shas = [sha for sha, _ in ranked]
                model_scores_map = {sha: score for sha, score in predictions[bug_id]}
                true_shas = set(ground_truths.get(bug_id, []))

                for gt_sha in true_shas:
                    faiss_rank = next(
                        (r + 1 for r, s in enumerate(faiss_ranked_shas) if s == gt_sha), -1
                    )
                    model_rank = next(
                        (r + 1 for r, s in enumerate(model_ranked_shas) if s == gt_sha), -1
                    )
                    rank_displacement = (
                        (faiss_rank - model_rank)
                        if (faiss_rank != -1 and model_rank != -1) else -1
                    )
                    faiss_score = (
                        float(faiss_sims[snap_shas.index(gt_sha)])
                        if gt_sha in snap_shas else -1.0
                    )
                    diagnostic_results.append({
                        'bug_id': bug_id,
                        'gt_blob_sha': gt_sha,
                        'faiss_rank': faiss_rank,
                        'model_rank': model_rank,
                        'rank_displacement': rank_displacement,
                        'faiss_score': faiss_score,
                        'model_score': model_scores_map.get(gt_sha, -1.0),
                    })

        if diagnostic_results:
            os.makedirs(os.path.join(RESULT_PATH, 'cooba_ph1_diagnostics'), exist_ok=True)
            diag_df = pd.DataFrame(diagnostic_results)
            src_safe = source_project.replace('/', '_')
            tgt_safe = target_project.replace('/', '_')
            diag_path = os.path.join(
                RESULT_PATH, 'cooba_ph1_diagnostics',
                f"{src_safe}_{tgt_safe}_{scenario}_diagnostics.csv",
            )
            try:
                diag_df.to_csv(diag_path, index=False)
                print(f"  Saved diagnostics → {diag_path}")
            except Exception as e:
                print(f"  Warning: could not save diagnostics: {e}")

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

    # load databases
    print("Loading project databases...")
    source_bug_db, source_blob_db, source_language = load_project_databases(source_project, df_meta)
    source_language = source_language.strip().lower()
    target_bug_db, target_blob_db, target_language = load_project_databases(target_project, df_meta)
    target_language = target_language.strip().lower()

    # Load GloVe embeddings (loaded once, cached in module-level _GLOVE_CACHE)
    print("Loading GloVe embeddings...")
    glove_dict = load_glove(GLOVE_PATH)

    # Initialize AST parsers with GloVe dict for per-node features
    parsers = {
        'java': JavaASTParser(),
        'python': PythonASTParser()
    }
    for p in parsers.values():
        p.glove_dict = glove_dict

    # Get repos
    source_repo = git.Repo(get_project_path(source_project, source_language))
    target_repo = git.Repo(get_project_path(target_project, target_language))

    # Pair-specific cache so each experiment is self-contained and can be cleaned up independently
    pair_key = f"{source_project.replace('/', '_')}__{target_project.replace('/', '_')}__{scenario}"
    source_cache = os.path.join(CACHE_DIR, pair_key, "source")
    target_cache = os.path.join(CACHE_DIR, pair_key, "target")

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
            TOP_K_CANDIDATES, is_training_data=True, bug_ids_to_process=target_train_ids
        )
    
    print("Generating test samples...")
    target_test_samples = create_labeled_samples(
        target_project, target_language, target_bug_db, target_blob_db,
        TOP_K_CANDIDATES, is_training_data=False, bug_ids_to_process=target_test_ids
    )

    # Create datasets with proper project_type
    source_dataset = None
    target_train_dataset = None

    if source_samples:
        source_dataset = CoobaBugLocalizationDataset(
            source_samples, df_bugs, source_repo, source_language,
            glove_dict, parsers[source_language], source_cache,
            project_type='source'
        )
    if target_train_samples:
        target_train_dataset = CoobaBugLocalizationDataset(
            target_train_samples, df_bugs, target_repo, target_language,
            glove_dict, parsers[target_language], target_cache,
            project_type='target'
        )

    target_test_dataset = CoobaBugLocalizationDataset(
        target_test_samples, df_bugs, target_repo, target_language,
        glove_dict, parsers[target_language], target_cache,
        project_type='target'
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
        target_test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=cooba_collate_fn, num_workers=0
    )

    # Determine mode based on scenario
    if 'WP' in scenario:
        mode = 'within-project'
    else:
        mode = 'cross-project'

    # Initialize model
    print("Initializing COOBA model...")
    model = COOBA(
        bug_embedding_dim=GLOVE_DIM,
        code_embedding_dim=GLOVE_DIM,
        bug_encoder_params=PARAMS['bug_encoder'],
        shared_extractor_params=PARAMS['shared_extractor'],
        individual_extractor_params=PARAMS['individual_extractor'],
        fusion_params=PARAMS['fusion'],
        mode=mode
    ).to(device)

    # Initialize discriminator + GRL for cross-project (paper Eq. 7-8)
    discriminator = None
    grl = None
    if mode == 'cross-project':
        public_feat_dim = PARAMS['shared_extractor']['num_filters'] * len(PARAMS['shared_extractor']['kernel_sizes'])
        discriminator = ProjectDiscriminator(
            input_dim=public_feat_dim,
            hidden_dim=128,
            dropout_prob=0.5
        ).to(device)
        grl = GradientReversalLayer(alpha=0.0)  # alpha ramped up each epoch

    # Single combined optimizer — paper: "adapt Adam to directly minimize L" (Eq. 12)
    all_params = list(model.parameters()) + (list(discriminator.parameters()) if discriminator else [])
    optimizer_main = optim.AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training
    print("Starting Training")
    for epoch in range(EPOCHS):
        if mode == 'within-project':
            train_loader = target_train_loader if target_train_loader else source_loader
            train_cooba(model, None, None, None, train_loader,
                        optimizer_main, epoch, device, mode)
        else:
            train_cooba(model, discriminator, grl, source_loader, target_train_loader,
                        optimizer_main, epoch, device, mode)
        
    # Evaluation
    print("\nEvaluating model...")
    metrics = evaluate_cooba(
        model, target_test_loader, device,
        source_project=source_project,
        target_project=target_project,
        scenario=scenario,
        target_bug_db=target_bug_db,
        target_blob_db=target_blob_db,
        target_repo=target_repo,
    )
    print(f"Final metrics: {metrics}")

    # Free GPU and CPU memory before returning so the next scenario starts clean
    del model
    if discriminator:
        del discriminator
    grl = None
    for obj in [source_loader, target_train_loader, source_dataset, target_train_dataset]:
        if obj is not None:
            del obj
    del target_test_loader, target_test_dataset
    del source_samples, target_train_samples, target_test_samples
    del source_bug_db, source_blob_db, target_bug_db, target_blob_db
    del df_meta, df_bugs
    torch.cuda.empty_cache()
    gc.collect()

    # Delete pair-specific cache — no longer needed after evaluation
    pair_cache_dir = os.path.join(CACHE_DIR, pair_key)
    if os.path.exists(pair_cache_dir):
        shutil.rmtree(pair_cache_dir)
        print(f"  Cleaned up pair cache: {pair_cache_dir}")

    return metrics