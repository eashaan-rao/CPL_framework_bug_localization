"""
BL-GAN training and evaluation pipeline — paper-faithful implementation.

Entry point: run_blgan_experiment()

Implements the semi-supervised GAN training strategy from:
  "BL-GAN: Semi-Supervised Bug Localization via Generative Adversarial Network"
  IEEE TKDE 2023.

Key differences from BGE-based version:
  - Model inputs are learned-vocabulary token IDs, not BGE vectors.
  - Generator uses directory-tree REINFORCE traversal (not candidate distribution).
  - Vocabulary covers bug report text, file path tokens, and AST node types.
  - BGE embeddings are used ONLY for FAISS pre-filtering of candidates.
"""

import os
from typing import Dict, List, Optional, Set, Tuple

import faiss
import git
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch as TGBatch, Data

from .model import BLGAN
from .data_prep import (
    BLGAN_CACHE_DIR,
    BLOB_DB_DIR,
    BUG_METADATA_DIR,
    BUG_REPORTS_PATH,
    PROJECTS_METADATA_PATH,
    REPO_BASE_PATH,
    MAX_BUG_LEN,
    MAX_PATH_LEN,
    MAX_AST_NODES,
    PYTHON_AST_TYPES,
    JAVA_AST_TYPES,
    LabeledBLGANDataset,
    UnlabeledBLGANDataset,
    build_vocab,
    collect_vocab_tokens,
    collate_labeled,
    create_labeled_samples,
    encode_ast_with_vocab,
    encode_tokens,
    get_blob_content,
    get_path_to_sha_map,
    get_project_path,
    load_project_databases,
    tokenize_text,
    build_directory_tree,
    _get_cached_raw_ast,
    _save_raw_ast,
)
from .ast_parsers import get_parser_for_language

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULT_PATH = "/home/cs21d002_eashaan/PhD/Objective1/results"

BGE_DIM = 1536
TOP_K_CANDIDATES = 300     # FAISS for labeled / evaluation
TOP_K_UNLABELED = 100      # FAISS for unlabeled candidate filtering (G training context)
K_GEN = 10                 # tree traversal episodes per unlabeled bug per G step
D_STEPS = 2                # discriminator updates per labeled batch
EPOCHS = 10
BATCH_SIZE = 32
UNLABELED_BATCH_SIZE = 8   # small — G step processes bugs sequentially
D_LR = 0.008               # paper's initial LR
G_LR = 0.008
WEIGHT_DECAY = 1e-5

PARAMS = {
    'embed_dim': 300,
    'nhead': 6,
    'transformer_layers': 2,
    'gcn_layers': 3,
    'path_hidden': 256,
    'path_lstm_layers': 2,
    'agent_hidden': 256,
    'agent_layers': 2,
    'dropout': 0.5,
    # 'vocab_size': set after building vocab
}

# AST cache size limit — above this we skip caching new entries
_AST_CACHE_MAX = 20000


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

def build_blgan_vocab(
    source_project: str,
    target_project: str,
    source_train_ids: List,
    target_train_ids: List,
    source_bug_db: dict,
    target_bug_db: dict,
    bug_reports_df: pd.DataFrame,
    source_language: str,
    target_language: str,
) -> Dict[str, int]:
    """Build a unified vocabulary for all BL-GAN model components.

    Sources:
      1. Bug report text tokens for all training bugs (source + target).
      2. Fixed AST node type tokens for Python and Java.

    File path tokens are included via tokenize_text applied to path strings.
    Does NOT parse source code ASTs (too slow for vocab building).

    Args:
        source_project    : Source project repo name.
        target_project    : Target project repo name.
        source_train_ids  : Bug IDs from source project used for training.
        target_train_ids  : Bug IDs from target project used for training.
        source_bug_db     : Source bug metadata database.
        target_bug_db     : Target bug metadata database.
        bug_reports_df    : DataFrame with 'bug_id' and 'bug_report_text'.
        source_language   : Source project language.
        target_language   : Target project language.
    Returns:
        {token: int_id} vocabulary dict.
    """
    token_lists: List[List[str]] = []

    if source_train_ids:
        token_lists.extend(
            collect_vocab_tokens(source_project, source_language, source_bug_db,
                                 source_train_ids, bug_reports_df)
        )
    if target_train_ids:
        token_lists.extend(
            collect_vocab_tokens(target_project, target_language, target_bug_db,
                                 target_train_ids, bug_reports_df)
        )

    extra_tokens: Set[str] = PYTHON_AST_TYPES | JAVA_AST_TYPES

    vocab = build_vocab(token_lists, extra_tokens=extra_tokens)
    return vocab


# ---------------------------------------------------------------------------
# Discriminator training step
# ---------------------------------------------------------------------------

def train_discriminator_step(
    model: BLGAN,
    labeled_batch: Tuple,
    optimizer_d: optim.Optimizer,
    device: torch.device,
    bce_loss: nn.BCELoss,
) -> float:
    """One discriminator update on a labeled batch.

    Args:
        model        : BLGAN model.
        labeled_batch: Output of collate_labeled:
                       (bug_ids, bug_lens, graph_batch, path_ids, path_lens, labels).
        optimizer_d  : Discriminator optimiser.
        device       : Torch device.
        bce_loss     : BCELoss instance.
    Returns:
        D loss value (float).
    """
    bug_ids, bug_lens, graph_batch, path_ids, path_lens, labels = labeled_batch

    bug_ids = bug_ids.to(device)
    bug_lens = bug_lens.to(device)
    graph_batch = graph_batch.to(device)
    path_ids = path_ids.to(device)
    path_lens = path_lens.to(device)
    labels = labels.to(device)

    optimizer_d.zero_grad()
    scores = model.discriminator(bug_ids, bug_lens, graph_batch, path_ids, path_lens)
    d_loss = bce_loss(scores, labels)
    d_loss.backward()
    optimizer_d.step()

    return d_loss.item()


# ---------------------------------------------------------------------------
# Generator training step
# ---------------------------------------------------------------------------

def train_generator_step(
    model: BLGAN,
    unlabeled_batch: Tuple,
    optimizer_g: optim.Optimizer,
    device: torch.device,
    target_repo: git.Repo,
    target_parser,
    path_sha_cache: dict,
    tree_cache: dict,
    ast_cache: dict,
    vocab: Dict[str, int],
) -> float:
    """One Generator REINFORCE update on an unlabeled batch.

    For each bug in the batch:
      1. Encode bug → b_G (with grad).
      2. Get/build directory tree for the bug's commit_sha.
      3. Run K_GEN tree traversal episodes, each producing a file path + log_prob.
      4. For each episode that yields a valid path: fetch AST, score with D (no_grad).
      5. reward = log1p(exp(disc_score)).
      6. baseline = mean(rewards over all valid episodes for this bug).
      7. g_loss += -sum(log_prob * (reward - baseline)).

    Args:
        model          : BLGAN model.
        unlabeled_batch: Output of UnlabeledBLGANDataset.collate_fn:
                         (bug_ids_t, bug_lens_t, commit_shas, bug_ids_str).
        optimizer_g    : Generator optimiser.
        device         : Torch device.
        target_repo    : gitpython Repo for the target project.
        target_parser  : AST parser for target project language.
        path_sha_cache : {commit_sha: {file_path: blob_sha}} — updated in-place.
        tree_cache     : {commit_sha: dir_tree dict} — updated in-place.
        ast_cache      : {blob_sha: encoded Data} — updated in-place.
        vocab          : Vocabulary dict.
    Returns:
        G loss value (float), or 0.0 if no valid episodes were found.
    """
    bug_ids_t, bug_lens_t, commit_shas, _ = unlabeled_batch

    bug_ids_t = bug_ids_t.to(device)
    bug_lens_t = bug_lens_t.to(device)

    optimizer_g.zero_grad()

    # Encode all bugs in the batch to get b_G with grad
    b_G_batch = model.generator.encode_bug(bug_ids_t, bug_lens_t)  # (B, embed_dim)

    total_loss = torch.zeros(1, device=device)
    valid_episodes = 0

    for i, commit_sha in enumerate(commit_shas):
        b_G = b_G_batch[i]  # (embed_dim,) — has grad

        # Get/build path→sha map for this commit
        if commit_sha not in path_sha_cache:
            path_sha_cache[commit_sha] = get_path_to_sha_map(target_repo, commit_sha)
        path_to_sha = path_sha_cache[commit_sha]
        if not path_to_sha:
            continue

        # Get/build directory tree
        if commit_sha not in tree_cache:
            tree_cache[commit_sha] = build_directory_tree(target_repo, commit_sha, vocab)
        dir_tree = tree_cache[commit_sha]
        if not dir_tree:
            continue

        # Run K_GEN traversal episodes
        episode_log_probs: List[torch.Tensor] = []
        episode_rewards: List[float] = []

        for _ in range(K_GEN):
            file_path, ep_log_prob = model.generator.traverse(b_G, dir_tree, device, greedy=False)

            if not file_path or file_path not in path_to_sha:
                # Traversal led to an empty/invalid path — skip this episode
                continue

            blob_sha = path_to_sha[file_path]

            # Get/build vocab-encoded AST for the selected file
            if blob_sha in ast_cache:
                encoded_graph = ast_cache[blob_sha]
            else:
                raw_graph = _get_cached_raw_ast(blob_sha, BLGAN_CACHE_DIR)
                if raw_graph is None:
                    code = get_blob_content(target_repo, blob_sha)
                    raw_graph = target_parser.parse(code) if code else Data(
                        node_tokens=[],
                        edge_index=torch.tensor([[], []], dtype=torch.long),
                    )
                    _save_raw_ast(raw_graph, blob_sha, BLGAN_CACHE_DIR)
                encoded_graph = encode_ast_with_vocab(raw_graph, vocab, MAX_AST_NODES)

                if len(ast_cache) < _AST_CACHE_MAX:
                    ast_cache[blob_sha] = encoded_graph

            # Score with discriminator (no grad — reward is treated as constant)
            path_tokens = re.findall(r'[a-zA-Z_]\w*|\d+', file_path.lower())
            path_ids_ep, path_len_ep = encode_tokens(path_tokens, vocab, MAX_PATH_LEN)
            path_ids_t_ep = torch.tensor(path_ids_ep, dtype=torch.long, device=device).unsqueeze(0)
            path_len_t_ep = torch.tensor([path_len_ep], dtype=torch.long, device=device)

            bug_ids_single = bug_ids_t[i].unsqueeze(0)
            bug_lens_single = bug_lens_t[i].unsqueeze(0)

            graph_batch_ep = TGBatch.from_data_list([encoded_graph]).to(device)

            with torch.no_grad():
                disc_score = model.discriminator.score(
                    bug_ids_single, bug_lens_single,
                    graph_batch_ep,
                    path_ids_t_ep, path_len_t_ep,
                ).item()

            reward = float(torch.log1p(torch.exp(torch.tensor(disc_score))).item())
            episode_log_probs.append(ep_log_prob)
            episode_rewards.append(reward)

        if not episode_rewards:
            continue

        # REINFORCE with baseline
        baseline = float(np.mean(episode_rewards))
        for ep_log_prob, reward in zip(episode_log_probs, episode_rewards):
            advantage = reward - baseline
            total_loss = total_loss - ep_log_prob * advantage
            valid_episodes += 1

    if valid_episodes == 0:
        optimizer_g.zero_grad()
        return 0.0

    # Normalise by number of episodes
    g_loss = total_loss / valid_episodes
    g_loss.backward()
    optimizer_g.step()

    return g_loss.item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_blgan(
    model: BLGAN,
    test_samples: List[Tuple],
    target_bug_db: dict,
    target_blob_db: dict,
    target_repo: git.Repo,
    target_parser,
    vocab: Dict[str, int],
    ast_cache: dict,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the Discriminator on test bugs.

    For each test bug: score all FAISS-filtered candidates with the Discriminator,
    rank by score, and compute Top-1/5/10, MAP, MRR.

    Note: The Generator is NOT used during evaluation.  The Discriminator scores
    pre-filtered FAISS candidates directly.

    Args:
        model          : Trained BLGAN model (eval mode set inside).
        test_samples   : List of (bug_id, blob_sha, label, file_path).
        target_bug_db  : {bug_id: metadata} for the target project.
        target_blob_db : {blob_sha: np.array} for the target project.
        target_repo    : gitpython Repo for the target project.
        target_parser  : AST parser for target project language.
        vocab          : Vocabulary dict.
        ast_cache      : {blob_sha: encoded Data} — updated in-place.
        device         : Torch device.
    Returns:
        {'Top-1': float, 'Top-5': float, 'Top-10': float, 'MAP': float, 'MRR': float}
    """
    model.eval()

    predictions: Dict[str, List[Tuple[str, float]]] = {}
    ground_truths: Dict[str, List[str]] = {}

    bug_reports_df = pd.read_parquet(BUG_REPORTS_PATH)

    # Pre-cache bug report token encodings
    bug_token_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_bug_tokens_eval(bug_id) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(bug_id)
        if key not in bug_token_cache:
            rows = bug_reports_df[bug_reports_df['bug_id'].astype(str) == key]
            text = ''
            if not rows.empty:
                text = str(rows.iloc[0].get('bug_report_text', '') or '')
            tokens = tokenize_text(text) if text else []
            ids, length = encode_tokens(tokens, vocab, MAX_BUG_LEN)
            bug_token_cache[key] = (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
            )
        return bug_token_cache[key]

    # Group test samples by bug_id
    bugs: Dict[str, List[Tuple[str, int, str]]] = {}
    for bug_id, blob_sha, label, file_path in test_samples:
        bugs.setdefault(str(bug_id), []).append((blob_sha, label, file_path))

    with torch.no_grad():
        for bug_id_str, candidates in tqdm(bugs.items(), desc='Evaluating BL-GAN'):
            bug_ids_t, bug_len_t = _get_bug_tokens_eval(bug_id_str)

            gt_shas: List[str] = []
            all_scores: List[Tuple[str, float]] = []

            for blob_sha, label, file_path in candidates:
                # Get encoded AST
                if blob_sha in ast_cache:
                    encoded_graph = ast_cache[blob_sha]
                else:
                    raw_graph = _get_cached_raw_ast(blob_sha, BLGAN_CACHE_DIR)
                    if raw_graph is None:
                        code = get_blob_content(target_repo, blob_sha)
                        raw_graph = target_parser.parse(code) if code else Data(
                            node_tokens=[],
                            edge_index=torch.tensor([[], []], dtype=torch.long),
                        )
                        _save_raw_ast(raw_graph, blob_sha, BLGAN_CACHE_DIR)
                    encoded_graph = encode_ast_with_vocab(raw_graph, vocab, MAX_AST_NODES)
                    if len(ast_cache) < _AST_CACHE_MAX:
                        ast_cache[blob_sha] = encoded_graph

                path_tokens = re.findall(r'[a-zA-Z_]\w*|\d+', file_path.lower())
                path_ids, path_len = encode_tokens(path_tokens, vocab, MAX_PATH_LEN)
                path_ids_t = torch.tensor(path_ids, dtype=torch.long, device=device).unsqueeze(0)
                path_len_t = torch.tensor([path_len], dtype=torch.long, device=device)

                graph_batch = TGBatch.from_data_list([encoded_graph]).to(device)

                score = model.discriminator.score(
                    bug_ids_t.unsqueeze(0).to(device),
                    bug_len_t.unsqueeze(0).to(device),
                    graph_batch,
                    path_ids_t,
                    path_len_t,
                ).item()

                all_scores.append((blob_sha, score))
                if label == 1:
                    gt_shas.append(blob_sha)

            predictions[bug_id_str] = all_scores
            ground_truths[bug_id_str] = gt_shas

    # Compute metrics
    top_k_hits: Dict[int, int] = {1: 0, 5: 0, 10: 0}
    mrr_scores: List[float] = []
    map_scores: List[float] = []

    for bug_id_str, preds in predictions.items():
        ranked = sorted(preds, key=lambda x: x[1], reverse=True)
        ranked_shas = [sha for sha, _ in ranked]
        true_shas = set(ground_truths.get(bug_id_str, []))

        for k in top_k_hits:
            if set(ranked_shas[:k]) & true_shas:
                top_k_hits[k] += 1

        # MRR
        for i, sha in enumerate(ranked_shas):
            if sha in true_shas:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0.0)

        # MAP
        precisions: List[float] = []
        hits = 0
        for i, sha in enumerate(ranked_shas):
            if sha in true_shas:
                hits += 1
                precisions.append(hits / (i + 1))
        map_scores.append(float(np.mean(precisions)) if precisions else 0.0)

    n = len(predictions)
    return {
        'Top-1': top_k_hits[1] / n if n > 0 else 0.0,
        'Top-5': top_k_hits[5] / n if n > 0 else 0.0,
        'Top-10': top_k_hits[10] / n if n > 0 else 0.0,
        'MRR': float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        'MAP': float(np.mean(map_scores)) if map_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_blgan(
    model: BLGAN,
    labeled_loader: DataLoader,
    unlabeled_loader: Optional[DataLoader],
    optimizer_d: optim.Optimizer,
    optimizer_g: optim.Optimizer,
    device: torch.device,
    epochs: int,
    target_repo: git.Repo,
    source_repo_optional: Optional[git.Repo],
    target_parser,
    source_parser_optional,
    target_bug_db: dict,
    source_bug_db: dict,
    path_sha_cache: dict,
    tree_cache: dict,
    ast_cache: dict,
    vocab: Dict[str, int],
) -> None:
    """Full BL-GAN training loop.

    Alternates between D steps (on labeled data) and G steps (on unlabeled data).
    Uses CosineAnnealingLR for both optimisers.

    Args:
        model                : BLGAN model.
        labeled_loader       : DataLoader for labeled data (collate_labeled).
        unlabeled_loader     : DataLoader for unlabeled data (UnlabeledBLGANDataset.collate_fn).
                               May be None if no unlabeled data.
        optimizer_d          : Discriminator optimiser.
        optimizer_g          : Generator optimiser.
        device               : Torch device.
        epochs               : Number of training epochs.
        target_repo          : gitpython Repo for target project (used by G step).
        source_repo_optional : Optional gitpython Repo for source project (unused by G step).
        target_parser        : AST parser for target project.
        source_parser_optional: Optional AST parser for source project (unused by G step).
        target_bug_db        : Target project bug DB.
        source_bug_db        : Source project bug DB.
        path_sha_cache       : {commit_sha: path_to_sha} cache — updated in-place.
        tree_cache           : {commit_sha: dir_tree} cache — updated in-place.
        ast_cache            : {blob_sha: encoded Data} cache — updated in-place.
        vocab                : Vocabulary dict.
    """
    bce = nn.BCELoss()
    total_steps = epochs * len(labeled_loader)

    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=total_steps, eta_min=1e-6)
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=total_steps, eta_min=1e-6)

    unlabeled_iter: Optional[iter] = iter(unlabeled_loader) if unlabeled_loader else None

    for epoch in range(epochs):
        model.train()
        total_d, total_g = 0.0, 0.0
        d_count, g_count = 0, 0

        for batch_idx, labeled_batch in enumerate(labeled_loader):
            # ----------------------------------------------------------------
            # Discriminator update(s)
            # ----------------------------------------------------------------
            for _ in range(D_STEPS):
                d_loss = train_discriminator_step(model, labeled_batch, optimizer_d, device, bce)
                total_d += d_loss
                d_count += 1
                scheduler_d.step()

            # ----------------------------------------------------------------
            # Generator update (if unlabeled data available)
            # ----------------------------------------------------------------
            if unlabeled_iter is not None:
                try:
                    ul_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    ul_batch = next(unlabeled_iter)

                g_loss = train_generator_step(
                    model, ul_batch, optimizer_g, device,
                    target_repo, target_parser,
                    path_sha_cache, tree_cache, ast_cache, vocab,
                )
                if g_loss > 0:
                    total_g += g_loss
                    g_count += 1
                scheduler_g.step()

            # Progress print
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(labeled_loader):
                avg_d = total_d / max(d_count, 1)
                avg_g = total_g / max(g_count, 1) if g_count > 0 else 0.0
                print(
                    f'  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(labeled_loader)} | '
                    f'D={avg_d:.4f} G={avg_g:.4f}',
                    flush=True,
                )

        print(
            f'  Epoch {epoch+1}/{epochs} done | '
            f'D={total_d/max(d_count,1):.4f} '
            f'G={total_g/max(g_count,1):.4f} '
            f'[{d_count} D-steps, {g_count} G-steps]'
        )


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_blgan_experiment(
    source_project: str,
    target_project: str,
    source_train_ids: List,
    target_train_ids: List,
    target_test_ids: List,
    scenario: str,
) -> Dict[str, float]:
    """Run a single BL-GAN CPL experiment.

    Matches the interface of run_cooba_experiment / run_tranpcnn_experiment.

    Args:
        source_project  : Source project repo name.
        target_project  : Target project repo name.
        source_train_ids: Bug IDs from source project for training.
        target_train_ids: Bug IDs from target project for training (labeled).
        target_test_ids : Bug IDs from target project for evaluation.
        scenario        : 'WP-small' | 'WP-large' | 'CPL-cold-start' | 'CPL-transfer'.
    Returns:
        {'Top-1': float, 'Top-5': float, 'Top-10': float, 'MAP': float, 'MRR': float}
    """
    import re as _re  # local alias to avoid shadowing module-level re import

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running BL-GAN experiment on {device}')
    print(f'Scenario: {scenario}  |  Source: {source_project}  |  Target: {target_project}')

    # -----------------------------------------------------------------------
    # 1. Load project metadata and databases
    # -----------------------------------------------------------------------
    df_meta = pd.read_parquet(PROJECTS_METADATA_PATH)
    bug_reports_df = pd.read_parquet(BUG_REPORTS_PATH)

    print('Loading project databases...')
    source_bug_db, source_blob_db, source_language = load_project_databases(source_project, df_meta)
    source_language = source_language.strip().lower()
    target_bug_db, target_blob_db, target_language = load_project_databases(target_project, df_meta)
    target_language = target_language.strip().lower()

    # Open git repos
    source_repo_path = get_project_path(source_project, source_language)
    target_repo_path = get_project_path(target_project, target_language)
    try:
        source_repo: Optional[git.Repo] = git.Repo(source_repo_path)
    except Exception:
        source_repo = None
    target_repo = git.Repo(target_repo_path)

    os.makedirs(BLGAN_CACHE_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 2. Build unified vocabulary
    # -----------------------------------------------------------------------
    print('Building vocabulary...')
    vocab = build_blgan_vocab(
        source_project=source_project,
        target_project=target_project,
        source_train_ids=source_train_ids or [],
        target_train_ids=target_train_ids or [],
        source_bug_db=source_bug_db,
        target_bug_db=target_bug_db,
        bug_reports_df=bug_reports_df,
        source_language=source_language,
        target_language=target_language,
    )
    print(f'  Vocabulary size: {len(vocab)}')

    # -----------------------------------------------------------------------
    # 3. Prepare labeled training samples (FAISS-filtered)
    # -----------------------------------------------------------------------
    print('Generating labeled training samples...')
    source_samples: List = []
    target_train_samples: List = []

    if source_train_ids:
        source_samples = create_labeled_samples(
            source_project, source_language, source_bug_db, source_blob_db,
            TOP_K_CANDIDATES, bug_ids_to_process=source_train_ids,
        )
    if target_train_ids:
        target_train_samples = create_labeled_samples(
            target_project, target_language, target_bug_db, target_blob_db,
            TOP_K_CANDIDATES, bug_ids_to_process=target_train_ids,
        )
    labeled_samples = source_samples + target_train_samples
    print(f'  Labeled samples: {len(labeled_samples)}')

    # -----------------------------------------------------------------------
    # 4. Prepare unlabeled bug IDs (target bugs not in train or test)
    # -----------------------------------------------------------------------
    exclude_ids: Set = set(map(str, target_train_ids or [])) | set(map(str, target_test_ids or []))
    unlabeled_bug_ids = [
        bid for bid in target_bug_db.keys()
        if str(bid) not in exclude_ids
    ]
    print(f'  Unlabeled bugs: {len(unlabeled_bug_ids)}')

    # -----------------------------------------------------------------------
    # 5. Prepare test samples
    # -----------------------------------------------------------------------
    print('Generating test samples...')
    test_samples = create_labeled_samples(
        target_project, target_language, target_bug_db, target_blob_db,
        TOP_K_CANDIDATES, bug_ids_to_process=target_test_ids,
    )
    print(f'  Test samples: {len(test_samples)}')

    if not labeled_samples:
        print('WARNING: No labeled samples available — returning zero metrics.')
        return {'Top-1': 0.0, 'Top-5': 0.0, 'Top-10': 0.0, 'MAP': 0.0, 'MRR': 0.0}

    # -----------------------------------------------------------------------
    # 6. Initialise AST parsers
    # -----------------------------------------------------------------------
    target_parser = get_parser_for_language(target_language)
    source_parser = get_parser_for_language(source_language) if source_language else None

    # -----------------------------------------------------------------------
    # 7. Build datasets and data loaders
    # -----------------------------------------------------------------------
    print('Building datasets...')
    combined_bug_db = {**source_bug_db, **target_bug_db}
    combined_blob_db = {**source_blob_db, **target_blob_db}

    # For labeled dataset we need source and target repos; use target_repo
    # for blobs from both projects (blob SHA lookups are content-addressed so
    # we can fall back to target_repo for source blobs that won't exist there,
    # but for training we still need AST from source blobs).
    # Strategy: create a combined dataset by splitting by project origin.
    # Simpler: create two datasets and concatenate.

    source_repo_for_ds = source_repo if source_repo is not None else target_repo

    labeled_dataset_source = LabeledBLGANDataset(
        samples=source_samples,
        bug_reports_df=bug_reports_df,
        bug_metadata_db=source_bug_db,
        blob_embedding_db=source_blob_db,
        repo=source_repo_for_ds,
        language=source_language,
        parser=get_parser_for_language(source_language),
        vocab=vocab,
        cache_dir=BLGAN_CACHE_DIR,
    ) if source_samples else None

    labeled_dataset_target = LabeledBLGANDataset(
        samples=target_train_samples,
        bug_reports_df=bug_reports_df,
        bug_metadata_db=target_bug_db,
        blob_embedding_db=target_blob_db,
        repo=target_repo,
        language=target_language,
        parser=target_parser,
        vocab=vocab,
        cache_dir=BLGAN_CACHE_DIR,
    ) if target_train_samples else None

    # Merge into a single ConcatDataset-like list
    from torch.utils.data import ConcatDataset
    datasets = [d for d in [labeled_dataset_source, labeled_dataset_target] if d is not None]
    if not datasets:
        print('WARNING: No labeled datasets built — returning zero metrics.')
        return {'Top-1': 0.0, 'Top-5': 0.0, 'Top-10': 0.0, 'MAP': 0.0, 'MRR': 0.0}

    labeled_combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    labeled_loader = DataLoader(
        labeled_combined,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_labeled,
        num_workers=0,
    )

    unlabeled_loader: Optional[DataLoader] = None
    if unlabeled_bug_ids:
        unlabeled_dataset = UnlabeledBLGANDataset(
            bug_ids=unlabeled_bug_ids,
            bug_reports_df=bug_reports_df,
            bug_metadata_db=target_bug_db,
            vocab=vocab,
        )
        if len(unlabeled_dataset) > 0:
            unlabeled_loader = DataLoader(
                unlabeled_dataset,
                batch_size=UNLABELED_BATCH_SIZE,
                shuffle=True,
                collate_fn=UnlabeledBLGANDataset.collate_fn,
                num_workers=0,
            )

    # -----------------------------------------------------------------------
    # 8. Initialise BLGAN model
    # -----------------------------------------------------------------------
    print('Initialising BL-GAN model...')
    params = dict(PARAMS)
    params['vocab_size'] = len(vocab)
    model = BLGAN(params).to(device)

    optimizer_d = optim.AdamW(model.discriminator_parameters(), lr=D_LR, weight_decay=WEIGHT_DECAY)
    optimizer_g = optim.AdamW(model.generator_parameters(), lr=G_LR, weight_decay=WEIGHT_DECAY)

    # -----------------------------------------------------------------------
    # 9. Initialise caches
    # -----------------------------------------------------------------------
    path_sha_cache: dict = {}
    tree_cache: dict = {}
    ast_cache: dict = {}

    # -----------------------------------------------------------------------
    # 10. Training
    # -----------------------------------------------------------------------
    print('Starting training...')
    train_blgan(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
        device=device,
        epochs=EPOCHS,
        target_repo=target_repo,
        source_repo_optional=source_repo,
        target_parser=target_parser,
        source_parser_optional=source_parser,
        target_bug_db=target_bug_db,
        source_bug_db=source_bug_db,
        path_sha_cache=path_sha_cache,
        tree_cache=tree_cache,
        ast_cache=ast_cache,
        vocab=vocab,
    )

    # -----------------------------------------------------------------------
    # 11. Evaluation
    # -----------------------------------------------------------------------
    print('\nEvaluating model...')
    metrics = evaluate_blgan(
        model=model,
        test_samples=test_samples,
        target_bug_db=target_bug_db,
        target_blob_db=target_blob_db,
        target_repo=target_repo,
        target_parser=target_parser,
        vocab=vocab,
        ast_cache=ast_cache,
        device=device,
    )
    print(f'Final metrics: {metrics}')

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# re import needed by evaluate_blgan and train_generator_step
# ---------------------------------------------------------------------------
import re
