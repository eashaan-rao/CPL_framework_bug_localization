"""
BL-GAN data preparation utilities — paper-faithful implementation.

Key differences from the BGE-based version:
  - No BGE vectors as model input. BGE embeddings are used ONLY for FAISS filtering.
  - A unified vocabulary is built from text tokens (bug reports + file paths + AST types).
  - Bug report text is loaded from bug_reports_clean.parquet.
  - AST graphs are parsed from source code and cached as raw token-string graphs
    (vocab-independent). Vocab encoding happens at dataset access time.
  - Directory trees are built for the Generator's tree traversal.
"""

import os
import pickle
import re
import collections
from typing import Dict, List, Optional, Set, Tuple

import faiss
import git
from git import Blob
from git.util import hex_to_bin
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch_geometric.data import Data

from .ast_parsers import get_parser_for_language, encode_ast_with_vocab

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
BLGAN_CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/blgan_cache"

# ---------------------------------------------------------------------------
# Tokenisation / vocabulary constants
# ---------------------------------------------------------------------------
MAX_BUG_LEN = 256       # max tokens in a bug report
MAX_PATH_LEN = 32       # max tokens in a file path
MAX_AST_NODES = 500     # max AST nodes per file
MAX_TREE_DEPTH = 15     # max directory tree traversal depth
MAX_VOCAB_SIZE = 50000  # top-N tokens kept in vocabulary
MIN_FREQ = 2            # minimum token frequency to keep
MAX_TREE_FILES = 1000   # max leaf files in directory tree (memory guard)

# Fixed AST node type tokens always included regardless of frequency
PYTHON_AST_TYPES: Set[str] = {
    'Module', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'Return', 'Delete',
    'Assign', 'AugAssign', 'AnnAssign', 'For', 'AsyncFor', 'While', 'If', 'With',
    'AsyncWith', 'Raise', 'Try', 'Assert', 'Import', 'ImportFrom', 'Global', 'Nonlocal',
    'Expr', 'Pass', 'Break', 'Continue', 'BoolOp', 'BinOp', 'UnaryOp', 'Lambda',
    'IfExp', 'Dict', 'Set', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp',
    'Await', 'Yield', 'YieldFrom', 'Compare', 'Call', 'FormattedValue', 'JoinedStr',
    'Constant', 'Attribute', 'Subscript', 'Starred', 'Name', 'List', 'Tuple', 'Slice',
}
JAVA_AST_TYPES: Set[str] = {
    'CompilationUnit', 'TypeDeclaration', 'ClassDeclaration', 'InterfaceDeclaration',
    'MethodDeclaration', 'ConstructorDeclaration', 'FieldDeclaration', 'VariableDeclarator',
    'LocalVariableDeclaration', 'IfStatement', 'WhileStatement', 'ForStatement',
    'ReturnStatement', 'ThrowStatement', 'TryStatement', 'CatchClause',
    'MethodInvocation', 'MemberReference', 'BinaryOperation', 'Assignment',
    'ClassCreator', 'ArrayCreator', 'Literal', 'ReferenceType', 'BasicType',
    'FormalParameter', 'StatementExpression', 'BlockStatement',
}

_PATH_SPLIT_RE = re.compile(r'[/\\.\-_]')


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize_text(text: str) -> List[str]:
    """Tokenise text by extracting alphanumeric words and numbers (lowercase).

    Args:
        text: Raw text (bug report, code identifier, etc.).
    Returns:
        List of lowercase token strings.
    """
    return re.findall(r'[a-zA-Z_]\w*|\d+', text.lower())


def encode_tokens(tokens: List[str], vocab: dict, max_len: int) -> Tuple[List[int], int]:
    """Convert a token list to a padded integer-ID list.

    Args:
        tokens : List of string tokens.
        vocab  : {token: int_id} with '<PAD>'=0 and '<UNK>'=1.
        max_len: Maximum sequence length; truncates and pads to this length.
    Returns:
        (ids, actual_length) where ids is len max_len (padded with 0) and
        actual_length is min(len(tokens), max_len) clamped to >= 1.
    """
    unk = vocab.get('<UNK>', 1)
    truncated = tokens[:max_len]
    actual_length = max(len(truncated), 1)
    ids = [vocab.get(t, unk) for t in truncated]
    ids = ids + [0] * (max_len - len(ids))
    return ids, actual_length


def encode_name_tokens(name: str, vocab: dict) -> List[int]:
    """Tokenise a file or directory name segment and encode to vocab IDs.

    Splits on path separators, underscores, dots, hyphens, then tokenises.

    Args:
        name : File or directory name string.
        vocab: {token: int_id}.
    Returns:
        List of int IDs (may be empty if name has no recognisable tokens).
    """
    parts = [p for p in _PATH_SPLIT_RE.split(name.lower()) if p]
    unk = vocab.get('<UNK>', 1)
    ids: List[int] = []
    for part in parts:
        sub_tokens = re.findall(r'[a-zA-Z_]\w*|\d+', part)
        for t in sub_tokens:
            ids.append(vocab.get(t, unk))
    return ids


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

def build_vocab(
    token_lists: List[List[str]],
    max_size: int = MAX_VOCAB_SIZE,
    min_freq: int = MIN_FREQ,
    extra_tokens: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """Build a frequency-based vocabulary from token lists.

    Reserved IDs:
        0 → '<PAD>'
        1 → '<UNK>'
        2+ → tokens in descending frequency order.

    Tokens in extra_tokens are always included regardless of frequency.

    Args:
        token_lists : Iterable of token lists.
        max_size    : Maximum total vocabulary size (including PAD + UNK).
        min_freq    : Minimum frequency threshold (applied to non-extra tokens).
        extra_tokens: Set of tokens to always include.
    Returns:
        {token: int_id} dict.
    """
    counter: collections.Counter = collections.Counter()
    for tokens in token_lists:
        counter.update(tokens)

    forced: Set[str] = set(extra_tokens) if extra_tokens else set()

    vocab: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
    idx = 2

    # Add forced tokens first (regardless of frequency)
    for token in sorted(forced):
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    # Add remaining tokens by descending frequency, respecting min_freq
    for token, freq in counter.most_common():
        if len(vocab) >= max_size:
            break
        if token in vocab:
            continue
        if freq < min_freq:
            break
        vocab[token] = idx
        idx += 1

    return vocab


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def load_project_databases(
    project_name: str,
    meta_df: pd.DataFrame,
) -> Tuple[dict, dict, str]:
    """Load pre-computed bug and blob embedding databases.

    Args:
        project_name: Repository name (may contain '/').
        meta_df     : Project metadata DataFrame with 'repo_name' and 'language'.
    Returns:
        (bug_db, blob_db, language)
    """
    language = meta_df[meta_df['repo_name'] == project_name]['language'].iloc[0]
    safe_name = project_name.replace('/', '_')
    blob_db_path = os.path.join(BLOB_DB_DIR, f'{safe_name}_blob_embeddings32.pkl')
    bug_db_path = os.path.join(BUG_METADATA_DIR, f'{safe_name}_bug_metadata32.pkl')
    with open(blob_db_path, 'rb') as f:
        blob_db = pickle.load(f)
    with open(bug_db_path, 'rb') as f:
        bug_db = pickle.load(f)
    return bug_db, blob_db, language


def get_project_path(repo_name: str, language: str) -> str:
    """Return the local filesystem path to a cloned git repository."""
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace('/', '_'))


def get_path_to_sha_map(repo: git.Repo, commit_sha: str) -> Dict[str, str]:
    """Build {file_path: blob_sha} for a commit snapshot. Returns {} on failure."""
    try:
        commit = repo.commit(commit_sha)
        return {blob.path: blob.hexsha for blob in commit.tree.traverse() if blob.type == 'blob'}
    except Exception:
        return {}


def get_blob_content(repo: git.Repo, blob_sha: str) -> str:
    """Fetch raw source code from a git blob SHA. Returns '' on any failure.

    Args:
        repo    : gitpython Repo object.
        blob_sha: Hexadecimal blob SHA string.
    Returns:
        Source code as a UTF-8 string (errors='ignore').
    """
    try:
        blob = Blob(repo, hex_to_bin(blob_sha))
        return blob.data_stream.read().decode('utf-8', errors='ignore')
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Vocab token collection (from bug reports only — fast)
# ---------------------------------------------------------------------------

def collect_vocab_tokens(
    project_name: str,
    language: str,
    bug_db: dict,
    bug_ids: List,
    bug_reports_df: pd.DataFrame,
) -> List[List[str]]:
    """Collect token lists from bug report texts for vocabulary building.

    Only processes bug report text (not AST) to keep vocab building fast.

    Args:
        project_name  : Repository name (unused but kept for interface consistency).
        language      : Programming language (unused).
        bug_db        : {bug_id: metadata} — used to verify which IDs exist.
        bug_ids       : List of bug IDs to process.
        bug_reports_df: DataFrame with 'bug_id' and 'bug_report_text' columns.
    Returns:
        List of token lists (one per valid bug report found).
    """
    token_lists: List[List[str]] = []
    for bug_id in bug_ids:
        rows = bug_reports_df[bug_reports_df['bug_id'].astype(str) == str(bug_id)]
        if rows.empty:
            continue
        text = str(rows.iloc[0].get('bug_report_text', '') or '')
        if text:
            token_lists.append(tokenize_text(text))
    return token_lists


# ---------------------------------------------------------------------------
# FAISS-based labeled sample generation
# ---------------------------------------------------------------------------

def create_labeled_samples(
    project_name: str,
    language: str,
    bug_metadata_db: dict,
    blob_embedding_db: dict,
    top_k: int,
    bug_ids_to_process: Optional[List] = None,
) -> List[Tuple]:
    """Generate (bug_id, blob_sha, label, file_path) tuples for labeled bugs.

    Uses per-commit FAISS to retrieve top_k candidates from BGE embeddings.
    BGE is used ONLY for FAISS filtering — it is NOT used as model input.
    Ground-truth files are force-included so positive training samples exist.

    Args:
        project_name      : Repository name.
        language          : Programming language.
        bug_metadata_db   : {bug_id: {'embedding': np.array, 'commit_sha': str,
                             'ground_truth_files': [str, ...]}}
        blob_embedding_db : {blob_sha: np.array(1536,)}
        top_k             : FAISS candidate count per bug.
        bug_ids_to_process: Subset of IDs; None means all bugs in db.
    Returns:
        List of (bug_id, blob_sha, label, file_path) tuples.
    """
    repo_path = get_project_path(project_name, language)
    repo = git.Repo(repo_path)

    if bug_ids_to_process is None:
        bug_ids_to_process = list(bug_metadata_db.keys())

    filtered_bug_db = {
        bid: bug_metadata_db[bid]
        for bid in bug_ids_to_process
        if bid in bug_metadata_db
    }

    bugs_by_commit: Dict[str, List] = {}
    for bug_id, meta in filtered_bug_db.items():
        sha = meta['commit_sha']
        bugs_by_commit.setdefault(sha, []).append(bug_id)

    labeled_samples: List[Tuple] = []

    for commit_sha, bug_ids in tqdm(bugs_by_commit.items(), desc=f'Labeled samples [{project_name}]'):
        path_to_sha_map = get_path_to_sha_map(repo, commit_sha)
        if not path_to_sha_map:
            continue

        sha_to_path: Dict[str, str] = {}
        for p, s in path_to_sha_map.items():
            sha_to_path.setdefault(s, p)

        snapshot_shas = list(path_to_sha_map.values())
        valid_shas = [s for s in snapshot_shas if s in blob_embedding_db]
        if not valid_shas:
            continue

        snapshot_embs = np.array(
            [blob_embedding_db[s] for s in valid_shas], dtype='float32'
        )
        faiss.normalize_L2(snapshot_embs)
        index = faiss.IndexFlatIP(snapshot_embs.shape[1])
        index.add(snapshot_embs)

        for bug_id in bug_ids:
            bug_info = bug_metadata_db[bug_id]
            bug_emb = bug_info['embedding'].astype('float32')
            query = np.expand_dims(bug_emb, 0)
            faiss.normalize_L2(query)

            k = min(top_k, len(valid_shas))
            _, idxs = index.search(query, k)
            candidate_shas: Set[str] = {valid_shas[i] for i in idxs[0]}

            gt_shas: Set[str] = {
                sha
                for path, sha in path_to_sha_map.items()
                if any(path.endswith(gt) for gt in bug_info.get('ground_truth_files', []))
            }

            # Force-include ground truth for training
            candidate_shas.update(gt_shas)

            for blob_sha in candidate_shas:
                label = 1 if blob_sha in gt_shas else 0
                file_path = sha_to_path.get(blob_sha, '')
                labeled_samples.append((bug_id, blob_sha, label, file_path))

    return labeled_samples


# ---------------------------------------------------------------------------
# Directory tree builder (for Generator)
# ---------------------------------------------------------------------------

def build_directory_tree(
    repo: git.Repo,
    commit_sha: str,
    vocab: dict,
) -> Optional[dict]:
    """Build a directory tree dict for Generator traversal.

    The tree is keyed by path strings. The root node has key ''.
    Each value contains:
        name        : Last path segment (directory or file name).
        name_tokens : List[int] vocab IDs for the name.
        is_leaf     : True if this node is a file (blob).
        sha         : Blob SHA if is_leaf else None.
        children    : List of child path strings.

    Limits to MAX_TREE_FILES leaf files to prevent memory issues.

    Args:
        repo      : gitpython Repo.
        commit_sha: Commit SHA string.
        vocab     : Vocabulary dict for encoding name tokens.
    Returns:
        Tree dict or None if commit cannot be resolved.
    """
    try:
        commit = repo.commit(commit_sha)
    except Exception:
        return None

    tree: dict = {
        '': {
            'name': '',
            'name_tokens': [],
            'is_leaf': False,
            'sha': None,
            'children': [],
        }
    }

    leaf_count = 0

    for item in commit.tree.traverse():
        if leaf_count >= MAX_TREE_FILES:
            break

        path_str: str = item.path

        # Ensure all ancestor directories exist in tree
        parts = path_str.split('/')
        for depth in range(len(parts)):
            ancestor_path = '/'.join(parts[:depth]) if depth > 0 else ''
            child_path = '/'.join(parts[:depth + 1])

            if child_path not in tree:
                child_name = parts[depth]
                is_leaf = (depth == len(parts) - 1) and (item.type == 'blob')
                sha_val = item.hexsha if is_leaf else None

                tree[child_path] = {
                    'name': child_name,
                    'name_tokens': encode_name_tokens(child_name, vocab),
                    'is_leaf': is_leaf,
                    'sha': sha_val,
                    'children': [],
                }

                # Register as child of ancestor
                if ancestor_path in tree and child_path not in tree[ancestor_path]['children']:
                    tree[ancestor_path]['children'].append(child_path)

            if depth == len(parts) - 1 and item.type == 'blob':
                leaf_count += 1

    return tree


# ---------------------------------------------------------------------------
# AST graph helpers
# ---------------------------------------------------------------------------

def _get_cached_raw_ast(blob_sha: str, cache_dir: str) -> Optional[Data]:
    """Load a cached raw AST (with node_tokens strings) from disk."""
    cache_path = os.path.join(cache_dir, f'{blob_sha}_ast.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _save_raw_ast(raw_graph: Data, blob_sha: str, cache_dir: str) -> None:
    """Save a raw AST (with node_tokens strings) to disk cache."""
    cache_path = os.path.join(cache_dir, f'{blob_sha}_ast.pkl')
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(raw_graph, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------------

class LabeledBLGANDataset(Dataset):
    """Labeled (bug, code file) pairs for Discriminator training.

    Each item returns:
        bug_token_ids : LongTensor (MAX_BUG_LEN,)
        bug_len       : LongTensor scalar
        ast_graph     : torch_geometric Data with x=LongTensor(N,) and edge_index
        path_ids      : LongTensor (MAX_PATH_LEN,)
        path_len      : LongTensor scalar
        label         : FloatTensor scalar

    AST graphs are parsed from source code lazily and cached on disk as
    vocab-independent raw graphs (node_tokens = strings).  Vocab encoding
    to integer IDs is done at __getitem__ time.

    Args:
        samples          : List of (bug_id, blob_sha, label, file_path).
        bug_reports_df   : DataFrame with 'bug_id' and 'bug_report_text'.
        bug_metadata_db  : {bug_id: {'commit_sha': str, ...}}
        blob_embedding_db: {blob_sha: np.array} — used ONLY to check existence.
        repo             : gitpython Repo for fetching blob content.
        language         : Programming language ('python' or 'java').
        parser           : BaseASTParser instance for the language.
        vocab            : {token: int_id}.
        cache_dir        : Directory for on-disk AST cache.
        bge_dim          : BGE dimension (unused by model, kept for compat).
    """

    def __init__(
        self,
        samples: List[Tuple],
        bug_reports_df: pd.DataFrame,
        bug_metadata_db: dict,
        blob_embedding_db: dict,
        repo: git.Repo,
        language: str,
        parser,
        vocab: Dict[str, int],
        cache_dir: str,
        bge_dim: int = 1536,
    ):
        self.samples = samples
        self.bug_reports_df = bug_reports_df
        self.bug_db = bug_metadata_db
        self.blob_db = blob_embedding_db
        self.repo = repo
        self.language = language
        self.parser = parser
        self.vocab = vocab
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        # In-memory bug text tokenisation cache (keyed by str(bug_id))
        self._bug_token_cache: Dict[str, Tuple[List[int], int]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_bug_tokens(self, bug_id) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (bug_token_ids LongTensor(MAX_BUG_LEN,), bug_len scalar)."""
        key = str(bug_id)
        if key not in self._bug_token_cache:
            rows = self.bug_reports_df[self.bug_reports_df['bug_id'].astype(str) == key]
            text = ''
            if not rows.empty:
                text = str(rows.iloc[0].get('bug_report_text', '') or '')
            tokens = tokenize_text(text) if text else []
            ids, length = encode_tokens(tokens, self.vocab, MAX_BUG_LEN)
            self._bug_token_cache[key] = (ids, length)
        ids, length = self._bug_token_cache[key]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

    def _get_ast_graph(self, blob_sha: str, file_path: str) -> Data:
        """Return a vocab-encoded AST graph for the given blob."""
        # Try on-disk cache first
        raw_graph = _get_cached_raw_ast(blob_sha, self.cache_dir)

        if raw_graph is None:
            # Fetch source code and parse
            code = get_blob_content(self.repo, blob_sha)
            raw_graph = self.parser.parse(code) if code else Data(
                node_tokens=[],
                edge_index=torch.tensor([[], []], dtype=torch.long),
            )
            _save_raw_ast(raw_graph, blob_sha, self.cache_dir)

        # Encode to integer IDs using current vocab
        return encode_ast_with_vocab(raw_graph, self.vocab, MAX_AST_NODES)

    def __getitem__(self, idx: int):
        bug_id, blob_sha, label, file_path = self.samples[idx]

        bug_ids_t, bug_len_t = self._get_bug_tokens(bug_id)

        ast_graph = self._get_ast_graph(blob_sha, file_path)

        path_tokens = re.findall(r'[a-zA-Z_]\w*|\d+', file_path.lower())
        path_ids, path_len = encode_tokens(path_tokens, self.vocab, MAX_PATH_LEN)
        path_ids_t = torch.tensor(path_ids, dtype=torch.long)
        path_len_t = torch.tensor(path_len, dtype=torch.long)

        label_t = torch.tensor(float(label), dtype=torch.float32)

        return bug_ids_t, bug_len_t, ast_graph, path_ids_t, path_len_t, label_t


def collate_labeled(batch: List) -> Tuple:
    """Custom collate_fn for LabeledBLGANDataset.

    Stacks tensors and uses torch_geometric Batch for AST graphs.

    Returns:
        (bug_ids, bug_lens, graph_batch, path_ids, path_lens, labels)
    """
    from torch_geometric.data import Batch as TGBatch

    bug_ids = torch.stack([x[0] for x in batch])
    bug_lens = torch.stack([x[1] for x in batch])
    graphs = TGBatch.from_data_list([x[2] for x in batch])
    path_ids = torch.stack([x[3] for x in batch])
    path_lens = torch.stack([x[4] for x in batch])
    labels = torch.stack([x[5] for x in batch])
    return bug_ids, bug_lens, graphs, path_ids, path_lens, labels


# ---------------------------------------------------------------------------
# Unlabeled dataset
# ---------------------------------------------------------------------------

class UnlabeledBLGANDataset(Dataset):
    """Unlabeled bugs for Generator training.

    Each item returns:
        bug_token_ids : LongTensor (MAX_BUG_LEN,)
        bug_len       : LongTensor scalar
        commit_sha    : str
        bug_id        : str

    Directory trees for Generator traversal are built/cached in the training
    loop — NOT in the dataset — to avoid large memory footprints at construction.

    Args:
        bug_ids       : List of bug IDs (unlabeled).
        bug_reports_df: DataFrame with 'bug_id' and 'bug_report_text'.
        bug_metadata_db: {bug_id: {'commit_sha': str, ...}}
        vocab         : {token: int_id}.
    """

    def __init__(
        self,
        bug_ids: List,
        bug_reports_df: pd.DataFrame,
        bug_metadata_db: dict,
        vocab: Dict[str, int],
    ):
        self.vocab = vocab
        self.items: List[Tuple] = []  # (bug_token_ids, bug_len, commit_sha, str_bug_id)

        for bug_id in bug_ids:
            key = str(bug_id)
            if bug_id not in bug_metadata_db:
                continue
            meta = bug_metadata_db[bug_id]
            commit_sha = meta.get('commit_sha', '')

            rows = bug_reports_df[bug_reports_df['bug_id'].astype(str) == key]
            text = ''
            if not rows.empty:
                text = str(rows.iloc[0].get('bug_report_text', '') or '')
            tokens = tokenize_text(text) if text else []
            ids, length = encode_tokens(tokens, vocab, MAX_BUG_LEN)

            bug_ids_t = torch.tensor(ids, dtype=torch.long)
            bug_len_t = torch.tensor(length, dtype=torch.long)
            self.items.append((bug_ids_t, bug_len_t, commit_sha, key))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple:
        return self.items[idx]

    @staticmethod
    def collate_fn(batch: List) -> Tuple:
        """Collate fn that handles string fields by keeping them as lists.

        Returns:
            (bug_ids_t, bug_lens_t, list_of_commit_shas, list_of_bug_ids)
        """
        bug_ids_t = torch.stack([x[0] for x in batch])
        bug_lens_t = torch.stack([x[1] for x in batch])
        commit_shas = [x[2] for x in batch]
        bug_ids_str = [x[3] for x in batch]
        return bug_ids_t, bug_lens_t, commit_shas, bug_ids_str
