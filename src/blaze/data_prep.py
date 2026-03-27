"""
BLAZE data preparation utilities.

Handles:
  - Dynamic AST-aware code chunking (DynamicCodeSplitter) with line-based fallback.
  - Loading pre-built blob/bug databases.
  - Building contrastive training triples: (issue_id_int, code_chunk_text, bug_text).
  - BlazeDataset for the training DataLoader.
"""

import os
import random
import pickle
from typing import Dict, List, Optional, Tuple

import git
import numpy as np
import pandas as pd
from git import Blob
from git.util import hex_to_bin
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths (all hardcoded, consistent with tranp_cnn / cooba)
# ---------------------------------------------------------------------------
REPO_BASE_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/repos"
BUG_METADATA_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
BLOB_DB_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs"
PROJECTS_METADATA_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/project_metadata.parquet"
BUG_REPORTS_PATH = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/bug_reports_clean.parquet"
BLAZE_CACHE_DIR = "/home/cs21d002_eashaan/PhD/Objective1/data/processed/blaze_cache"

# Chunker settings (mirrors BLAZE paper defaults)
CHUNK_MAX_WINDOW_SIZE = 20   # max lines per chunk (~512 tokens at ~25 chars/line)
CHUNK_GENERAL_COST = 10      # cost for splitting at non-AST boundary
CHUNK_FALLBACK_LINES = 60    # lines per chunk when tree-sitter is unavailable

# AST node types and their split costs (lower cost = preferred split point)
_CLASS_TYPES = {
    "class_definition", "class_specifier", "class_declaration",
    "enum_specifier", "interface_declaration", "struct_type",
}
_FUNC_TYPES = {
    "function_definition", "function_declaration",
    "generator_function_declaration", "method_definition",
}
_TYPES_OF_INTEREST = _CLASS_TYPES | _FUNC_TYPES
_TYPE_COST = {t: 3 for t in _CLASS_TYPES}
_TYPE_COST.update({t: 5 for t in _FUNC_TYPES})

# tree_sitter language tag by project language
_LANG_MAP = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "c++": "cpp",
    "go": "go",
    "kotlin": "kotlin",
}


# ---------------------------------------------------------------------------
# Dynamic Code Splitter (ported from blaze/utils/processor.py, no llama_index)
# ---------------------------------------------------------------------------

class DynamicCodeSplitter:
    """
    AST-aware code splitter using dynamic programming to find split points
    at natural code boundaries (classes, functions) as described in BLAZE.

    Falls back to fixed line-count chunking if tree_sitter_languages is
    not installed or the file cannot be parsed.
    """

    def __init__(
        self,
        language: str,
        general_cost: int = CHUNK_GENERAL_COST,
        max_window_size: int = CHUNK_MAX_WINDOW_SIZE,
    ):
        self.language = language
        self.general_cost = general_cost
        self.max_window_size = max_window_size

    # ---- core DP split -------------------------------------------------------

    @staticmethod
    def _min_cost_split(
        splits: List[Tuple[int, int]],
        max_length: int,
        max_window_size: int,
        general_cost: int,
    ) -> Tuple[List[Tuple[int, int]], int]:
        """DP minimising continuity-loss cost for a sequence of split candidates."""
        split_cost = {point: cost for point, cost in splits}
        dp = [(0, -1)] * (max_length + 1)  # (accumulated_cost, prev_breakpoint)

        for i in range(1, max_length + 1):
            min_cost, best_j = float("inf"), -1
            for j in range(max(i - max_window_size, 0), i):
                cost = dp[j][0] + split_cost.get(i, general_cost)
                if cost < min_cost:
                    min_cost, best_j = cost, j
            dp[i] = (min_cost, best_j)

        breakpoints, current = [], max_length
        total_cost = dp[max_length][0]
        while current > 0:
            prev = dp[current][1]
            breakpoints.append((prev, current - 1))
            current = prev
        breakpoints.reverse()
        return breakpoints, total_cost

    def _extract_nodes(self, node, result=None, depth=0):
        """Recursively collect AST nodes of interest with their line numbers."""
        if result is None:
            result = []
        if depth > 2000:
            return result
        if node.type in _TYPES_OF_INTEREST and "lambda" not in node.text.decode("utf-8", "ignore"):
            line_no = node.start_point[0] + 1  # 1-indexed
            result.append((line_no, _TYPE_COST[node.type]))
        for child in node.children:
            self._extract_nodes(child, result, depth + 1)
        return result

    def _find_chunks(self, root_node, text: str) -> List[str]:
        lines = text.split("\n")
        splits = self._extract_nodes(root_node)
        breakpoints, _ = self._min_cost_split(
            splits, max_length=len(lines),
            max_window_size=self.max_window_size,
            general_cost=self.general_cost,
        )
        return ["\n".join(lines[start:end + 1]).strip() for start, end in breakpoints]

    # ---- public API ----------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """Split source code into chunks. Returns non-empty chunks."""
        try:
            import tree_sitter_languages  # optional dependency
        except ImportError:
            return _fallback_chunk(text, CHUNK_FALLBACK_LINES)

        try:
            parser = tree_sitter_languages.get_parser(self.language)
        except Exception:
            return _fallback_chunk(text, CHUNK_FALLBACK_LINES)

        tree = parser.parse(bytes(text, "utf-8"))
        if tree.root_node.children and tree.root_node.children[0].type == "ERROR":
            return _fallback_chunk(text, CHUNK_FALLBACK_LINES)

        chunks = [c for c in self._find_chunks(tree.root_node, text) if c]
        return chunks if chunks else _fallback_chunk(text, CHUNK_FALLBACK_LINES)


def _fallback_chunk(text: str, lines_per_chunk: int) -> List[str]:
    """Split text into fixed-size line chunks when AST parsing is unavailable."""
    lines = text.splitlines()
    chunks = []
    for i in range(0, max(len(lines), 1), lines_per_chunk):
        chunk = "\n".join(lines[i : i + lines_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text[:2000]]


# ---------------------------------------------------------------------------
# Shared helper functions (mirrors tranp_cnn / cooba)
# ---------------------------------------------------------------------------

def get_project_path(repo_name: str, language: str) -> str:
    return os.path.join(REPO_BASE_PATH, language.lower(), repo_name.replace("/", "_"))


def get_path_to_sha_map(repo: git.Repo, commit_sha: str) -> Dict[str, str]:
    """Map file_path → blob_sha for every source file at the given commit."""
    try:
        commit = repo.commit(commit_sha)
        return {
            blob.path: blob.hexsha
            for blob in commit.tree.traverse()
            if blob.type == "blob"
        }
    except Exception:
        return {}


def load_project_databases(project_name: str, meta_df: pd.DataFrame):
    """Load pre-computed blob and bug embedding databases for a project."""
    language = meta_df[meta_df["repo_name"] == project_name]["language"].iloc[0]
    blob_db_path = os.path.join(
        BLOB_DB_DIR, project_name.replace("/", "_") + "_blob_embeddings32.pkl"
    )
    bug_db_path = os.path.join(
        BUG_METADATA_DIR, project_name.replace("/", "_") + "_bug_metadata32.pkl"
    )
    with open(blob_db_path, "rb") as f:
        blob_db = pickle.load(f)
    with open(bug_db_path, "rb") as f:
        bug_db = pickle.load(f)
    return bug_db, blob_db, language


def read_blob_content(repo: git.Repo, blob_sha: str) -> str:
    """Read raw text content of a git blob, ignoring decode errors."""
    try:
        return Blob(repo, hex_to_bin(blob_sha)).data_stream.read().decode("utf-8", "ignore")
    except Exception:
        return ""


def get_file_language(file_path: str, project_language: str) -> str:
    """Infer tree-sitter language tag from file extension, fallback to project language."""
    ext_map = {
        ".py": "python", ".java": "java", ".js": "javascript",
        ".ts": "javascript", ".cpp": "cpp", ".cc": "cpp", ".go": "go", ".kt": "kotlin",
    }
    ext = os.path.splitext(file_path)[1].lower()
    return ext_map.get(ext, _LANG_MAP.get(project_language.lower(), "python"))


# ---------------------------------------------------------------------------
# Contrastive training sample creation
# ---------------------------------------------------------------------------

def create_contrastive_samples(
    project_name: str,
    language: str,
    bug_metadata_db: dict,
    bug_reports_df: pd.DataFrame,
    bug_ids_to_process: List,
    seed: int = 42,
) -> List[Tuple[int, str, str]]:
    """
    Build contrastive training triples for BLAZE.

    For each bug ID:
      1. Look up ground-truth files from bug_metadata_db.
      2. Read blob content from git.
      3. Chunk with DynamicCodeSplitter (one random chunk per ground-truth file).
      4. Yield (issue_id_int, code_chunk_text, bug_text).

    Returns:
        List of (issue_id_int, code_chunk_text, bug_report_text).
        issue_id_int is a consecutive integer assigned per bug_id for NTXent loss.
    """
    random.seed(seed)
    repo_path = get_project_path(project_name, language)
    try:
        repo = git.Repo(repo_path)
    except Exception as e:
        print(f"  [BLAZE] Cannot open repo {repo_path}: {e}")
        return []

    # Build bug text lookup
    bug_texts: Dict = pd.Series(
        bug_reports_df["bug_report_text"].values,
        index=bug_reports_df["bug_id"],
    ).to_dict()

    splitter = DynamicCodeSplitter(language=_LANG_MAP.get(language.lower(), "python"))
    samples: List[Tuple[int, str, str]] = []
    bug_id_to_int: Dict = {}

    filtered_db = {
        bid: bug_metadata_db[bid]
        for bid in bug_ids_to_process
        if bid in bug_metadata_db
    }

    for bug_id, meta in tqdm(filtered_db.items(), desc=f"Creating contrastive samples [{project_name}]"):
        bug_text = bug_texts.get(bug_id, "")
        if not bug_text:
            continue

        commit_sha = meta.get("commit_sha", "")
        if not commit_sha:
            continue

        path_to_sha = get_path_to_sha_map(repo, commit_sha)
        if not path_to_sha:
            continue

        # Map ground-truth file paths → blob SHAs
        gt_shas = set()
        for gt_file in meta.get("ground_truth_files", []):
            for file_path, blob_sha in path_to_sha.items():
                if file_path.endswith(gt_file):
                    gt_shas.add(blob_sha)
                    break

        if not gt_shas:
            continue

        # Assign a stable integer ID for this bug
        if bug_id not in bug_id_to_int:
            bug_id_to_int[bug_id] = len(bug_id_to_int)
        issue_id_int = bug_id_to_int[bug_id]

        for blob_sha in gt_shas:
            content = read_blob_content(repo, blob_sha)
            if not content:
                continue
            chunks = splitter.split_text(content)
            if not chunks:
                continue
            # Pick one random chunk per ground-truth file
            chunk = random.choice(chunks)
            samples.append((issue_id_int, chunk, bug_text))

    return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BlazeDataset(Dataset):
    """
    Minimal dataset returning raw strings.
    The model's internal tokenizer handles tokenization on the fly.

    Each sample: (issue_id_int, code_chunk_text, bug_report_text).
    Returned dict keys: 'bug_text', 'code_text', 'issue_id'.
    """

    def __init__(self, samples: List[Tuple[int, str, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        issue_id, code_text, bug_text = self.samples[idx]
        return {
            "bug_text": bug_text,
            "code_text": code_text,
            "issue_id": issue_id,
        }
